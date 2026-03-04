import torch
from torch import nn
from torch.nn import Module
from pytorch_custom_utils import save_load
from beartype.typing import Union, Tuple
from model.custom_transformers_training import FlexAttentionTransformers as Transformers
from fns import dequantize_verts_tensor
import math
from model.pc_encoder import CloudEncoder

def get_positional_encoding(L, D, device='cpu'):
    # Create a tensor to hold the positional encodings
    position = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / D))
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

@save_load()
class TreeMeshGPT(Module):
    def __init__(
        self,
        *,
        dim: Union[int, Tuple[int, int]] = 1024,
        flash_attn = True,
        attn_depth = 24,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
        ),
        dropout = 0.,
        quant_bit = 7,
        pad_id = -1,
    ):
        super().__init__()

        self.quant_bit = quant_bit
        self.dim = dim

        self.sos_emb = nn.Parameter(torch.randn(dim))
        self.fc_edges = nn.Linear(1024, dim)
        self.sos_emb_2 = nn.Parameter(torch.randn(512))
        self.fc_edges_2 = nn.Linear(1024, dim)
        
        self.decoder = Transformers(
            dim = dim,
            depth = attn_depth,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )
        
        self.head_coord1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit+2)
        )
        
        self.coord1_emb = nn.Embedding(2**quant_bit, dim)
        self.head_coord2 = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit)
        )
        
        self.coord2_emb = nn.Embedding(2**quant_bit, dim)
        self.head_coord3 = nn.Sequential(
            nn.Linear(dim*3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2**quant_bit)
        )

        self.pad_id = pad_id
        self.pc_encoder = CloudEncoder()
        self.pc_adapter = nn.Linear(64, dim)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        vertices,
        edges,
        gt_ind,
        pc,
        **kwargs
    ):
        
        quant_bit = self.quant_bit
        dim = self.dim
        
        vertices_cont = dequantize_verts_tensor(vertices, n_bits=self.quant_bit)
        
        # construct mask  
        ##########################      
        gt_ind[gt_ind==-2] = -1
        
        indices = (gt_ind != -1).cumsum(dim=1).eq((gt_ind != -1).sum(dim=1, keepdim=True)) & (gt_ind != -1)
        non_zeros = indices.nonzero()
        
        if non_zeros.size(0) == 0:
            a = self.sos_emb * 0
            return a.sum(), a.sum()
        
        last_indices = non_zeros[:, 1].view(gt_ind.size(0))
        
        # assume batch = 1
        eos_ind = int(last_indices+1)        
        gt_ind = gt_ind[:,:eos_ind+1]
        gt_ind[:, -1] = -2
        B, N = gt_ind.shape
        edges = edges[:,:N]
        range_tensor = torch.arange(N, device=vertices.device).unsqueeze(0).expand(B, N)
        mask = range_tensor <= N
        
        # gather gt, split into gt_x, gt_y, gt_z
        gt_ind_copy = gt_ind.clone()
        gt_ind_copy[gt_ind_copy == -1] = 0
        gt_ind_copy[gt_ind_copy == -2] = 0
        _, M, C = vertices.shape  # C = 3 (the last dimension)
        expanded_indices = gt_ind_copy.unsqueeze(-1).expand(B, N, C)
        gt = torch.gather(vertices, dim=1, index=expanded_indices)
        gt[gt_ind==-1] = 2**quant_bit
        gt[gt_ind==-2] = 2**quant_bit + 1
        
        gt_x = gt[:,:,0].clamp(max = 2**quant_bit-1)
        gt_y = gt[:,:,1].clamp(max = 2**quant_bit-1)
        gt_z = gt[:,:,2]
        
        # construct input features (edges_emb)
        edges_emb = torch.zeros(B, N, dim, device = edges.device)
        edge_type = (edges==torch.tensor([-1,-1], device = edges.device)).sum(-1)
        edges_emb[edge_type==2] = self.sos_emb
        
        edges_ind = edges.clone()
        edges_ind = edges_ind.clamp(min = 0)
        index_1 = edges_ind[:, :, 0].unsqueeze(-1).expand(B, N, C)
        gather_1 = torch.gather(vertices_cont, dim=1, index=index_1)
        
        gather_1 = self.pc_encoder.point_embed(gather_1)        
        index_2 = edges_ind[:, :, 1].unsqueeze(-1).expand(B, N, C)
        gather_2 = torch.gather(vertices_cont, dim=1, index=index_2)
        gather_2 = self.pc_encoder.point_embed(gather_2)
        
        init_edges = torch.cat([gather_1, gather_2], dim=-1)
        edges_emb_seq = self.fc_edges(init_edges).float()
        
        edges_emb[edge_type==0] = edges_emb_seq[edge_type==0]
        edges_2 = init_edges[edge_type==1][:,:512]
        edges_2_fea = torch.cat([edges_2, self.sos_emb_2.unsqueeze(0).expand(edges_2.shape[0], -1)], dim=-1)
        edges_emb[edge_type==1] = self.fc_edges_2(edges_2_fea).float()        
        ####################################################
        
        pc_embed = self.pc_encoder(pc.float())
        l2_norm_loss = torch.mean(torch.sum(pc_embed ** 2, dim=-1))
        
        pc_embed = self.pc_adapter(pc_embed)
        prefix_len = pc_embed.shape[1]
        
        B, L, D = edges_emb.shape
        pos_encoding = get_positional_encoding(L, D, device = edges_emb.device)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1) 
        edges_emb = edges_emb + pos_encoding
        
        start_emb = torch.cat([pc_embed, edges_emb], dim=1)       
        res = self.decoder(start_emb)
        res = res[:,prefix_len:]
        
        out_z = self.head_coord1(res)
        
        z_force_inp = gt_z.clamp(max=2**quant_bit-1)
        z_force = self.coord1_emb(z_force_inp)
        inp_y = torch.cat([res, z_force], dim=-1)
        out_y = self.head_coord2(inp_y)
        
        y_force_inp = gt_y.clamp(max=2**quant_bit-1)
        y_force = self.coord2_emb(y_force_inp)
        inp_x = torch.cat([res, z_force, y_force], dim=-1)
        out_x = self.head_coord3(inp_x)
        
        coord_mask = gt_z < 2**quant_bit
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        res_z = out_z.view(B * N, 2**quant_bit+2)
        target_z = gt_z.view(B * N).long()
        loss_z = criterion(res_z, target_z)
        loss_z = loss_z * mask.view(-1)
        loss_z = loss_z.sum() / mask.sum()

        res_y = out_y.view(B*N, 2**quant_bit)
        target_y = gt_y.view(B * N).long()
        loss_y = criterion(res_y, target_y)
        loss_y = loss_y * mask.view(-1)
        loss_y = loss_y * coord_mask.view(-1)
        loss_y = loss_y.sum() / (mask & coord_mask).sum()

        res_x = out_x.view(B*N, 2**quant_bit)
        target_x = gt_x.view(B * N).long()
        loss_x = criterion(res_x, target_x)
        loss_x = loss_x * mask.view(-1)
        loss_x = loss_x * coord_mask.view(-1)
        loss_x = loss_x.sum() / (mask & coord_mask).sum()

        loss_ce = loss_x + loss_y + loss_z
        
        return loss_ce, l2_norm_loss
