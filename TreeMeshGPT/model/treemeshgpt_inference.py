import torch
from torch import nn
from torch.nn import Module
from pytorch_custom_utils import save_load
from beartype.typing import Union, Tuple
from einops import pack
from model.custom_transformers_inference import FlashAttentionTransformers as Transformers
from model.custom_transformers_inference import eval_decorator
from fns import dequantize_verts_tensor
import math
import sys
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
        topk = 10,
        max_seq_len = 30000
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
        self.n = 0
        self.topk = topk
        self.max_seq_len = max_seq_len
        
    @eval_decorator
    @torch.no_grad()
    def generate(
        self,
        pc,
        n = 0,
    ):
        
        device = self.sos_emb.device
        self.n = -n        
        
        def add_stack(edges):
            node = {}
            node['edges'] = edges
            stack.append(node)
            
        def initialize_connected_component(edges, acc_fea, pred, p, cache, first, t_init=1):
            
            # Step 0
            fea = self.sos() + pe(p)
            acc_fea = pack([acc_fea, fea], 'b * d')[0]
            xyz_0, eos, cache = self.predict(acc_fea, t=t_init, init_mask=False, first=first, kv_cache=cache)
            pred = pack([pred, xyz_0], 'b * d')[0]
            p += 1
            if eos: return edges, acc_fea, pred, p, cache, eos, first
            
            edges = torch.cat([edges, torch.cat([xyz_0, pad], dim=-1)], dim=0)

            # Step 1
            fea = self.sos1(xyz_0) + pe(p)
            acc_fea = pack([acc_fea, fea], 'b * d')[0]
            xyz_1, eos, cache = self.predict(acc_fea, t=t_init, init_mask=False, first=first, kv_cache=cache)
            pred = pack([pred, xyz_1], 'b * d')[0]
            p += 1
            if eos: return edges, acc_fea, pred, p, cache, eos, first

            first = False  # Ensure first-time flag is reset

            edges = torch.cat([edges, torch.cat([xyz_0, xyz_1], dim=-1)], dim=0)
            add_stack(edges=[xyz_0, xyz_1])

            # Step 2
            fea = self.encode_edge(xyz_0, xyz_1) + pe(p)
            acc_fea = pack([acc_fea, fea], 'b * d')[0]
            xyz_2, eos, cache = self.predict(acc_fea, t=t_init, init_mask=False, kv_cache=cache)
            pred = pack([pred, xyz_2], 'b * d')[0]
            p += 1
            if eos: return edges, acc_fea, pred, p, cache, eos, first

            add_stack(edges=[xyz_2, xyz_0])  # L
            add_stack(edges=[xyz_1, xyz_2])  # R

            return edges, acc_fea, pred, p, cache, eos, first
                    
                
        dim = self.dim  
        pad = torch.tensor([[-1 ,-1, -1]], device = device)
        edges = torch.empty((0, 6), device = device).long()  
        
        edge_pad = torch.cat([pad, pad], dim=-1)
        pred = torch.empty((1, 0, 3), device = device).long()   
        init_pe = get_positional_encoding(30000, 1024, device=device).unsqueeze(0)
        
        acc_fea = torch.empty((1, 0, dim), device = device)
        
        def pe(id):
            return init_pe[:, id][:, None]
        p = 0
        
        eos = False
        
        pc_embed = self.pc_encoder(pc.float())
        pc_embed = self.pc_adapter(pc_embed)
        acc_fea = pack([acc_fea, pc_embed], 'b * d')[0]
        _, cache = self.decoder(acc_fea, return_hiddens = True)
        
        
        ###
        first = True
        max_seq = self.max_seq_len
        
        while eos == False and pred.shape[1] < max_seq:
            
            self.n += n
            stack = [] 
            edges = torch.cat([edges, edge_pad], dim=0)             
            edges, acc_fea, pred, p, cache, eos, first = initialize_connected_component(edges, acc_fea, pred, p, cache, first, t_init = 1)
            if eos:
                break
                        
            while stack and pred.shape[1] < max_seq:
                cur_node = stack.pop()
                cur_edges = torch.cat([cur_node['edges'][1], cur_node['edges'][0]], dim=-1)
                
                prev_faces = torch.cat([edges.unsqueeze(0), pred], dim=-1).reshape(-1, 3, 3)
                face_mask = (prev_faces != -1).all(dim=(1, 2))
                prev_faces = prev_faces[face_mask]
                                
                edges = torch.cat([edges, cur_edges], dim=0)
                fea = self.encode_edge(cur_node['edges'][1], cur_node['edges'][0]) + pe(p)
                acc_fea = pack([acc_fea, fea], 'b * d')[0]            
                    
                te = self.adjust_temperature(len(stack))        
                xyz_res, eos, cache = self.predict(acc_fea, t = te, kv_cache = cache)
                
                if xyz_res.sum() != -3:
                    cur_face = torch.cat([cur_edges, xyz_res], dim=-1).reshape(-1, 3, 3)[0]
                    exists = self.check_duplicate(prev_faces, cur_face)
                    
                    if exists and len(stack) > 0:
                        xyz_res = torch.tensor([-1, -1, -1], device=fea.device).unsqueeze(0)
                    else:
                        tt = 0.5
                        while exists:
                            xyz_res, eos, cache_inloop = self.predict(acc_fea, t = tt, kv_cache = cache)
                            cur_face = torch.cat([cur_edges, xyz_res], dim=-1).reshape(-1, 3, 3)[0]
                            exists = self.check_duplicate(prev_faces, cur_face)
                            tt += 0.1
                            
                            if not exists:
                                cache = cache_inloop
                            
                sys.stdout.write(f"\rSequence length: {pred.shape[1]}/{max_seq} | Stack length: {len(stack):<4}")
                sys.stdout.flush()
                pred = pack([pred, xyz_res], 'b * d')[0]
                p += 1
                
                if xyz_res.sum() != -3 and xyz_res.sum() != -6:
                    add_stack(edges=[xyz_res, cur_node['edges'][1]]) # L
                    add_stack(edges=[cur_node['edges'][0], xyz_res]) # R

                if eos:
                    break
                
        mask1 = ~(pred[0] < 0).any(dim=-1)
        mask2 = ~(edges < 0).any(dim=-1)
        mask = mask1 & mask2
        edges_valid = edges[mask]
        pred_valid = pred[0][mask]
        triangles = torch.cat([edges_valid, pred_valid], dim=-1)
        triangles = triangles.reshape(-1, 3, 3)
        triangles = dequantize_verts_tensor(triangles, n_bits=self.quant_bit)
        
        return triangles

    def sos(self):
        return self.sos_emb.unsqueeze(0).unsqueeze(0)

    def sos1(self, xyz):
        xyz = dequantize_verts_tensor(xyz, n_bits=self.quant_bit).unsqueeze(1)
        fea = torch.cat([self.pc_encoder.point_embed(xyz), self.sos_emb_2.unsqueeze(0).unsqueeze(0)], dim=-1)
        fea = self.fc_edges_2(fea)
        return fea
    
    def encode_edge(self, xyz_0, xyz_1):
        a = dequantize_verts_tensor(xyz_0, n_bits=self.quant_bit).unsqueeze(0)
        b = dequantize_verts_tensor(xyz_1, n_bits=self.quant_bit).unsqueeze(0)
        a = self.pc_encoder.point_embed(a)
        b = self.pc_encoder.point_embed(b)
        c = torch.cat([a, b], dim=-1)
        return self.fc_edges(c)
    
    def predict_xyz(self, res, dequantize=False, top_k=10, temperature=1, init_mask = False, first = False):
        # Get logits from head_x
        logits_z = self.head_coord1(res)            
        logits_z[0][-1] = logits_z[0][-1] + self.n
        logits_z = logits_z / temperature
        
        if init_mask:
            logits_z[0][-2] = -999
        
        if first:
            logits_z[0][-2:] = -999

        # Apply softmax to get probabilities
        probs_z = torch.softmax(logits_z, dim=-1)
        
        # Top-k sampling: Get top-k probabilities and their corresponding indices
        topk_probs_z, topk_indices_z = torch.topk(probs_z, k=top_k, dim=-1)
        if (2**self.quant_bit + 1) in topk_indices_z[:5]:
            self.n += 0.001

        if topk_indices_z[0][0] ==  2**self.quant_bit + 1:
            z = torch.tensor([2**self.quant_bit + 1], device = res.device)
        else:
            mask = topk_indices_z != 2**self.quant_bit + 1
            masked_probs = topk_probs_z * mask.float()
            masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)
            z = topk_indices_z[torch.arange(topk_indices_z.size(0)), torch.multinomial(masked_probs, num_samples=1).squeeze()]
        eos = False

        if z < 2**self.quant_bit:
            emb_z = self.coord1_emb(z)
            inp_y = torch.cat([res, emb_z], dim=-1)

            logits_y = self.head_coord2(inp_y)
            logits_y = logits_y / temperature
            probs_y = torch.softmax(logits_y, dim=-1)
            topk_probs_y, topk_indices_y = torch.topk(probs_y, k=top_k, dim=-1)
            y = topk_indices_y[torch.arange(topk_indices_y.size(0)), torch.multinomial(topk_probs_y, num_samples=1).squeeze()]

            emb_y = self.coord2_emb(y)
            inp_x = torch.cat([res, emb_z, emb_y], dim=-1)

            logits_x = self.head_coord3(inp_x)
            logits_x = logits_x / temperature
            probs_x = torch.softmax(logits_x, dim=-1)
            topk_probs_x, topk_indices_x = torch.topk(probs_x, k=top_k, dim=-1)
            x = topk_indices_x[torch.arange(topk_indices_x.size(0)), torch.multinomial(topk_probs_x, num_samples=1).squeeze()]

            xyz = torch.cat([x,y,z], dim=-1)

            if dequantize:
                xyz = dequantize_verts_tensor(xyz, n_bits=self.quant_bit)

        elif z == 2**self.quant_bit:
            xyz = torch.tensor([-1, -1, -1], device=z.device)
        elif z == 2**self.quant_bit + 1:
            xyz = torch.tensor([-2, -2, -2], device=z.device)
            eos = True

        return xyz, eos
    
    def predict(self, acc_fea, t = 0.1, init_mask = False, first = False, kv_cache = None):
        res, intermediates = self.decoder(acc_fea, cache = kv_cache, return_hiddens = True)
        res = res[0]
        xyz, eos = self.predict_xyz(res, dequantize=False, top_k=self.topk, temperature=t, init_mask=init_mask, first = first)
        return xyz.unsqueeze(0), eos, intermediates
    
    def check_duplicate(self, prev_faces, cur_face):
        rotated_faces = torch.cat([
            prev_faces, 
            prev_faces[:, [1, 2, 0]], 
            prev_faces[:, [2, 0, 1]]
        ], dim=0)
        return (rotated_faces == cur_face).all(dim=(1, 2)).any()
    
    def adjust_temperature(self, stack_size):
        if stack_size < 10:
            return 0.7
        elif stack_size < 100:
            return 0.5
        return 0.2