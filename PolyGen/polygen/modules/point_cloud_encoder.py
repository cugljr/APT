from typing import List, Tuple

import torch
import torch.nn as nn


class APESPointCloudEncoder(nn.Module):
    """APES-inspired point-cloud encoder for conditional generation.

    The module first embeds per-point features, applies self-attention,
    then performs attention-score-based downsampling to produce context tokens
    for decoder cross-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        num_context_tokens: int = 256,
        knn_scales: Tuple[int, ...] = (8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        super(APESPointCloudEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_context_tokens = num_context_tokens
        self.knn_scales = tuple(sorted(knn_scales))
        self.max_k = max(self.knn_scales)

        self.point_embed = nn.Sequential(
            nn.Linear(3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.pre_norm = nn.LayerNorm(hidden_size)

        self.local_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in self.knn_scales
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * (1 + len(self.knn_scales)), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.score_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in self.knn_scales])

        self.token_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.self_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @staticmethod
    def _gather_neighbors(feats: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = feats.shape
        k = knn_idx.shape[-1]
        batch_idx = torch.arange(batch_size, device=feats.device)[:, None, None].expand(batch_size, num_points, k)
        return feats[batch_idx, knn_idx]

    @staticmethod
    def _token_allocation(total_tokens: int, n_scales: int) -> List[int]:
        if n_scales <= 0:
            return []
        base = total_tokens // n_scales
        rem = total_tokens - base * n_scales
        alloc = [base] * n_scales
        for i in range(rem):
            alloc[i] += 1
        return alloc

    def _build_multiscale_features(self, point_cloud: torch.Tensor, feats: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            dists = torch.cdist(point_cloud, point_cloud)
            knn_all = torch.topk(dists, k=self.max_k + 1, dim=-1, largest=False).indices[..., 1:]

        scale_feats = []
        for k, mlp in zip(self.knn_scales, self.local_mlps):
            knn_idx = knn_all[..., :k]
            neighbors = self._gather_neighbors(feats, knn_idx)
            centers = feats.unsqueeze(2)
            rel = neighbors - centers
            pooled = torch.cat([torch.mean(rel, dim=2), torch.max(rel, dim=2).values], dim=-1)
            local = mlp(pooled)
            scale_feats.append(local)
        return scale_feats

    def _multiscale_tokens(self, scale_feats: List[torch.Tensor]) -> torch.Tensor:
        if len(scale_feats) == 0:
            raise RuntimeError("scale_feats must not be empty.")
        n_scales = len(scale_feats)
        batch_size, num_points, _ = scale_feats[0].shape
        token_alloc = self._token_allocation(min(self.num_context_tokens, num_points), n_scales)
        token_chunks = []
        for feats, score_head, n_tok in zip(scale_feats, self.score_heads, token_alloc):
            if n_tok <= 0:
                continue
            scores = score_head(feats).squeeze(-1)
            _, indices = torch.topk(scores, k=n_tok, dim=1)
            gather_idx = indices.unsqueeze(-1).expand(batch_size, n_tok, self.hidden_size)
            token_chunks.append(torch.gather(feats, dim=1, index=gather_idx))
        if len(token_chunks) == 0:
            return scale_feats[0][:, :1]
        return torch.cat(token_chunks, dim=1)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Encodes point cloud into sequential context embeddings.

        Args:
            point_cloud: Tensor of shape [batch_size, num_points, 3].
        Returns:
            context: Tensor of shape [batch_size, num_context_tokens, hidden_size].
        """
        point_cloud = point_cloud.to(torch.float32)
        base_feats = self.pre_norm(self.point_embed(point_cloud))
        scale_feats = self._build_multiscale_features(point_cloud, base_feats)
        fused = self.fusion(torch.cat([base_feats] + scale_feats, dim=-1))
        fused_attn, _ = self.self_attn1(fused, fused, fused)
        fused = fused + fused_attn

        multi_scale_feats = [fused + s for s in scale_feats]
        tokens = self._multiscale_tokens(multi_scale_feats)
        tokens = self.token_norm(tokens)
        tokens_attn, _ = self.self_attn2(tokens, tokens, tokens)
        context = self.post_norm(tokens + tokens_attn)
        return context
