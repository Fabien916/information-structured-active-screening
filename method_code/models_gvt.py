#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import softmax, to_dense_adj, to_dense_batch

from gvt_gt_pyg_backbone import GraphTransformerNet, MLP
from vq import VectorQuantize


def _pick_num_heads(model_dim: int, prefer: int) -> int:
    for h in [int(prefer), 8, 4, 2, 1]:
        if h > 0 and int(model_dim) % int(h) == 0:
            return int(h)
    return 1


def _scatter_mean_no_ext(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    cnt = torch.zeros((dim_size, 1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt.index_add_(
        0,
        index,
        torch.ones((index.size(0), 1), device=src.device, dtype=src.dtype),
    )
    return out / cnt.clamp_min(1.0)


class VectorQuantizer(nn.Module):
    """VectorQuantize wrapper with project-local stats/output API."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.8,
        eps: float = 1.0e-5,
        threshold_ema_dead_code: float = 2.0,
        use_cosine_sim: bool = True,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sample_codebook_temp: float = 0.0,
        orthogonal_reg_weight: float = 0.0,
        codebook_dim: int | None = None,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.commitment_weight = float(commitment_weight)
        self.vq = VectorQuantize(
            dim=int(code_dim),
            codebook_size=int(codebook_size),
            codebook_dim=None if codebook_dim is None else int(codebook_dim),
            decay=float(decay),
            eps=float(eps),
            kmeans_init=bool(kmeans_init),
            kmeans_iters=int(kmeans_iters),
            use_cosine_sim=bool(use_cosine_sim),
            threshold_ema_dead_code=float(threshold_ema_dead_code),
            commitment_weight=float(commitment_weight),
            orthogonal_reg_weight=float(orthogonal_reg_weight),
            sample_codebook_temp=float(sample_codebook_temp),
        )

    @property
    def embedding(self) -> torch.Tensor:
        return self.vq.codebook

    def get_codebook(self) -> torch.Tensor:
        return self.vq.codebook

    def forward(self, z_e: torch.Tensor):
        quantized, indices, vq_loss, _dist, _embed = self.vq(z_e)
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1.0e-10)))
        usage = torch.count_nonzero(avg_probs > 0).float() / float(self.codebook_size)

        with torch.no_grad():
            loss_commit = F.mse_loss(z_e, quantized.detach())
            loss_commit_weighted = loss_commit * float(self.commitment_weight)

        return quantized, indices, vq_loss, {
            "perplexity": float(perplexity.detach().cpu().item()),
            "usage": float(usage.detach().cpu().item()),
            "loss_commit": float(loss_commit.detach().cpu().item()),
            "loss_commit_weighted": float(loss_commit_weighted.detach().cpu().item()),
            "commitment_weight": float(self.commitment_weight),
            "loss_vq_total": float(vq_loss.detach().cpu().item()),
        }


class Encoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        num_layers: int,
        pe_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = GraphTransformerNet(
            node_dim_in=int(node_in_dim),
            edge_dim_in=int(edge_dim),
            pe_in_dim=int(pe_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(output_dim),
            norm="bn",
            num_gt_layers=int(num_layers),
            num_heads=int(num_heads),
            act="relu",
            dropout=float(dropout),
        )
        self.fusion_layer = nn.Linear(int(output_dim) * 2, int(output_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        pe: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_x, encoded_edge_attr = self.net(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pe=pe,
            batch=batch,
        )
        aggregated_edge_info = _scatter_mean_no_ext(
            encoded_edge_attr, edge_index[1], dim_size=x.size(0)
        )
        fused_node_features = torch.cat([encoded_x, aggregated_edge_info], dim=-1)
        final_node_features = self.fusion_layer(fused_node_features)
        return final_node_features, encoded_edge_attr


class SequenceLanguageEncoder(nn.Module):
    """
    Pure sequence LM-style encoder over node token sequence in each graph.
    Ignores graph edges during encoding and uses causal/bidirectional self-attention.
    """

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.causal = bool(causal)
        self.input_proj = nn.Linear(int(node_in_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        n_heads = _pick_num_heads(int(hidden_dim), int(num_heads))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(n_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.output_proj = nn.Linear(int(hidden_dim), int(output_dim))

    @staticmethod
    def _sinusoidal_pos_enc(seq_len: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / max(1, dim))
        )
        pe = torch.zeros((seq_len, dim), device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        pe: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dense, mask_nodes = to_dense_batch(x, batch=batch)
        seq_len = int(x_dense.size(1))
        h = self.input_proj(x_dense)
        pos = self._sinusoidal_pos_enc(
            seq_len=seq_len,
            dim=int(self.hidden_dim),
            device=h.device,
            dtype=h.dtype,
        ).unsqueeze(0)
        h = self.dropout(h + pos)

        pad_mask = ~mask_nodes
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=h.device, dtype=torch.bool), diagonal=1
            )
        h = self.backbone(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        h = self.output_proj(h)
        node_out = h[mask_nodes]

        edge_out = torch.zeros(
            (int(edge_index.size(1)), int(node_out.size(-1))),
            device=node_out.device,
            dtype=node_out.dtype,
        )
        return node_out, edge_out


class EdgeReconstructor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 64, edge_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(edge_dim) + 1),
        )
        self.edge_dim = int(edge_dim)

    def score_pairs(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_i = embeddings[edge_index[0]]
        node_j = embeddings[edge_index[1]]
        pairs_ij = torch.cat([node_i, node_j], dim=-1)
        pairs_ji = torch.cat([node_j, node_i], dim=-1)
        return (self.mlp(pairs_ij) + self.mlp(pairs_ji)) / 2.0

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_nodes = embeddings.size(0)
        edge_index = torch.nonzero(mask, as_tuple=False).t()
        edge_logits = self.score_pairs(embeddings, edge_index)

        adj_logits = torch.zeros(
            n_nodes,
            n_nodes,
            self.edge_dim + 1,
            device=embeddings.device,
            dtype=edge_logits.dtype,
        )
        adj_logits[:, :, -1] = 1.0
        adj_logits[edge_index[0], edge_index[1]] = edge_logits
        return adj_logits


class RoPE(nn.Module):
    def __init__(self, dim: int, theta_base: float = 10000.0):
        super().__init__()
        if int(dim) % 2 != 0:
            raise ValueError("RoPE dim must be even")
        self.dim = int(dim)
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / float(self.dim))
        )
        self.register_buffer("inv_freq", inv_freq.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        freqs = positions.float().unsqueeze(1) * self.inv_freq
        cos_val = torch.cos(freqs)
        sin_val = torch.sin(freqs)

        half_dim = self.dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        rotated_x1 = x1 * cos_val - x2 * sin_val
        rotated_x2 = x1 * sin_val + x2 * cos_val
        return torch.cat((rotated_x1, rotated_x2), dim=-1)


class ReorderTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.heads = _pick_num_heads(int(dim), int(heads))
        self.head_dim = int(dim) // self.heads
        self.scale = float(self.head_dim) ** -0.5
        self.norm1 = nn.LayerNorm(int(dim))
        self.norm2 = nn.LayerNorm(int(dim))
        self.qkv = nn.Linear(int(dim), int(dim) * 3)
        self.out_proj = nn.Linear(int(dim), int(dim))
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), int(dim) * 4),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim) * 4, int(dim)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, n_nodes, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1).to(dtype=logits.dtype)
        valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
        logits = logits.masked_fill(~valid_pair.unsqueeze(1), -1.0e4)
        attn = torch.softmax(logits, dim=-1)
        attn = attn * valid_pair.unsqueeze(1).to(dtype=attn.dtype)
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        attn = attn / denom
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(bsz, n_nodes, self.dim)
        x = x + self.dropout(self.out_proj(out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class LearnedReorderPositioner(nn.Module):
    """
    Learn permutation-equivariant node scores and output both differentiable
    (soft/ST) permutation and hard permutation indices.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        tau: float = 0.1,
        sinkhorn_iter: int = 8,
        noise_scale: float = 1.0,
        entropy_weight: float = 0.01,
        hard: bool = True,
        scorer: str = "mlp",
        use_rrwp: bool = False,
        rrwp_dim: int = 0,
        gin_layers: int = 3,
        gin_dropout: float = 0.1,
        gt_layers: int = 2,
        gt_heads: int = 4,
        gt_dropout: float = 0.1,
        use_degree: bool = True,
        bandwidth_weight: float = 0.0,
        soft_bandwidth_weight: float = 0.05,
        bandwidth_p: int = 1,
    ):
        super().__init__()
        self.tau = float(tau)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.noise_scale = float(noise_scale)
        self.entropy_weight = float(entropy_weight)
        self.hard = bool(hard)
        self.scorer = str(scorer).strip().lower()
        self.use_rrwp = bool(use_rrwp)
        self.rrwp_dim = int(rrwp_dim)
        self.use_degree = bool(use_degree)
        self.gin_dropout = float(gin_dropout)
        self.gt_dropout = float(gt_dropout)
        self.bandwidth_weight = float(bandwidth_weight)
        self.soft_bandwidth_weight = float(soft_bandwidth_weight)
        self.bandwidth_p = int(bandwidth_p)
        self.noise_dim = 1

        node_dim = int(input_dim)
        if self.use_rrwp:
            node_dim += int(self.rrwp_dim)
        if self.use_degree:
            node_dim += 1
        node_dim += self.noise_dim

        if self.scorer == "gin":
            layers = max(1, int(gin_layers))
            self.gin_layers = nn.ModuleList()
            in_dim = node_dim
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                self.gin_layers.append(GINConv(mlp, train_eps=True))
                in_dim = int(hidden_dim)
            self.score_head = nn.Linear(int(hidden_dim), 1)
            self.score_mlp = None
            self.gt_in = None
            self.gt_blocks = None
            self.gt_adj_bias = None
            self.gt_rrwp_bias = None
        elif self.scorer in {"gt", "transformer"}:
            self.gin_layers = None
            self.score_mlp = None
            self.gt_in = nn.Linear(node_dim, int(hidden_dim))
            self.gt_blocks = nn.ModuleList(
                [
                    ReorderTransformerBlock(
                        dim=int(hidden_dim),
                        heads=int(gt_heads),
                        dropout=float(gt_dropout),
                    )
                    for _ in range(max(1, int(gt_layers)))
                ]
            )
            self.gt_adj_bias = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.gt_rrwp_bias = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.score_head = nn.Linear(int(hidden_dim), 1)
        else:
            self.gin_layers = None
            self.score_head = None
            self.gt_in = None
            self.gt_blocks = None
            self.gt_adj_bias = None
            self.gt_rrwp_bias = None
            self.score_mlp = nn.Sequential(
                nn.Linear(node_dim, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )

    def _soft_perm_from_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(scores.numel())
        if n <= 1:
            eye = torch.eye(n, device=scores.device, dtype=scores.dtype)
            idx = torch.arange(n, device=scores.device, dtype=torch.long)
            return eye, eye, idx

        s = scores.view(n, 1)
        s_sorted, indices = scores.sort(descending=True)
        s_sorted = s_sorted.view(1, n)
        log_perm = -(s - s_sorted).abs() / max(self.tau, 1.0e-6)

        for _ in range(max(0, self.sinkhorn_iter)):
            log_perm = log_perm - torch.logsumexp(log_perm, dim=1, keepdim=True)
            log_perm = log_perm - torch.logsumexp(log_perm, dim=0, keepdim=True)
        perm_soft = torch.exp(log_perm)

        if not self.hard:
            return perm_soft, perm_soft, indices

        with torch.no_grad():
            perm_hard = torch.zeros_like(perm_soft)
            perm_hard[torch.arange(n, device=scores.device), indices] = 1.0
        perm_st = (perm_hard - perm_soft).detach() + perm_soft
        return perm_st, perm_soft, indices

    def _entropy_penalty(self, perm_soft: torch.Tensor) -> torch.Tensor:
        eps = 1.0e-12
        row_prob = perm_soft / perm_soft.sum(dim=1, keepdim=True).clamp_min(eps)
        col_prob = perm_soft / perm_soft.sum(dim=0, keepdim=True).clamp_min(eps)
        row_ent = -(row_prob * row_prob.clamp_min(eps).log()).sum(dim=1).mean()
        col_ent = -(col_prob * col_prob.clamp_min(eps).log()).sum(dim=0).mean()
        return row_ent + col_ent

    def _build_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        rrwp: torch.Tensor | None,
    ) -> torch.Tensor:
        feats = [x]
        if self.use_rrwp and rrwp is not None:
            rrwp_use = rrwp
            if rrwp_use.dim() == 1:
                rrwp_use = rrwp_use.unsqueeze(-1)
            if int(rrwp_use.size(-1)) > int(self.rrwp_dim):
                rrwp_use = rrwp_use[:, : int(self.rrwp_dim)]
            feats.append(rrwp_use.to(device=x.device, dtype=x.dtype))
        if self.use_degree:
            if edge_index is None or edge_index.numel() == 0:
                deg = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
            else:
                deg_raw = torch.bincount(edge_index[0], minlength=x.size(0)).to(x.dtype)
                deg = deg_raw.unsqueeze(-1)
            feats.append(deg)
        if self.training and self.noise_scale > 0.0:
            noise = torch.rand((x.size(0), self.noise_dim), device=x.device, dtype=x.dtype)
            noise = noise * float(self.noise_scale)
        else:
            noise = torch.zeros((x.size(0), self.noise_dim), device=x.device, dtype=x.dtype)
        feats.append(noise)
        return torch.cat(feats, dim=-1)

    def _build_gt_bias(
        self,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None,
        rrwp: torch.Tensor | None,
        max_num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        bias = torch.zeros((num_graphs, max_num_nodes, max_num_nodes), device=device, dtype=dtype)
        if edge_index is not None and edge_index.numel() > 0:
            dense_adj = to_dense_adj(edge_index=edge_index, batch=batch, max_num_nodes=max_num_nodes)
            dense_adj = (dense_adj > 0).to(dtype=dtype)
            bias = bias + dense_adj * self.gt_adj_bias.to(device=device, dtype=dtype)
        if self.use_rrwp and rrwp is not None:
            rrwp_use = rrwp
            if rrwp_use.dim() == 1:
                rrwp_use = rrwp_use.unsqueeze(-1)
            if int(rrwp_use.size(-1)) > int(self.rrwp_dim):
                rrwp_use = rrwp_use[:, : int(self.rrwp_dim)]
            rrwp_use = rrwp_use.to(device=device, dtype=dtype)
            dense_rrwp, dense_mask = to_dense_batch(rrwp_use, batch=batch, max_num_nodes=max_num_nodes)
            rrwp_sim = torch.matmul(dense_rrwp, dense_rrwp.transpose(1, 2))
            rrwp_sim = rrwp_sim / max(int(dense_rrwp.size(-1)), 1)
            pair_mask = dense_mask.unsqueeze(1) & dense_mask.unsqueeze(2)
            rrwp_sim = rrwp_sim * pair_mask.to(dtype=dtype)
            bias = bias + rrwp_sim * self.gt_rrwp_bias.to(device=device, dtype=dtype)
        return bias

    def _score_nodes(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None,
        rrwp: torch.Tensor | None,
    ) -> torch.Tensor:
        node_feat = self._build_features(x, edge_index=edge_index, rrwp=rrwp)
        if self.scorer == "gin":
            if edge_index is None:
                raise ValueError("GIN scorer requires edge_index")
            h = node_feat
            for conv in self.gin_layers:
                h = conv(h, edge_index)
                h = F.relu(h)
                if self.gin_dropout > 0.0 and self.training:
                    h = F.dropout(h, p=self.gin_dropout, training=True)
            return self.score_head(h).squeeze(-1)
        if self.scorer in {"gt", "transformer"}:
            h = self.gt_in(node_feat)
            dense_h, dense_mask = to_dense_batch(h, batch=batch)
            attn_bias = self._build_gt_bias(
                batch=batch,
                edge_index=edge_index,
                rrwp=rrwp,
                max_num_nodes=int(dense_h.size(1)),
                dtype=dense_h.dtype,
                device=dense_h.device,
            )
            encoded = dense_h
            for block in self.gt_blocks:
                encoded = block(encoded, attn_bias=attn_bias, mask=dense_mask)
            return self.score_head(encoded[dense_mask]).squeeze(-1)
        return self.score_mlp(node_feat).squeeze(-1)

    def _positions_from_perm(self, perm: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        count = int(perm.size(0))
        rank_vals = torch.arange(count, device=perm.device, dtype=dtype)
        return (rank_vals.view(count, 1) * perm.to(dtype=dtype)).sum(dim=0)

    def _hard_positions_from_idx(self, perm_idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        count = int(perm_idx.numel())
        out = torch.empty((count,), device=perm_idx.device, dtype=dtype)
        out[perm_idx] = torch.arange(count, device=perm_idx.device, dtype=dtype)
        return out

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        rrwp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, object]]:
        n_nodes = int(x.size(0))
        device = x.device
        dtype = x.dtype
        positions_hard = torch.zeros((n_nodes,), device=device, dtype=dtype)
        perm_idx_list: list[torch.Tensor] = []
        scores_all = self._score_nodes(x, batch=batch, edge_index=edge_index, rrwp=rrwp)

        _g, _uc_ptr, uc_counts = torch.unique_consecutive(
            batch,
            return_inverse=True,
            return_counts=True,
        )
        graph_starts = torch.cat(
            (
                torch.tensor([0], device=device, dtype=uc_counts.dtype),
                uc_counts[:-1].cumsum(0),
            )
        )

        rayleigh_terms: list[torch.Tensor] = []
        numerator_terms: list[torch.Tensor] = []
        denom_terms: list[torch.Tensor] = []
        bandwidth_terms: list[torch.Tensor] = []

        edge_src = edge_index[0] if edge_index is not None and edge_index.numel() > 0 else None
        edge_dst = edge_index[1] if edge_index is not None and edge_index.numel() > 0 else None
        edge_graph = batch[edge_src] if edge_src is not None else None

        for gid in range(int(uc_counts.numel())):
            start = int(graph_starts[gid].item())
            count = int(uc_counts[gid].item())
            end = start + count
            scores = scores_all[start:end]
            tie_break = torch.arange(count, device=device, dtype=dtype) * 1.0e-8
            perm_idx = torch.argsort(scores + tie_break, descending=True)
            positions_hard[start:end] = self._hard_positions_from_idx(perm_idx, dtype=dtype)
            perm_idx_list.append(perm_idx.long())

            if count <= 1 or edge_graph is None:
                rayleigh_terms.append(torch.zeros((), device=device, dtype=dtype))
                numerator_terms.append(torch.zeros((), device=device, dtype=dtype))
                denom_terms.append(torch.ones((), device=device, dtype=dtype))
                bandwidth_terms.append(torch.zeros((), device=device, dtype=dtype))
                continue

            graph_mask = edge_graph == gid
            if int(graph_mask.sum().item()) == 0:
                rayleigh_terms.append(torch.zeros((), device=device, dtype=dtype))
                numerator_terms.append(torch.zeros((), device=device, dtype=dtype))
                denom_terms.append(torch.ones((), device=device, dtype=dtype))
                bandwidth_terms.append(torch.zeros((), device=device, dtype=dtype))
                continue

            local_src = edge_src[graph_mask] - start
            local_dst = edge_dst[graph_mask] - start
            centered = scores - scores.mean()
            numerator = ((centered[local_src] - centered[local_dst]) ** 2).mean()
            denom = centered.pow(2).mean().clamp_min(1.0e-6)
            rayleigh = numerator / denom
            hard_span = (positions_hard[start:end][local_src] - positions_hard[start:end][local_dst]).abs().mean()
            rayleigh_terms.append(rayleigh)
            numerator_terms.append(numerator)
            denom_terms.append(denom)
            bandwidth_terms.append(hard_span)

        rayleigh = torch.stack(rayleigh_terms).mean() if rayleigh_terms else torch.zeros((), device=device, dtype=dtype)
        numerator_mean = torch.stack(numerator_terms).mean() if numerator_terms else torch.zeros((), device=device, dtype=dtype)
        denom_mean = torch.stack(denom_terms).mean() if denom_terms else torch.ones((), device=device, dtype=dtype)
        hard_bandwidth = torch.stack(bandwidth_terms).mean() if bandwidth_terms else torch.zeros((), device=device, dtype=dtype)

        rayleigh_loss = rayleigh * float(self.bandwidth_weight)
        reorder_loss = rayleigh_loss
        return positions_hard, {
            "reorder_loss": reorder_loss,
            "reorder_entropy": torch.zeros((), device=device, dtype=dtype),
            "reorder_bandwidth": hard_bandwidth,
            "reorder_soft_bandwidth": torch.zeros((), device=device, dtype=dtype),
            "reorder_bandwidth_loss": rayleigh_loss,
            "reorder_soft_bandwidth_loss": torch.zeros((), device=device, dtype=dtype),
            "reorder_laplacian": numerator_mean,
            "reorder_laplacian_loss": rayleigh_loss,
            "reorder_spread": denom_mean,
            "reorder_spread_loss": torch.zeros((), device=device, dtype=dtype),
            "reorder_rayleigh": rayleigh,
            "reorder_rayleigh_loss": rayleigh_loss,
            "perm_st_list": [],
            "perm_idx_list": perm_idx_list,
            "uc_counts": uc_counts,
            "scores_all": scores_all,
            "positions_soft": positions_hard,
            "positions_hard": positions_hard,
        }

class EGTLayerSparse(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        if int(hidden_dim) % int(heads) != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim) // int(heads)
        self.dropout = nn.Dropout(float(dropout))
        self.attn_dropout = nn.Dropout(float(dropout))

        self.node_norm1 = nn.LayerNorm(int(hidden_dim))
        self.node_norm2 = nn.LayerNorm(int(hidden_dim))
        self.edge_norm1 = nn.LayerNorm(int(hidden_dim))
        self.edge_norm2 = nn.LayerNorm(int(hidden_dim))

        self.q_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=True)
        self.k_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=True)
        self.v_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=True)
        self.edge_bias_proj = nn.Linear(int(hidden_dim), int(heads), bias=True)
        self.edge_gate_proj = nn.Linear(int(hidden_dim), int(heads), bias=True)
        self.edge_update_proj = nn.Linear(int(heads), int(hidden_dim), bias=True)

        self.node_out_proj = nn.Linear(int(hidden_dim), int(hidden_dim), bias=True)
        self.node_ffn = MLP(
            input_dim=int(hidden_dim),
            output_dim=int(hidden_dim),
            hidden_dims=int(hidden_dim) * 4,
            num_hidden_layers=1,
            dropout=float(dropout),
            act="relu",
        )
        self.edge_ffn = MLP(
            input_dim=int(hidden_dim),
            output_dim=int(hidden_dim),
            hidden_dims=int(hidden_dim) * 4,
            num_hidden_layers=1,
            dropout=float(dropout),
            act="relu",
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.size(1) != edge_attr.size(0):
            raise RuntimeError("edge_index and edge_attr must have the same number of edges")
        if edge_index.numel() == 0:
            raise RuntimeError("EGTLayerSparse requires non-empty edge_index")

        src, dst = edge_index[0], edge_index[1]
        x_ln = self.node_norm1(x)
        e_ln = self.edge_norm1(edge_attr)

        q = self.q_proj(x_ln).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x_ln).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x_ln).view(-1, self.heads, self.head_dim)
        edge_bias = self.edge_bias_proj(e_ln)
        edge_gate = self.edge_gate_proj(e_ln)

        q_dst = q[dst]
        k_src = k[src]
        v_src = v[src]
        attn_scores = (q_dst * k_src).sum(dim=-1) / (float(self.head_dim) ** 0.5)
        attn_scores = attn_scores.clamp(-5.0, 5.0)
        attn_hat = attn_scores + edge_bias
        gate_values = torch.sigmoid(edge_gate)
        attn_alpha = softmax(attn_hat, dst, num_nodes=x.size(0))
        attn_tilde = self.attn_dropout(attn_alpha * gate_values)

        node_gate_degree = torch.zeros((x.size(0), self.heads), device=x.device, dtype=x.dtype)
        node_gate_degree.index_add_(0, dst, gate_values)
        degree_scale = torch.log1p(node_gate_degree).unsqueeze(-1)

        messages = attn_tilde.unsqueeze(-1) * v_src
        node_agg = torch.zeros_like(v)
        node_agg.index_add_(0, dst, messages)
        node_agg = node_agg * degree_scale
        node_delta = self.node_out_proj(node_agg.reshape(x.size(0), self.hidden_dim))

        x = x + self.dropout(node_delta)
        x = x + self.dropout(self.node_ffn(self.node_norm2(x)))

        edge_delta = self.edge_update_proj(attn_hat)
        edge_attr = edge_attr + self.dropout(edge_delta)
        edge_attr = edge_attr + self.dropout(self.edge_ffn(self.edge_norm2(edge_attr)))
        return x, edge_attr


class EGTNetworkSparse(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 8,
        edge_output_dim: int | None = None,
    ):
        super().__init__()
        edge_output_dim = int(edge_dim) if edge_output_dim is None else int(edge_output_dim)
        self.node_input_proj = nn.Linear(int(input_dim), int(hidden_dim), bias=True)
        self.edge_input_proj = nn.Linear(int(edge_dim), int(hidden_dim), bias=True)
        self.layers = nn.ModuleList(
            [
                EGTLayerSparse(
                    hidden_dim=int(hidden_dim),
                    heads=int(heads),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.node_output_proj = nn.Linear(int(hidden_dim), int(output_dim), bias=True)
        self.edge_output_proj = nn.Linear(int(hidden_dim), int(edge_output_dim), bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
        return_dense_edges: bool = True,
    ):
        if edge_index.size(1) == 0:
            raise RuntimeError("EGTNetworkSparse requires non-empty edge_index")
        h_node = self.node_input_proj(x)
        h_edge = self.edge_input_proj(edge_attr)
        for layer in self.layers:
            h_node, h_edge = layer(h_node, edge_index, h_edge)
        out_node = self.node_output_proj(h_node)
        out_edge = self.edge_output_proj(h_edge)

        x_dense, mask_nodes = to_dense_batch(out_node, batch)
        if not return_dense_edges:
            return x_dense, out_edge, mask_nodes
        max_num_nodes = int(x_dense.size(1))
        edge_dense = to_dense_adj(
            edge_index=edge_index,
            batch=batch,
            edge_attr=out_edge,
            max_num_nodes=max_num_nodes,
        )
        return x_dense, edge_dense, mask_nodes


class EGTDecoderRoPE(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
        heads: int = 8,
        rope_theta_base: float = 10000.0,
        edge_mlp_hidden_dim: int | None = None,
        edge_mlp_out_dim: int = 16,
        edge_recon_hidden_dim: int = 64,
        use_learned_reorder: bool = False,
        learned_reorder_hidden_dim: int = 64,
        learned_reorder_tau: float = 0.1,
        learned_reorder_sinkhorn_iter: int = 8,
        learned_reorder_noise_scale: float = 1.0,
        learned_reorder_entropy_weight: float = 0.01,
        learned_reorder_hard: bool = True,
        learned_reorder_scorer: str = "mlp",
        learned_reorder_use_rrwp: bool = False,
        learned_reorder_rrwp_dim: int = 0,
        learned_reorder_gin_layers: int = 3,
        learned_reorder_gin_dropout: float = 0.1,
        learned_reorder_gt_layers: int = 2,
        learned_reorder_gt_heads: int = 4,
        learned_reorder_gt_dropout: float = 0.1,
        learned_reorder_use_degree: bool = True,
        learned_reorder_bandwidth_weight: float = 0.0,
        learned_reorder_soft_bandwidth_weight: float = 0.05,
        learned_reorder_bandwidth_p: int = 1,
    ):
        super().__init__()
        edge_mlp_hidden_dim = int(hidden_dim) if edge_mlp_hidden_dim is None else int(edge_mlp_hidden_dim)
        self.edge_mlp = MLP(
            input_dim=int(input_dim),
            output_dim=int(edge_mlp_out_dim),
            hidden_dims=int(edge_mlp_hidden_dim),
            num_hidden_layers=1,
            dropout=float(dropout),
            act="relu",
        )
        self.adj_decoder = EdgeReconstructor(
            int(edge_mlp_out_dim),
            hidden_dim=int(edge_recon_hidden_dim),
            edge_dim=int(edge_dim),
        )
        self.net = EGTNetworkSparse(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(output_dim),
            edge_dim=int(edge_dim) + 1,
            num_layers=int(num_layers),
            dropout=float(dropout),
            heads=int(heads),
        )
        self.rope = RoPE(int(input_dim), theta_base=float(rope_theta_base))
        self.use_learned_reorder = bool(use_learned_reorder)
        if self.use_learned_reorder:
            self.reorder_positioner = LearnedReorderPositioner(
                input_dim=int(input_dim),
                hidden_dim=int(learned_reorder_hidden_dim),
                tau=float(learned_reorder_tau),
                sinkhorn_iter=int(learned_reorder_sinkhorn_iter),
                noise_scale=float(learned_reorder_noise_scale),
                entropy_weight=float(learned_reorder_entropy_weight),
                hard=bool(learned_reorder_hard),
                scorer=str(learned_reorder_scorer),
                use_rrwp=bool(learned_reorder_use_rrwp),
                rrwp_dim=int(learned_reorder_rrwp_dim),
                gin_layers=int(learned_reorder_gin_layers),
                gin_dropout=float(learned_reorder_gin_dropout),
                gt_layers=int(learned_reorder_gt_layers),
                gt_heads=int(learned_reorder_gt_heads),
                gt_dropout=float(learned_reorder_gt_dropout),
                use_degree=bool(learned_reorder_use_degree),
                bandwidth_weight=float(learned_reorder_bandwidth_weight),
                soft_bandwidth_weight=float(learned_reorder_soft_bandwidth_weight),
                bandwidth_p=int(learned_reorder_bandwidth_p),
            )
        else:
            self.reorder_positioner = None
        self._upper_tri_index_cache: dict[tuple[str, int], torch.Tensor] = {}

    @staticmethod
    def _graph_starts_from_counts(counts: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.tensor([0], device=counts.device), counts[:-1].cumsum(0)))

    def _upper_tri_indices(self, n_local: int, device: torch.device) -> torch.Tensor:
        key = (str(device), int(n_local))
        pair_local = self._upper_tri_index_cache.get(key, None)
        if pair_local is None:
            pair_local = torch.triu_indices(
                int(n_local),
                int(n_local),
                offset=1,
                device=device,
                dtype=torch.long,
            )
            self._upper_tri_index_cache[key] = pair_local
        return pair_local

    def _build_complete_graph_edges(
        self,
        counts: torch.Tensor,
        starts: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        upper_list: list[torch.Tensor] = []
        full_list: list[torch.Tensor] = []
        for gid in range(int(counts.numel())):
            n_local = int(counts[gid].item())
            if n_local <= 1:
                continue
            start = int(starts[gid].item())
            pair_local = self._upper_tri_indices(n_local=n_local, device=device)
            pair_global = pair_local + start
            upper_list.append(pair_global)
            full_list.append(pair_global)
            full_list.append(pair_global.flip(0))
        if not upper_list:
            empty = torch.empty((2, 0), device=device, dtype=torch.long)
            return empty, empty
        return torch.cat(upper_list, dim=1), torch.cat(full_list, dim=1)

    def forward(
        self,
        quantized: torch.Tensor,
        batch: torch.Tensor,
        use_internal_reorder: bool = True,
        edge_index_for_reorder: torch.Tensor | None = None,
        rrwp_for_reorder: torch.Tensor | None = None,
        return_dense_adj: bool = True,
    ):
        num_nodes = quantized.size(0)
        node_arange = torch.arange(num_nodes, device=quantized.device)
        _uc_val, uc_ptr, uc_counts = torch.unique_consecutive(
            batch, return_inverse=True, return_counts=True
        )
        dense_graph_id_starts = self._graph_starts_from_counts(uc_counts)
        positions = node_arange - dense_graph_id_starts[uc_ptr]
        aux_loss = {
            "reorder_loss": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_entropy": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_bandwidth": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_bandwidth_loss": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_soft_bandwidth": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_soft_bandwidth_loss": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
        }
        if use_internal_reorder and self.reorder_positioner is not None:
            positions, aux_loss = self.reorder_positioner(
                quantized,
                batch,
                edge_index=edge_index_for_reorder,
                rrwp=rrwp_for_reorder,
            )
        quantized = self.rope(quantized, positions)
        quantized_edge = self.edge_mlp(quantized)
        upper_pair_index, complete_edge_index = self._build_complete_graph_edges(
            counts=uc_counts,
            starts=dense_graph_id_starts,
            device=quantized.device,
        )
        if complete_edge_index.numel() > 0:
            complete_edge_attr = self.adj_decoder.score_pairs(quantized_edge, complete_edge_index)
        else:
            complete_edge_attr = torch.empty(
                (0, self.adj_decoder.edge_dim + 1),
                device=quantized.device,
                dtype=quantized.dtype,
            )

        x, edge_attr_sparse, mask_nodes = self.net(
            quantized,
            complete_edge_index,
            batch,
            complete_edge_attr,
            return_dense_edges=False,
        )
        x = x[mask_nodes]
        if complete_edge_index.numel() > 0:
            upper_mask = complete_edge_index[0] < complete_edge_index[1]
            lower_mask = complete_edge_index[0] > complete_edge_index[1]
            packed_edge_logits = (edge_attr_sparse[upper_mask] + edge_attr_sparse[lower_mask]) / 2.0
            if packed_edge_logits.size(0) != upper_pair_index.size(1):
                raise RuntimeError("decoder packed edge logits do not align with complete pair graph")
        else:
            packed_edge_logits = torch.empty(
                (0, self.adj_decoder.edge_dim + 1),
                device=quantized.device,
                dtype=quantized.dtype,
            )
        edge_attr_dense = None
        if return_dense_adj:
            dense_mask_nodes = to_dense_batch(quantized, batch=batch)[1]
            bsz, max_nodes = dense_mask_nodes.shape
            edge_attr_dense = torch.zeros(
                (bsz, max_nodes, max_nodes, self.adj_decoder.edge_dim + 1),
                device=quantized.device,
                dtype=quantized.dtype,
            )
            edge_attr_dense[..., -1] = 1.0
            pair_offset = 0
            for gid in range(int(uc_counts.numel())):
                n_local = int(dense_mask_nodes[gid].sum().item())
                if n_local <= 1:
                    continue
                pair_local = self._upper_tri_indices(n_local=n_local, device=quantized.device)
                pair_count = int(pair_local.size(1))
                local_upper = packed_edge_logits[pair_offset : pair_offset + pair_count]
                pair_offset += pair_count
                edge_attr_dense[gid, pair_local[0], pair_local[1], :] = local_upper
                edge_attr_dense[gid, pair_local[1], pair_local[0], :] = local_upper
        aux_loss["packed_edge_logits"] = packed_edge_logits
        return x, edge_attr_dense, mask_nodes, aux_loss


class GraphVQVAE(nn.Module):
    """
    GVT VQ-AE:
    node/edge graph reconstruction with per-node discrete latents.
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        edge_dim: int,
        codebook_size: int,
        lamb_edge: float = 1.0,
        lamb_node: float = 1.0,
        pe_dim: int = 6,
        vq_commitment_weight: float = 0.25,
        vq_decay: float = 0.8,
        vq_dead_code_threshold: float = 2.0,
        vq_use_cosine_sim: bool = True,
        vq_kmeans_init: bool = False,
        vq_kmeans_iters: int = 10,
        vq_sample_codebook_temp: float = 0.0,
        vq_orthogonal_reg_weight: float = 0.0,
        vq_codebook_dim: int | None = None,
        gvt_heads: int = 8,
        gvt_dropout: float = 0.1,
        encoder_backbone: str = "seq_lm",
        lm_causal: bool = True,
        lm_dropout: float = 0.1,
        lm_heads: int = 8,
        decoder_dropout: float = 0.1,
        decoder_heads: int = 8,
        decoder_rope_theta_base: float = 10000.0,
        decoder_edge_mlp_hidden_dim: int | None = None,
        decoder_edge_mlp_out_dim: int = 16,
        decoder_edge_recon_hidden_dim: int = 64,
        decoder_use_learned_reorder: bool = False,
        decoder_learned_reorder_hidden_dim: int = 64,
        decoder_learned_reorder_tau: float = 0.1,
        decoder_learned_reorder_sinkhorn_iter: int = 8,
        decoder_learned_reorder_noise_scale: float = 1.0,
        decoder_learned_reorder_entropy_weight: float = 0.01,
        decoder_learned_reorder_hard: bool = True,
        decoder_learned_reorder_scorer: str = "gin",
        decoder_learned_reorder_use_rrwp: bool = True,
        decoder_learned_reorder_rrwp_dim: int | None = None,
        decoder_learned_reorder_gin_layers: int = 3,
        decoder_learned_reorder_gin_dropout: float = 0.1,
        decoder_learned_reorder_gt_layers: int = 2,
        decoder_learned_reorder_gt_heads: int = 4,
        decoder_learned_reorder_gt_dropout: float = 0.1,
        decoder_learned_reorder_use_degree: bool = True,
        decoder_learned_reorder_bandwidth_weight: float = 0.0,
        decoder_learned_reorder_soft_bandwidth_weight: float = 0.05,
        decoder_learned_reorder_bandwidth_p: int = 1,
        decoder_reorder_apply_to_codes: bool = True,

        reorder_use_learned: bool | None = None,
        reorder_hidden_dim: int | None = None,
        reorder_tau: float | None = None,
        reorder_sinkhorn_iter: int | None = None,
        reorder_noise_scale: float | None = None,
        reorder_entropy_weight: float | None = None,
        reorder_hard: bool | None = None,
        reorder_scorer: str | None = None,
        reorder_use_rrwp: bool | None = None,
        reorder_rrwp_dim: int | None = None,
        reorder_gin_layers: int | None = None,
        reorder_gin_dropout: float | None = None,
        reorder_gt_layers: int | None = None,
        reorder_gt_heads: int | None = None,
        reorder_gt_dropout: float | None = None,
        reorder_use_degree: bool | None = None,
        reorder_bandwidth_weight: float | None = None,
        reorder_soft_bandwidth_weight: float | None = None,
        reorder_bandwidth_p: int | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.lamb_edge = float(lamb_edge)
        self.lamb_node = float(lamb_node)
        self.encoder_backbone = str(encoder_backbone).strip().lower()
        self.decoder_reorder_apply_to_codes = bool(decoder_reorder_apply_to_codes)

        resolved_reorder_use_learned = bool(
            decoder_use_learned_reorder if reorder_use_learned is None else reorder_use_learned
        )
        resolved_reorder_hidden_dim = int(
            decoder_learned_reorder_hidden_dim if reorder_hidden_dim is None else reorder_hidden_dim
        )
        resolved_reorder_tau = float(
            decoder_learned_reorder_tau if reorder_tau is None else reorder_tau
        )
        resolved_reorder_sinkhorn_iter = int(
            decoder_learned_reorder_sinkhorn_iter if reorder_sinkhorn_iter is None else reorder_sinkhorn_iter
        )
        resolved_reorder_noise_scale = float(
            decoder_learned_reorder_noise_scale if reorder_noise_scale is None else reorder_noise_scale
        )
        resolved_reorder_entropy_weight = float(
            decoder_learned_reorder_entropy_weight if reorder_entropy_weight is None else reorder_entropy_weight
        )
        resolved_reorder_hard = bool(
            decoder_learned_reorder_hard if reorder_hard is None else reorder_hard
        )
        resolved_reorder_scorer = str(
            decoder_learned_reorder_scorer if reorder_scorer is None else reorder_scorer
        )
        resolved_reorder_use_rrwp = bool(
            decoder_learned_reorder_use_rrwp if reorder_use_rrwp is None else reorder_use_rrwp
        )
        resolved_reorder_rrwp_dim = int(
            (pe_dim if decoder_learned_reorder_rrwp_dim is None else decoder_learned_reorder_rrwp_dim)
            if reorder_rrwp_dim is None else reorder_rrwp_dim
        )
        resolved_reorder_gin_layers = int(
            decoder_learned_reorder_gin_layers if reorder_gin_layers is None else reorder_gin_layers
        )
        resolved_reorder_gin_dropout = float(
            decoder_learned_reorder_gin_dropout if reorder_gin_dropout is None else reorder_gin_dropout
        )
        resolved_reorder_gt_layers = int(
            decoder_learned_reorder_gt_layers if reorder_gt_layers is None else reorder_gt_layers
        )
        resolved_reorder_gt_heads = int(
            decoder_learned_reorder_gt_heads if reorder_gt_heads is None else reorder_gt_heads
        )
        resolved_reorder_gt_dropout = float(
            decoder_learned_reorder_gt_dropout if reorder_gt_dropout is None else reorder_gt_dropout
        )
        resolved_reorder_use_degree = bool(
            decoder_learned_reorder_use_degree if reorder_use_degree is None else reorder_use_degree
        )
        resolved_reorder_bandwidth_weight = float(
            decoder_learned_reorder_bandwidth_weight
            if reorder_bandwidth_weight is None else reorder_bandwidth_weight
        )
        resolved_reorder_soft_bandwidth_weight = float(
            decoder_learned_reorder_soft_bandwidth_weight
            if reorder_soft_bandwidth_weight is None else reorder_soft_bandwidth_weight
        )
        resolved_reorder_bandwidth_p = int(
            decoder_learned_reorder_bandwidth_p if reorder_bandwidth_p is None else reorder_bandwidth_p
        )

        if self.encoder_backbone in {"pyg_gt", "graph", "graph_transformer"}:
            self.encoder = Encoder(
                node_in_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                output_dim=int(output_dim),
                edge_dim=int(edge_dim),
                num_layers=int(num_layers),
                pe_dim=int(pe_dim),
                num_heads=int(gvt_heads),
                dropout=float(gvt_dropout),
            )
        elif self.encoder_backbone in {"seq_lm", "lm", "transformer_lm"}:
            self.encoder = SequenceLanguageEncoder(
                node_in_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                output_dim=int(output_dim),
                num_layers=int(num_layers),
                num_heads=int(lm_heads),
                dropout=float(lm_dropout),
                causal=bool(lm_causal),
            )
        else:
            raise ValueError(
                f"Unsupported encoder_backbone: {encoder_backbone}. "
                "Use one of: seq_lm, pyg_gt"
            )
        self.vq = VectorQuantizer(
            codebook_size=int(codebook_size),
            code_dim=int(output_dim),
            commitment_weight=float(vq_commitment_weight),
            decay=float(vq_decay),
            threshold_ema_dead_code=float(vq_dead_code_threshold),
            use_cosine_sim=bool(vq_use_cosine_sim),
            kmeans_init=bool(vq_kmeans_init),
            kmeans_iters=int(vq_kmeans_iters),
            sample_codebook_temp=float(vq_sample_codebook_temp),
            orthogonal_reg_weight=float(vq_orthogonal_reg_weight),
            codebook_dim=None if vq_codebook_dim is None else int(vq_codebook_dim),
        )
        self.decoder = EGTDecoderRoPE(
            num_layers=int(num_layers),
            input_dim=int(output_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(input_dim),
            edge_dim=int(edge_dim),
            dropout=float(decoder_dropout),
            heads=int(decoder_heads),
            rope_theta_base=float(decoder_rope_theta_base),
            edge_mlp_hidden_dim=decoder_edge_mlp_hidden_dim,
            edge_mlp_out_dim=int(decoder_edge_mlp_out_dim),
            edge_recon_hidden_dim=int(decoder_edge_recon_hidden_dim),
            use_learned_reorder=False,
        )
        if resolved_reorder_use_learned:
            self.reorder_positioner = LearnedReorderPositioner(
                input_dim=int(output_dim),
                hidden_dim=resolved_reorder_hidden_dim,
                tau=resolved_reorder_tau,
                sinkhorn_iter=resolved_reorder_sinkhorn_iter,
                noise_scale=resolved_reorder_noise_scale,
                entropy_weight=resolved_reorder_entropy_weight,
                hard=resolved_reorder_hard,
                scorer=resolved_reorder_scorer,
                use_rrwp=resolved_reorder_use_rrwp,
                rrwp_dim=resolved_reorder_rrwp_dim,
                gin_layers=resolved_reorder_gin_layers,
                gin_dropout=resolved_reorder_gin_dropout,
                gt_layers=resolved_reorder_gt_layers,
                gt_heads=resolved_reorder_gt_heads,
                gt_dropout=resolved_reorder_gt_dropout,
                use_degree=resolved_reorder_use_degree,
                bandwidth_weight=resolved_reorder_bandwidth_weight,
                soft_bandwidth_weight=resolved_reorder_soft_bandwidth_weight,
                bandwidth_p=resolved_reorder_bandwidth_p,
            )
        else:
            self.reorder_positioner = None
        self._upper_tri_index_cache: dict[tuple[str, int], torch.Tensor] = {}

    @staticmethod
    def _apply_soft_perm_nodes(
        x: torch.Tensor,
        batch: torch.Tensor,
        perm_st_list: list[torch.Tensor],
    ) -> torch.Tensor:
        out = x.clone()
        _g, _ptr, counts = torch.unique_consecutive(batch, return_inverse=True, return_counts=True)
        starts = torch.cat((torch.tensor([0], device=batch.device), counts[:-1].cumsum(0)))
        for gid in range(int(counts.numel())):
            s = int(starts[gid].item())
            e = s + int(counts[gid].item())
            P = perm_st_list[gid].to(device=x.device, dtype=x.dtype)
            out[s:e] = P @ x[s:e]
        return out

    @staticmethod
    def _apply_hard_perm_nodes(
        x: torch.Tensor,
        batch: torch.Tensor,
        perm_idx_list: list[torch.Tensor],
    ) -> torch.Tensor:
        out = x.clone()
        _g, _ptr, counts = torch.unique_consecutive(batch, return_inverse=True, return_counts=True)
        starts = torch.cat((torch.tensor([0], device=batch.device), counts[:-1].cumsum(0)))
        for gid in range(int(counts.numel())):
            s = int(starts[gid].item())
            e = s + int(counts[gid].item())
            idx = perm_idx_list[gid].to(device=x.device, dtype=torch.long)
            out[s:e] = x[s:e][idx]
        return out

    @staticmethod
    def _apply_hard_perm_dense_adj(
        adj: torch.Tensor,
        batch: torch.Tensor,
        perm_idx_list: list[torch.Tensor],
    ) -> torch.Tensor:
        out = adj.clone()
        _g, _ptr, counts = torch.unique_consecutive(batch, return_inverse=True, return_counts=True)
        for gid in range(int(counts.numel())):
            n = int(counts[gid].item())
            idx = perm_idx_list[gid].to(device=adj.device, dtype=torch.long)
            out[gid, :n, :n, :] = adj[gid, :n, :n, :][idx][:, idx]
        return out
    def _build_gt_adj(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        max_num_nodes: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        gt_adj = torch.zeros_like(like)
        gt_adj[..., -1] = 0.9
        if edge_index.numel() == 0 or edge_attr.numel() == 0:
            return gt_adj
        dense = to_dense_adj(
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            max_num_nodes=int(max_num_nodes),
        )
        if dense.dim() == 3:
            dense = dense.unsqueeze(-1)
        gt_adj[..., :-1] = dense
        return gt_adj

    @staticmethod
    def _apply_hard_perm_edge_index(
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        perm_idx_list: list[torch.Tensor],
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        out = edge_index.clone()
        _g, _ptr, counts = torch.unique_consecutive(batch, return_inverse=True, return_counts=True)
        starts = torch.cat((torch.tensor([0], device=batch.device), counts[:-1].cumsum(0)))
        for gid in range(int(counts.numel())):
            s = int(starts[gid].item())
            e = s + int(counts[gid].item())
            local_mask = (edge_index[0] >= s) & (edge_index[0] < e)
            if not bool(local_mask.any()):
                continue
            idx = perm_idx_list[gid].to(device=edge_index.device, dtype=torch.long)
            inv = torch.empty_like(idx)
            inv[idx] = torch.arange(idx.numel(), device=idx.device, dtype=torch.long)
            out[:, local_mask] = inv[(edge_index[:, local_mask] - s)] + s
        return out

    def _upper_tri_indices(self, n_local: int, device: torch.device) -> torch.Tensor:
        key = (str(device), int(n_local))
        pair_local = self._upper_tri_index_cache.get(key, None)
        if pair_local is None:
            pair_local = torch.triu_indices(
                int(n_local),
                int(n_local),
                offset=1,
                device=device,
                dtype=torch.long,
            )
            self._upper_tri_index_cache[key] = pair_local
        return pair_local

    @staticmethod
    def _upper_tri_flat_index(src: torch.Tensor, dst: torch.Tensor, n_local: int) -> torch.Tensor:
        src = src.to(dtype=torch.long)
        dst = dst.to(dtype=torch.long)
        row_start = src * (2 * int(n_local) - src - 1) // 2
        return row_start + (dst - src - 1)

    def _build_packed_edge_labels(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        device = batch.device
        no_bond_class = int(self.edge_dim)
        _g, _ptr, counts = torch.unique_consecutive(batch, return_inverse=True, return_counts=True)
        starts = torch.cat((torch.tensor([0], device=device), counts[:-1].cumsum(0)))
        edge_labels = (
            torch.argmax(edge_attr, dim=-1).to(device=device, dtype=torch.long)
            if edge_attr.numel() > 0
            else torch.empty((0,), device=device, dtype=torch.long)
        )

        pair_label_list: list[torch.Tensor] = []
        edge_graph = batch[edge_index[0]] if edge_index.numel() > 0 else None
        for gid in range(int(counts.numel())):
            start = int(starts[gid].item())
            n_local = int(counts[gid].item())
            if n_local <= 1:
                continue
            pair_local = self._upper_tri_indices(n_local=n_local, device=device)
            pair_labels = torch.full(
                (pair_local.shape[1],),
                no_bond_class,
                device=device,
                dtype=torch.long,
            )
            if edge_index.numel() > 0 and edge_attr.numel() > 0:
                local_mask = edge_graph == gid
                local_edge_index = edge_index[:, local_mask] - start
                local_edge_labels = edge_labels[local_mask]
                if local_edge_index.numel() > 0:
                    upper_mask = local_edge_index[0] < local_edge_index[1]
                    local_edge_index = local_edge_index[:, upper_mask]
                    local_edge_labels = local_edge_labels[upper_mask]
                    if local_edge_index.numel() > 0:
                        flat_idx = self._upper_tri_flat_index(
                            src=local_edge_index[0],
                            dst=local_edge_index[1],
                            n_local=n_local,
                        )
                        pair_labels[flat_idx] = local_edge_labels
            pair_label_list.append(pair_labels)

        if not pair_label_list:
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.cat(pair_label_list, dim=0)

    def forward(
        self,
        x,
        edge_attr,
        edge_index,
        pe,
        batch,
        rcm_warm_start_positions: torch.Tensor | None = None,
        rcm_warm_start_weight: float = 0.0,
    ):
        encoded_x, _encoded_edge = self.encoder(x, edge_attr, edge_index, pe, batch)
        quantized, embed_ind, commit_loss, vq_stats = self.vq(encoded_x)

        use_external_reorder = bool(self.decoder_reorder_apply_to_codes and self.reorder_positioner is not None)
        aux_loss = {
            "reorder_loss": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_entropy": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_bandwidth": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
            "reorder_bandwidth_loss": torch.zeros((), device=quantized.device, dtype=quantized.dtype),
        }
        target_node = torch.argmax(x, dim=-1)
        embed_out = embed_ind
        quantized_pre_reorder = quantized
        reorder_soft_hard_gap = torch.zeros((), device=quantized.device, dtype=quantized.dtype)
        reorder_pool_mse = torch.zeros((), device=quantized.device, dtype=quantized.dtype)

        if use_external_reorder:
            _pos, aux_loss = self.reorder_positioner(
                encoded_x,
                batch,
                edge_index=edge_index,
                rrwp=pe,
            )
            perm_idx_list = aux_loss.get("perm_idx_list", [])
            quantized = self._apply_hard_perm_nodes(quantized, batch=batch, perm_idx_list=perm_idx_list)
            target_node = self._apply_hard_perm_nodes(target_node, batch=batch, perm_idx_list=perm_idx_list)
            embed_out = self._apply_hard_perm_nodes(embed_ind, batch=batch, perm_idx_list=perm_idx_list)


        pred_node, pred_adj, _mask_node, dec_aux = self.decoder(
            quantized,
            batch=batch,
            use_internal_reorder=not use_external_reorder,
            edge_index_for_reorder=edge_index,
            rrwp_for_reorder=pe,
            return_dense_adj=False,
        )
        if not use_external_reorder:
            aux_loss = dec_aux

        node_rec_loss = self.lamb_node * F.cross_entropy(pred_node, target_node)

        edge_index_loss = edge_index
        if use_external_reorder:
            edge_index_loss = self._apply_hard_perm_edge_index(
                edge_index=edge_index,
                batch=batch,
                perm_idx_list=aux_loss.get("perm_idx_list", []),
            )
        packed_edge_labels = self._build_packed_edge_labels(
            edge_index=edge_index_loss,
            edge_attr=edge_attr,
            batch=batch,
        )
        packed_edge_logits = dec_aux["packed_edge_logits"]
        if packed_edge_logits.size(0) != packed_edge_labels.size(0):
            raise RuntimeError("decoder packed edge logits do not align with packed edge labels")
        if packed_edge_labels.numel() == 0:
            edge_rec_loss = torch.zeros_like(node_rec_loss)
        else:
            edge_rec_loss = self.lamb_edge * F.cross_entropy(packed_edge_logits, packed_edge_labels)

        reorder_loss = aux_loss.get("reorder_loss", torch.zeros_like(commit_loss))
        reorder_rcm_warm_loss = torch.zeros_like(reorder_loss)
        if (
            use_external_reorder
            and rcm_warm_start_positions is not None
            and float(rcm_warm_start_weight) > 0.0
        ):
            if rcm_warm_start_positions.shape[0] == encoded_x.shape[0]:
                pred_scores = aux_loss.get("scores_all", None)
                target_positions = rcm_warm_start_positions.to(
                    device=encoded_x.device, dtype=encoded_x.dtype
                )
                if pred_scores is not None and edge_index.numel() > 0:
                    undirected_mask = edge_index[0] < edge_index[1]
                    pair_src = edge_index[0, undirected_mask]
                    pair_dst = edge_index[1, undirected_mask]
                    if pair_src.numel() > 0:
                        target_dir = torch.sign(target_positions[pair_dst] - target_positions[pair_src])
                        valid = target_dir != 0
                        if valid.any():
                            score_margin = pred_scores[pair_src[valid]] - pred_scores[pair_dst[valid]]
                            reorder_rcm_warm_loss = (
                                F.softplus(-target_dir[valid] * score_margin).mean()
                                * float(rcm_warm_start_weight)
                            )
                            reorder_loss = reorder_loss + reorder_rcm_warm_loss
        loss = node_rec_loss + edge_rec_loss + commit_loss + reorder_loss
        vq_stats["reorder_loss"] = float(reorder_loss.detach().cpu().item())
        vq_stats["reorder_entropy"] = float(
            aux_loss.get("reorder_entropy", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_bandwidth"] = float(
            aux_loss.get("reorder_bandwidth", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_bandwidth_loss"] = float(
            aux_loss.get("reorder_bandwidth_loss", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_soft_bandwidth"] = float(
            aux_loss.get("reorder_soft_bandwidth", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_soft_bandwidth_loss"] = float(
            aux_loss.get("reorder_soft_bandwidth_loss", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_laplacian"] = float(
            aux_loss.get("reorder_laplacian", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_laplacian_loss"] = float(
            aux_loss.get("reorder_laplacian_loss", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_spread"] = float(
            aux_loss.get("reorder_spread", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_spread_loss"] = float(
            aux_loss.get("reorder_spread_loss", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_rayleigh"] = float(
            aux_loss.get("reorder_rayleigh", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_rayleigh_loss"] = float(
            aux_loss.get("reorder_rayleigh_loss", torch.zeros_like(reorder_loss)).detach().cpu().item()
        )
        vq_stats["reorder_rcm_warm_loss"] = float(reorder_rcm_warm_loss.detach().cpu().item())
        vq_stats["reorder_soft_hard_gap"] = float(reorder_soft_hard_gap.detach().cpu().item())
        vq_stats["reorder_pool_mse"] = float(reorder_pool_mse.detach().cpu().item())
        return (
            (loss, node_rec_loss, edge_rec_loss, commit_loss),
            pred_node,
            packed_edge_logits,
            embed_out,
            vq_stats,
        )
    @torch.no_grad()
    def encode(self, x, edge_attr, edge_index, pe, batch):
        encoded_x, _ = self.encoder(x, edge_attr, edge_index, pe, batch)
        quantized, embed_ind, commit_loss, vq_stats = self.vq(encoded_x)

        if bool(self.decoder_reorder_apply_to_codes and self.reorder_positioner is not None):
            _pos, aux_loss = self.reorder_positioner(
                encoded_x,
                batch,
                edge_index=edge_index,
                rrwp=pe,
            )
            perm_idx_list = aux_loss.get("perm_idx_list", [])
            quantized = self._apply_hard_perm_nodes(quantized, batch=batch, perm_idx_list=perm_idx_list)
            embed_ind = self._apply_hard_perm_nodes(embed_ind, batch=batch, perm_idx_list=perm_idx_list)
            vq_stats["reorder_entropy"] = float(
                aux_loss.get("reorder_entropy", torch.zeros((), device=quantized.device, dtype=quantized.dtype)).detach().cpu().item()
            )

        return quantized, embed_ind, commit_loss, vq_stats





























