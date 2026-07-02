#!/usr/bin/env python3
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, dropout, latent_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            input_dim = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))
        self.dropout = dropout
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_add_pool(x, batch)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar


class SelfiesDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        latent_dim,
        layers,
        heads,
        dropout,
        max_len,
        latent_prefix_len=1,
        latent_cross_attn=True,
    ):
        super().__init__()
        if int(latent_prefix_len) <= 0:
            raise ValueError("latent_prefix_len must be >= 1")
        self.hidden_dim = int(hidden_dim)
        self.latent_prefix_len = int(latent_prefix_len)
        self.latent_cross_attn_enabled = bool(latent_cross_attn)
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim * self.latent_prefix_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        if self.latent_cross_attn_enabled:
            self.latent_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=False,
            )
            self.latent_cross_attn_drop = nn.Dropout(dropout)
            self.latent_cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_ids, z, tgt_key_padding_mask=None):
        batch_size, seq_len = tgt_ids.shape
        positions = torch.arange(seq_len, device=tgt_ids.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        tgt = self.token_emb(tgt_ids) + self.pos_emb(positions)
        tgt = tgt.transpose(0, 1)
        memory = self.latent_proj(z).view(batch_size, self.latent_prefix_len, self.hidden_dim)
        memory = memory.transpose(0, 1)
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt_ids.device), diagonal=1
        ).bool()
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        if self.latent_cross_attn_enabled:
            # Keep decoder memory path and add a late latent cross-attn residual to
            # strengthen z-conditioning and reduce latent information loss.
            lat_out, _ = self.latent_cross_attn(
                query=out,
                key=memory,
                value=memory,
                need_weights=False,
            )
            out = self.latent_cross_attn_norm(
                out + self.latent_cross_attn_drop(lat_out)
            )
        out = out.transpose(0, 1)
        return self.out(out)


class GraphSelfiesVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, graph_batch, seq_in, tgt_key_padding_mask=None):
        mu, logvar = self.encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(seq_in, z, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits, mu, logvar
