#!/usr/bin/env python3
import argparse
import atexit
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GINEConv
import pandas as pd
from rdkit import Chem, rdBase

from data.pocket_dataset import PocketLigandDockingDataset
from data.ligand_only_dataset import LigandOnlyDataset
from data.ligand_only_3d_dataset import LigandOnly3DDataset
from mobo.config_utils import load_config
from mobo.pretrain_targets import PRETRAIN_PROPERTY_NAMES

TORCHMD_AVAILABLE = False
TensorNet = None
scatter = None
act_class_mapping = None
torchmd_import_error = None
try:
    from torchmd_local.tensornet import TensorNet  # type: ignore
    from torchmd_local.utils import scatter, act_class_mapping  # type: ignore

    TORCHMD_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    torchmd_import_error = exc

rdBase.DisableLog("rdApp.error")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _compute_lr(epoch: int, base_lr: float, total_epochs: int, warmup_epochs: int, min_lr: float) -> float:
    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)
    if total_epochs <= 1:
        return base_lr
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    t = min(max(t, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cosine


def _print_epoch_header() -> None:
    print("epoch | lr        | train_loss | val_rmse | val_mae | val_r2")
    print("------+-----------+------------+----------+---------+--------")


def masked_mean_pool(x: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        sums = torch.zeros(num_graphs, x.size(-1), device=x.device)
        sums.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).unsqueeze(-1)
        return sums / counts
    mask = mask.to(dtype=x.dtype)
    x_masked = x * mask.unsqueeze(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    sums = torch.zeros(num_graphs, x.size(-1), device=x.device)
    sums.index_add_(0, batch, x_masked)
    counts = torch.zeros(num_graphs, device=x.device)
    counts.index_add_(0, batch, mask)
    return sums / counts.clamp_min(1).unsqueeze(-1)


class GINSurrogate(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        use_ligand_mask: bool = True,
        fp_dim: int = 0,
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.use_ligand_mask = use_ligand_mask
        self.fp_dim = int(fp_dim)
        self.node_emb = nn.Linear(node_dim, hidden_dim)

        convs = []
        norms = []
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if use_edge_attr:
                conv = GINEConv(mlp, edge_dim=edge_dim)
            else:
                conv = GINConv(mlp)
            convs.append(conv)
            norms.append(nn.BatchNorm1d(hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        out_dim = hidden_dim * 2 if use_ligand_mask else hidden_dim
        fp_out_dim = 0
        if self.fp_dim > 0:
            fp_out_dim = hidden_dim
            self.fp_mlp = nn.Sequential(
                nn.Linear(self.fp_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.fp_mlp = None
        self.readout = nn.Sequential(
            nn.Linear(out_dim + fp_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x = data.x
        if x is None:
            raise ValueError("Data.x is required for GINSurrogate.")
        x = self.node_emb(x)
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        if self.use_edge_attr and edge_attr is None:
            raise ValueError("edge_attr is required when use_edge_attr=True.")

        for conv, norm in zip(self.convs, self.norms):
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        batch = data.batch
        if self.use_ligand_mask and hasattr(data, "mask_ligand"):
            mask = data.mask_ligand.to(dtype=x.dtype)
            lig_feat = masked_mean_pool(x, batch, mask)
            pocket_feat = masked_mean_pool(x, batch, 1.0 - mask)
            graph_feat = torch.cat([lig_feat, pocket_feat], dim=-1)
        else:
            graph_feat = masked_mean_pool(x, batch, None)
        if self.fp_mlp is not None:
            fp = getattr(data, "fp", None)
            if fp is None:
                raise ValueError("Fingerprint feature 'fp' is required when fp_dim>0.")
            fp_feat = self.fp_mlp(fp)
            graph_feat = torch.cat([graph_feat, fp_feat], dim=-1)
        return self.readout(graph_feat)


class GaussianPredictor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        act=None,
        num_layers: int = 3,
    ):
        super().__init__()
        num_layers = max(1, int(num_layers))
        hidden_dim = int(hidden_dim)
        if act is None:
            act_factory = nn.ReLU
        elif isinstance(act, type):
            act_factory = act
        else:
            act_factory = lambda: act
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_features, 2))
        else:
            layers.append(nn.Linear(in_features, hidden_dim))
            for idx in range(num_layers - 1):
                layers.append(act_factory())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                out_dim = 2 if idx == (num_layers - 2) else hidden_dim
                layers.append(nn.Linear(hidden_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_mu: float = 0.0, prior_sigma: float = 1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self._cached_weight: torch.Tensor | None = None
        self._cached_bias: torch.Tensor | None = None
        self._use_cache = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_cache and self._cached_weight is not None and self._cached_bias is not None:
            weight = self._cached_weight
            bias = self._cached_bias
        else:
            w_sigma = torch.exp(self.weight_log_sigma)
            b_sigma = torch.exp(self.bias_log_sigma)
            weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
            bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        return nn.functional.linear(x, weight, bias)

    def sample_params(self) -> None:
        w_sigma = torch.exp(self.weight_log_sigma)
        b_sigma = torch.exp(self.bias_log_sigma)
        self._cached_weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        self._cached_bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        self._use_cache = True

    def clear_params(self) -> None:
        self._cached_weight = None
        self._cached_bias = None
        self._use_cache = False

    def kl_loss(self) -> torch.Tensor:
        def _kl(q_mu: torch.Tensor, q_log_sigma: torch.Tensor) -> torch.Tensor:
            q_sigma = torch.exp(q_log_sigma)
            p_sigma = self.prior_sigma
            p_mu = self.prior_mu
            return (
                torch.log(torch.tensor(p_sigma, device=q_mu.device, dtype=q_mu.dtype) / q_sigma)
                + (q_sigma**2 + (q_mu - p_mu) ** 2) / (2 * p_sigma**2)
                - 0.5
            ).sum()

        return _kl(self.weight_mu, self.weight_log_sigma) + _kl(self.bias_mu, self.bias_log_sigma)


class BayesianPredictor(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        act=None,
        num_layers: int = 3,
    ):
        super().__init__()
        num_layers = max(1, int(num_layers))
        hidden_dim = int(hidden_dim)
        if act is None:
            act_factory = nn.ReLU
        elif isinstance(act, type):
            act_factory = act
        else:
            act_factory = lambda: act
        layers = []
        if num_layers == 1:
            layers.append(BayesianLinear(in_features, out_features))
        else:
            layers.append(BayesianLinear(in_features, hidden_dim))
            for idx in range(num_layers - 1):
                layers.append(act_factory())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                out_dim = out_features if idx == (num_layers - 2) else hidden_dim
                layers.append(BayesianLinear(hidden_dim, out_dim))
        self.bnn_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bnn_head(x)

    def sample_bayes_params(self) -> None:
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                module.sample_params()

    def clear_bayes_params(self) -> None:
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                module.clear_params()


def get_total_kl_loss(module: nn.Module) -> torch.Tensor:
    kl_loss = torch.zeros((), device=next(module.parameters()).device)
    for sub in module.modules():
        if isinstance(sub, BayesianLinear):
            kl_loss = kl_loss + sub.kl_loss()
    return kl_loss


class NIGPredictor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        act=None,
        num_layers: int = 3,
    ):
        super().__init__()
        num_layers = max(1, int(num_layers))
        hidden_dim = int(hidden_dim)
        if act is None:
            act_factory = nn.ReLU
        elif isinstance(act, type):
            act_factory = act
        else:
            act_factory = lambda: act
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_features, 4))
        else:
            layers.append(nn.Linear(in_features, hidden_dim))
            for idx in range(num_layers - 1):
                layers.append(act_factory())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                out_dim = 4 if idx == (num_layers - 2) else hidden_dim
                layers.append(nn.Linear(hidden_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def split_gaussian_output(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.ndim == 1 or pred.size(-1) != 2:
        raise ValueError(f"Gaussian surrogate output must have shape [N, 2], got {tuple(pred.shape)}.")
    return pred[..., 0], pred[..., 1]


def gaussian_variance_from_raw(
    raw_log_var: torch.Tensor,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> torch.Tensor:
    raw = raw_log_var.clamp(min=float(logvar_min), max=float(logvar_max))
    return torch.nn.functional.softplus(raw) + float(min_var)



def raw_logvar_from_variance(var: torch.Tensor, min_var: float = 1e-6) -> torch.Tensor:
    shifted = torch.clamp(var - float(min_var), min=1e-12)
    return torch.log(torch.expm1(shifted))



def split_nig_output(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.ndim == 1 or pred.size(-1) != 4:
        raise ValueError(f"NIG surrogate output must have shape [N, 4], got {tuple(pred.shape)}.")
    gamma = pred[..., 0]
    nu = torch.nn.functional.softplus(pred[..., 1]) + 1e-6
    alpha = torch.nn.functional.softplus(pred[..., 2]) + 1.0 + 1e-6
    beta = torch.nn.functional.softplus(pred[..., 3]) + 1e-6
    return gamma, nu, alpha, beta


def nig_variances_from_params(
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    min_var: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = torch.clamp(alpha - 1.0, min=1e-6)
    ale_var = torch.clamp(beta / denom, min=min_var)
    epi_var = torch.clamp(beta / (nu * denom), min=0.0)
    total_var = torch.clamp(ale_var + epi_var, min=min_var)
    return ale_var, epi_var, total_var


def nig_nll_loss(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    two_blambda = 2.0 * beta * (1.0 + nu)
    err2 = (target - gamma) ** 2
    loss = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log(nu * err2 + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return loss.mean()


def nig_evidence_regularizer(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    evidence = 2.0 * nu + alpha
    return (torch.abs(target - gamma) * evidence).mean()


def gaussian_nll_loss(
    mu: torch.Tensor,
    raw_log_var: torch.Tensor,
    target: torch.Tensor,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
) -> torch.Tensor:
    var = gaussian_variance_from_raw(raw_log_var, logvar_min=logvar_min, logvar_max=logvar_max, min_var=min_var)
    return 0.5 * (torch.log(var) + ((target - mu) ** 2) / var).mean()


class TensorNetEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        fp_dim: int = 0,
    ):
        super().__init__()
        if not TORCHMD_AVAILABLE or TensorNet is None:
            raise ImportError(f"TensorNet not available: {torchmd_import_error}")
        if node_feat_dim <= 0:
            raise ValueError("TensorNetEncoder requires node_feat_dim > 0 for atom vocab features.")
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")

        self.repr = TensorNet(
            hidden_channels=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            node_feat_dim=int(node_feat_dim) if node_feat_dim else None,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
        )
        act = act_class_mapping[activation]
        self.fp_dim = int(fp_dim)
        fp_out_dim = 0
        if self.fp_dim > 0:
            fp_out_dim = embedding_dim
            self.fp_mlp = nn.Sequential(
                nn.Linear(self.fp_dim, embedding_dim),
                act(),
                nn.Dropout(dropout),
            )
        else:
            self.fp_mlp = None
        self.dropout = nn.Dropout(dropout)
        self.reduce_op = reduce_op
        self.output_dim = int(embedding_dim + fp_out_dim)

    def forward(self, data):
        pos = getattr(data, "pos", None)
        batch = getattr(data, "batch", None)
        x = getattr(data, "x", None)
        if pos is None or batch is None or x is None:
            raise ValueError("TensorNetEncoder requires data.pos, data.batch, and data.x")
        h, _, _, pos, batch = self.repr(None, pos, batch, x=x)
        h = self.dropout(h)
        graph_feat = scatter(h, batch, dim=0, reduce=self.reduce_op)
        if self.fp_mlp is not None:
            fp = getattr(data, "fp", None)
            if fp is None:
                raise ValueError("Fingerprint feature 'fp' is required when fp_dim>0.")
            fp_feat = self.fp_mlp(fp)
            graph_feat = torch.cat([graph_feat, fp_feat], dim=-1)
        return graph_feat


class TensorNetSurrogate(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        fp_dim: int = 0,
        head_hidden_dim: int | None = None,
        head_num_layers: int = 3,
    ):
        super().__init__()
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")
        act = act_class_mapping[activation]
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim is not None else int(embedding_dim)
        self.encoder = TensorNetEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            node_feat_dim=node_feat_dim,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=fp_dim,
        )
        self.readout = GaussianPredictor(
            self.encoder.output_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            act=act,
            num_layers=head_num_layers,
        )

    def encode(self, data):
        return self.encoder(data)

    def forward_encoded(self, feat: torch.Tensor) -> torch.Tensor:
        return self.readout(feat)

    def forward(self, data):
        return self.forward_encoded(self.encode(data))


class TensorNetBayesSurrogate(nn.Module):
    uncertainty_mode = "bayes"

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        fp_dim: int = 0,
        head_hidden_dim: int | None = None,
        head_num_layers: int = 3,
    ):
        super().__init__()
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")
        act = act_class_mapping[activation]
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim is not None else int(embedding_dim)
        self.encoder = TensorNetEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            node_feat_dim=node_feat_dim,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=fp_dim,
        )
        self.readout = BayesianPredictor(
            self.encoder.output_dim,
            1,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            act=act,
            num_layers=head_num_layers,
        )

    def encode(self, data):
        return self.encoder(data)

    def forward_encoded(self, feat: torch.Tensor) -> torch.Tensor:
        return self.readout(feat).squeeze(-1)

    def forward(self, data):
        return self.forward_encoded(self.encode(data))

    def kl_loss(self) -> torch.Tensor:
        return get_total_kl_loss(self.readout)

    def sample_bayes_params(self) -> None:
        self.readout.sample_bayes_params()

    def clear_bayes_params(self) -> None:
        self.readout.clear_bayes_params()


class TensorNetNIGSurrogate(nn.Module):
    uncertainty_mode = "nig"

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        fp_dim: int = 0,
        head_hidden_dim: int | None = None,
        head_num_layers: int = 3,
    ):
        super().__init__()
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")
        act = act_class_mapping[activation]
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim is not None else int(embedding_dim)
        self.encoder = TensorNetEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            node_feat_dim=node_feat_dim,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=fp_dim,
        )
        self.readout = NIGPredictor(
            self.encoder.output_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            act=act,
            num_layers=head_num_layers,
        )

    def encode(self, data):
        return self.encoder(data)

    def forward_encoded(self, feat: torch.Tensor) -> torch.Tensor:
        return self.readout(feat)

    def forward(self, data):
        return self.forward_encoded(self.encode(data))

    def decompose_encoded(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.forward_encoded(feat)
        gamma, nu, alpha, beta = split_nig_output(raw)
        ale_var, epi_var, total_var = nig_variances_from_params(nu, alpha, beta)
        return gamma, ale_var, epi_var, total_var

    def decompose(self, data):
        return self.decompose_encoded(self.encode(data))


class TensorNetEnsembleSurrogate(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        fp_dim: int = 0,
        head_hidden_dim: int | None = None,
        head_num_layers: int = 3,
        num_heads: int = 3,
    ):
        super().__init__()
        if int(num_heads) < 2:
            raise ValueError("TensorNetEnsembleSurrogate requires num_heads >= 2.")
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")
        act = act_class_mapping[activation]
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim is not None else int(embedding_dim)
        self.encoder = TensorNetEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            node_feat_dim=node_feat_dim,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=fp_dim,
        )
        self.num_heads = int(num_heads)
        self.readouts = nn.ModuleList(
            [
                GaussianPredictor(
                    self.encoder.output_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=dropout,
                    act=act,
                    num_layers=head_num_layers,
                )
                for _ in range(self.num_heads)
            ]
        )

    def encode(self, data):
        return self.encoder(data)

    def forward_heads_from_encoded(self, feat: torch.Tensor) -> torch.Tensor:
        preds = [head(feat) for head in self.readouts]
        return torch.stack(preds, dim=1)

    def forward_heads(self, data):
        return self.forward_heads_from_encoded(self.encode(data))

    def forward_encoded(self, feat: torch.Tensor) -> torch.Tensor:
        pred = self.forward_heads_from_encoded(feat)
        mu = pred[..., 0]
        raw_log_var = pred[..., 1]
        var = gaussian_variance_from_raw(raw_log_var)
        mu_mean = mu.mean(dim=1)
        ale_var = var.mean(dim=1)
        epi_var = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1)
        total_var = ale_var + epi_var
        raw_total = raw_logvar_from_variance(total_var)
        return torch.stack([mu_mean, raw_total], dim=-1)

    def forward(self, data):
        return self.forward_encoded(self.encode(data))

    def decompose_encoded(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.forward_heads_from_encoded(feat)
        mu = pred[..., 0]
        raw_log_var = pred[..., 1]
        var = gaussian_variance_from_raw(raw_log_var)
        mu_mean = mu.mean(dim=1)
        ale_var = var.mean(dim=1)
        epi_var = ((mu - mu_mean.unsqueeze(1)) ** 2).mean(dim=1)
        total_var = ale_var + epi_var
        return mu_mean, ale_var, epi_var, total_var

    def decompose(self, data):
        return self.decompose_encoded(self.encode(data))


class TensorNetFingerprintPretrainModel(nn.Module):
    def __init__(
        self,
        fp_bits: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 4.5,
        node_feat_dim: int = 0,
        max_num_neighbors: int = 64,
        equivariance_invariance_group: str = "O(3)",
        static_shapes: bool = True,
        check_errors: bool = True,
        dropout: float = 0.1,
        reduce_op: str = "sum",
        head_hidden_dim: int | None = None,
        property_dim: int = len(PRETRAIN_PROPERTY_NAMES),
    ):
        super().__init__()
        if activation not in act_class_mapping:
            raise ValueError(f"Unknown activation: {activation}")
        act = act_class_mapping[activation]
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim is not None else int(embedding_dim)
        self.encoder = TensorNetEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            node_feat_dim=node_feat_dim,
            max_num_neighbors=max_num_neighbors,
            equivariance_invariance_group=equivariance_invariance_group,
            static_shapes=static_shapes,
            check_errors=check_errors,
            dropout=dropout,
            reduce_op=reduce_op,
            fp_dim=0,
        )
        self.fp_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, head_hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, int(fp_bits)),
        )
        self.property_dim = int(property_dim)
        if self.property_dim <= 0:
            raise ValueError("property_dim must be > 0 for TensorNet fingerprint/property pretraining.")
        self.property_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, head_hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, self.property_dim),
        )

    def encode(self, data):
        return self.encoder(data)

    def forward(self, data):
        feat = self.encode(data)
        return self.fp_head(feat), self.property_head(feat)

def get_targets(batch) -> torch.Tensor | None:
    if hasattr(batch, "dock_score"):
        y = batch.dock_score
    elif hasattr(batch, "vina_score"):
        y = batch.vina_score
    else:
        return None
    return y.to(torch.float32)


def fit_target_stats(
    loader,
    device,
    dock_valid_max: float | None = None,
) -> Tuple[float, float]:
    values = []
    for batch in loader:
        y = get_targets(batch)
        if y is None:
            continue
        y = y.to(device)
        mask = torch.isfinite(y)
        if dock_valid_max is not None:
            mask = mask & (y <= dock_valid_max)
        y = y[mask]
        if y.numel() == 0:
            continue
        values.append(y)
    if not values:
        return 0.0, 1.0
    y_all = torch.cat(values, dim=0)
    mean = float(y_all.mean().item())
    std = float(y_all.std(unbiased=False).item())
    if std <= 0:
        std = 1.0
    return mean, std


def _pick_smiles_column(columns: List[str]) -> str | None:
    col_map = {c.lower(): c for c in columns}
    for key in ["smiles_canonical", "smiles", "smile", "smiles_clean", "smiles_cleaned", "smiles_norm", "smiles_std"]:
        if key in col_map:
            return col_map[key]
    for key in ["smiles", "smile", "smiles_canonical", "smiles_clean"]:
        if key in col_map:
            return col_map[key]
    return None


def build_ligand_vocab(root: str) -> List[str]:
    csv_path = f"{root}/smiles.csv"
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    if df.empty:
        return []
    col = _pick_smiles_column(list(df.columns))
    if col is None:
        col = df.columns[0]
    symbols = set()
    for smi in df[col].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            symbols.add(atom.GetSymbol())
    return sorted(symbols)


def clear_processed_if_vocab_mismatch(root: str, ligand_vocab: List[str]) -> None:
    proc_dir = f"{root}/processed"
    meta_paths = [f"{proc_dir}/pocket_dataset_{split}.pt.meta" for split in ("train", "valid", "test")]
    mismatch = False
    for meta_path in meta_paths:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        existing = meta.get("ligand_vocab", [])
        if list(existing) != list(ligand_vocab):
            mismatch = True
            break
    if not mismatch:
        return
    for split in ("train", "valid", "test"):
        base = f"{proc_dir}/pocket_dataset_{split}.pt"
        for suffix in ("", ".meta", ".atom_per_mol.json", ".pocket.pt"):
            path = base + suffix
            try:
                Path(path).unlink()
            except Exception:
                pass

def _bayes_predictive_stats(
    model,
    batch,
    eval_samples: int,
    min_var: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    draws = []
    eval_samples = max(1, int(eval_samples))
    with torch.no_grad():
        for _ in range(eval_samples):
            model.sample_bayes_params()
            draws.append(model(batch).detach())
            model.clear_bayes_params()
    stacked = torch.stack(draws, dim=0)
    mean = stacked.mean(dim=0)
    if stacked.size(0) == 1:
        var = torch.full_like(mean, float(min_var))
    else:
        var = stacked.var(dim=0, unbiased=False).clamp_min(float(min_var))
    return mean, var


def _safe_tensor_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).item()
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(torch.sum(x * y).item() / denom)

def evaluate(
    model,
    loader,
    device,
    mean,
    std,
    dock_valid_max: float | None = None,
    eval_samples: int = 1,
    logvar_min: float = -8.0,
    logvar_max: float = 4.0,
    min_var: float = 1e-6,
):
    model.eval()
    preds = []
    targets = []
    nll_terms = []

    total_std_terms = []
    ale_std_terms = []
    epi_std_terms = []

    # NEW: raw NIG parameter logs
    nu_terms = []
    alpha_terms = []
    beta_terms = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = get_targets(batch)
            if y is None:
                continue
            y = y.to(device)

            mask = torch.isfinite(y)
            if dock_valid_max is not None:
                mask = mask & (y <= dock_valid_max)
            if mask.sum() == 0:
                continue

            y_z = (y[mask] - mean) / std

            mode = getattr(model, "uncertainty_mode", "gaussian")
            if mode == "bayes":
                mu_z, var_z = _bayes_predictive_stats(model, batch, eval_samples=eval_samples, min_var=min_var)
                mu_z = mu_z[mask]
                var_z = var_z[mask].clamp_min(min_var)
                mu = mu_z * std + mean
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z) ** 2) / var_z)
                total_std = torch.sqrt(var_z * (std ** 2))
                ale_std = total_std
                epi_std = torch.full_like(total_std, float("nan"))

            elif mode == "nig":
                raw = model(batch)
                gamma, nu, alpha, beta = split_nig_output(raw)
                gamma = gamma[mask]
                nu = nu[mask]
                alpha = alpha[mask]
                beta = beta[mask]

                ale_var_z, epi_var_z, total_var_z = nig_variances_from_params(
                    nu, alpha, beta, min_var=min_var
                )
                mu = gamma * std + mean
                nll = nig_nll_loss(gamma, nu, alpha, beta, y_z)

                total_std = torch.sqrt(total_var_z * (std ** 2))
                ale_std = torch.sqrt(ale_var_z * (std ** 2))
                epi_std = torch.sqrt(epi_var_z * (std ** 2))

                nu_terms.append(torch.as_tensor(nu.detach().cpu()).reshape(-1))
                alpha_terms.append(torch.as_tensor(alpha.detach().cpu()).reshape(-1))
                beta_terms.append(torch.as_tensor(beta.detach().cpu()).reshape(-1))

            elif hasattr(model, "decompose"):
                mu_z, ale_var_z, epi_var_z, total_var_z = model.decompose(batch)
                mu = mu_z[mask] * std + mean

                total_var_z = total_var_z[mask].clamp_min(min_var)
                ale_var_z = ale_var_z[mask].clamp_min(min_var)
                epi_var_z = epi_var_z[mask].clamp_min(0.0)

                nll = 0.5 * (torch.log(total_var_z) + ((y_z - mu_z[mask]) ** 2) / total_var_z)
                total_std = torch.sqrt(total_var_z * (std ** 2))
                ale_std = torch.sqrt(ale_var_z * (std ** 2))
                epi_std = torch.sqrt(epi_var_z * (std ** 2))

            else:
                raw = model(batch)
                mu_z, raw_log_var = split_gaussian_output(raw)
                var_z = gaussian_variance_from_raw(
                    raw_log_var[mask],
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    min_var=min_var,
                )
                mu = mu_z[mask] * std + mean
                nll = 0.5 * (torch.log(var_z) + ((y_z - mu_z[mask]) ** 2) / var_z)
                total_std = torch.sqrt(var_z * (std ** 2))
                ale_std = total_std
                epi_std = torch.full_like(total_std, float("nan"))

            preds.append(torch.as_tensor(mu.detach().cpu()).reshape(-1))
            targets.append(torch.as_tensor(y[mask].detach().cpu()).reshape(-1))
            nll_terms.append(torch.as_tensor(nll.detach().cpu()).reshape(-1))
            total_std_terms.append(torch.as_tensor(total_std.detach().cpu()).reshape(-1))
            ale_std_terms.append(torch.as_tensor(ale_std.detach().cpu()).reshape(-1))
            epi_std_terms.append(torch.as_tensor(epi_std.detach().cpu()).reshape(-1))

    if not preds:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "nll": float("nan"),
            "mean_std": float("nan"),
            "median_std": float("nan"),
            "p90_std": float("nan"),
            "corr_abs_error_std": float("nan"),
            "coverage_1sigma": float("nan"),
            "coverage_2sigma": float("nan"),
            "mean_ale_std": float("nan"),
            "mean_epi_std": float("nan"),
            "corr_abs_error_ale_std": float("nan"),
            "corr_abs_error_epi_std": float("nan"),
            "corr_dock_ale_std": float("nan"),
            "corr_dock_epi_std": float("nan"),
            "mean_epi_over_ale": float("nan"),
            "median_epi_over_ale": float("nan"),
            "std_log_epi_over_ale": float("nan"),
            "spearman_epi_ale": float("nan"),
            "corr_abs_error_epi_over_ale": float("nan"),
            "corr_dock_epi_over_ale": float("nan"),
            "mean_nu": float("nan"),
            "mean_alpha": float("nan"),
            "mean_beta": float("nan"),
        }

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(targets, dim=0)
    total_std = torch.cat(total_std_terms, dim=0)
    ale_std = torch.cat(ale_std_terms, dim=0)
    epi_std = torch.cat(epi_std_terms, dim=0)

    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()

    denom = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1.0 - (torch.sum((y_pred - y_true) ** 2).item() / denom) if denom > 0 else float("nan")

    # 支持 batch 标量 NLL / 逐样本 NLL 两种情况
    nll_sum = 0.0
    nll_count = 0
    target_cursor = 0
    for nll_flat in nll_terms:
        nll_flat = torch.as_tensor(nll_flat).reshape(-1)
        if nll_flat.numel() == 0:
            continue
        # 如果是逐样本 NLL
        if nll_flat.numel() > 1:
            nll_sum += float(nll_flat.sum().item())
            nll_count += int(nll_flat.numel())
        else:
            # 如果是 batch mean 标量，就按“一个 batch 的均值”乘回该 batch 的样本数
            # 这里无法从 nll 本身恢复 batch 大小，只能退化为按 1 计；
            # 但你当前 nig_nll_loss 若以后改成逐样本返回，这里会自动走上面的分支。
            nll_sum += float(nll_flat.item())
            nll_count += 1
    nll = float(nll_sum / max(nll_count, 1))

    abs_err = torch.abs(y_pred - y_true)
    corr_abs_error_std = _safe_tensor_corr(abs_err, total_std)
    corr_abs_error_ale_std = _safe_tensor_corr(abs_err, ale_std)

    epi_finite_mask = torch.isfinite(epi_std)
    epi_finite = epi_std[epi_finite_mask]
    mean_epi_std = epi_finite.mean().item() if epi_finite.numel() > 0 else float("nan")
    corr_abs_error_epi_std = _safe_tensor_corr(abs_err[epi_finite_mask], epi_finite) if epi_finite.numel() > 2 else float("nan")

    coverage_1sigma = torch.mean((abs_err <= total_std).to(torch.float32)).item()
    coverage_2sigma = torch.mean((abs_err <= 2.0 * total_std).to(torch.float32)).item()

    finite_pair = torch.isfinite(epi_std) & torch.isfinite(ale_std) & (ale_std > 0)
    if finite_pair.sum().item() > 2:
        epi_pair = epi_std[finite_pair]
        ale_pair = ale_std[finite_pair]
        abs_pair = abs_err[finite_pair]
        y_pair = y_true[finite_pair]

        epi_over_ale = epi_pair / ale_pair.clamp_min(1e-12)
        log_epi_over_ale = torch.log(epi_over_ale.clamp_min(1e-12))

        mean_epi_over_ale = epi_over_ale.mean().item()
        median_epi_over_ale = epi_over_ale.median().item()
        std_log_epi_over_ale = log_epi_over_ale.std().item() if log_epi_over_ale.numel() > 1 else 0.0
        spearman_epi_ale = _safe_tensor_corr(epi_pair, ale_pair)
        corr_abs_error_epi_over_ale = _safe_tensor_corr(abs_pair, epi_over_ale)
        corr_dock_epi_over_ale = _safe_tensor_corr(y_pair, epi_over_ale)
    else:
        mean_epi_over_ale = float("nan")
        median_epi_over_ale = float("nan")
        std_log_epi_over_ale = float("nan")
        spearman_epi_ale = float("nan")
        corr_abs_error_epi_over_ale = float("nan")
        corr_dock_epi_over_ale = float("nan")

    corr_dock_ale_std = _safe_tensor_corr(y_true, ale_std)
    corr_dock_epi_std = _safe_tensor_corr(y_true[epi_finite_mask], epi_finite) if epi_finite.numel() > 2 else float("nan")

    if nu_terms:
        nu_all = torch.cat(nu_terms, dim=0)
        alpha_all = torch.cat(alpha_terms, dim=0)
        beta_all = torch.cat(beta_terms, dim=0)
        mean_nu = nu_all.mean().item()
        mean_alpha = alpha_all.mean().item()
        mean_beta = beta_all.mean().item()
    else:
        mean_nu = float("nan")
        mean_alpha = float("nan")
        mean_beta = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nll": nll,
        "mean_std": total_std.mean().item(),
        "median_std": total_std.median().item(),
        "p90_std": torch.quantile(total_std, 0.9).item(),
        "corr_abs_error_std": corr_abs_error_std,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
        "mean_ale_std": ale_std.mean().item(),
        "mean_epi_std": mean_epi_std,
        "corr_abs_error_ale_std": corr_abs_error_ale_std,
        "corr_abs_error_epi_std": corr_abs_error_epi_std,
        "corr_dock_ale_std": corr_dock_ale_std,
        "corr_dock_epi_std": corr_dock_epi_std,
        "mean_epi_over_ale": mean_epi_over_ale,
        "median_epi_over_ale": median_epi_over_ale,
        "std_log_epi_over_ale": std_log_epi_over_ale,
        "spearman_epi_ale": spearman_epi_ale,
        "corr_abs_error_epi_over_ale": corr_abs_error_epi_over_ale,
        "corr_dock_epi_over_ale": corr_dock_epi_over_ale,
        "mean_nu": mean_nu,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
    }

def _apply_config_defaults(args, defaults, cfg: dict) -> tuple[argparse.Namespace, dict]:
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    torchmd_cfg = model_cfg.get("torchmd", {}) if isinstance(model_cfg, dict) else {}

    def set_if_default(name: str, value):
        if value is None:
            return
        if getattr(args, name) == getattr(defaults, name):
            setattr(args, name, value)

    set_if_default("root", train_cfg.get("root"))
    set_if_default("device", train_cfg.get("device"))
    set_if_default("batch_size", train_cfg.get("batch_size"))
    set_if_default("epochs", train_cfg.get("epochs"))
    set_if_default("lr", train_cfg.get("lr"))
    set_if_default("weight_decay", train_cfg.get("weight_decay"))
    set_if_default("hidden_dim", train_cfg.get("hidden_dim"))
    set_if_default("num_layers", train_cfg.get("num_layers"))
    set_if_default("dropout", train_cfg.get("dropout"))
    set_if_default("standardize", train_cfg.get("standardize"))
    set_if_default("save_path", train_cfg.get("save_path"))
    set_if_default("auto_prepare", train_cfg.get("auto_prepare"))
    set_if_default("confgen_num_confs", train_cfg.get("confgen_num_confs"))
    set_if_default("confgen_max_attempts", train_cfg.get("confgen_max_attempts"))
    set_if_default("confgen_max_opt_iters", train_cfg.get("confgen_max_opt_iters"))
    set_if_default("fp_dim", train_cfg.get("fp_dim"))
    set_if_default("fp_radius", train_cfg.get("fp_radius"))
    set_if_default("gaussian_warmup_epochs", train_cfg.get("gaussian_warmup_epochs"))
    set_if_default("gaussian_var_reg_beta", train_cfg.get("gaussian_var_reg_beta"))
    set_if_default("gaussian_logvar_min", train_cfg.get("gaussian_logvar_min"))
    set_if_default("gaussian_logvar_max", train_cfg.get("gaussian_logvar_max"))
    set_if_default("gaussian_min_var", train_cfg.get("gaussian_min_var"))
    set_if_default("backbone", train_cfg.get("backbone"))
    set_if_default("eval_samples", train_cfg.get("eval_samples"))
    set_if_default("scheduler", train_cfg.get("scheduler"))
    set_if_default("warmup_epochs", train_cfg.get("warmup_epochs"))
    set_if_default("min_lr", train_cfg.get("min_lr"))
    set_if_default("early_stop_patience", train_cfg.get("early_stop_patience"))
    set_if_default("early_stop_min_delta", train_cfg.get("early_stop_min_delta"))
    if args.confgen_optimize == defaults.confgen_optimize and "confgen_optimize" in train_cfg:
        args.confgen_optimize = bool(train_cfg.get("confgen_optimize", False))
    if args.confgen_prefer_mmff == defaults.confgen_prefer_mmff and "confgen_prefer_mmff" in train_cfg:
        args.confgen_prefer_mmff = bool(train_cfg.get("confgen_prefer_mmff", False))

    if args.no_edge_attr == defaults.no_edge_attr and "use_edge_attr" in train_cfg:
        args.no_edge_attr = not bool(train_cfg.get("use_edge_attr", True))
    if args.no_ligand_mask == defaults.no_ligand_mask and "use_ligand_mask" in train_cfg:
        args.no_ligand_mask = not bool(train_cfg.get("use_ligand_mask", True))
    if args.pocket_graph == defaults.pocket_graph and "pocket_graph" in train_cfg:
        args.pocket_graph = bool(train_cfg.get("pocket_graph", False))
    if args.ligand_only == defaults.ligand_only and "ligand_only" in train_cfg:
        args.ligand_only = bool(train_cfg.get("ligand_only", True))

    return args, torchmd_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a GIN surrogate for docking score prediction.")
    parser.add_argument("--config", default="config/surrogate/config_surrogate.yaml", help="YAML config for surrogate training.")
    parser.add_argument("--backbone", default="gin", help="Surrogate backbone: gin or tensornet.")
    parser.add_argument("--root", default="dataset/6KRO", help="Dataset root (contains processed/).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-edge-attr", action="store_true")
    parser.add_argument("--no-ligand-mask", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-path", default="checkpoints/gin_surrogate.pt")
    parser.add_argument("--auto-prepare", action="store_true")
    parser.add_argument("--confgen-num-confs", type=int, default=10)
    parser.add_argument("--confgen-max-attempts", type=int, default=10)
    parser.add_argument("--confgen-max-opt-iters", type=int, default=200)
    parser.add_argument("--confgen-optimize", action="store_true")
    parser.add_argument("--confgen-prefer-mmff", action="store_true")
    parser.add_argument("--fp-dim", type=int, default=2048, help="Morgan fingerprint size (0 to disable).")
    parser.add_argument("--fp-radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--gaussian-warmup-epochs", type=int, default=5)
    parser.add_argument("--gaussian-var-reg-beta", type=float, default=1e-4)
    parser.add_argument("--gaussian-logvar-min", type=float, default=-8.0)
    parser.add_argument("--gaussian-logvar-max", type=float, default=4.0)
    parser.add_argument("--gaussian-min-var", type=float, default=1e-6)
    parser.add_argument("--eval-samples", type=int, default=1, help="Reserved for compatibility; Gaussian head uses analytic mean/variance.")
    parser.add_argument("--scheduler", default="none", help="LR scheduler: none|cosine.")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--ligand-only", action="store_true", default=True, help="Train on ligand-only subgraphs.")
    parser.add_argument("--pocket-graph", action="store_true", help="Train on full pocket+ligand graphs.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="gin_surrogate", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team (optional).")
    parser.add_argument("--wandb-name", default=None, help="W&B run name (optional).")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", default="online", help="W&B mode: online/offline/disabled.")
    defaults = parser.parse_args([])
    args = parser.parse_args()
    log_path = Path("train_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    atexit.register(log_file.close)
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = load_config(str(cfg_path))
            if isinstance(cfg, dict):
                args, torchmd_cfg = _apply_config_defaults(args, defaults, cfg)
            else:
                torchmd_cfg = {}
        else:
            torchmd_cfg = {}
    else:
        torchmd_cfg = {}

    use_edge_attr = not args.no_edge_attr
    use_ligand_mask = not args.no_ligand_mask
    if args.pocket_graph and args.fp_dim > 0:
        print("[warn] pocket-graph mode does not attach fingerprints; disabling fp_dim.")
        args.fp_dim = 0

    backbone = str(args.backbone).lower()
    torchmd_cfg_used = {}
    if backbone in {"tensornet", "tsa", "tensor"}:
        train_ds = LigandOnly3DDataset(
            args.root,
            split="train",
            confgen_max_attempts=args.confgen_max_attempts,
            confgen_num_confs=args.confgen_num_confs,
            confgen_max_opt_iters=args.confgen_max_opt_iters,
            confgen_optimize=args.confgen_optimize,
            confgen_prefer_mmff=args.confgen_prefer_mmff,
            fp_dim=args.fp_dim,
            fp_radius=args.fp_radius,
        )
        val_ds = LigandOnly3DDataset(
            args.root,
            split="valid",
            confgen_max_attempts=args.confgen_max_attempts,
            confgen_num_confs=args.confgen_num_confs,
            confgen_max_opt_iters=args.confgen_max_opt_iters,
            confgen_optimize=args.confgen_optimize,
            confgen_prefer_mmff=args.confgen_prefer_mmff,
            fp_dim=args.fp_dim,
            fp_radius=args.fp_radius,
        )
        test_ds = LigandOnly3DDataset(
            args.root,
            split="test",
            confgen_max_attempts=args.confgen_max_attempts,
            confgen_num_confs=args.confgen_num_confs,
            confgen_max_opt_iters=args.confgen_max_opt_iters,
            confgen_optimize=args.confgen_optimize,
            confgen_prefer_mmff=args.confgen_prefer_mmff,
            fp_dim=args.fp_dim,
            fp_radius=args.fp_radius,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        node_feat_dim = int(getattr(train_ds, "num_node_classes_", 0))
        if node_feat_dim <= 0:
            raise RuntimeError("TensorNet training requires non-empty atom vocab features.")

        head_hidden_dim = torchmd_cfg.get("head_hidden_dim", None)
        head_hidden_dim = None if head_hidden_dim in (None, "", "null") else int(head_hidden_dim)
        head_num_layers = int(torchmd_cfg.get("head_num_layers", 2))

        torchmd_cfg_used = {
            "embedding_dim": int(torchmd_cfg.get("embedding_dim", args.hidden_dim)),
            "num_layers": int(torchmd_cfg.get("num_layers", 3)),
            "num_rbf": int(torchmd_cfg.get("num_rbf", 32)),
            "rbf_type": str(torchmd_cfg.get("rbf_type", "expnorm")),
            "trainable_rbf": bool(torchmd_cfg.get("trainable_rbf", False)),
            "activation": str(torchmd_cfg.get("activation", "silu")),
            "cutoff_lower": float(torchmd_cfg.get("cutoff_lower", 0.0)),
            "cutoff_upper": float(torchmd_cfg.get("cutoff_upper", 4.5)),
            "node_feat_dim": node_feat_dim,
            "max_num_neighbors": int(torchmd_cfg.get("max_num_neighbors", 64)),
            "equivariance_invariance_group": str(
                torchmd_cfg.get("equivariance_group", torchmd_cfg.get("equivariance_invariance_group", "O(3)"))
            ),
            "static_shapes": bool(torchmd_cfg.get("static_shapes", True)),
            "check_errors": bool(torchmd_cfg.get("check_errors", True)),
            "dropout": float(torchmd_cfg.get("dropout", args.dropout)),
            "reduce_op": str(torchmd_cfg.get("reduce", torchmd_cfg.get("reduce_op", "sum"))),
            "head_hidden_dim": head_hidden_dim,
            "head_num_layers": head_num_layers,
            "gaussian_warmup_epochs": int(torchmd_cfg.get("gaussian_warmup_epochs", args.gaussian_warmup_epochs)),
            "gaussian_var_reg_beta": float(torchmd_cfg.get("gaussian_var_reg_beta", args.gaussian_var_reg_beta)),
            "gaussian_logvar_min": float(torchmd_cfg.get("gaussian_logvar_min", args.gaussian_logvar_min)),
            "gaussian_logvar_max": float(torchmd_cfg.get("gaussian_logvar_max", args.gaussian_logvar_max)),
            "gaussian_min_var": float(torchmd_cfg.get("gaussian_min_var", args.gaussian_min_var)),
        }
        if torchmd_cfg_used["head_hidden_dim"] is None:
            torchmd_cfg_used["head_hidden_dim"] = torchmd_cfg_used["embedding_dim"]

        model = TensorNetSurrogate(
            embedding_dim=torchmd_cfg_used["embedding_dim"],
            num_layers=torchmd_cfg_used["num_layers"],
            num_rbf=torchmd_cfg_used["num_rbf"],
            rbf_type=torchmd_cfg_used["rbf_type"],
            trainable_rbf=torchmd_cfg_used["trainable_rbf"],
            activation=torchmd_cfg_used["activation"],
            cutoff_lower=torchmd_cfg_used["cutoff_lower"],
            cutoff_upper=torchmd_cfg_used["cutoff_upper"],
            node_feat_dim=node_feat_dim,
            max_num_neighbors=torchmd_cfg_used["max_num_neighbors"],
            equivariance_invariance_group=torchmd_cfg_used["equivariance_invariance_group"],
            static_shapes=torchmd_cfg_used["static_shapes"],
            check_errors=torchmd_cfg_used["check_errors"],
            dropout=torchmd_cfg_used["dropout"],
            reduce_op=torchmd_cfg_used["reduce_op"],
            fp_dim=args.fp_dim,
            head_hidden_dim=torchmd_cfg_used["head_hidden_dim"],
            head_num_layers=torchmd_cfg_used["head_num_layers"],
        )
        node_dim = node_feat_dim
        edge_dim = 0
        atom_extra_dim = int(getattr(train_ds, "atom_extra_dim", 0))
        bond_extra_dim = 0
    else:
        ligand_vocab = build_ligand_vocab(args.root)
        if ligand_vocab:
            clear_processed_if_vocab_mismatch(args.root, ligand_vocab)

        train_base = PocketLigandDockingDataset(
            args.root,
            split="train",
            auto_prepare=args.auto_prepare,
            ligand_vocab_override=ligand_vocab or None,
        )
        val_base = PocketLigandDockingDataset(
            args.root,
            split="valid",
            auto_prepare=False,
            ligand_vocab_override=ligand_vocab or None,
        )
        test_base = PocketLigandDockingDataset(
            args.root,
            split="test",
            auto_prepare=False,
            ligand_vocab_override=ligand_vocab or None,
        )
        if args.pocket_graph:
            train_ds = train_base
            val_ds = val_base
            test_ds = test_base
        elif args.ligand_only:
            train_ds = LigandOnlyDataset(train_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)
            val_ds = LigandOnlyDataset(val_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)
            test_ds = LigandOnlyDataset(test_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)
        else:
            train_ds = LigandOnlyDataset(train_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)
            val_ds = LigandOnlyDataset(val_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)
            test_ds = LigandOnlyDataset(test_base, fp_dim=args.fp_dim, fp_radius=args.fp_radius)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        node_dim = train_ds.num_node_classes_
        sample = train_ds[0]
        if use_edge_attr and hasattr(sample, "edge_attr") and sample.edge_attr is not None:
            edge_dim = int(sample.edge_attr.size(-1))
        else:
            edge_dim = train_ds.num_edge_classes_
        model = GINSurrogate(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_edge_attr=use_edge_attr,
            use_ligand_mask=use_ligand_mask,
            fp_dim=args.fp_dim,
        )
        atom_extra_dim = int(getattr(train_ds, "atom_extra_dim", 0))
        bond_extra_dim = int(getattr(train_ds, "bond_extra_dim", max(edge_dim - 5, 0)))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_name = str(args.scheduler).lower()
    warmup_epochs = int(args.warmup_epochs)
    min_lr = float(args.min_lr)
    early_stop_patience = int(args.early_stop_patience)
    early_stop_min_delta = float(args.early_stop_min_delta)
    gaussian_warmup_epochs = int(args.gaussian_warmup_epochs)
    gaussian_var_reg_beta = float(args.gaussian_var_reg_beta)
    gaussian_logvar_min = float(args.gaussian_logvar_min)
    gaussian_logvar_max = float(args.gaussian_logvar_max)
    gaussian_min_var = float(args.gaussian_min_var)

    if args.standardize:
        mean, std = fit_target_stats(train_loader, device)
    else:
        mean, std = 0.0, 1.0

    print(f"train_size: {len(train_ds)} val_size: {len(val_ds)} test_size: {len(test_ds)}")
    print(f"node_dim: {node_dim} edge_dim: {edge_dim} device: {device}")
    print(f"target_mean: {mean:.4f} target_std: {std:.4f}")

    wandb_run = None
    if args.wandb:
        os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
        os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
        os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")
        try:
            import wandb  # type: ignore
        except Exception as exc:
            print(f"[warn] wandb not available: {exc}")
        else:
            run_cfg = dict(vars(args))
            run_cfg.update(
                {
                    "train_size": len(train_ds),
                    "val_size": len(val_ds),
                    "test_size": len(test_ds),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "atom_extra_dim": atom_extra_dim,
                    "bond_extra_dim": bond_extra_dim,
                    "target_mean": mean,
                    "target_std": std,
                }
            )
            run_cfg["backbone"] = backbone
            if torchmd_cfg_used:
                run_cfg["torchmd"] = torchmd_cfg_used
            tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_name or None,
                config=run_cfg,
                tags=tags if tags else None,
                mode=args.wandb_mode,
            )

    best_val = None
    best_epoch = None
    best_report = {}
    _print_epoch_header()
    for epoch in range(1, args.epochs + 1):
        if scheduler_name == "cosine":
            lr = _compute_lr(epoch, args.lr, args.epochs, warmup_epochs, min_lr)
            _set_optimizer_lr(optimizer, lr)
        model.train()
        losses = []
        mse_vals = []
        nll_vals = []
        var_vals = []
        for batch in train_loader:
            batch = batch.to(device)
            y = get_targets(batch)
            if y is None:
                continue
            y = y.to(device)
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            y = (y[mask] - mean) / std
            raw = model(batch)
            mu, raw_log_var = split_gaussian_output(raw)
            mu = mu[mask]
            raw_log_var = raw_log_var[mask]
            mse = torch.mean((mu - y) ** 2)
            if epoch <= gaussian_warmup_epochs:
                loss = mse
            else:
                nll = gaussian_nll_loss(
                    mu,
                    raw_log_var,
                    y,
                    logvar_min=gaussian_logvar_min,
                    logvar_max=gaussian_logvar_max,
                    min_var=gaussian_min_var,
                )
                loss = nll + gaussian_var_reg_beta * torch.mean(raw_log_var ** 2)
                nll_vals.append(nll.item())
                var_vals.append(
                    gaussian_variance_from_raw(
                        raw_log_var,
                        logvar_min=gaussian_logvar_min,
                        logvar_max=gaussian_logvar_max,
                        min_var=gaussian_min_var,
                    ).mean().item()
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            mse_vals.append(mse.item())
        train_loss = sum(losses) / max(len(losses), 1)
        train_mse = sum(mse_vals) / max(len(mse_vals), 1)
        train_nll = sum(nll_vals) / max(len(nll_vals), 1) if nll_vals else float('nan')
        train_var = sum(var_vals) / max(len(var_vals), 1) if var_vals else float('nan')

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            mean,
            std,
            eval_samples=args.eval_samples,
            logvar_min=gaussian_logvar_min,
            logvar_max=gaussian_logvar_max,
            min_var=gaussian_min_var,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 1 and epoch != 1:
            _print_epoch_header()
        print(
            f"{epoch:4d} | {current_lr:9.2e} | {train_loss:10.4f} | "
            f"{val_metrics['rmse']:8.4f} | {val_metrics['mae']:7.4f} | {val_metrics['r2']:6.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/mse": train_mse,
                    "train/nll": train_nll,
                    "train/pred_var": train_var,
                    "train/lr": current_lr,
                    "val/rmse": val_metrics["rmse"],
                    "val/mae": val_metrics["mae"],
                    "val/r2": val_metrics["r2"],
                }
            )

        if best_val is None or val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_epoch = epoch
            best_report = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "lr": current_lr,
            }
            config = dict(vars(args))
            config["atom_extra_dim"] = atom_extra_dim
            config["bond_extra_dim"] = bond_extra_dim
            config["backbone"] = backbone
            if torchmd_cfg_used:
                config["torchmd"] = torchmd_cfg_used
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "mean": mean,
                    "std": std,
                    "config": config,
                },
                args.save_path,
            )
        if (
            early_stop_patience > 0
            and best_epoch is not None
            and (epoch - best_epoch) >= early_stop_patience
            and val_metrics["rmse"] > (best_val - early_stop_min_delta)
        ):
            print(f"[early_stop] epoch={epoch} best_epoch={best_epoch} best_val_rmse={best_val:.4f}")
            break

    test_metrics = evaluate(model, test_loader, device, mean, std, eval_samples=args.eval_samples)
    print("test  | rmse      | mae      | r2")
    print("------+-----------+----------+--------")
    print(
        f"test  | {test_metrics['rmse']:9.4f} | {test_metrics['mae']:8.4f} | {test_metrics['r2']:6.4f}"
    )
    best_payload = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_report.get("val_rmse"),
        "best_val_mae": best_report.get("val_mae"),
        "best_val_r2": best_report.get("val_r2"),
        "best_train_loss": best_report.get("train_loss"),
        "best_lr": best_report.get("lr"),
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
    }
    try:
        best_path = Path("train_best.json")
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2)
        if best_epoch is not None:
            print("best  | epoch | val_rmse | val_mae | val_r2 | lr")
            print("------+-------+----------+---------+--------+-----------")
            print(
                f"best  | {best_epoch:5d} | {best_report.get('val_rmse', float('nan')):8.4f} | "
                f"{best_report.get('val_mae', float('nan')):7.4f} | {best_report.get('val_r2', float('nan')):6.4f} | "
                f"{best_report.get('lr', float('nan')):9.2e}"
            )
        print(f"[best] saved report: {best_path}")
    except Exception as exc:
        print(f"[warn] failed to write train_best.json: {exc}")
    if wandb_run is not None:
        wandb_run.log(
            {
                "test/rmse": test_metrics["rmse"],
                "test/mae": test_metrics["mae"],
                "test/r2": test_metrics["r2"],
            }
        )
        wandb_run.finish()
    try:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


