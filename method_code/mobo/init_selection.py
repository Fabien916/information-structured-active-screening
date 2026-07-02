from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.loader import DataLoader

from data.ligand_only_3d_dataset import LigandOnly3DStore
from mobo.constants import ATOM_EXTRA_DIM
from mobo.io_utils import _load_ligand_vocab_override, _read_smiles_csv, _scan_ligand_vocab
from train_gin_surrogate import TensorNetEncoder


@dataclass(frozen=True)
class FingerprintSelectionResult:
    selected: pd.DataFrame
    assignments: pd.DataFrame
    quotas: pd.DataFrame
    fp_mat: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float


@dataclass(frozen=True)
class LatentSelectionResult:
    selected: pd.DataFrame
    assignments: pd.DataFrame
    quotas: pd.DataFrame
    latent_z: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    ligand_vocab: tuple[str, ...]
    atom_extra_dim: int
    training_dataset_root: str | None
    input_library_size: int
    encoded_library_size: int
    failed_library_size: int
    failure_report_path: str | None


def build_fingerprint_library(
    df: pd.DataFrame,
    smiles_col: str,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> tuple[pd.DataFrame, np.ndarray]:
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column: {smiles_col}")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(fp_radius), fpSize=int(fp_bits))
    rows = []
    fps = []
    seen: set[str] = set()
    failed_rows = []
    for row in df.to_dict(orient="records"):
        smiles = str(row.get(smiles_col, "")).strip()
        if not smiles:
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError("MolFromSmiles returned None")
            mol = Chem.RemoveHs(mol, sanitize=True, implicitOnly=False)
            canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception as exc:
            failed_rows.append({"smiles": smiles, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not canon or canon in seen:
            continue
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            failed_rows.append({"smiles": smiles, "error": "MolFromSmiles returned None after canonicalization"})
            continue
        seen.add(canon)
        out = dict(row)
        out["smiles"] = canon
        out["smiles_canonical"] = canon
        rows.append(out)
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((int(fp_bits),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    if not rows:
        raise RuntimeError("No valid molecules available for fingerprint selection.")
    if failed_rows:
        example = failed_rows[0]
        print(
            f"[init_selection] skipped {len(failed_rows)} molecules that RDKit could not canonicalize; "
            f"first error={example['error']} smiles={example['smiles']}"
        )
    return pd.DataFrame(rows), np.stack(fps, axis=0)


def assign_labels(x: np.ndarray, centroids: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    labels = np.empty(x.shape[0], dtype=np.int32)
    inertia = 0.0
    c_norm = np.sum(centroids * centroids, axis=1, dtype=np.float32)
    for start in range(0, x.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), x.shape[0])
        xb = x[start:stop].astype(np.float32, copy=False)
        x_norm = np.sum(xb * xb, axis=1, dtype=np.float32)
        dots = xb @ centroids.T
        dists = x_norm[:, None] + c_norm[None, :] - 2.0 * dots
        lab = np.argmin(dists, axis=1)
        labels[start:stop] = lab
        inertia += float(np.take_along_axis(dists, lab[:, None], axis=1).sum())
    return labels, float(inertia)


def fit_minibatch_kmeans(
    x: np.ndarray,
    n_clusters: int,
    batch_size: int,
    epochs: int,
    n_init: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D fingerprint matrix, got {tuple(x.shape)}")
    if x.shape[0] < int(n_clusters):
        raise RuntimeError(f"n_clusters={n_clusters} exceeds sample count={x.shape[0]}")
    best_centroids = None
    best_labels = None
    best_inertia = None
    rng_master = np.random.default_rng(int(seed))
    for _ in range(int(n_init)):
        init_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        centroids = x[init_rng.choice(x.shape[0], size=int(n_clusters), replace=False)].astype(np.float32, copy=True)
        counts = np.zeros(int(n_clusters), dtype=np.int64)
        order = np.arange(x.shape[0], dtype=np.int64)
        epoch_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        for _epoch in range(int(epochs)):
            epoch_rng.shuffle(order)
            for start in range(0, x.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), x.shape[0])
                batch_idx = order[start:stop]
                xb = x[batch_idx].astype(np.float32, copy=False)
                x_norm = np.sum(xb * xb, axis=1, dtype=np.float32)
                c_norm = np.sum(centroids * centroids, axis=1, dtype=np.float32)
                dots = xb @ centroids.T
                dists = x_norm[:, None] + c_norm[None, :] - 2.0 * dots
                labels = np.argmin(dists, axis=1)
                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    pts = xb[mask]
                    n_pts = int(pts.shape[0])
                    if n_pts <= 0:
                        continue
                    counts[cluster_id] += n_pts
                    eta = float(n_pts) / float(counts[cluster_id])
                    centroids[cluster_id] = (1.0 - eta) * centroids[cluster_id] + eta * np.mean(pts, axis=0, dtype=np.float32)
        labels, inertia = assign_labels(x, centroids, batch_size=int(batch_size))
        if best_inertia is None or inertia < best_inertia:
            best_centroids = centroids.copy()
            best_labels = labels.copy()
            best_inertia = float(inertia)
    if best_centroids is None or best_labels is None or best_inertia is None:
        raise RuntimeError("Failed to fit fingerprint k-means.")
    return best_centroids, best_labels, best_inertia


def allocate_cluster_quota(cluster_sizes: np.ndarray, total_size: int, min_per_cluster: int) -> np.ndarray:
    n = int(cluster_sizes.shape[0])
    total_size = int(total_size)
    min_per_cluster = int(min_per_cluster)
    sizes = cluster_sizes.astype(np.int64, copy=False)
    if np.any(sizes < 0):
        raise RuntimeError("Cluster sizes must be non-negative.")
    if int(sizes.sum()) < total_size:
        raise RuntimeError(f"total_size={total_size} exceeds total clustered samples={int(sizes.sum())}")

    base = np.minimum(sizes, int(min_per_cluster)).astype(np.int64)
    if total_size < int(base.sum()):
        raise RuntimeError(
            f"total_size={total_size} is smaller than capped cluster minimum allocation={int(base.sum())}"
        )
    quotas = base.copy()
    remain = int(total_size - quotas.sum())
    if remain <= 0:
        return quotas.astype(np.int32)

    capacity = sizes - quotas
    weights = np.sqrt(sizes.astype(np.float64))
    weights[capacity <= 0] = 0.0
    weights_sum = float(weights.sum())
    if weights_sum <= 0.0:
        raise RuntimeError("No cluster capacity remains for quota allocation.")
    frac = remain * weights / weights_sum
    add = np.minimum(np.floor(frac).astype(np.int64), capacity)
    quotas += add
    leftover = int(total_size - quotas.sum())
    if leftover > 0:
        residual = frac - add.astype(np.float64)
        while leftover > 0:
            remaining_capacity = sizes - quotas
            if int(remaining_capacity.sum()) < leftover:
                raise RuntimeError(
                    f"Insufficient remaining cluster capacity for quota leftover={leftover}; "
                    f"remaining_capacity={int(remaining_capacity.sum())}"
                )
            residual[remaining_capacity <= 0] = -1.0
            order = np.argsort(-residual)
            progressed = False
            for cluster_id in order:
                if leftover <= 0:
                    break
                if remaining_capacity[int(cluster_id)] <= 0:
                    continue
                quotas[int(cluster_id)] += 1
                leftover -= 1
                progressed = True
            if not progressed:
                raise RuntimeError("Failed to distribute cluster quota leftover despite available capacity.")
    return quotas.astype(np.int32)


def select_cluster_members(
    fp_mat: np.ndarray,
    quotas: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    picks = []
    for cluster_id in range(int(centroids.shape[0])):
        quota = int(quotas[cluster_id])
        idx = np.flatnonzero(labels == cluster_id)
        if idx.size == 0 or quota <= 0:
            continue
        quota = min(quota, int(idx.size))
        sub = fp_mat[idx].astype(np.float32, copy=False)
        center = centroids[cluster_id][None, :].astype(np.float32, copy=False)
        d_center = np.sum((sub - center) ** 2, axis=1)
        first_local = int(np.argmin(d_center))
        chosen_local = [first_local]
        chosen_global = [int(idx[first_local])]
        if quota > 1:
            min_dist = np.sum((sub - sub[first_local:first_local + 1]) ** 2, axis=1)
            min_dist[first_local] = -1.0
            while len(chosen_local) < quota:
                next_local = int(np.argmax(min_dist))
                chosen_local.append(next_local)
                chosen_global.append(int(idx[next_local]))
                d_new = np.sum((sub - sub[next_local:next_local + 1]) ** 2, axis=1)
                min_dist = np.minimum(min_dist, d_new)
                min_dist[chosen_local] = -1.0
        picks.extend(chosen_global)
    if len(set(picks)) != len(picks):
        raise RuntimeError("Duplicate indices detected in fingerprint initialization selection.")
    return np.asarray(picks, dtype=np.int64)


def select_fingerprint_init_set(
    df: pd.DataFrame,
    smiles_col: str,
    n_clusters: int,
    init_size: int,
    min_per_cluster: int = 1,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    batch_size: int = 2048,
    epochs: int = 10,
    n_init: int = 2,
    seed: int = 42,
) -> FingerprintSelectionResult:
    lib_df, fp_mat = build_fingerprint_library(df, smiles_col=smiles_col, fp_radius=fp_radius, fp_bits=fp_bits)
    centroids, labels, inertia = fit_minibatch_kmeans(
        fp_mat,
        n_clusters=int(n_clusters),
        batch_size=int(batch_size),
        epochs=int(epochs),
        n_init=int(n_init),
        seed=int(seed),
    )
    lib_df = lib_df.copy()
    lib_df["cluster"] = labels.astype(np.int32)
    cluster_sizes = np.bincount(labels.astype(np.int32), minlength=int(n_clusters))
    quotas = allocate_cluster_quota(cluster_sizes, total_size=int(init_size), min_per_cluster=int(min_per_cluster))
    pick_idx = select_cluster_members(fp_mat, quotas=quotas, labels=labels, centroids=centroids)
    selected = lib_df.iloc[pick_idx].copy().reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, selected.shape[0] + 1, dtype=np.int32)
    quota_df = pd.DataFrame({
        "cluster": np.arange(int(n_clusters), dtype=np.int32),
        "size": cluster_sizes.astype(np.int32),
        "quota": quotas.astype(np.int32),
    })
    return FingerprintSelectionResult(
        selected=selected,
        assignments=lib_df,
        quotas=quota_df,
        fp_mat=fp_mat,
        labels=labels,
        centroids=centroids,
        inertia=float(inertia),
    )


def build_latent_library(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column: {smiles_col}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in df.to_dict(orient="records"):
        source_smiles = str(row.get(smiles_col, "")).strip()
        if not source_smiles:
            raise RuntimeError("Encountered empty SMILES while loading the latent initialization library.")
        canon = canonicalize_smiles_noh(source_smiles)
        if not canon:
            raise RuntimeError(f"Invalid SMILES encountered in the latent initialization library: {source_smiles}")
        if canon in seen:
            continue
        seen.add(canon)
        item = dict(row)
        item["smiles"] = canon
        item["smiles_canonical"] = canon
        rows.append(item)
    if not rows:
        raise RuntimeError("No valid molecules available in the latent initialization library.")
    out = pd.DataFrame(rows).reset_index(drop=True)
    if "library_id" not in out.columns:
        out.insert(0, "library_id", np.arange(1, out.shape[0] + 1, dtype=np.int32))
    return out


def _load_vocab_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"Expected a list in ligand vocab JSON: {path}")
        vocab = [str(x).strip() for x in data if str(x).strip()]
    else:
        vocab = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not vocab:
        raise RuntimeError(f"Ligand vocab file is empty: {path}")
    return sorted(set(vocab))


def _resolve_training_root(training_dataset_root: str | Path | None, ckpt_cfg: dict, cfg: dict) -> Path | None:
    if training_dataset_root:
        path = Path(training_dataset_root)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    dataset_root = ckpt_cfg.get("dataset_root")
    if dataset_root:
        path = Path(str(dataset_root))
        if path.exists():
            return path
    general_cfg = dict(cfg.get("general") or {})
    dataset_root = general_cfg.get("dataset_root")
    if dataset_root:
        path = Path(str(dataset_root))
        if path.exists():
            return path
    return None


def _resolve_ligand_vocab(
    training_dataset_root: str | Path | None,
    ligand_vocab_file: str | Path | None,
    ckpt_cfg: dict,
    cfg: dict,
    expected_node_feat_dim: int | None,
    library_smiles: list[str],
) -> tuple[list[str], int, Path | None]:
    training_root = _resolve_training_root(training_dataset_root, ckpt_cfg, cfg)
    if ligand_vocab_file:
        ligand_vocab = _load_vocab_file(Path(ligand_vocab_file))
    else:
        ligand_vocab = None
        if training_root is not None:
            ligand_vocab = _load_ligand_vocab_override(training_root)
        if ligand_vocab is None:
            ligand_vocab = _scan_ligand_vocab(library_smiles)
    if not ligand_vocab:
        raise RuntimeError("Resolved ligand vocabulary is empty.")
    if expected_node_feat_dim is None:
        atom_extra_dim = int(ATOM_EXTRA_DIM)
    else:
        atom_extra_dim = int(expected_node_feat_dim) - len(ligand_vocab)
        if atom_extra_dim < 0:
            raise RuntimeError(
                f"Resolved ligand vocab size={len(ligand_vocab)} exceeds checkpoint node_feat_dim={expected_node_feat_dim}."
            )
    return ligand_vocab, atom_extra_dim, training_root


def _extract_encoder_state(ckpt: dict) -> dict[str, torch.Tensor]:
    encoder_state = ckpt.get("encoder_state")
    if encoder_state is not None:
        return encoder_state
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Checkpoint missing both encoder_state and model_state.")
    extracted = {
        key[len("encoder.") :]: value
        for key, value in model_state.items()
        if str(key).startswith("encoder.")
    }
    if not extracted:
        raise RuntimeError("Checkpoint model_state does not contain encoder.* parameters.")
    return extracted


def _build_encoder(
    checkpoint_path: Path,
    cfg: dict,
    ligand_vocab: list[str],
    atom_extra_dim: int,
) -> tuple[TensorNetEncoder, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = dict(ckpt.get("config") or {})
    torchmd_cfg = dict(ckpt_cfg.get("torchmd") or {})
    if not torchmd_cfg:
        torchmd_cfg = dict(((cfg.get("model") or {}).get("torchmd") or {}))
    if not torchmd_cfg:
        raise RuntimeError("Checkpoint/config missing model.torchmd settings for TensorNet encoder reconstruction.")
    node_feat_dim = torchmd_cfg.get("node_feat_dim", ckpt_cfg.get("node_feat_dim"))
    if node_feat_dim is None:
        node_feat_dim = len(ligand_vocab) + int(atom_extra_dim)
    node_feat_dim = int(node_feat_dim)
    expected_node_feat_dim = len(ligand_vocab) + int(atom_extra_dim)
    if node_feat_dim != expected_node_feat_dim:
        raise RuntimeError(
            f"TensorNet node_feat_dim mismatch: checkpoint expects {node_feat_dim}, resolved vocab implies {expected_node_feat_dim}."
        )
    encoder = TensorNetEncoder(
        embedding_dim=int(torchmd_cfg.get("embedding_dim", 128)),
        num_layers=int(torchmd_cfg.get("num_layers", 2)),
        num_rbf=int(torchmd_cfg.get("num_rbf", 32)),
        rbf_type=str(torchmd_cfg.get("rbf_type", "expnorm")),
        trainable_rbf=bool(torchmd_cfg.get("trainable_rbf", False)),
        activation=str(torchmd_cfg.get("activation", "silu")),
        cutoff_lower=float(torchmd_cfg.get("cutoff_lower", 0.0)),
        cutoff_upper=float(torchmd_cfg.get("cutoff_upper", 4.5)),
        node_feat_dim=node_feat_dim,
        max_num_neighbors=int(torchmd_cfg.get("max_num_neighbors", 64)),
        equivariance_invariance_group=str(
            torchmd_cfg.get("equivariance_invariance_group", torchmd_cfg.get("equivariance_group", "O(3)"))
        ),
        static_shapes=bool(torchmd_cfg.get("static_shapes", True)),
        check_errors=bool(torchmd_cfg.get("check_errors", True)),
        dropout=float(torchmd_cfg.get("dropout", 0.0)),
        reduce_op=str(torchmd_cfg.get("reduce_op", torchmd_cfg.get("reduce", "sum"))),
        fp_dim=0,
    )
    encoder.load_state_dict(_extract_encoder_state(ckpt), strict=True)
    return encoder, ckpt_cfg


def _encode_library_with_store(
    library_df: pd.DataFrame,
    encoder: TensorNetEncoder,
    device: torch.device,
    ligand_vocab: list[str],
    atom_extra_dim: int,
    batch_size: int,
    num_workers: int,
    cache_dir: str | Path | None,
    confgen_cfg: dict,
    mp_chunksize: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    work_df = library_df.copy().reset_index(drop=True)
    internal_id_col = "__latent_row_id__"
    original_ligand_col = "__latent_orig_ligand_id__"
    work_df[internal_id_col] = [f"latent_row_{i}" for i in range(work_df.shape[0])]
    if "ligand_id" in work_df.columns:
        work_df[original_ligand_col] = work_df["ligand_id"]
    work_df["ligand_id"] = work_df[internal_id_col]
    rows = work_df.to_dict(orient="records")
    store = LigandOnly3DStore(
        root=Path(cache_dir or "."),
        ligand_vocab_override=ligand_vocab,
        atom_extra_dim=atom_extra_dim,
        cache_dir=cache_dir,
        fp_dim=0,
        rows=rows,
        confgen_max_attempts=int(confgen_cfg["max_attempts"]),
        confgen_seed=int(confgen_cfg["seed"]),
        confgen_num_confs=int(confgen_cfg["num_confs"]),
        confgen_max_opt_iters=int(confgen_cfg["max_opt_iters"]),
        confgen_optimize=bool(confgen_cfg["optimize"]),
        confgen_prefer_mmff=bool(confgen_cfg["prefer_mmff"]),
        build_num_workers=int(num_workers),
        build_mp_chunksize=int(mp_chunksize),
    )
    success_rows: list[dict] = []
    row_by_id = {str(row[internal_id_col]): row for row in work_df.to_dict(orient="records")}
    for data in store.data_list:
        lig_id = str(getattr(data, "ligand_id", "")).strip()
        row = row_by_id.get(lig_id)
        if row is None:
            raise RuntimeError(f"Failed to map latent-initialization success row: ligand_id={lig_id}")
        success_rows.append(dict(row))
    encoded_df = pd.DataFrame(success_rows)
    failed_rows: list[dict] = []
    for row in work_df.to_dict(orient="records"):
        cache_key = store._cache_key(str(row.get("ligand_id", "")), str(row.get("smiles_canonical", "")))
        cache_row = store._db.execute(
            "SELECT status, error_stage, error_message FROM ligand3d_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if cache_row is None:
            raise RuntimeError(f"Missing ligand3d cache row for latent initialization: {cache_key}")
        status = str(cache_row["status"])
        if status == "ready":
            continue
        out = dict(row)
        out["cache_key"] = cache_key
        out["error_stage"] = None if cache_row["error_stage"] is None else str(cache_row["error_stage"])
        out["error_message"] = None if cache_row["error_message"] is None else str(cache_row["error_message"])
        failed_rows.append(out)
    failed_df = pd.DataFrame(failed_rows)
    if encoded_df.shape[0] + failed_df.shape[0] != work_df.shape[0]:
        raise RuntimeError(
            f"Latent initialization library bookkeeping mismatch: encoded={encoded_df.shape[0]} failed={failed_df.shape[0]} total={work_df.shape[0]}"
        )
    for frame in (encoded_df, failed_df):
        if internal_id_col in frame.columns:
            frame.drop(columns=[internal_id_col], inplace=True)
        if original_ligand_col in frame.columns:
            frame["ligand_id"] = frame[original_ligand_col]
            frame.drop(columns=[original_ligand_col], inplace=True)
    loader = DataLoader(
        store.get_dataset_from_indices(list(range(len(store.data_list)))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    encoder = encoder.to(device)
    encoder.eval()
    embeds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            graph_z = encoder(batch).detach().cpu().numpy()
            embeds.append(graph_z.astype(np.float32, copy=False))
    latent_z = np.concatenate(embeds, axis=0)
    if latent_z.shape[0] != encoded_df.shape[0]:
        raise RuntimeError(
            f"TensorNet latent row count does not match the encoded latent library size: latent={latent_z.shape[0]} encoded={encoded_df.shape[0]}"
        )
    return latent_z, encoded_df.reset_index(drop=True), failed_df.reset_index(drop=True)


def select_tensornet_latent_init_set(
    df: pd.DataFrame,
    smiles_col: str,
    checkpoint_path: str | Path,
    cfg: dict,
    n_clusters: int,
    init_size: int,
    min_per_cluster: int = 1,
    training_dataset_root: str | Path | None = None,
    ligand_vocab_file: str | Path | None = None,
    encode_batch_size: int = 256,
    encode_num_workers: int = 0,
    kmeans_batch_size: int = 2048,
    kmeans_epochs: int = 10,
    kmeans_n_init: int = 2,
    seed: int = 42,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
    confgen_cfg: Optional[dict] = None,
    build_mp_chunksize: int = 16,
    failure_report_path: str | Path | None = None,
) -> LatentSelectionResult:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    library_df = build_latent_library(df, smiles_col=smiles_col)

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_ckpt_cfg = dict(raw_ckpt.get("config") or {})
    raw_torchmd_cfg = dict(raw_ckpt_cfg.get("torchmd") or {})
    expected_node_feat_dim = raw_torchmd_cfg.get("node_feat_dim", raw_ckpt_cfg.get("node_feat_dim"))
    if expected_node_feat_dim is not None:
        expected_node_feat_dim = int(expected_node_feat_dim)
    ligand_vocab, atom_extra_dim, resolved_training_root = _resolve_ligand_vocab(
        training_dataset_root=training_dataset_root,
        ligand_vocab_file=ligand_vocab_file,
        ckpt_cfg=raw_ckpt_cfg,
        cfg=cfg,
        expected_node_feat_dim=expected_node_feat_dim,
        library_smiles=library_df["smiles_canonical"].astype(str).tolist(),
    )
    encoder, _ckpt_cfg = _build_encoder(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        ligand_vocab=ligand_vocab,
        atom_extra_dim=atom_extra_dim,
    )
    if confgen_cfg is None:
        candidate_cfg = dict(((cfg.get("candidate") or {}).get("candidate_3d") or {}))
        confgen_cfg = {
            "max_attempts": int(candidate_cfg.get("max_attempts", 3)),
            "seed": int(seed),
            "num_confs": int(candidate_cfg.get("num_confs", 4)),
            "max_opt_iters": int(candidate_cfg.get("max_opt_iters", 100)),
            "optimize": bool(candidate_cfg.get("optimize", True)),
            "prefer_mmff": bool(candidate_cfg.get("prefer_mmff", False)),
        }
    device_obj = torch.device(device)
    latent_z, encoded_library_df, failed_library_df = _encode_library_with_store(
        library_df=library_df,
        encoder=encoder,
        device=device_obj,
        ligand_vocab=ligand_vocab,
        atom_extra_dim=atom_extra_dim,
        batch_size=int(encode_batch_size),
        num_workers=int(encode_num_workers),
        cache_dir=cache_dir,
        confgen_cfg=confgen_cfg,
        mp_chunksize=int(build_mp_chunksize),
    )
    failure_report = None
    if failure_report_path is not None and not failed_library_df.empty:
        failure_report = Path(failure_report_path)
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        failed_library_df.to_csv(failure_report, index=False)
    if encoded_library_df.shape[0] < int(init_size):
        raise RuntimeError(
            f"Latent initialization library after 3D filtering is too small: encoded={encoded_library_df.shape[0]} init_size={init_size} failed={failed_library_df.shape[0]}"
        )
    if encoded_library_df.shape[0] < int(n_clusters):
        raise RuntimeError(
            f"Latent initialization library after 3D filtering has fewer rows than clusters: encoded={encoded_library_df.shape[0]} n_clusters={n_clusters} failed={failed_library_df.shape[0]}"
        )
    centroids, labels, inertia = fit_minibatch_kmeans(
        latent_z,
        n_clusters=int(n_clusters),
        batch_size=int(kmeans_batch_size),
        epochs=int(kmeans_epochs),
        n_init=int(kmeans_n_init),
        seed=int(seed),
    )
    ordered_df = encoded_library_df.copy().reset_index(drop=True)
    ordered_df["cluster"] = labels.astype(np.int32)
    cluster_sizes = np.bincount(labels.astype(np.int32), minlength=int(n_clusters))
    quotas = allocate_cluster_quota(cluster_sizes, total_size=int(init_size), min_per_cluster=int(min_per_cluster))
    pick_idx = select_cluster_members(latent_z, quotas=quotas, labels=labels, centroids=centroids)
    selected = ordered_df.iloc[pick_idx].copy().reset_index(drop=True)
    if selected.shape[0] != int(init_size):
        raise RuntimeError(
            f"TensorNet latent selection returned {selected.shape[0]} molecules, expected {init_size}. "
            f"cluster_sizes={cluster_sizes.tolist()} quotas={quotas.tolist()}"
        )
    selected["selection_rank"] = np.arange(1, selected.shape[0] + 1, dtype=np.int32)
    quota_df = pd.DataFrame({
        "cluster": np.arange(int(n_clusters), dtype=np.int32),
        "size": cluster_sizes.astype(np.int32),
        "quota": quotas.astype(np.int32),
    })
    return LatentSelectionResult(
        selected=selected,
        assignments=ordered_df,
        quotas=quota_df,
        latent_z=latent_z,
        labels=labels,
        centroids=centroids,
        inertia=float(inertia),
        ligand_vocab=tuple(ligand_vocab),
        atom_extra_dim=int(atom_extra_dim),
        training_dataset_root=None if resolved_training_root is None else str(resolved_training_root),
        input_library_size=int(library_df.shape[0]),
        encoded_library_size=int(encoded_library_df.shape[0]),
        failed_library_size=int(failed_library_df.shape[0]),
        failure_report_path=None if failure_report is None else str(failure_report),
    )


def load_latent_library_csv(csv_path: str | Path, smiles_col: str | None = None) -> tuple[pd.DataFrame, str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = _read_smiles_csv(str(csv_path))
    if df.empty:
        raise RuntimeError(f"Latent initialization library is empty: {csv_path}")
    if smiles_col is None:
        candidates = {str(col).lower(): str(col) for col in df.columns}
        for key in ("smiles_canonical", "smiles", "smile"):
            if key in candidates:
                smiles_col = candidates[key]
                break
    if smiles_col is None or smiles_col not in df.columns:
        raise RuntimeError(f"Failed to resolve SMILES column in latent initialization library: {csv_path}")
    return df, str(smiles_col)
