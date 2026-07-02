from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem
from tqdm.auto import tqdm

from graph.schema import compute_mask_ligand_edge
from data.pocket_dataset import _smiles_to_3d_mol
from mobo.constants import BOND_TYPE_CLASSES, ATOM_EXTRA_DIM, BOND_EXTRA_DIM
from mobo.smiles_utils import (
    compute_qed,
    compute_sa,
    _adjust_feature_vec,
    _atom_feature_vec,
    _bond_feature_vec,
    _smiles_to_fp,
)
from mobo.io_utils import _read_smiles_csv, _pick_smiles_column
from mobo.pretrain_targets import build_pretrain_props

_MP_CFG: dict | None = None


def _mp_init_graphs(cfg: dict) -> None:
    global _MP_CFG
    _MP_CFG = cfg
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _mp_build_graph_3d_worker(smiles: str) -> Tuple[str, Data | None, str | None]:
    cfg = _MP_CFG or {}
    try:
        data = smiles_to_ligand_3d(
            smiles,
            atom_index=cfg.get("atom_index"),
            atom_feat_dim=cfg.get("atom_feat_dim", 0),
            atom_extra_dim=cfg.get("atom_extra_dim", 0),
            fp_dim=cfg.get("fp_dim", 0),
            fp_radius=cfg.get("fp_radius", 2),
            max_attempts=cfg.get("max_attempts", 10),
            seed=cfg.get("seed", 0),
            num_confs=cfg.get("num_confs", 10),
            max_opt_iters=cfg.get("max_opt_iters", 200),
            optimize=cfg.get("optimize", True),
            prefer_mmff=cfg.get("prefer_mmff", True),
        )
        return (smiles, data, None)
    except Exception as exc:
        return (smiles, None, f"{type(exc).__name__}: {exc}")


def _mp_init_ligand_graphs(cfg: dict) -> None:
    _mp_init_graphs(cfg)


def _mp_build_ligand_graph_worker(item: Tuple[str, float, str]) -> Data:
    smi, dock_val, lig_id = item
    cfg = _MP_CFG or {}
    data = smiles_to_ligand_graph(
        smi,
        cfg.get("atom_index", {}),
        int(cfg.get("node_dim", 0)),
        int(cfg.get("edge_dim", 0)),
        atom_extra_dim=cfg.get("atom_extra_dim", 0),
        bond_extra_dim=cfg.get("bond_extra_dim", 0),
        fp_dim=cfg.get("fp_dim", 0),
        fp_radius=cfg.get("fp_radius", 2),
    )
    data.dock_score = dock_val
    data.vina_score = dock_val
    data.ligand_id = lig_id
    return data


def smiles_to_ligand_graph(
    smiles: str,
    atom_index: Dict[str, int],
    node_dim: int,
    edge_dim: int,
    atom_extra_dim: int | None = None,
    bond_extra_dim: int | None = None,
    fp_dim: int = 0,
    fp_radius: int = 2,
) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in graph conversion: {smiles}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(f"Empty molecule in graph conversion: {smiles}")
    x = torch.zeros((num_atoms, node_dim), dtype=torch.float32)
    atom_vocab_dim = len(atom_index)
    if atom_extra_dim is None:
        extra_dim = max(int(node_dim - atom_vocab_dim), 0)
    else:
        extra_dim = max(min(int(atom_extra_dim), node_dim - atom_vocab_dim), 0)
    for i, atom in enumerate(mol.GetAtoms()):
        idx = atom_index.get(atom.GetSymbol())
        if idx is None or idx >= node_dim:
            raise ValueError(f"Atom symbol '{atom.GetSymbol()}' missing in ligand vocab.")
        x[i, idx] = 1.0
        if extra_dim > 0:
            vec = _adjust_feature_vec(_atom_feature_vec(atom), extra_dim)
            if vec:
                x[i, atom_vocab_dim : atom_vocab_dim + extra_dim] = torch.tensor(vec, dtype=torch.float32)

    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []
    edge_feat: List[List[float]] = []
    ligand_pairs = set()
    for bond in mol.GetBonds():
        btype_raw = bond.GetBondType()
        if btype_raw not in bond_type_map:
            raise ValueError(f"Unsupported bond type in SMILES: {smiles}")
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond_type_map[btype_raw]
        ligand_pairs.add(tuple(sorted((i, j))))
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_type.extend([btype, btype])
        bfeat = _bond_feature_vec(bond)
        edge_feat.extend([bfeat, bfeat])

    for a in range(num_atoms):
        for b in range(a + 1, num_atoms):
            if (a, b) in ligand_pairs:
                continue
            edge_src.extend([a, b])
            edge_dst.extend([b, a])
            edge_type.extend([0, 0])
            edge_feat.extend([[0.0] * BOND_EXTRA_DIM, [0.0] * BOND_EXTRA_DIM])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty((2, 0), dtype=torch.long)
    if edge_type:
        edge_type_t = torch.tensor(edge_type, dtype=torch.long)
        edge_onehot = F.one_hot(edge_type_t, num_classes=BOND_TYPE_CLASSES).to(torch.float32)
        if edge_feat:
            edge_extra_dim = max(int(edge_dim - edge_onehot.size(1)), 0) if bond_extra_dim is None else max(min(int(bond_extra_dim), edge_dim - edge_onehot.size(1)), 0)
            if edge_extra_dim > 0:
                extra = torch.tensor([_adjust_feature_vec(f, edge_extra_dim) for f in edge_feat], dtype=torch.float32)
                edge_attr = torch.cat([edge_onehot, extra], dim=-1)
            else:
                edge_attr = edge_onehot
        else:
            edge_attr = edge_onehot
    else:
        edge_attr = torch.zeros((0, edge_dim), dtype=torch.float32)
        edge_type_t = torch.zeros((0,), dtype=torch.long)
    mask_ligand = torch.ones(num_atoms, dtype=torch.bool)
    mask_ligand_edge = compute_mask_ligand_edge(edge_index, mask_ligand)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type_t,
        mask_ligand=mask_ligand,
        mask_ligand_edge=mask_ligand_edge,
        smiles=smiles,
    )
    fp = _smiles_to_fp(smiles, fp_dim, fp_radius=fp_radius)
    if fp is not None:
        data.fp = fp.unsqueeze(0)
    return data


def smiles_to_ligand_3d(
    smiles: str,
    atom_index: Dict[str, int] | None = None,
    atom_feat_dim: int = 0,
    atom_extra_dim: int = 0,
    fp_dim: int = 0,
    fp_radius: int = 2,
    max_attempts: int = 10,
    seed: int = 0,
    num_confs: int = 10,
    max_opt_iters: int = 200,
    optimize: bool = True,
    prefer_mmff: bool = True,
) -> Data:
    mol = _smiles_to_3d_mol(
        smiles,
        max_attempts=max_attempts,
        seed=seed,
        num_confs=num_confs,
        max_opt_iters=max_opt_iters,
        optimize=optimize,
        prefer_mmff=prefer_mmff,
    )
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        raise RuntimeError(f"3D conformer generation failed for SMILES: {smiles}")
    if atom_index is None:
        raise ValueError("atom_index is required for 3D ligand features.")
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    x = None
    if atom_feat_dim > 0:
        num_atoms = mol.GetNumAtoms()
        x = torch.zeros((num_atoms, atom_feat_dim), dtype=torch.float32)
        atom_dim = len(atom_index)
        extra_dim = max(int(atom_extra_dim), 0)
        for idx, atom in enumerate(mol.GetAtoms()):
            sym_idx = atom_index.get(atom.GetSymbol())
            if sym_idx is None:
                raise ValueError(f"Atom symbol '{atom.GetSymbol()}' missing in ligand vocab.")
            x[idx, sym_idx] = 1.0
            if extra_dim > 0:
                vec = _adjust_feature_vec(_atom_feature_vec(atom), extra_dim)
                if vec:
                    x[idx, atom_dim : atom_dim + extra_dim] = torch.tensor(vec, dtype=torch.float32)
    data = Data(pos=pos, x=x, smiles=smiles)
    if fp_dim > 0:
        fp = _smiles_to_fp(smiles, fp_dim, fp_radius=fp_radius)
        if fp is not None:
            data.fp = fp.unsqueeze(0)
    props = build_pretrain_props(smiles, mol=mol, pos=pos)
    if props is not None:
        data.pretrain_props = props.unsqueeze(0)
    return data


def build_graphs_3d(
    smiles_list: Sequence[str],
    atom_index: Dict[str, int] | None = None,
    atom_feat_dim: int = 0,
    atom_extra_dim: int = 0,
    fp_dim: int = 0,
    fp_radius: int = 2,
    max_attempts: int = 10,
    seed: int = 0,
    num_confs: int = 10,
    max_opt_iters: int = 200,
    optimize: bool = True,
    prefer_mmff: bool = True,
    progress_desc: str | None = None,
    num_workers: int = 1,
    mp_chunksize: int = 16,
    skip_failed: bool = True,
    max_error_logs: int = 10,
):
    graphs = []
    kept_smiles = []
    failed = []
    if num_workers and num_workers > 1:
        import multiprocessing as mp
        cfg = {
            "atom_index": atom_index,
            "atom_feat_dim": atom_feat_dim,
            "atom_extra_dim": atom_extra_dim,
            "fp_dim": fp_dim,
            "fp_radius": fp_radius,
            "max_attempts": max_attempts,
            "seed": seed,
            "num_confs": num_confs,
            "max_opt_iters": max_opt_iters,
            "optimize": optimize,
            "prefer_mmff": prefer_mmff,
        }
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(num_workers), initializer=_mp_init_graphs, initargs=(cfg,)) as pool:
            it = pool.imap_unordered(_mp_build_graph_3d_worker, smiles_list, chunksize=max(int(mp_chunksize), 1))
            if progress_desc:
                it = tqdm(it, total=len(smiles_list), desc=progress_desc, unit="mol")
            for res in it:
                smi, data, err = res
                if err is not None:
                    if not skip_failed:
                        raise RuntimeError(f"3D graph build failed for SMILES {smi}: {err}")
                    failed.append((smi, err))
                    continue
                graphs.append(data)
                kept_smiles.append(smi)
    else:
        iterable = smiles_list
        if progress_desc:
            iterable = tqdm(smiles_list, desc=progress_desc, unit="mol")
        for smi in iterable:
            try:
                data = smiles_to_ligand_3d(
                    smi,
                    atom_index=atom_index,
                    atom_feat_dim=atom_feat_dim,
                    atom_extra_dim=atom_extra_dim,
                    fp_dim=fp_dim,
                    fp_radius=fp_radius,
                    max_attempts=max_attempts,
                    seed=seed,
                    num_confs=num_confs,
                    max_opt_iters=max_opt_iters,
                    optimize=optimize,
                    prefer_mmff=prefer_mmff,
                )
            except Exception as exc:
                if not skip_failed:
                    raise
                failed.append((smi, f"{type(exc).__name__}: {exc}"))
                continue
            graphs.append(data)
            kept_smiles.append(smi)
    if failed:
        shown = failed[: max(0, int(max_error_logs))]
        print(f"[candidate_3d] skipped {len(failed)} molecules due to 3D build failures.")
        for smi, err in shown:
            print(f"[candidate_3d][skip] {smi} | {err}")
        if len(failed) > len(shown):
            print(f"[candidate_3d][skip] ... {len(failed) - len(shown)} more")
    return graphs, kept_smiles


def build_graphs(
    smiles_list: Sequence[str],
    atom_index: Dict[str, int],
    node_dim: int,
    edge_dim: int,
    atom_extra_dim: int | None = None,
    bond_extra_dim: int | None = None,
    fp_dim: int = 0,
    fp_radius: int = 2,
):
    graphs = []
    kept_smiles = []
    for smi in smiles_list:
        data = smiles_to_ligand_graph(
            smi,
            atom_index,
            node_dim,
            edge_dim,
            atom_extra_dim=atom_extra_dim,
            bond_extra_dim=bond_extra_dim,
            fp_dim=fp_dim,
            fp_radius=fp_radius,
        )
        graphs.append(data)
        kept_smiles.append(smi)
    return graphs, kept_smiles


def compute_qed_sa(
    graphs: Sequence[Data],
    smiles: Sequence[str],
    use_sa: bool = True,
    sa_clamp_min: float | None = None,
    sa_clamp_max: float | None = None,
):
    qed_values = []
    sa_values = []
    for data, smi in zip(graphs, smiles):
        qed_val = compute_qed(smi)
        if not np.isfinite(qed_val):
            qed_val = float("nan")
        sa_val = 0.0
        if use_sa:
            sa_val = compute_sa(smi, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max)
        qed_values.append(qed_val)
        sa_values.append(sa_val)
    if not graphs:
        return torch.zeros(0), torch.zeros(0)
    return (
        torch.tensor(qed_values, dtype=torch.float32),
        torch.tensor(sa_values, dtype=torch.float32),
    )


def build_ligand_graphs_from_smiles_csv(
    dataset_root: str,
    split: str,
    ligand_vocab: Sequence[str],
    atom_extra_dim: int,
    bond_extra_dim: int,
    fp_dim: int,
    fp_radius: int,
    include_ids: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
    num_workers: int = 1,
    mp_chunksize: int = 16,
    progress_desc: str | None = None,
) -> list[Data]:
    csv_path = str(Path(dataset_root) / "smiles.csv")
    df = _read_smiles_csv(csv_path)
    if df.empty:
        return []
    if include_ids is not None:
        include_set = {str(x).strip() for x in include_ids if str(x).strip()}
        if include_set and "ligand_id" in df.columns:
            df = df[df["ligand_id"].astype(str).isin(include_set)]
    else:
        if "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split.lower()]
        else:
            if split.lower() != "train":
                return []
    if exclude_ids is not None:
        exclude_set = {str(x).strip() for x in exclude_ids if str(x).strip()}
        if exclude_set and "ligand_id" in df.columns:
            df = df[~df["ligand_id"].astype(str).isin(exclude_set)]
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        return []
    atom_index = {sym: i for i, sym in enumerate(ligand_vocab)}
    node_dim = len(ligand_vocab) + max(int(atom_extra_dim), 0)
    edge_dim = BOND_TYPE_CLASSES + max(int(bond_extra_dim), 0)
    graphs: list[Data] = []
    rows_data = []
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            raise ValueError("Encountered empty SMILES in smiles.csv while building ligand graphs.")
        dock_val = float(row.get("dock_score", float("nan")))
        lig_id = str(row.get("ligand_id", ""))
        rows_data.append((smi, dock_val, lig_id))

    if num_workers and num_workers > 1:
        import multiprocessing as mp

        cfg = {
            "atom_index": atom_index,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "atom_extra_dim": atom_extra_dim,
            "bond_extra_dim": bond_extra_dim,
            "fp_dim": fp_dim,
            "fp_radius": fp_radius,
        }
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(num_workers), initializer=_mp_init_ligand_graphs, initargs=(cfg,)) as pool:
            it = pool.imap_unordered(_mp_build_ligand_graph_worker, rows_data, chunksize=max(int(mp_chunksize), 1))
            if progress_desc:
                it = tqdm(it, total=len(rows_data), desc=progress_desc, unit="mol")
            for data in it:
                graphs.append(data)
    else:
        iterable = rows_data
        if progress_desc:
            iterable = tqdm(rows_data, desc=progress_desc, unit="mol")
        for smi, dock_val, lig_id in iterable:
            data = smiles_to_ligand_graph(
                smi,
                atom_index,
                node_dim,
                edge_dim,
                atom_extra_dim=atom_extra_dim,
                bond_extra_dim=bond_extra_dim,
                fp_dim=fp_dim,
                fp_radius=fp_radius,
            )
            data.dock_score = dock_val
            data.vina_score = dock_val
            data.ligand_id = lig_id
            graphs.append(data)
    return graphs
