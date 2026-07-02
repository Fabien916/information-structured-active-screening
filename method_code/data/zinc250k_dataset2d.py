from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


def _kekulize_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    # After kekulize, aromatic bonds should be gone
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            return None
    return mol


class Zinc250kDataset2D(InMemoryDataset):
    """
    Ligand-only ZINC250k with two-pass vocab build (atoms/bonds), fully connected graph edges.
    """

    def __init__(
        self,
        root: str = "datasets/zinc250k",
        split: str = "train",
        rare_atom_threshold: float = 0.0,
        rare_edge_threshold: float = 0.0,
        kekulize_required: bool = True,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform=None,
        pre_transform=None,
    ):
        self.split = split
        self.rare_atom_threshold = float(rare_atom_threshold)
        self.rare_edge_threshold = float(rare_edge_threshold)
        self.kekulize_required = bool(kekulize_required)
        self.split_ratio = split_ratio
        super().__init__(root, transform, pre_transform)
        obj = torch.load(self.processed_paths[0], weights_only=False)
        if isinstance(obj, tuple) and len(obj) == 3:
            data, self.slices, meta = obj
        else:
            # stale format, reprocess
            Path(self.processed_paths[0]).unlink(missing_ok=True)
            self._process_once()
            data, self.slices, meta = torch.load(self.processed_paths[0], weights_only=False)
        self.data = data
        self.meta = meta
        self.ligand_vocab = meta["ligand_vocab"]
        self.edge_decoder_ = meta["edge_decoder"]
        self.num_node_classes_ = len(self.ligand_vocab)
        self.num_edge_classes_ = len(self.edge_decoder_)
        self.x_prob_ = torch.tensor(meta["x_prob"], dtype=torch.float32)
        self.e_prob_ = torch.tensor(meta["e_prob"], dtype=torch.float32)
        self._node_counts_cache: torch.Tensor | None = None
        hist = meta.get("node_count_hist", None)
        self._node_hist = torch.tensor(hist, dtype=torch.float32) if hist is not None else None

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.split}.pt"]

    @property
    def raw_file_names(self) -> List[str]:
        return ["smiles.csv"]

    def _process_once(self):
        proc_dir = Path(self.processed_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)
        # pass-1: scan
        raw_path = Path(self.raw_paths[0])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw smiles.csv not found: {raw_path}")
        df = pd.read_csv(raw_path)
        smiles_list = df[df.columns[0]].tolist()
        total_smiles = len(smiles_list)
        drop_none = 0
        drop_aromatic = 0
        atom_counts = {}
        bond_counts = {}
        mol_cache = []
        for smi in smiles_list:
            mol = _kekulize_mol(smi) if self.kekulize_required else Chem.MolFromSmiles(smi)
            if mol is None:
                drop_none += 1
                continue
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            for a in atoms:
                atom_counts[a] = atom_counts.get(a, 0) + 1
            bond_map = {}
            bad = False
            for b in mol.GetBonds():
                if b.GetIsAromatic():
                    bad = True
                    break
                tok = b.GetBondType()
                bond_counts[tok] = bond_counts.get(tok, 0) + 1
                u = b.GetBeginAtomIdx()
                v = b.GetEndAtomIdx()
                bond_map[(u, v)] = tok
                bond_map[(v, u)] = tok
            if bad:
                drop_aromatic += 1
                continue
            mol_cache.append((smi, atoms, bond_map))
        print(
            "[zinc2d] pass1 total=%d kept=%d drop_invalid=%d drop_aromatic=%d"
            % (total_smiles, len(mol_cache), drop_none, drop_aromatic)
        )
        total_atoms = sum(atom_counts.values())
        total_bonds = sum(bond_counts.values())
        atom_vocab = [a for a, c in atom_counts.items() if c >= self.rare_atom_threshold * total_atoms]
        bond_vocab = [b for b, c in bond_counts.items() if c >= self.rare_edge_threshold * total_bonds]
        atom_vocab = sorted(atom_vocab)
        bond_vocab = sorted(bond_vocab, key=lambda bt: float(bt))
        # ensure tokens exist
        if not atom_vocab:
            raise RuntimeError("atom_vocab empty after filtering")
        edge_decoder = [None] + bond_vocab
        print(
            "[zinc2d] vocab atoms=%d bonds=%d edge_classes=%d"
            % (len(atom_vocab), len(bond_vocab), len(edge_decoder))
        )
        atom_to_idx = {a: i for i, a in enumerate(atom_vocab)}
        bond_to_idx = {b: i + 1 for i, b in enumerate(bond_vocab)}  # +1 because None at 0

        # pass-2: encode
        data_list = []
        x_hist = torch.zeros(len(atom_vocab), dtype=torch.long)
        e_hist = torch.zeros(len(edge_decoder), dtype=torch.long)
        node_count_hist = {}
        drop_atom_vocab = 0
        drop_bond_vocab = 0
        total_n = 0
        total_bonds_undir = 0
        total_edges = 0
        total_no_bond = 0
        for smi, atoms, bond_map in mol_cache:
            if any(a not in atom_to_idx for a in atoms):
                drop_atom_vocab += 1
                continue
            x_idx = torch.tensor([atom_to_idx[a] for a in atoms], dtype=torch.long)
            n = x_idx.numel()
            node_count_hist[n] = node_count_hist.get(n, 0) + 1
            edge_pairs = []
            et_list = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    tok = bond_map.get((i, j), None)
                    if tok is None:
                        et_list.append(0)  # no bond
                    else:
                        if tok not in bond_to_idx:
                            edge_pairs = None
                            drop_bond_vocab += 1
                            break
                        et_list.append(bond_to_idx[tok])
                    edge_pairs.append([i, j])
                if edge_pairs is None:
                    break
            if edge_pairs is None:
                continue
            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
                edge_type = torch.tensor(et_list, dtype=torch.long)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                edge_type = torch.empty(0, dtype=torch.long)
            d = Data(
                x_idx=x_idx,
                x=torch.nn.functional.one_hot(x_idx, num_classes=len(atom_vocab)).to(torch.float32),
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=torch.nn.functional.one_hot(edge_type, num_classes=len(edge_decoder)).to(torch.float32),
                mask_ligand=torch.ones(n, dtype=torch.bool),
                mask_ligand_edge=torch.ones(edge_type.numel(), dtype=torch.bool),
                smiles=smi,
            )
            data_list.append(d)
            total_n += int(n)
            total_bonds_undir += int(len(bond_map) // 2)
            x_hist += torch.bincount(x_idx, minlength=len(atom_vocab))
            if edge_type.numel() > 0:
                total_edges += int(edge_type.numel())
                total_no_bond += int((edge_type == 0).sum().item())
                e_hist += torch.bincount(edge_type, minlength=len(edge_decoder))
        if not data_list:
            raise RuntimeError("No molecules retained after pass-2.")
        kept = len(data_list)
        avg_n = float(total_n / kept) if kept > 0 else 0.0
        avg_bonds = float(total_bonds_undir / kept) if kept > 0 else 0.0
        avg_no_bond = float(total_no_bond / max(total_edges, 1))
        print(
            "[zinc2d] pass2 kept=%d drop_atom_vocab=%d drop_bond_vocab=%d"
            % (kept, drop_atom_vocab, drop_bond_vocab)
        )
        print(
            "[zinc2d] avg_n=%.4f avg_bonds=%.4f avg_no_bond=%.6f total_edges=%d total_no_bond=%d"
            % (avg_n, avg_bonds, avg_no_bond, total_edges, total_no_bond)
        )

        # priors (include no-bond at idx 0)
        x_prob = x_hist.float() / x_hist.sum().clamp_min(1).float()
        pe = (
            e_hist.float() / e_hist.sum().clamp_min(1).float()
            if e_hist.sum() > 0
            else torch.zeros_like(e_hist, dtype=torch.float32)
        )
        print("[zinc2d] e_prob:", pe.tolist())
        max_n = max(node_count_hist.keys()) if node_count_hist else 0
        node_count_hist_list = [node_count_hist.get(i, 0) for i in range(max_n + 1)]

        # split
        torch.manual_seed(42)
        perm = torch.randperm(len(data_list))
        n = len(data_list)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        r_train, r_val, r_test = self.split_ratio
        total_ratio = r_train + r_val + r_test
        if total_ratio <= 0:
            raise ValueError("split_ratio must sum to > 0")
        r_train /= total_ratio
        r_val /= total_ratio
        n_train = int(r_train * n)
        n_val = int(r_val * n)
        splits = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        meta = {
            "ligand_vocab": atom_vocab,
            "edge_decoder": edge_decoder,
            "x_prob": x_prob.tolist(),
            "e_prob": pe.tolist(),
            "atom_counts": atom_counts,
            "bond_counts": bond_counts,
            "node_count_hist": node_count_hist_list,
        }
        for split, idxs in splits.items():
            out = [data_list[i] for i in idxs.tolist()]
            data, slices = self.collate(out)
            torch.save((data, slices, meta), Path(self.processed_dir) / f"{split}.pt")

    def process(self):
        # Only run once on first instantiation
        any_missing = any(not Path(p).exists() for p in self.processed_paths)
        print(
            "[zinc2d] process any_missing=%s processed_dir=%s raw=%s"
            % (any_missing, str(self.processed_dir), str(self.raw_paths[0]))
        )
        if any_missing:
            self._process_once()

    def sample_num_nodes(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Sample number of nodes according to empirical histogram saved in meta.
        """
        dev = device or torch.device("cpu")
        probs = self._node_hist.to(dev)
        probs = probs / probs.sum().clamp_min(1e-12)
        support = torch.arange(probs.numel(), device=dev)
        idx = torch.multinomial(probs, batch_size, replacement=True)
        return support[idx]




def build_zinc250k_dataset2d(root: str, split: str | None = None, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1), **kwargs):
    """
    Build/load ZINC250k 2D datasets (train/val/test) from a raw smiles.csv under root/raw/.
    """
    raw_path = Path(root) / "raw" / "smiles.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Expected raw smiles.csv at {raw_path}")
    if split is not None:
        return Zinc250kDataset2D(root=root, split=split, split_ratio=split_ratio, **kwargs)
    train = Zinc250kDataset2D(root=root, split="train", split_ratio=split_ratio, **kwargs)
    val = Zinc250kDataset2D(root=root, split="val", split_ratio=split_ratio, **kwargs)
    test = Zinc250kDataset2D(root=root, split="test", split_ratio=split_ratio, **kwargs)
    return train, val, test


__all__ = ["Zinc250kDataset2D", "build_zinc250k_dataset2d"]
