"""Helpers for loading pocket-conditioned datasets."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Tuple

from torch_geometric.loader import DataLoader

from .pocket_dataset import PocketLigandDockingDataset


def build_pocket_dataloaders(cfg: Dict,
                             batch_size: int,
                             num_workers: int,
                             shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    target_root = Path(cfg.get('target_root', 'datasets/colon_cancer')).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    smiles_src = cfg.get('smiles_csv', None)
    if smiles_src:
        src_path = Path(smiles_src).expanduser().resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Pocket dataset smiles_csv not found: {src_path}")
        dest_path = target_root / "smiles.csv"
        if not dest_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dest_path)

    dataset_kwargs = dict(
        radius=cfg.get('radius', 10.0),
        auto_prepare=cfg.get('auto_prepare', True),
        overwrite_docking=cfg.get('overwrite_docking', False),
        exhaustiveness=cfg.get('exhaustiveness', 32),
        vina_executable=cfg.get('vina_executable', "bin/vina_1.2.7_win.exe"),
        tmp_dir=cfg.get('tmp_dir', None),
        context_radius=cfg.get('context_radius', 6.0),
        context_knn=cfg.get('context_knn', 24),
        rare_atom_threshold=cfg.get('rare_atom_threshold', 0.01),
        ligand_vocab_override=cfg.get('ligand_vocab_override', None),
    )

    train_ds = PocketLigandDockingDataset(target_root, split="train", **dataset_kwargs)
    valid_ds = PocketLigandDockingDataset(target_root, split="valid", **dataset_kwargs)
    test_ds = PocketLigandDockingDataset(target_root, split="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
