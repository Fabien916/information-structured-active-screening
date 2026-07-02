from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, QED, AllChem, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3, CalcNumHBA, CalcNumHBD, CalcTPSA

PRETRAIN_PROPERTY_NAMES = [
    "qed",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "fsp3",
    "conf_energy",
    "radius_gyration",
    "asphericity",
    "max_pairwise_distance",
]


def _coerce_positions(pos: torch.Tensor | np.ndarray | Iterable[Iterable[float]] | None) -> np.ndarray | None:
    if pos is None:
        return None
    if isinstance(pos, torch.Tensor):
        arr = pos.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(pos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return None
    return arr


def _max_pairwise_distance(arr: np.ndarray | None) -> float:
    if arr is None or arr.shape[0] < 2:
        return 0.0
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
    return float(np.max(dist))


def _conformer_energy(mol: Chem.Mol | None) -> float:
    if mol is None or mol.GetNumConformers() == 0:
        return 0.0
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
                if ff is not None:
                    return float(ff.CalcEnergy())
    except Exception:
        pass
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is not None:
            return float(ff.CalcEnergy())
    except Exception:
        pass
    return 0.0


def build_pretrain_props(
    smiles: str,
    *,
    mol: Chem.Mol | None = None,
    pos: torch.Tensor | np.ndarray | Iterable[Iterable[float]] | None = None,
) -> torch.Tensor | None:
    base_mol = Chem.MolFromSmiles(str(smiles))
    if base_mol is None:
        return None
    try:
        base_mol = Chem.RemoveHs(base_mol, sanitize=True)
    except Exception:
        return None

    arr = _coerce_positions(pos)
    conf_mol = mol
    if conf_mol is not None and conf_mol.GetNumConformers() > 0 and arr is None:
        try:
            arr = np.asarray(conf_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        except Exception:
            arr = None

    try:
        radius_gyration = float(rdMolDescriptors.CalcRadiusOfGyration(conf_mol)) if conf_mol is not None else 0.0
    except Exception:
        radius_gyration = 0.0
    try:
        asphericity = float(rdMolDescriptors.CalcAsphericity(conf_mol)) if conf_mol is not None else 0.0
    except Exception:
        asphericity = 0.0

    values = [
        float(QED.qed(base_mol)),
        float(Crippen.MolLogP(base_mol)),
        float(CalcTPSA(base_mol)),
        float(CalcNumHBD(base_mol)),
        float(CalcNumHBA(base_mol)),
        float(CalcFractionCSP3(base_mol)),
        float(_conformer_energy(conf_mol)),
        float(radius_gyration),
        float(asphericity),
        float(_max_pairwise_distance(arr)),
    ]
    values = [0.0 if not math.isfinite(v) else float(v) for v in values]
    return torch.tensor(values, dtype=torch.float32)
