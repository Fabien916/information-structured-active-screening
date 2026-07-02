from __future__ import annotations

from typing import List

import math
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, QED, rdFingerprintGenerator
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError as exc:
    raise ImportError(
        "RDKit Contrib SA_Score is required. Install an RDKit build that includes `rdkit.Contrib.SA_Score`."
    ) from exc
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as exc:
    raise ImportError("RDKit MolStandardize is required (`rdkit.Chem.MolStandardize`).") from exc

def calc_sa_score_mol(
    mol: Chem.Mol,
    clamp_min: float | None = 1.0,
    clamp_max: float | None = 10.0,
) -> float:
    if mol is None:
        raise ValueError("SA score requires a valid RDKit Mol.")
    sascore = float(sascorer.calculateScore(mol))
    if clamp_min is not None and clamp_max is not None:
        lo = float(min(clamp_min, clamp_max))
        hi = float(max(clamp_min, clamp_max))
        sascore = min(hi, max(lo, sascore))
    return float(sascore)


def compute_qed(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for QED: {smiles}")
    mol = Chem.RemoveHs(mol, sanitize=True)
    return float(QED.qed(mol))


def compute_sa(smiles: str, clamp_min: float | None = None, clamp_max: float | None = None) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for SA: {smiles}")
    mol = Chem.RemoveHs(mol, sanitize=True)
    return float(calc_sa_score_mol(mol, clamp_min=clamp_min, clamp_max=clamp_max))


def _adjust_feature_vec(vec: List[float], target_dim: int) -> List[float]:
    if target_dim <= 0:
        return []
    if len(vec) >= target_dim:
        return vec[:target_dim]
    return vec + [0.0] * (target_dim - len(vec))


def _atom_feature_vec(atom: Chem.Atom) -> List[float]:
    deg = min(int(atom.GetTotalDegree()), 6) / 6.0
    val = min(int(atom.GetTotalValence()), 6) / 6.0
    charge = max(min(int(atom.GetFormalCharge()), 3), -3) / 3.0
    num_h = min(int(atom.GetTotalNumHs()), 4) / 4.0
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    in_ring = 1.0 if atom.IsInRing() else 0.0
    hyb = atom.GetHybridization()
    hyb_onehot = [
        1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3D else 0.0,
        1.0 if hyb == Chem.rdchem.HybridizationType.SP3D2 else 0.0,
    ]
    chiral = 1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0
    gasteiger = 0.0
    if atom.HasProp("_GasteigerCharge"):
        try:
            gasteiger = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            gasteiger = 0.0
    if not math.isfinite(gasteiger):
        gasteiger = 0.0
    gasteiger = max(min(gasteiger, 3.0), -3.0) / 3.0
    return [deg, val, charge, num_h, aromatic, in_ring, *hyb_onehot, chiral, gasteiger]


def _bond_feature_vec(bond: Chem.Bond) -> List[float]:
    order = bond.GetBondTypeAsDouble()
    order = min(float(order), 3.0) / 3.0
    conj = 1.0 if bond.GetIsConjugated() else 0.0
    ring = 1.0 if bond.IsInRing() else 0.0
    aromatic = 1.0 if bond.GetIsAromatic() else 0.0
    aromatic_oh = 1.0 if bond.GetBondType() == Chem.BondType.AROMATIC else 0.0
    return [order, conj, ring, aromatic, aromatic_oh]


def _smiles_to_fp(smiles: str, fp_dim: int, fp_radius: int = 2) -> torch.Tensor | None:
    if fp_dim <= 0:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Avoid RDKit deprecation warning: prefer MorganGenerator over legacy AllChem.GetMorganFingerprintAsBitVect.
    cache = getattr(_smiles_to_fp, "_morgan_gen_cache", None)
    if cache is None:
        cache = {}
        setattr(_smiles_to_fp, "_morgan_gen_cache", cache)
    key = (int(fp_radius), int(fp_dim))
    gen = cache.get(key)
    if gen is None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=key[0], fpSize=key[1])
        cache[key] = gen
    fp = gen.GetFingerprint(mol)
    arr = torch.zeros(fp_dim, dtype=torch.float32)
    # ExplicitBitVect supports fast bit iteration.
    for idx in fp.GetOnBits():
        arr[int(idx)] = 1.0
    return arr


def normalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for normalization: {smiles}")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    mol = Chem.RemoveHs(mol, sanitize=True, implicitOnly=False)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles_noh(smiles: str) -> str:
    return normalize_smiles(smiles)
