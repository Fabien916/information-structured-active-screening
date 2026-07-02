from __future__ import annotations

from typing import Optional

from rdkit import Chem


def canonicalize_smiles(smiles: str, *, remove_hs: bool = True) -> Optional[str]:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=True)
        Chem.SanitizeMol(mol)
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return canon or None
    except Exception:
        return None
