from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import logging
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pkgutil
import gzip
import io
import pickle
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, QED, rdPartialCharges
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception:  # pragma: no cover - optional dependency
    sascorer = None
_MEEKO_IMPORT_ERROR = None
try:
    import meeko
    from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
except Exception as exc:  # pragma: no cover - optional dependency
    meeko = None
    MoleculePreparation = None
    PDBQTWriterLegacy = None
    PDBQTMolecule = None
    RDKitMolCreate = None
    _MEEKO_IMPORT_ERROR = exc
from torch_geometric.data import Data, InMemoryDataset
from data.pocket_utils import canonicalize_smiles

AtomArray = Tuple[np.ndarray, List[str]]  # (N,3) coordinates, element symbols

rdBase.DisableLog("rdApp.warning")

_WINDOWS_OBABEL_PATH = Path(r"C:\Program Files (x86)\OpenBabel-3.1.1\obabel.exe")
_WSL_OBABEL_PATH = Path("/mnt/c/Program Files (x86)/OpenBabel-3.1.1/obabel.exe")

# 压缩后的配体 vocab：仅保留 C / N / O
COMPRESSED_LIGAND_VOCAB = ["C", "N", "O"]


_SA_FRAGMENT_SCORES = None

# Feature dimensions for surrogate graphs
BOND_TYPE_CLASSES = 5  # no_bond, single, double, triple, aromatic
ATOM_EXTRA_DIM = 13
BOND_EXTRA_DIM = 5


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


def _load_sa_fragment_scores():
    global _SA_FRAGMENT_SCORES
    if _SA_FRAGMENT_SCORES is not None:
        return _SA_FRAGMENT_SCORES
    data = pkgutil.get_data("rdkit.Chem", "fpscores.pkl.gz")
    if data is None:
        raise RuntimeError("RDKit fragment score file fpscores.pkl.gz not found; install RDKit with contrib data.")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
            raw = pickle.load(f)
        fscores = {}
        # Support both known RDKit formats:
        # 1) list of lists: each row is [score, fid1, fid2, ...]
        # 2) legacy: raw[0] scores array; raw[i] (i>=1) contains fragment ids
        if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
            # Format 1
            if len(raw[0]) > 1 and isinstance(raw[0][1], (int, np.integer)):
                for row in raw:
                    if not row:
                        continue
                    score = float(row[0])
                    for fid in row[1:]:
                        fscores[int(fid)] = score
            else:
                # Format 2
                for i in range(1, len(raw)):
                    score = raw[0][i]
                    for fid in raw[i]:
                        fscores[int(fid)] = float(score)
        elif isinstance(raw, dict):
            fscores = {int(k): float(v) for k, v in raw.items()}
        else:
            raise RuntimeError("Unrecognized fpscores.pkl.gz format.")
        _SA_FRAGMENT_SCORES = fscores
    except Exception as exc:
        raise RuntimeError(f"Failed to load RDKit SA fragment scores: {exc}") from exc
    return _SA_FRAGMENT_SCORES


def _calc_sa_score(
    mol: Chem.Mol,
    clamp_min: float | None = 1.0,
    clamp_max: float | None = 10.0,
) -> float:
    """Ertl SA_score implementation (RDKit contrib)."""
    if mol is None:
        raise ValueError("SA score requires a valid RDKit Mol.")
    if sascorer is None:
        raise RuntimeError("RDKit contrib sascorer not available; install RDKit with Contrib/SA_Score.")
    sascore = float(sascorer.calculateScore(mol))
    if clamp_min is not None and clamp_max is not None:
        lo = float(min(clamp_min, clamp_max))
        hi = float(max(clamp_min, clamp_max))
        sascore = min(hi, max(lo, sascore))
    return float(sascore)


def _resolve_obabel_executable() -> str:
    for candidate in (_WSL_OBABEL_PATH, _WINDOWS_OBABEL_PATH):
        if candidate.exists():
            return str(candidate)
    resolved = which("obabel")
    if resolved is not None:
        return resolved
    return str(_WSL_OBABEL_PATH if os.name != "nt" else _WINDOWS_OBABEL_PATH)


OBABEL_EXECUTABLE = _resolve_obabel_executable()
log = logging.getLogger(__name__)
log.disabled = True


@dataclass(frozen=True)
class PocketDatasetPaths:
    """Utility container describing a target directory layout."""

    root: Path
    pocket_pdb: Path
    protein_pdb: Path
    reference_ligand: Optional[Path]
    smiles_csv: Path
    docking_dir: Path
    pocket_radius: float = 10.0

    @staticmethod
    def from_root(root: Path, *, pocket_filename: str = "pocket.pdb",
                  protein_filename: str = "protein.pdb",
                  reference_ligand: str = "reference_ligand.sdf",
                  smiles_filename: str = "smiles.csv",
                  docking_subdir: str = "docking",
                  pocket_radius: float = 10.0) -> "PocketDatasetPaths":
        root = Path(root).resolve()
        return PocketDatasetPaths(
            root=root,
            pocket_pdb=root / pocket_filename,
            protein_pdb=root / protein_filename,
            reference_ligand=(root / reference_ligand if (root / reference_ligand).exists() else None),
            smiles_csv=root / smiles_filename,
            docking_dir=root / docking_subdir,
            pocket_radius=float(pocket_radius),
        )


def _load_ligand_vocab_override(root: Path) -> Optional[List[str]]:
    json_path = root / "ligand_vocab.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                vocab = [str(x).strip() for x in data if str(x).strip()]
                return sorted(set(vocab))
        except Exception:
            pass
    txt_path = root / "ligand_vocab.txt"
    if txt_path.exists():
        try:
            raw = txt_path.read_text(encoding="utf-8").splitlines()
            vocab = [line.strip() for line in raw if line.strip()]
            if vocab:
                return sorted(set(vocab))
        except Exception:
            pass
    return None


def _read_pdb_atoms(path: Path) -> AtomArray:
    coords, elems = [], []
    with path.open("r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip()[0]
            coords.append((x, y, z))
            elems.append(element)
    if not coords:
        raise ValueError(f"No atoms parsed from {path}")
    return np.asarray(coords, dtype=np.float64), elems


def _read_pocket_residues(path: Path) -> Tuple[np.ndarray, List[str]]:
    residues: dict = {}
    with path.open("r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip()[-1]
            if element.upper() == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip() or ""
            key = (chain, resseq, icode)
            entry = residues.setdefault(key, {"coords": [], "name": resname})
            entry["coords"].append((x, y, z))

    coords: List[np.ndarray] = []
    names: List[str] = []
    for entry in residues.values():
        arr = np.asarray(entry["coords"], dtype=np.float64)
        if arr.size == 0:
            continue
        coords.append(arr.mean(axis=0))
        names.append(entry["name"])

    if not coords:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.vstack(coords), names


def _normalize_smiles_dataframe(
    df: pd.DataFrame,
    split_ratio: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    assign_split: bool = True,
) -> Tuple[pd.DataFrame, bool]:
    df = df.copy()
    modified = False
    col_map = {c.lower(): c for c in df.columns}

    def find_column(keys):
        for key in keys:
            if key in col_map:
                return col_map[key]
        return None

    smiles_col = find_column(["smiles", "smile"])
    if smiles_col is None:
        if df.shape[1] == 0:
            raise ValueError("smiles.csv must contain at least one column with SMILES strings.")
        smiles_col = df.columns[0]
        modified = True
    df["smiles"] = df[smiles_col].astype(str)

    ligand_col = find_column(["ligand_id", "ligandid", "id"])
    if ligand_col is None:
        df["ligand_id"] = [f"LIG_{i:05d}" for i in range(len(df))]
        modified = True
    else:
        df["ligand_id"] = df[ligand_col].astype(str)

    split_col = find_column(["split"])
    if split_col is None:
        if assign_split:
            # 按比例自动划分 train/valid/test
            p_train, p_valid, p_test = split_ratio
            p_sum = max(p_train + p_valid + p_test, 1e-8)
            p_train, p_valid, p_test = (p_train / p_sum, p_valid / p_sum, p_test / p_sum)
            total = len(df)
            n_train = int(round(total * p_train))
            n_valid = int(round(total * p_valid))
            # 确保总数不超
            if n_train + n_valid > total:
                n_valid = max(0, total - n_train)
            n_test = max(0, total - n_train - n_valid)
            splits = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
            while len(splits) < total:
                splits.append("train")
            df["split"] = splits[:total]
            modified = True
    else:
        df["split"] = df[split_col].astype(str)

    if "dock_pose" not in df.columns:
        df["dock_pose"] = ""
        modified = True
    if "dock_score" not in df.columns:
        df["dock_score"] = np.nan
        modified = True

    # Fill SA if missing
    if "sa_score" not in df.columns or df["sa_score"].isna().any():
        sa_vals: List[float] = []
        for idx, smi in enumerate(df["smiles"].tolist()):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError("MolFromSmiles returned None")
                mol = Chem.RemoveHs(mol, sanitize=True)
                sa_vals.append(_calc_sa_score(mol))
            except Exception as exc:
                logging.warning("[sa_score] compute failed for idx=%d smiles=%s err=%s", idx, smi, exc)
                sa_vals.append(float("nan"))
        df["sa_score"] = sa_vals
        modified = True

    # Fill logP if missing (Crippen MolLogP)
    logp_col = None
    for cand in ["logp", "logP", "log_p"]:
        if cand in df.columns:
            logp_col = cand
            break
    need_logp = (logp_col is None) or df[logp_col].isna().any()
    if need_logp:
        logp_vals: List[float] = []
        for idx, smi in enumerate(df["smiles"].tolist()):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError("MolFromSmiles returned None")
                mol = Chem.RemoveHs(mol, sanitize=True)
                logp_vals.append(float(Crippen.MolLogP(mol)))
            except Exception as exc:
                logging.warning("[logP] compute failed for idx=%d smiles=%s err=%s", idx, smi, exc)
                logp_vals.append(float("nan"))
        df["logp"] = logp_vals
        modified = True

    # Fill QED if missing
    qed_col = None
    for cand in ["qed", "QED"]:
        if cand in df.columns:
            qed_col = cand
            break
    need_qed = (qed_col is None) or df[qed_col].isna().any()
    if need_qed:
        qed_vals: List[float] = []
        for idx, smi in enumerate(df["smiles"].tolist()):
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    raise ValueError("MolFromSmiles returned None")
                mol = Chem.RemoveHs(mol, sanitize=True)
                qed_vals.append(float(QED.qed(mol)))
            except Exception as exc:
                logging.warning("[QED] compute failed for idx=%d smiles=%s err=%s", idx, smi, exc)
                qed_vals.append(float("nan"))
        df["qed"] = qed_vals
        modified = True

    return df, modified


def _deduplicate_canonical_rows(df: pd.DataFrame, merge_col: str = "_2") -> Tuple[pd.DataFrame, bool]:
    """
    按 canonical SMILES 去重。对重复条目：
      - 保留首次出现的整行
      - 若存在 merge_col（默认 "_2"，对应“中药”来源），将重复行该列合并为用中文逗号连接的去重字符串。
    返回 (去重后的 df, 是否发生合并)。
    """
    if "smiles_canonical" not in df.columns:
        return df, False
    seen: Dict[str, Dict] = {}
    merged = False
    cols = list(df.columns)
    for row in df.itertuples(index=False):
        can = str(getattr(row, "smiles_canonical"))
        if can not in seen:
            base = row._asdict()
            herbs = []
            if merge_col in cols:
                val = getattr(row, merge_col, "")
                if isinstance(val, str) and val.strip():
                    herbs.append(val.strip())
            seen[can] = {"row": base, "herb": herbs}
        else:
            merged = True
            if merge_col in cols:
                val = getattr(row, merge_col, "")
                if isinstance(val, str) and val.strip():
                    herbs = seen[can]["herb"]
                    if val.strip() not in herbs:
                        herbs.append(val.strip())
    rows_out: List[Dict] = []
    for entry in seen.values():
        row_dict = entry["row"]
        if merge_col in cols:
            row_dict[merge_col] = "，".join(entry["herb"]) if entry["herb"] else ""
        rows_out.append(row_dict)
    if not merged:
        return df, False
    return pd.DataFrame(rows_out, columns=cols), True


def _assign_split_column(df: pd.DataFrame, split_ratio: Tuple[float, float, float]) -> pd.DataFrame:
    if "split" in df.columns:
        return df
    p_train, p_valid, p_test = split_ratio
    p_sum = max(p_train + p_valid + p_test, 1e-8)
    p_train, p_valid, p_test = (p_train / p_sum, p_valid / p_sum, p_test / p_sum)
    total = len(df)
    n_train = int(round(total * p_train))
    n_valid = int(round(total * p_valid))
    if n_train + n_valid > total:
        n_valid = max(0, total - n_train)
    n_test = max(0, total - n_train - n_valid)
    splits = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    while len(splits) < total:
        splits.append("train")
    df = df.copy()
    df["split"] = splits[:total]
    return df


def _write_pocket_pdb(path: Path, coords: np.ndarray, elements: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for idx, (coord, elem) in enumerate(zip(coords, elements), start=1):
            x, y, z = coord
            handle.write(
                f"ATOM  {idx:5d} {elem:>2s}  POC A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {elem:>2s}\n"
            )
        handle.write("END\n")


def _parse_protein_atoms(path: Path, *, include_hetatm: bool = True) -> List[dict]:
    """Parse protein PDB, preserving residue/chain info (skip hydrogens)."""
    atoms: List[dict] = []
    with path.open("r") as handle:
        for line in handle:
            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                if rec == "ENDMDL":
                    break
                continue
            if rec == "HETATM" and not include_hetatm:
                continue
            element = line[76:78].strip()
            if not element:
                element = line[12:16].strip()[0]
            if element.upper() == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            atoms.append(
                {
                    "line": line.rstrip("\n"),
                    "coords": np.array([x, y, z], dtype=np.float64),
                    "element": element,
                    "atom_name": atom_name,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                }
            )
    return atoms


def _clean_pdb_for_meeko(src: Path, dst: Path, *, default_altloc: str | None = None) -> None:
    """Write a simplified PDB for Meeko: drop HETATM, resolve altloc, ensure element."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r") as handle_in, dst.open("w") as handle_out:
        for line in handle_in:
            rec = line[0:6].strip()
            if rec != "ATOM":
                continue
            # resolve alternate locations (column 17, 0-based 16)
            if len(line) > 16:
                altloc = line[16]
                if altloc.strip():
                    if default_altloc is None:
                        continue
                    if altloc != str(default_altloc):
                        continue
                    # normalize altloc to blank
                    line = line[:16] + " " + line[17:]
            line = line.rstrip("\n")
            if len(line) < 78:
                line = line.ljust(78)
            element = line[76:78].strip()
            if not element:
                atom_name = line[12:16].strip()
                if atom_name and atom_name[0].isdigit():
                    element = atom_name[1:3].strip().upper()
                else:
                    element = atom_name[:2].strip().upper()
                if not element:
                    element = "C"
                line = line[:76] + f"{element:>2s}" + line[78:]
            handle_out.write(line + "\n")
        handle_out.write("END\n")


def _extract_pocket_atoms(protein_atoms: List[dict], ligand_atoms: AtomArray, radius: float) -> List[dict]:
    """Select entire residues whose nearest atom is within radius of any ligand atom."""
    if not protein_atoms:
        return []
    lig_coords, _ = ligand_atoms
    # group atoms by residue key
    residues: dict = {}
    for atom in protein_atoms:
        key = (atom["chain"], atom["resseq"], atom["icode"])
        residues.setdefault(key, {"atoms": [], "resname": atom["resname"]})
        residues[key]["atoms"].append(atom)
    selected_atoms: List[dict] = []
    for res in residues.values():
        coords = np.stack([a["coords"] for a in res["atoms"]], axis=0)
        dist = np.linalg.norm(coords[:, None, :] - lig_coords[None, :, :], axis=-1)
        if np.any(dist <= radius):
            selected_atoms.extend(res["atoms"])
    return selected_atoms


def _write_pocket_pdb_from_atoms(atoms: List[dict], path: Path) -> None:
    """Write pocket atoms to PDB, preserving residue info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("HEADER    pocket\nCOMPND    pocket\n")
        for idx, atom in enumerate(atoms, start=1):
            line = atom["line"]
            # replace serial field with new contiguous index
            new_line = f"{line[:6]}{idx:5d}{line[11:]}"
            handle.write(new_line.rstrip("\n") + "\n")
        handle.write("END\n")


def _write_atomarray_to_sdf(atom_array: AtomArray, path: Path, title: str = "reference_ligand") -> None:
    coords, elements = atom_array
    if coords.shape[0] != len(elements):
        raise ValueError("Mismatch between coordinates and element symbols when writing SDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write("Generated by MultiPocket lazy preparation\n")
        handle.write("\n")
        n_atoms = coords.shape[0]
        handle.write(f"{n_atoms:>3}{0:>3}  0  0  0  0  0  0  0  0  0  0  0  0\n")
        for (x, y, z), elem in zip(coords, elements):
            handle.write(f"{x:10.4f}{y:10.4f}{z:10.4f} {elem:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n")
        handle.write("M  END\n")
        handle.write("$$$$\n")


def _extract_pocket(protein_atoms: AtomArray, ligand_atoms: AtomArray,
                    radius: float) -> AtomArray:
    prot_coords, prot_elems = protein_atoms
    lig_coords, _ = ligand_atoms
    distances = np.linalg.norm(
        prot_coords[:, None, :] - lig_coords[None, :, :],
        axis=-1,
    )
    min_dist = distances.min(axis=1)
    keep_mask = min_dist <= radius
    if not np.count_nonzero(keep_mask):
        raise ValueError("Pocket extraction produced empty atom set. Increase radius.")
    return prot_coords[keep_mask], [elem for elem, keep in zip(prot_elems, keep_mask) if keep]


def _load_ligand_conformer(path: Path) -> Chem.Mol:
    # Accept SDF/SDF.GZ or other RDKit-supported Mol block
    if not path.exists():
        raise FileNotFoundError(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError(f"Failed to read reference ligand from {path}")
    return mol



def _smiles_to_3d_mol(
    smiles: str,
    max_attempts: int = 10,
    seed: int = 0,
    num_confs: int = 10,
    max_opt_iters: int = 200,
    optimize: bool = True,
    prefer_mmff: bool = True,
) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 只保留最大有机片段（把 .N、对离子、溶剂片段去掉）
    mol = LargestFragmentChooser(preferOrganic=True).choose(mol)
    # 强制去除所有显式氢，保持全程无氢
    mol = Chem.RemoveHs(mol, sanitize=True, implicitOnly=False)

    def _build_params(attempt_seed: int, random_coords: bool, use_small_ring: bool) -> Any:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(attempt_seed)
        params.useRandomCoords = bool(random_coords)
        params.useSmallRingTorsions = bool(use_small_ring)
        params.enforceChirality = True
        params.pruneRmsThresh = 0.1
        return params

    def _try_embed_multi(base_mol: Chem.Mol, random_coords: bool = False, use_small_ring: bool = False) -> tuple[Optional[Chem.Mol], list[int]]:
        for attempt in range(max_attempts):
            mol_copy = Chem.Mol(base_mol)
            params = _build_params(seed + attempt, random_coords=random_coords, use_small_ring=use_small_ring)
            try:
                conf_ids = list(AllChem.EmbedMultipleConfs(mol_copy, numConfs=max(1, num_confs), params=params))
            except Exception:
                conf_ids = []
            if conf_ids:
                return mol_copy, conf_ids
        return None, []

    # Strategy 1: no-H ETKDG variants
    embedded, conf_ids = _try_embed_multi(mol, random_coords=False, use_small_ring=False)
    if embedded is None:
        embedded, conf_ids = _try_embed_multi(mol, random_coords=True, use_small_ring=False)
    if embedded is None:
        embedded, conf_ids = _try_embed_multi(mol, random_coords=True, use_small_ring=True)

    # Strategy 2: add-H then ETKDG (often more stable for bridged/caged skeletons)
    if embedded is None:
        mol_h = Chem.AddHs(mol, addCoords=False)
        embedded_h, conf_ids_h = _try_embed_multi(mol_h, random_coords=True, use_small_ring=True)
        if embedded_h is not None:
            try:
                embedded = Chem.RemoveHs(embedded_h, sanitize=True, implicitOnly=False)
                conf_ids = [0] if embedded.GetNumConformers() > 0 else []
            except Exception:
                embedded = None
                conf_ids = []

    if embedded is None:
        raise RuntimeError(
            f"ETKDG failed to embed molecule after multiple strategies: {smiles}"
        )
    if not conf_ids:
        conf_ids = [0]

    def _optimize_and_score(m: Chem.Mol, ids: list[int]) -> tuple[list[float], str]:
        if not optimize:
            return [0.0] * len(ids), "none"
        energies = [float("inf")] * len(ids)
        method = "UFF"
        if prefer_mmff:
            try:
                if AllChem.MMFFHasAllMoleculeParams(m):
                    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
                    if props is not None:
                        method = "MMFF"
                        for i, cid in enumerate(ids):
                            try:
                                ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
                                if ff is None:
                                    continue
                                ff.Minimize(maxIts=max_opt_iters)
                                energies[i] = float(ff.CalcEnergy())
                            except Exception:
                                continue
                        return energies, method
            except Exception:
                pass
        for i, cid in enumerate(ids):
            try:
                ff = AllChem.UFFGetMoleculeForceField(m, confId=cid)
                if ff is None:
                    continue
                ff.Minimize(maxIts=max_opt_iters)
                energies[i] = float(ff.CalcEnergy())
            except Exception:
                continue
        return energies, method

    energies, _method = _optimize_and_score(embedded, conf_ids)
    best_idx = 0
    try:
        best_idx = min(range(len(energies)), key=lambda i: energies[i])
    except Exception:
        best_idx = 0
    best_conf_id = conf_ids[best_idx] if conf_ids else 0
    try:
        best_conf = Chem.Conformer(embedded.GetConformer(best_conf_id))
        embedded.RemoveAllConformers()
        embedded.AddConformer(best_conf, assignId=True)
    except Exception:
        pass

    try:
        embedded = Chem.RemoveHs(embedded, sanitize=True, implicitOnly=False)
    except Exception:
        embedded = Chem.RemoveHs(embedded, sanitize=False, implicitOnly=False)
        try:
            Chem.SanitizeMol(embedded)
        except Exception:
            pass
    # 最终确保不含显式氢
    if any(atom.GetAtomicNum() == 1 for atom in embedded.GetAtoms()):
        raise ValueError("Failed to strip hydrogens from embedded molecule")
    try:
        rdPartialCharges.ComputeGasteigerCharges(embedded)
    except Exception:
        pass
    for atom in embedded.GetAtoms():
        if atom.HasProp("_GasteigerCharge"):
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                charge = 0.0
        else:
            charge = 0.0
        if not math.isfinite(charge):
            charge = 0.0
        atom.SetProp("_GasteigerCharge", f"{charge:.6f}")
    return embedded


def _ensure_receptor_pdbqt(
    pocket_pdb: Path,
    pocket_pdbqt: Path,
    *,
    meeko_allow_bad_res: bool = False,
    meeko_default_altloc: str | None = None,
) -> None:
    if pocket_pdbqt.exists():
        return
    if meeko is None:
        raise RuntimeError(f"Meeko not available; cannot prepare receptor PDBQT. import_error={_MEEKO_IMPORT_ERROR}")
    script = Path(meeko.__file__).resolve().parent / "cli" / "mk_prepare_receptor.py"
    clean_pdb = pocket_pdb.with_suffix(".clean.pdb")
    _clean_pdb_for_meeko(pocket_pdb, clean_pdb, default_altloc=meeko_default_altloc)
    mk_config = pocket_pdbqt.parent / "meeko_receptor_config.json"
    if not mk_config.exists():
        # Ensure atom types are assigned for receptor PDBQT.
        mk_config.write_text(
            json.dumps({"load_atom_params": "ad4_types"}, indent=2)
        )
    cmd = [
        sys.executable,
        str(script),
        "--read_pdb", str(clean_pdb),
        "--write_pdbqt", str(pocket_pdbqt),
        "--mk_config", str(mk_config),
        "--rigid_macrocycles",
    ]
    if meeko_allow_bad_res:
        cmd.append("--allow_bad_res")
    if meeko_default_altloc:
        cmd += ["--default_altloc", str(meeko_default_altloc)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Meeko receptor prep failed:\n{result.stdout}\n{result.stderr}")
    if not pocket_pdbqt.exists() or pocket_pdbqt.stat().st_size == 0:
        raise RuntimeError("Meeko receptor prep produced empty PDBQT.")


def _write_pdbqt_from_rdkit(mol: Chem.Mol, path: Path) -> None:
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot export empty ligand to PDBQT")
    if MoleculePreparation is None or PDBQTWriterLegacy is None:
        raise RuntimeError(f"Meeko not available; cannot prepare ligand PDBQT. import_error={_MEEKO_IMPORT_ERROR}")
    mol = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure partial charge props are finite before Meeko writer.
    # Some molecules can carry NaN charges, which hard-fails PDBQT export.
    for atom in mol.GetAtoms():
        for prop in ("_GasteigerCharge", "_GasteigerHCharge", "_TriposPartialCharge"):
            if atom.HasProp(prop):
                try:
                    v = float(atom.GetProp(prop))
                except Exception:
                    atom.ClearProp(prop)
                    continue
                if not math.isfinite(v):
                    atom.ClearProp(prop)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    for atom in mol.GetAtoms():
        c = 0.0
        h = 0.0
        if atom.HasProp("_GasteigerCharge"):
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                c = 0.0
        if atom.HasProp("_GasteigerHCharge"):
            try:
                h = float(atom.GetProp("_GasteigerHCharge"))
            except Exception:
                h = 0.0
        if not math.isfinite(c):
            c = 0.0
        if not math.isfinite(h):
            h = 0.0
        atom.SetProp("_GasteigerCharge", f"{c:.6f}")
        atom.SetProp("_GasteigerHCharge", f"{h:.6f}")

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    if not setups:
        raise RuntimeError("Meeko failed to prepare ligand.")
    pdbqt_string, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"Meeko PDBQT writer failed: {err}")
    path.write_text(pdbqt_string)


def _write_pre_sdf_from_mol(mol: Chem.Mol, path: Path) -> None:
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot export empty ligand to pre.sdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    except Exception:
        mol_h = mol
    try:
        block = Chem.MolToMolBlock(mol_h)
    except Exception:
        block = Chem.MolToMolBlock(mol)
    path.write_text(block)


def _prune_bad_hydrogens(mol: Chem.Mol, max_bond_len: float = 2.2, origin_tol: float = 1.0e-3) -> Chem.Mol:
    if mol is None or mol.GetNumConformers() == 0:
        return mol
    conf = mol.GetConformer()
    remove = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
            h = a if a.GetAtomicNum() == 1 else b
            heavy = b if a.GetAtomicNum() == 1 else a
            ph = conf.GetAtomPosition(h.GetIdx())
            p = conf.GetAtomPosition(heavy.GetIdx())
            d = ph.Distance(p)
            if d > max_bond_len:
                remove.append(h.GetIdx())
                continue
            if (abs(ph.x) + abs(ph.y) + abs(ph.z)) < origin_tol and (abs(p.x) + abs(p.y) + abs(p.z)) > origin_tol * 10.0:
                remove.append(h.GetIdx())
    if not remove:
        return mol
    emol = Chem.RWMol(mol)
    for idx in sorted(set(remove), reverse=True):
        try:
            emol.RemoveAtom(idx)
        except Exception:
            pass
    new = emol.GetMol()
    try:
        Chem.SanitizeMol(new)
    except Exception:
        pass
    return new


def _pose_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next(iter(supplier), None)
        if mol is None or mol.GetNumAtoms() == 0:
            return False
        return True
    except Exception:
        return False
    finally:
        try:
            supplier = None
            del supplier
        except Exception:
            pass


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        log.warning("Failed to delete file in use: %s", path)


def _best_match_by_bond_distance(
    templ: Chem.Mol, pose: Chem.Mol, max_bond_len: float = 2.0
) -> tuple[int, ...] | None:
    try:
        matches = pose.GetSubstructMatches(templ, useChirality=False)
    except Exception:
        matches = ()
    if not matches:
        return None
    conf = pose.GetConformer()
    best_within = None
    best_within_score = None
    best_max = None
    best_sum = None
    best = None
    for match in matches:
        score = 0.0
        max_len = 0.0
        for bond in templ.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = match[i]
            pj = match[j]
            pos_i = conf.GetAtomPosition(pi)
            pos_j = conf.GetAtomPosition(pj)
            d = pos_i.Distance(pos_j)
            score += d
            if d > max_len:
                max_len = d
        if max_len <= max_bond_len:
            if best_within_score is None or score < best_within_score:
                best_within_score = score
                best_within = match
            continue
        if best_max is None or max_len < best_max or (max_len == best_max and (best_sum is None or score < best_sum)):
            best_max = max_len
            best_sum = score
            best = match
    if best_within is not None:
        return best_within
    return best


def _build_ligand_graph(
    row,
    lig_mol: Chem.Mol,
    pocket_x: torch.Tensor,
    pocket_pos: torch.Tensor,
    symbol2idx: dict,
    feat_dim: int,
    atom_dim: int,
    atom_feat_dim: int,
    pose_path: Path,
) -> Data:
    # 去除氢原子，保持与 DiffGui / 3D 构图一致
    try:
        lig_mol = Chem.RemoveHs(lig_mol, sanitize=True)
    except Exception:
        lig_mol = Chem.RemoveHs(lig_mol, sanitize=False)
    if lig_mol.GetNumConformers() == 0:
        raise ValueError(f"Ligand from {pose_path} has no conformer after RemoveHs")
    lig_conf = lig_mol.GetConformer()
    lig_coords = np.asarray(lig_conf.GetPositions(), dtype=np.float32)
    lig_pos = torch.tensor(lig_coords, dtype=torch.float32)
    lig_atoms = [atom.GetSymbol() for atom in lig_mol.GetAtoms()]
    lig_feat = torch.zeros(len(lig_atoms), feat_dim, dtype=torch.float32)
    atom_indices_list: List[int] = []
    for sym in lig_atoms:
        if sym not in symbol2idx:
            raise ValueError(f"Atom '{sym}' not in ligand vocabulary")
        atom_indices_list.append(symbol2idx[sym])
    atom_indices = torch.tensor(atom_indices_list, dtype=torch.long)
    lig_feat[torch.arange(len(lig_atoms)), atom_indices] = 1.0
    # Append extra atom features
    extra_dim = max(int(atom_feat_dim - atom_dim), 0)
    if extra_dim > 0:
        extra_vals = torch.zeros((len(lig_atoms), extra_dim), dtype=torch.float32)
        for idx, atom in enumerate(lig_mol.GetAtoms()):
            vec = _adjust_feature_vec(_atom_feature_vec(atom), extra_dim)
            if vec:
                extra_vals[idx] = torch.tensor(vec, dtype=torch.float32)
        lig_feat[:, atom_dim:atom_feat_dim] = extra_vals

    num_pocket = pocket_x.size(0)
    num_ligand = lig_feat.size(0)
    total_nodes = num_pocket + num_ligand
    global_offset = num_pocket

    x = torch.cat([pocket_x, lig_feat], dim=0)
    pos = torch.cat([pocket_pos, lig_pos], dim=0)
    residue_dim = max(int(feat_dim - atom_feat_dim), 0)
    if residue_dim > 0:
        pocket_x_idx = pocket_x[:, atom_feat_dim:].argmax(dim=-1) + atom_feat_dim
    else:
        pocket_x_idx = torch.zeros(num_pocket, dtype=torch.long)
    x_idx = torch.cat([pocket_x_idx, atom_indices], dim=0)

    mask_context_node = torch.zeros(total_nodes, dtype=torch.bool)
    mask_context_node[:num_pocket] = True
    mask_ligand = ~mask_context_node
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_type: List[int] = []

    bond_type_map = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    no_bond_idx = 0

    ligand_pairs: Set[Tuple[int, int]] = set()
    edge_feat: List[List[float]] = []
    for bond in lig_mol.GetBonds():
        btype_raw = bond.GetBondType()
        if btype_raw not in bond_type_map:
            log.warning("[pocket_dataset] drop molecule containing unsupported bond type: %s", str(btype_raw))
            return None
        i = global_offset + bond.GetBeginAtomIdx()
        j = global_offset + bond.GetEndAtomIdx()
        btype = bond_type_map[btype_raw]
        ligand_pairs.add(tuple(sorted((i, j))))
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_type.extend([btype, btype])
        bfeat = _bond_feature_vec(bond)
        edge_feat.extend([bfeat, bfeat])

    # 完全图：为未显式存在的配体节点对补充 NO_BOND 边
    for a in range(num_ligand):
        ia = global_offset + a
        for b in range(a + 1, num_ligand):
            ib = global_offset + b
            if (ia, ib) in ligand_pairs:
                continue
            ligand_pairs.add((ia, ib))
            edge_src.extend([ia, ib])
            edge_dst.extend([ib, ia])
            edge_type.extend([no_bond_idx, no_bond_idx])
            edge_feat.extend([[0.0] * BOND_EXTRA_DIM, [0.0] * BOND_EXTRA_DIM])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty((2, 0), dtype=torch.long)
    if edge_type:
        edge_type_t = torch.tensor(edge_type, dtype=torch.long)
        edge_onehot = F.one_hot(edge_type_t, num_classes=BOND_TYPE_CLASSES).to(torch.float32)
        if edge_feat:
            extra = torch.tensor([_adjust_feature_vec(f, BOND_EXTRA_DIM) for f in edge_feat], dtype=torch.float32)
            edge_attr = torch.cat([edge_onehot, extra], dim=-1)
        else:
            edge_attr = edge_onehot
    else:
        edge_attr = torch.zeros((0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM), dtype=torch.float32)
    raw_smi = getattr(row, "smiles", "")
    canonical_smi = raw_smi
    try:
        smi_mol = Chem.MolFromSmiles(raw_smi)
        if smi_mol is not None:
            smi_mol = Chem.RemoveHs(smi_mol, sanitize=True)
            canonical_smi = Chem.MolToSmiles(smi_mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        log.warning("[pocket_dataset] canonicalize smiles failed smiles=%s err=%s", raw_smi, str(exc))

    from graph.schema import compute_mask_ligand_edge

    data = Data(
        x=x,
        x_idx=x_idx,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.zeros(0, dtype=torch.long),
        mask_ligand=mask_ligand,
        mask_ligand_edge=compute_mask_ligand_edge(edge_index, mask_ligand),
        smiles=canonical_smi,
        ligand_id=getattr(row, "ligand_id", ""),
        pose_path=str(pose_path),
    )
    try:
        dock_score_val = float(getattr(row, "dock_score", float("nan")))
    except Exception:
        dock_score_val = float("nan")
    data.vina_score = dock_score_val
    data.dock_score = dock_score_val
    try:
        sa_val = float(getattr(row, "sa_score", float("nan")))
    except Exception:
        sa_val = float("nan")
    data.sa_score = sa_val
    try:
        if hasattr(row, "logp"):
            logp_val = float(getattr(row, "logp", float("nan")))
        elif hasattr(row, "logP"):
            logp_val = float(getattr(row, "logP", float("nan")))
        else:
            logp_val = float("nan")
    except Exception:
        logp_val = float("nan")
    data.logp = logp_val
    return data

def _vina_box_from_coords(coords: np.ndarray, buffer: float = 2.0) -> Tuple[List[float], List[float]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = ((maxs + mins) * 0.5).tolist()
    size = (np.maximum(maxs - mins, 1.0) + buffer).tolist()
    return center, size


def _run_vina_binary(vina_exec: Union[str, Path],
                     receptor_pdbqt: Path,
                     ligand_pdbqt: Path,
                     center: List[float],
                     box_size: List[float],
                     exhaustiveness: int = 32,
                     seed: int = 0,
                     n_poses: int = 1) -> Tuple[float, str]:
    exec_path = Path(vina_exec)
    if not exec_path.exists():
        resolved = which(str(vina_exec))
        if resolved is None:
            raise FileNotFoundError(f"Vina executable not found: {vina_exec}")
        exec_path = Path(resolved)
    fd, pose_name = tempfile.mkstemp(suffix=".pdbqt")
    os.close(fd)
    pose_file = Path(pose_name)
    cmd = [
        str(exec_path),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.3f}",
        "--size_y", f"{box_size[1]:.3f}",
        "--size_z", f"{box_size[2]:.3f}",
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(int(seed)),
        "--num_modes", str(max(n_poses, 1)),
        "--out", str(pose_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pose_file.unlink(missing_ok=True)
        log_text = result.stdout + "\n" + result.stderr
        raise RuntimeError(f"Vina docking failed:\n{log_text}")

    log_text = result.stdout or ""

    score = None
    for line in log_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                score = float(parts[1])
                break
            except ValueError:
                continue
    if score is None:
        score = float("nan")

    pose_data = pose_file.read_text()
    pose_file.unlink(missing_ok=True)
    return score, pose_data


def _pose_to_sdf(pdbqt_pose: List[str], rdkit_mol: Chem.Mol) -> str:
    if not pdbqt_pose:
        raise ValueError("Empty docking pose received from vina.")
    if PDBQTMolecule is None or RDKitMolCreate is None:
        raise RuntimeError(f"Meeko not available; cannot parse docking pose. import_error={_MEEKO_IMPORT_ERROR}")
    pose_block = pdbqt_pose[0]
    pdbqt_mol = PDBQTMolecule(pose_block)
    mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not mol_list or mol_list[0] is None:
        raise ValueError("Meeko failed to parse docking pose.")
    pose_mol = mol_list[0]
    pose_noh = Chem.RemoveHs(pose_mol, sanitize=False, implicitOnly=False)
    base = Chem.RemoveHs(Chem.Mol(rdkit_mol), sanitize=False, implicitOnly=False)
    n_pose = pose_noh.GetNumAtoms()
    n_base = base.GetNumAtoms()
    if n_pose != n_base:
        raise ValueError(
            f"Pose atom count mismatch (pose={n_pose} vs base={n_base}); "
            "check docking pose hydrogens/atom ordering."
        )
    conf_in = pose_noh.GetConformer()
    match = _best_match_by_bond_distance(base, pose_noh, max_bond_len=2.0)
    if match is None:
        log.warning("[pose_to_sdf] substructure match failed; using identity mapping for pose coords")
        match = tuple(range(n_base))
    conf = Chem.Conformer(n_base)
    for i in range(n_base):
        pos = conf_in.GetAtomPosition(match[i])
        conf.SetAtomPosition(i, pos)
    base.RemoveAllConformers()
    base.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(base)
    except Exception:
        pass
    # Add Hs for downstream visualization consistency.
    try:
        base_h = Chem.AddHs(base, addCoords=True)
    except Exception:
        base_h = base
    base_h = _prune_bad_hydrogens(base_h)
    return Chem.MolToMolBlock(base_h, includeStereo=True)



class PocketLigandDockingDataset(InMemoryDataset):
    # 0 表示无键/接触，其余 1-4 为单/双/三/芳
    NO_BOND_IDX: int = 0
    """
    Dataset that auto-generates pocket PDB, performs docking with AutoDock Vina,
    and returns protein-pocket conditioned ligand graphs.

    The expected directory layout of `root` is:
        root/
            protein.pdb              # full protein structure
            reference_ligand.sdf     # optional reference pose (defines pocket center)
            pocket.pdb               # generated if missing
            docking/                 # docking outputs
                {ligand_id}.sdf
                {ligand_id}.json
            smiles.csv               # ligand metadata
    """

    def __init__(self,
                 target_root: Path | str,
                 split: str = "train",
                 radius: float = 10.0,
                 auto_prepare: bool = True,
                 overwrite_docking: bool = False,
                 exhaustiveness: int = 8,
                 vina_executable: Optional[Path | str] = None,
                 tmp_dir: Optional[Path | str] = None,
                 context_radius: float = 6.0,
                 context_knn: int = 24,
                 split_ratio: Tuple[float, float, float] = (0.9, 0.1, 0.0),
                 rare_atom_threshold: float = 0.01,
                 ligand_vocab_override: Optional[List[str]] = None,
                 transform=None,
                 pre_transform=None):
        self.paths = PocketDatasetPaths.from_root(Path(target_root),pocket_radius=radius)
        self.split = split.lower()
        if tmp_dir is None:
            tmp_dir = self.paths.root / "tmp"
        self.tmp_dir = Path(tmp_dir)
        self.auto_prepare = auto_prepare
        self.overwrite_docking = overwrite_docking
        self.exhaustiveness = int(exhaustiveness)
        self.context_radius = float(context_radius)
        self.context_knn = 0  # no pocket-pockect KNN edges to match DiffGui (atom-level, edge-free)
        self.split_ratio = split_ratio
        self.rare_atom_threshold = float(rare_atom_threshold)
        if ligand_vocab_override:
            self.ligand_vocab_override = list(ligand_vocab_override)
        else:
            self.ligand_vocab_override = _load_ligand_vocab_override(self.paths.root)
        vina_exec = vina_executable
        if vina_exec is None:
            local_vina = self.paths.root / "bin" / "vina"
            alt_vina = self.paths.root / "bin" / "vina.exe"
            repo_root = Path(__file__).resolve().parents[1]
            repo_vina = repo_root / "bin" / "vina_1.2.7_win.exe"
            repo_vina_plain = repo_root / "bin" / "vina"
            repo_vina_exe = repo_root / "bin" / "vina.exe"
            if local_vina.exists():
                log.info("Using target-local Vina executable at %s", local_vina)
                vina_exec = local_vina
            elif alt_vina.exists():
                log.info("Using target-local Vina executable at %s", alt_vina)
                vina_exec = alt_vina
            elif repo_vina_plain.exists():
                log.info("Using repository Vina executable at %s", repo_vina_plain)
                vina_exec = repo_vina_plain
            elif repo_vina_exe.exists():
                log.info("Using repository Vina executable at %s", repo_vina_exe)
                vina_exec = repo_vina_exe
            elif repo_vina.exists():
                log.info("Using repository Vina executable at %s", repo_vina)
                vina_exec = repo_vina
            else:
                log.warning("Falling back to system 'vina' executable; may fail if not on PATH")
                vina_exec = "vina"
        self.vina_exec: Union[str, Path] = vina_exec
        if self.auto_prepare:
            self._ensure_preparation()
        super().__init__(self.paths.root, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        from graph.schema import assert_graph_schema, assert_no_legacy_fields

        assert_no_legacy_fields(self._data, where="PocketLigandDockingDataset.load_processed(self._data)")
        assert_graph_schema(self._data, where="PocketLigandDockingDataset.load_processed(self._data)", require_edge_attr=True)
        meta_path = Path(self.processed_paths[0] + ".meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")
        with meta_path.open("r") as handle:
            meta = json.load(handle)
        self.ligand_vocab = meta.get("ligand_vocab", [])
        self.residue_vocab = meta.get("residue_vocab", [])
        self.atom_dim = int(meta.get("atom_dim", len(self.ligand_vocab)))
        self.atom_extra_dim = int(meta.get("atom_extra_dim", max(int(meta.get("atom_feat_dim", self.atom_dim)) - self.atom_dim, 0)))
        self.atom_feat_dim = int(meta.get("atom_feat_dim", self.atom_dim + self.atom_extra_dim))
        self.residue_dim = int(meta.get("residue_dim", len(self.residue_vocab)))
        self.context_radius = float(meta.get("context_radius", context_radius))
        self.context_knn = int(meta.get("context_knn", context_knn))
        self.bond_length_bins_ = meta.get("bond_length_bins", None)
        self.bond_length_hist_ = meta.get("bond_length_hist", None)
        EDGE_DECODER_BASE = [
            None,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        EDGE_CLASS_NAMES_BASE = ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        self.ligand_vocab_index = {sym: i for i, sym in enumerate(self.ligand_vocab)}
        extra_names = [f"FEAT_{i}" for i in range(max(self.atom_feat_dim - self.atom_dim, 0))]
        self.atom_vocab_ = [*self.ligand_vocab, *extra_names, *[f"RES_{r}" for r in self.residue_vocab]]
        self.num_node_classes_ = self.atom_feat_dim + self.residue_dim
        self.edge_decoder_ = list(EDGE_DECODER_BASE)
        self.num_edge_classes_ = int(meta.get("edge_attr_dim", BOND_TYPE_CLASSES + BOND_EXTRA_DIM))
        self.bond_extra_dim = int(meta.get("bond_extra_dim", max(self.num_edge_classes_ - BOND_TYPE_CLASSES, 0)))
        self.edge_classes_ = list(EDGE_CLASS_NAMES_BASE)
        # Featurizer vocab for protein/AA
        self.protein_atomic_numbers_ = torch.tensor([6, 7, 8, 16, 34, 1], dtype=torch.long)
        self.max_num_aa_ = 21
        periodic_table = Chem.GetPeriodicTable()
        mass_list: List[float] = []
        for sym in self.ligand_vocab:
            try:
                mass_list.append(float(periodic_table.GetAtomicWeight(periodic_table.GetAtomicNumber(sym))))
            except Exception:
                mass_list.append(0.0)
        mass_list.extend([0.0 for _ in range(max(self.atom_feat_dim - self.atom_dim, 0))])
        mass_list.extend([0.0 for _ in self.residue_vocab])
        self.atom_masses_ = torch.tensor(mass_list, dtype=torch.float32)
        # 键长参考直方图（可选）
        self.bond_length_bins_ = meta.get("bond_length_bins", None)
        self.bond_length_hist_ = meta.get("bond_length_hist", None)
        self.smiles_list_ = [getattr(self.get(i), "smiles", "") for i in range(len(self))]
        # 预处理时已对 SMILES 去氢并 canonical
        self.reference_smiles_is_canonical_ = True
        self.num_graphs_ = len(self)
        self._bond_length_cache = None
        # 若缺失统计，现算一份以供可视化使用
        if self.bond_length_bins_ is None or self.bond_length_hist_ is None:
            try:
                bins_np, hist_dict = self._compute_bond_length_reference()
                self.bond_length_bins_ = torch.tensor(bins_np, dtype=torch.float32)
                hist_list = [hist_dict.get(t, np.zeros(len(bins_np) - 1, dtype=np.float64)) for t in range(self.num_edge_classes_)]
                self.bond_length_hist_ = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
            except Exception as exc:
                log.warning("Failed to compute bond length reference on the fly: %s", exc)
                self.bond_length_bins_ = torch.empty(0, dtype=torch.float32)
                self.bond_length_hist_ = torch.empty(0, 0, dtype=torch.float32)

        # 统计节点/边类别（仅配体-配体边）
        self._recompute_priors()
        # 确认先验中 no_bond (index 0) 概率最高，否则可能混用旧编码
        try:
            if self.e_prob_.argmax().item() != self.NO_BOND_IDX:
                raise ValueError("Edge prior suggests no_bond is not index 0; please reprocess dataset with NO_BOND_IDX=0.")
        except Exception as exc:
            raise
        ligand_sizes: List[int] = []
        for idx in range(len(self)):
            data_i = super().__getitem__(idx)
            if hasattr(data_i, "num_ligand_nodes"):
                ligand_sizes.append(int(data_i.num_ligand_nodes))
            elif hasattr(data_i, "mask_ligand"):
                ligand_sizes.append(int(data_i.mask_ligand.sum().item()))
            else:
                ligand_sizes.append(int(data_i.num_nodes))
        if ligand_sizes:
            unique_sizes, counts = torch.unique(torch.tensor(ligand_sizes, dtype=torch.long), return_counts=True)
        else:
            unique_sizes = torch.zeros(0, dtype=torch.long)
            counts = torch.zeros(0, dtype=torch.long)
        self.size_bins_ = unique_sizes
        self.size_counts_ = counts
        self.pocket_template_: Data | None = self._load_pocket_template()

    @property
    def x_prob(self) -> torch.Tensor:
        return self.x_prob_

    @property
    def e_prob(self) -> torch.Tensor:
        return self.e_prob_

    @property
    def x_hist_total(self) -> torch.Tensor:
        return self.x_hist_total_

    @property
    def e_hist_total(self) -> torch.Tensor:
        return self.e_hist_total_

    @staticmethod
    def _make_pocket_template(pocket_x: torch.Tensor,
                              pocket_pos: torch.Tensor,
                              pocket_name: str | None = None,
                              pocket_index: int | None = None) -> Data:
        """Build a pocket-only Data object (atom-level) with no internal edges."""
        num_pocket = int(pocket_x.size(0))
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty(0, dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM), dtype=torch.float32)
        mask_ligand = torch.zeros(num_pocket, dtype=torch.bool)
        from graph.schema import compute_mask_ligand_edge

        template = Data(
            x=pocket_x.clone(),
            x_idx=torch.argmax(pocket_x, dim=-1),
            pos=pocket_pos.clone(),
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            mask_ligand=mask_ligand,
            mask_ligand_edge=compute_mask_ligand_edge(edge_index, mask_ligand),
            pocket_name=pocket_name or "",
        )
        template.num_pocket_nodes = num_pocket
        if pocket_index is not None:
            template.pocket_index = torch.tensor([pocket_index], dtype=torch.long)
        return template

    def _load_pocket_template(self) -> Data | None:
        path = Path(self.processed_paths[0] + ".pocket.pt")
        if path.exists():
            tpl = torch.load(path, weights_only=False)
            from graph.schema import assert_no_legacy_fields
            assert_no_legacy_fields(tpl, where=f"Pocket template load: {path}")
            return tpl
        # fallback: rebuild directly from pocket.pdb so it matches preprocessing
        return self._build_pocket_template_from_files()

    def _build_pocket_template_from_files(self) -> Data | None:
        """Rebuild pocket template from pocket.pdb contents."""
        try:
            pocket_res_coords, pocket_res_names = _read_pocket_residues(self.paths.pocket_pdb)
        except Exception as exc:
            log.warning("Failed to read pocket residues from %s: %s", self.paths.pocket_pdb, exc)
            return None
        feat_dim = self.atom_feat_dim + self.residue_dim
        pocket_x = torch.zeros(len(pocket_res_names), feat_dim, dtype=torch.float32)
        for idx, res in enumerate(pocket_res_names):
            res_upper = res.upper()
            if res_upper not in self.residue_vocab:
                continue
            rid = self.residue_vocab.index(res_upper)
            pocket_x[idx, self.atom_feat_dim + rid] = 1.0
        pocket_pos = torch.tensor(pocket_res_coords, dtype=torch.float32)
        return self._make_pocket_template(
            pocket_x=pocket_x,
            pocket_pos=pocket_pos,
            pocket_name=self.paths.root.name,
        )

    @staticmethod
    def _extract_pocket_template(data: Data, pocket_name: str | None = None) -> Data | None:
        if not hasattr(data, "mask_ligand"):
            raise RuntimeError("Cannot extract pocket template: missing required field mask_ligand.")
        mask_ligand = data.mask_ligand.bool()
        mask_context_node = ~mask_ligand
        if mask_context_node.numel() == 0:
            return None
        pocket_x = data.x[mask_context_node]
        pocket_pos = data.pos[mask_context_node]
        edge_mask = None
        if hasattr(data, "edge_index"):
            src = data.edge_index[0]
            dst = data.edge_index[1]
            edge_mask = (mask_context_node[src] & mask_context_node[dst])
        if edge_mask is not None and edge_mask.numel() > 0:
            edge_index = data.edge_index[:, edge_mask].clone()
            edge_type = data.edge_type[edge_mask].clone() if hasattr(data, "edge_type") else torch.zeros(edge_index.size(1), dtype=torch.long)
            if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_attr = data.edge_attr[edge_mask].clone()
            else:
                edge_onehot = F.one_hot(edge_type.clamp(min=0), num_classes=BOND_TYPE_CLASSES).float()
                edge_attr = torch.cat(
                    [edge_onehot, torch.zeros((edge_onehot.size(0), BOND_EXTRA_DIM), dtype=edge_onehot.dtype)],
                    dim=-1,
                )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)
            edge_attr = torch.zeros((0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM), dtype=torch.float32)
        mask_ligand_tpl = torch.zeros(pocket_x.size(0), dtype=torch.bool)
        from graph.schema import compute_mask_ligand_edge
        tpl = Data(
            x=pocket_x.clone(),
            x_idx=torch.argmax(pocket_x, dim=-1),
            pos=pocket_pos.clone(),
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            mask_ligand=mask_ligand_tpl,
            mask_ligand_edge=compute_mask_ligand_edge(edge_index, mask_ligand_tpl),
            pocket_name=pocket_name or getattr(data, "pocket_name", ""),
        )
        tpl.num_pocket_nodes = pocket_x.size(0)
        return tpl

    def sample_pocket(self, device: torch.device | str | None = None) -> Data:
        """Return a pocket-only template for sampling."""
        pocket = self.pocket_template_
        if pocket is None:
            pocket = self._load_pocket_template()
        if pocket is None:
            raise RuntimeError("Pocket template unavailable; dataset may be empty.")
        tpl = pocket.clone()
        if device is not None:
            tpl = tpl.to(device)
        if getattr(tpl, "batch", None) is None or tpl.batch.numel() == 0:
            tpl.batch = torch.zeros(tpl.num_nodes, dtype=torch.long, device=tpl.pos.device)
        if not hasattr(tpl, "sample_num_nodes"):
            tpl.sample_num_nodes = lambda batch_size, device=None: self.sample_num_nodes(batch_size, device)  # type: ignore[attr-defined]
        return tpl

    @property
    def processed_file_names(self) -> List[str]:
        return [f"pocket_dataset_{self.split}.pt"]

    def _process(self) -> None:
        if all(Path(path).exists() for path in self.processed_paths):
            return
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.process()

    def process(self) -> None:  # noqa: D401
        if self.auto_prepare:
            self._ensure_preparation()
        df_raw = pd.read_csv(self.paths.smiles_csv)
        df, _ = _normalize_smiles_dataframe(df_raw, split_ratio=self.split_ratio, assign_split=False)
        pocket_coords, pocket_elems = _read_pdb_atoms(self.paths.pocket_pdb)
        if len(pocket_elems) > 0:
            mask_heavy = [elem.upper() != "H" for elem in pocket_elems]
            pocket_coords = pocket_coords[mask_heavy]
            pocket_elems = [e for e, keep in zip(pocket_elems, mask_heavy) if keep]
        pocket_res_coords, pocket_res_names = _read_pocket_residues(self.paths.pocket_pdb)

        ligand_mols = []
        residue_set = set(name.upper() for name in pocket_res_names)
        symbol_set = set(pocket_elems)

        # 使用 full vocab 初值（扫描得到），后续根据频率筛选 major_atoms
        if self.ligand_vocab_override:
            ligand_vocab = list(self.ligand_vocab_override)
        else:
            ligand_vocab = None  # 将在扫描后确定

        # --- 仅当前 split 构建图（第一遍：读取并缓存 ligand mol，用于后续统计/过滤） ---
        for row in df.itertuples(index=False):
            if hasattr(row, "split") and str(row.split).lower() != self.split:
                continue
            pose_raw = getattr(row, "dock_pose", "")
            # pandas may surface missing values as floats (NaN); coerce safely to string
            if isinstance(pose_raw, str):
                pose_rel = pose_raw.strip()
            else:
                try:
                    val_num = float(pose_raw)
                    pose_rel = "" if not np.isfinite(val_num) else str(pose_raw).strip()
                except Exception:
                    pose_rel = str(pose_raw).strip()
            if not pose_rel or pose_rel.lower() in {"nan", "none"}:
                log.warning(
                    "Docking pose missing/NaN for ligand %s in split %s; skipping",
                    getattr(row, "ligand_id", "?"),
                    self.split,
                )
                continue
            pose_path = self.paths.root / pose_rel
            if not pose_path.exists():
                log.warning(
                    "Docking pose missing for ligand %s in split %s: %s; skipping",
                    getattr(row, "ligand_id", "?"),
                    self.split,
                    pose_rel,
                )
                continue
            supplier = Chem.SDMolSupplier(str(pose_path), removeHs=True)
            lig_mol = next(iter(supplier), None)
            if lig_mol is None:
                log.warning(
                    "Failed to read docking pose for ligand %s in split %s: %s; skipping",
                    getattr(row, "ligand_id", "?"),
                    self.split,
                    pose_path,
                )
                continue
            lig_mol = Chem.RemoveHs(lig_mol, sanitize=True)
            symbol_set.update(atom.GetSymbol() for atom in lig_mol.GetAtoms())
            ligand_mols.append((row, lig_mol, pose_path))

        # 如果未指定覆盖 vocab，则使用扫描得到的完整词表
        if ligand_vocab is None:
            ligand_vocab = sorted(symbol_set)
        self.ligand_vocab_full = list(ligand_vocab)

        residue_vocab = sorted(residue_set)
        atom_dim = len(ligand_vocab)
        residue_dim = len(residue_vocab)
        atom_feat_dim = atom_dim + ATOM_EXTRA_DIM
        feat_dim = atom_feat_dim + residue_dim
        symbol2idx = {sym: idx for idx, sym in enumerate(ligand_vocab)}
        residue2idx = {res: idx for idx, res in enumerate(residue_vocab)}

        # --- 第一遍统计：per-molecule 原子频率（仅 ligand 部分）；不再做 vocab 压缩 ---
        ligand_symbol_sets: List[Set[str] | None] = []
        atom_mol_counts = {sym: 0 for sym in self.ligand_vocab_full}
        num_molecules = 0
        for _, lig_mol, _ in ligand_mols:
            try:
                symbols = {atom.GetSymbol() for atom in lig_mol.GetAtoms()}
            except Exception:
                ligand_symbol_sets.append(None)
                continue
            ligand_symbol_sets.append(symbols)
            num_molecules += 1
            for sym in symbols:
                if sym in atom_mol_counts:
                    atom_mol_counts[sym] += 1
        atom_mol_freq = {sym: (atom_mol_counts[sym] / float(num_molecules)) if num_molecules > 0 else 0.0 for sym in self.ligand_vocab_full}
        rare_atom_threshold = getattr(self, "rare_atom_threshold", 0.03)
        major_atoms = list(self.ligand_vocab_full)
        # silent
        report = {
            "full_ligand_vocab": self.ligand_vocab_full,
            "num_molecules": int(num_molecules),
            "per_mol_counts": {sym: int(atom_mol_counts[sym]) for sym in self.ligand_vocab_full},
            "per_mol_freq": {sym: float(atom_mol_freq[sym]) for sym in self.ligand_vocab_full},
            "rare_atom_threshold": float(rare_atom_threshold),
            "major_atoms": major_atoms,
        }
        report_path = Path(self.processed_paths[0] + ".atom_per_mol.json")
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # --- 第二遍：使用完整 vocab 构建图 ---
        self.ligand_vocab = list(self.ligand_vocab_full)
        self.ligand_vocab_index = {sym: i for i, sym in enumerate(self.ligand_vocab)}
        # 重新计算维度与口袋特征
        atom_dim = len(self.ligand_vocab)
        atom_feat_dim = atom_dim + ATOM_EXTRA_DIM
        feat_dim = atom_feat_dim + residue_dim
        symbol2idx = {sym: idx for idx, sym in enumerate(self.ligand_vocab)}
        pocket_x = torch.zeros(len(pocket_res_names), feat_dim, dtype=torch.float32)
        for idx, res in enumerate(pocket_res_names):
            rid = residue2idx[res.upper()]
            pocket_x[idx, atom_feat_dim + rid] = 1.0
        pocket_pos = torch.tensor(pocket_res_coords, dtype=torch.float32)
        pocket_template = self._make_pocket_template(
            pocket_x=pocket_x,
            pocket_pos=pocket_pos,
            pocket_name=self.paths.root.name,
        )
        data_list: List[Data] = []
        for row_idx, ((row, lig_mol, pose_path), sym_set) in enumerate(zip(ligand_mols, ligand_symbol_sets)):
            try:
                lig_data = _build_ligand_graph(
                    row,
                    lig_mol,
                    pocket_x,
                    pocket_pos,
                    symbol2idx,
                    feat_dim,
                    atom_dim,
                    atom_feat_dim,
                    pose_path,
                )
            except ValueError as exc:
                log.warning("[Pocket %s] Skip ligand %s due to vocab mismatch: %s", self.paths.root.name, getattr(row, "ligand_id", "?"), exc)
                continue
            if lig_data is None:
                log.warning("[Pocket %s] Skip ligand %s due to unsupported bond types", self.paths.root.name, getattr(row, "ligand_id", "?"))
                continue
            if sym_set is None:
                continue
            # 记录该样本在当前 split 中的顺序，便于后续 batch 直接索引 SMILES
            lig_data.smiles_idx = len(data_list)  # type: ignore[attr-defined]
            data_list.append(lig_data)
        if not data_list:
            empty = Data(
                x=torch.zeros(0, feat_dim),
                pos=torch.zeros(0, 3),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                edge_attr=torch.zeros(0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM),
                mask_ligand=torch.zeros(0, dtype=torch.bool),
                mask_ligand_edge=torch.zeros(0, dtype=torch.bool),
                x_idx=torch.zeros(0, dtype=torch.long),
            )
            data_list.append(empty)
        # 统计节点/边类别直方图（仅配体-配体边）
        x_counts = torch.zeros(atom_feat_dim + residue_dim, dtype=torch.float64)
        e_counts = torch.zeros(BOND_TYPE_CLASSES, dtype=torch.float64)
        bond_len_lists: Dict[int, List[float]] = {}
        bins_np = np.linspace(0.0, 3.0, 101)
        for g in data_list:
            ml = g.mask_ligand if hasattr(g, "mask_ligand") else torch.zeros(g.num_nodes, dtype=torch.bool)
            x_counts += torch.bincount(g.x_idx[ml], minlength=x_counts.numel()).double()
            if g.edge_type.numel() > 0:
                ei = g.edge_index
                edge_keep = ml[ei[0]] & ml[ei[1]]
                if edge_keep.any():
                    e_counts += torch.bincount(g.edge_type[edge_keep], minlength=e_counts.numel()).double()
                    src = ei[0][edge_keep]
                    dst = ei[1][edge_keep]
                    et = g.edge_type[edge_keep]
                    # 仅上三角，避免重复
                    tri_mask = src < dst
                    src = src[tri_mask]
                    dst = dst[tri_mask]
                    et = et[tri_mask]
                    if src.numel() > 0:
                        lengths = torch.norm(g.pos[src] - g.pos[dst], dim=1).cpu().numpy()
                        et_np = et.cpu().numpy()
                        for l, t in zip(lengths.tolist(), et_np.tolist()):
                            if t == self.NO_BOND_IDX:
                                continue
                            bond_len_lists.setdefault(int(t), []).append(float(l))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # 预计算键长直方图
        hist_dict = {}
        max_edge_class = BOND_TYPE_CLASSES
        bond_classes = [c for c in range(max_edge_class) if c != self.NO_BOND_IDX]
        hist_stack = np.zeros((max_edge_class, len(bins_np) - 1), dtype=np.float64)
        for t in bond_classes:
            vals = bond_len_lists.get(t, [])
            if vals:
                hist_counts, _ = np.histogram(vals, bins=bins_np, density=False)
                hist = hist_counts.astype(np.float64)
                total = hist.sum()
                if total > 0:
                    hist /= total
            else:
                hist = np.zeros(len(bins_np) - 1, dtype=np.float64)
            hist_dict[t] = hist
            hist_stack[t] = hist
        meta = {
            "ligand_vocab": self.ligand_vocab,
            "residue_vocab": residue_vocab,
            "atom_dim": atom_dim,
            "atom_feat_dim": atom_feat_dim,
            "atom_extra_dim": ATOM_EXTRA_DIM,
            "residue_dim": residue_dim,
            "context_radius": self.context_radius,
            "context_knn": self.context_knn,
            "bond_length_bins": bins_np.tolist(),
            "bond_length_hist": hist_stack.tolist(),
            "x_hist_total": x_counts.tolist() if "x_counts" in locals() else None,
            "e_hist_total": e_counts.tolist() if "e_counts" in locals() else None,
            "edge_attr_dim": BOND_TYPE_CLASSES + BOND_EXTRA_DIM,
            "bond_extra_dim": BOND_EXTRA_DIM,
            "bond_type_classes": BOND_TYPE_CLASSES,
        }
        meta_path = Path(self.processed_paths[0] + ".meta")
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2)
        torch.save(pocket_template, self.processed_paths[0] + ".pocket.pt")

    def bond_length_reference_by_type(self):
        """
        返回（bins, hist_dict），对每类显式键（排除 NO_BOND）给出键长直方图参考。
        若预处理未保存，则懒加载计算并缓存。
        """
        if hasattr(self, "_bond_length_cache") and self._bond_length_cache is not None:
            return self._bond_length_cache
        # 若已有存储的 bins/hist，则直接转换
        if isinstance(self.bond_length_bins_, torch.Tensor) and self.bond_length_bins_.numel() > 0:
            bins = self.bond_length_bins_.cpu().numpy()
        else:
            bins = np.linspace(0.0, 3.0, 101)
        hist_dict = {}
        if isinstance(self.bond_length_hist_, torch.Tensor) and self.bond_length_hist_.numel() > 0:
            hist_arr = self.bond_length_hist_.cpu().numpy()
            bond_classes = [c for c in range(hist_arr.shape[0]) if c != self.NO_BOND_IDX]
            for t in bond_classes:
                hist_dict[t] = hist_arr[t]
        # 若缺失，现算
        if not hist_dict:
            bins, hist_dict = self._compute_bond_length_reference()
        self._bond_length_cache = (bins, hist_dict)
        return self._bond_length_cache

    def _compute_bond_length_reference(self):
        """
        现算键长分布（仅真实键类型，排除 NO_BOND），返回 (bins, hist_dict)。
        """
        bins = np.linspace(0.0, 3.0, 101)
        max_edge_class = 5
        bond_classes = [c for c in range(max_edge_class) if c != self.NO_BOND_IDX]
        length_lists: Dict[int, List[float]] = {i: [] for i in bond_classes}
        for data in self:
            if not hasattr(data, "edge_index") or data.edge_index.numel() == 0:
                continue
            if not hasattr(data, "pos") or data.pos.numel() == 0:
                continue
            edge_type = getattr(data, "edge_type", None)
            if edge_type is None or edge_type.numel() == 0:
                continue
            edge_index = data.edge_index
            pos = data.pos
            mask = (edge_index[0] < edge_index[1]) & (edge_type != self.NO_BOND_IDX)
            if mask.sum().item() == 0:
                continue
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            etypes = edge_type[mask].cpu().numpy()
            lengths = torch.norm(pos[src] - pos[dst], dim=1).cpu().numpy()
            for l, t in zip(lengths.tolist(), etypes.tolist()):
                if t == self.NO_BOND_IDX:
                    continue
                length_lists.setdefault(int(t), []).append(float(l))

        hist_dict: Dict[int, np.ndarray] = {}
        for t in bond_classes:
            values = length_lists.get(t, [])
            if values:
                hist_counts, _ = np.histogram(values, bins=bins, density=False)
                hist = hist_counts.astype(np.float64)
                total = hist.sum()
                if total > 0:
                    hist /= total
            else:
                hist = np.zeros(len(bins) - 1, dtype=np.float64)
            hist_dict[t] = hist
        return bins, hist_dict

    def _recompute_priors(self):
        """根据当前编码统计配体节点/边先验与键长直方图。"""
        x_counts = torch.zeros(self.num_node_classes_, dtype=torch.float64)
        e_counts = torch.zeros(self.num_edge_classes_, dtype=torch.float64)
        bond_len_lists: Dict[int, List[float]] = {}
        bins_np = np.linspace(0.0, 3.0, 101)
        num_ligand_atom_classes = len(self.ligand_vocab) if hasattr(self, "ligand_vocab") else len(self.atom_vocab_)
        atom_mol_counts = torch.zeros(num_ligand_atom_classes, dtype=torch.long)
        num_molecules = 0
        for idx in range(len(self)):
            data_i = super().__getitem__(idx)
            ml = getattr(data_i, "mask_ligand", torch.zeros(data_i.num_nodes, dtype=torch.bool))
            x_counts += torch.bincount(data_i.x_idx[ml], minlength=self.num_node_classes_).double()
            # 每个分子出现过哪些 ligand 原子类别
            num_molecules += 1
            if ml.any():
                x_idx_lig = data_i.x_idx[ml]
                mask_valid = (x_idx_lig >= 0) & (x_idx_lig < num_ligand_atom_classes)
                if mask_valid.any():
                    present_classes = torch.unique(x_idx_lig[mask_valid])
                    atom_mol_counts[present_classes] += 1
            if data_i.edge_type.numel() > 0:
                ei = data_i.edge_index
                edge_keep = ml[ei[0]] & ml[ei[1]]
                if edge_keep.any():
                    et = data_i.edge_type[edge_keep]
                    e_counts += torch.bincount(et, minlength=self.num_edge_classes_).double()
                    src = ei[0][edge_keep]
                    dst = ei[1][edge_keep]
                    tri_mask = src < dst
                    src = src[tri_mask]
                    dst = dst[tri_mask]
                    et = et[tri_mask]
                    if src.numel() > 0:
                        lengths = torch.norm(data_i.pos[src] - data_i.pos[dst], dim=1).cpu().numpy()
                        et_np = et.cpu().numpy()
                        for l, t in zip(lengths.tolist(), et_np.tolist()):
                            if t == self.NO_BOND_IDX:
                                continue
                            bond_len_lists.setdefault(int(t), []).append(float(l))
        self.x_hist_total_ = x_counts.long()
        self.e_hist_total_ = e_counts.long()
        self.x_prob_ = (x_counts / x_counts.sum().clamp_min(1e-12)).float()
        self.e_prob_ = (e_counts / e_counts.sum().clamp_min(1e-12)).float()
        # 键长参考
        hist_dict = {}
        max_edge_class = self.num_edge_classes_
        bond_classes = [c for c in range(max_edge_class) if c != self.NO_BOND_IDX]
        hist_stack = np.zeros((max_edge_class, len(bins_np) - 1), dtype=np.float64)
        for t in bond_classes:
            vals = bond_len_lists.get(t, [])
            if vals:
                hist_counts, _ = np.histogram(vals, bins=bins_np, density=False)
                hist = hist_counts.astype(np.float64)
                total = hist.sum()
                if total > 0:
                    hist /= total
            else:
                hist = np.zeros(len(bins_np) - 1, dtype=np.float64)
            hist_dict[t] = hist
            hist_stack[t] = hist
        self.bond_length_bins_ = torch.tensor(bins_np, dtype=torch.float32)
        self.bond_length_hist_ = torch.tensor(hist_stack, dtype=torch.float32)
        self._bond_length_cache = (bins_np, hist_dict)
        # per-molecule atom report（仅 ligand 原子）
        if num_molecules > 0:
            atom_mol_freq = (atom_mol_counts.float() / float(num_molecules)).tolist()
        else:
            atom_mol_freq = [0.0] * int(num_ligand_atom_classes)
        if hasattr(self, "ligand_vocab"):
            atom_vocab = list(self.ligand_vocab)
        else:
            atom_vocab = [str(v) for v in self.atom_vocab_[:num_ligand_atom_classes]]
        atom_report = {
            "atom_vocab": atom_vocab,
            "num_molecules": int(num_molecules),
            "per_mol_counts": [int(x) for x in atom_mol_counts.tolist()],
            "per_mol_freq": atom_mol_freq,
        }
        report_path = Path(self.processed_paths[0] + ".atom_per_mol.json")
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(atom_report, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _ensure_preparation(self) -> None:
        log.info("Preparing pocket dataset at %s (split=%s)", self.paths.root, self.split)
        self.paths.docking_dir.mkdir(parents=True, exist_ok=True)
        protein_atoms = _read_pdb_atoms(self.paths.protein_pdb)
        reference_mol = None
        ligand_atoms = None
        if self.paths.reference_ligand is not None:
            reference_mol = _load_ligand_conformer(self.paths.reference_ligand)
            lig_coords = np.asarray(reference_mol.GetConformer().GetPositions(), dtype=np.float64)
            ligand_atoms = (lig_coords, [atom.GetSymbol() for atom in reference_mol.GetAtoms()])

        if ligand_atoms is not None:
            protein_full = _parse_protein_atoms(self.paths.protein_pdb)
            pocket_atoms = _extract_pocket_atoms(protein_full, ligand_atoms, radius=self.paths.pocket_radius)
            if not pocket_atoms:
                raise ValueError("Pocket extraction produced empty atom set. Increase radius.")
            _write_pocket_pdb_from_atoms(pocket_atoms, self.paths.pocket_pdb)
        elif not self.paths.pocket_pdb.exists():
            protein_full = _parse_protein_atoms(self.paths.protein_pdb)
            pocket_atoms = protein_full
            _write_pocket_pdb_from_atoms(pocket_atoms, self.paths.pocket_pdb)
        pocket_coords, pocket_elems = _read_pdb_atoms(self.paths.pocket_pdb)

        pocket_pdbqt = self.paths.pocket_pdb.with_suffix(".pdbqt")
        _ensure_receptor_pdbqt(self.paths.pocket_pdb, pocket_pdbqt)
        center, box_size = _vina_box_from_coords(pocket_coords)

        try:
            df_raw = pd.read_csv(self.paths.smiles_csv)
        except pd.errors.ParserError:
            log.warning("pandas failed to parse %s with default engine; retry with python engine (skip bad lines)", self.paths.smiles_csv)
            df_raw = pd.read_csv(self.paths.smiles_csv, engine="python", on_bad_lines="skip")
        if df_raw.empty:
            raise RuntimeError(f"smiles.csv at {self.paths.smiles_csv} contains no rows; cannot build dataset.")
        df, modified_base = _normalize_smiles_dataframe(df_raw, split_ratio=self.split_ratio, assign_split=False)
        canonical_values: List[str] = []
        canonical_changed = False
        for smi in df["smiles"].astype(str):
            can = canonicalize_smiles(smi) or smi
            # 去氢后再 canonical，保持与无氢建图/对接一致
            try:
                mol_tmp = Chem.MolFromSmiles(can)
                if mol_tmp is not None:
                    # 强制移除所有显式氢（包括连接在杂原子上的 H），保持与无氢建图一致
                    mol_tmp = Chem.RemoveHs(mol_tmp, sanitize=True, implicitOnly=False)
                    can_noh = Chem.MolToSmiles(mol_tmp, canonical=True, isomericSmiles=True)
                    if can_noh:
                        can = can_noh
            except Exception:
                pass
            canonical_values.append(can)
        if "smiles_canonical" in df.columns:
            if list(df["smiles_canonical"].astype(str)) != canonical_values:
                canonical_changed = True
        else:
            canonical_changed = True
        df["smiles_canonical"] = canonical_values
        debug_dedup = os.environ.get("DEBUG_SMILES_DEDUP", "0") == "1"
        if debug_dedup:
            total = len(df)
            canon_series = pd.Series(canonical_values, dtype="object")
            empty_canon = canon_series.isna() | canon_series.astype(str).str.strip().isin({"", "nan", "none"})
            uniq = int(canon_series.nunique(dropna=False))
            dup = int(total - uniq)
            print(
                f"[debug] smiles rows={total} unique_canonical={uniq} "
                f"dup={dup} empty_canonical={int(empty_canon.sum())}"
            )
            if dup > 0:
                top_counts = canon_series.value_counts().head(5)
                print(f"[debug] top_canonical_counts: {top_counts.to_dict()}")
        # 按 canonical SMILES 去重，并合并中药来源列
        df, merged_dup = _deduplicate_canonical_rows(df, merge_col="Chinese (汉字)")
        if debug_dedup and merged_dup:
            print(f"[debug] dedup merged -> rows={len(df)}")
        if merged_dup:
            modified_base = True
        # 去重后再分配 split（若原始文件未提供）
        df = _assign_split_column(df, self.split_ratio)
        if modified_base or canonical_changed or merged_dup:
            df.to_csv(self.paths.smiles_csv, index=False)
            modified_base = True
        updated = False
        skipped_ligands: List[str] = []
        best_score = float('inf')
        best_pose_path: Optional[Path] = None
        best_mol: Optional[Chem.Mol] = None

        canonical_cache: Dict[str, Dict[str, Union[str, float, List[float], None]]] = {}
        for row in df.itertuples(index=False):
            can = getattr(row, "smiles_canonical", "")
            if not can or can in canonical_cache:
                continue
            # robustly coerce dock_pose to string path; pandas may give NaN (float)
            pose_val = getattr(row, "dock_pose", "")
            pose_rel_existing = ""
            if isinstance(pose_val, str):
                pose_rel_existing = pose_val.strip()
            else:
                try:
                    # if it's a number (NaN included), treat as missing
                    val_num = float(pose_val)
                    if not np.isfinite(val_num):
                        pose_rel_existing = ""
                except Exception:
                    # fallback: stringify non-numeric object
                    pose_rel_existing = str(pose_val).strip()
            if not pose_rel_existing or pose_rel_existing.lower() in {"", "nan", "none"}:
                continue
            pose_existing = self.paths.root / pose_rel_existing
            if not pose_existing.exists():
                continue
            score_existing = getattr(row, "dock_score", np.nan)
            try:
                score_existing = float(score_existing)
            except (TypeError, ValueError):
                score_existing = float("nan")
            ligand_existing = getattr(row, "ligand_id", "")
            meta_existing = self.paths.docking_dir / f"{ligand_existing}.json"
            pdbqt_rel = ""
            center_existing = None
            box_existing = None
            if meta_existing.exists():
                try:
                    with meta_existing.open("r") as handle:
                        meta_payload = json.load(handle)
                    pdbqt_rel = meta_payload.get("pdbqt", "") or ""
                    center_existing = meta_payload.get("center")
                    box_existing = meta_payload.get("box_size")
                    if isinstance(score_existing, float) and math.isnan(score_existing):
                        score_existing = meta_payload.get("vina_score", score_existing)
                except Exception:
                    pass
            canonical_cache[can] = {
                "pose_rel": pose_rel_existing,
                "score": score_existing,
                "pdbqt_rel": pdbqt_rel,
                "center": center_existing,
                "box_size": box_existing,
            }

        failed_canonical: Set[str] = set()
        failed_map: Dict[str, str] = {}
        failed_file = self.paths.docking_dir / "failed.json"
        if failed_file.exists():
            try:
                with failed_file.open("r") as handle:
                    failed_map = json.load(handle) or {}
            except Exception:
                failed_map = {}
        if self.overwrite_docking:
            failed_map = {}
            failed_canonical.clear()

        if "split" in df.columns:
            split_mask = df["split"].astype(str).str.lower() == self.split
            df_iter = df[split_mask]
        else:
            df_iter = df

        if df_iter.empty:
            log.warning("No ligands found for split '%s' in %s; skipping docking", self.split, self.paths.smiles_csv)
            iterator = []
        else:
            iterator = tqdm(
                df_iter.itertuples(index=True),
                total=len(df_iter),
                desc=f"Dock {self.paths.root.name}",
                leave=False,
            )

        for row in iterator:
            ligand_id = getattr(row, "ligand_id", None)
            smiles = getattr(row, "smiles", None)
            if ligand_id is None or smiles is None:
                raise ValueError("smiles.csv must contain columns 'ligand_id' and 'smiles'.")
            row_index = getattr(row, "Index", None)
            if row_index is None:
                raise ValueError("smiles.csv row index unavailable; cannot update docking results safely.")
            if ligand_id in failed_map:
                skipped_ligands.append(ligand_id)
                log.warning("[Pocket %s] Skip ligand %s due to previous failure: %s", self.paths.root.name, ligand_id, failed_map.get(ligand_id, "unknown"))
                updated = True
                continue
            canonical_key = getattr(row, "smiles_canonical", "")
            if canonical_key and canonical_key in failed_canonical:
                skipped_ligands.append(ligand_id)
                log.warning("[Pocket %s] Skip ligand %s due to previous failures for canonical %s", self.paths.root.name, ligand_id, canonical_key)
                updated = True
                continue
            pose_rel_raw = getattr(row, "dock_pose", "")
            pose_rel = str(pose_rel_raw).strip()
            if pose_rel.lower() in {"nan", "none"}:
                pose_rel = ""
            score_raw = getattr(row, "dock_score", np.nan)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = np.nan
            orig_pose = pose_rel
            orig_score = score
            pose_path = self.paths.root / pose_rel if pose_rel else None
            meta_path = self.paths.docking_dir / f"{ligand_id}.json"
            if (pose_path is None or not pose_path.exists()) and meta_path.exists():
                try:
                    with meta_path.open("r") as handle:
                        meta_cached = json.load(handle)
                    pose_meta_rel = meta_cached.get("pose_path", "")
                    if pose_meta_rel:
                        pose_meta = self.paths.root / pose_meta_rel
                        if pose_meta.exists():
                            pose_path = pose_meta
                            pose_rel = pose_meta_rel
                    if (score is None) or (isinstance(score, float) and np.isnan(score)):
                        score = meta_cached.get("vina_score", score)
                except Exception:
                    pass
            cached_entry = canonical_cache.get(canonical_key) if canonical_key else None
            reused_from_canonical = False
            if (pose_path is None or not pose_path.exists()) and cached_entry:
                cached_pose_rel = cached_entry.get("pose_rel", "")
                if cached_pose_rel:
                    candidate_cached = self.paths.root / cached_pose_rel
                    if candidate_cached.exists() and _pose_is_valid(candidate_cached):
                        pose_path = candidate_cached
                        pose_rel = cached_pose_rel
                        cached_score = cached_entry.get("score")
                        if cached_score is not None and not (isinstance(cached_score, float) and math.isnan(cached_score)):
                            score = cached_score
                        reused_from_canonical = True
            candidate_pose = self.paths.docking_dir / f"{ligand_id}.sdf"
            if (pose_path is None or not pose_path.exists()) and candidate_pose.exists():
                pose_path = candidate_pose
                pose_rel = str(candidate_pose.relative_to(self.paths.root))
                if (score is None) or (isinstance(score, float) and np.isnan(score)):
                    if meta_path.exists():
                        try:
                            with meta_path.open("r") as handle:
                                meta = json.load(handle)
                            score = meta.get("vina_score", score)
                        except Exception:
                            pass
            if pose_path is not None and not _pose_is_valid(pose_path):
                log.warning("[Pocket %s] Invalid stored pose for ligand %s, re-docking", self.paths.root.name, ligand_id)
                _safe_unlink(pose_path)
                pose_path = None
                pose_rel = ""
                score = float("nan")
            needs_docking = self.overwrite_docking or pose_path is None or not pose_path.exists()
            if needs_docking:
                try:
                    lig_mol = _smiles_to_3d_mol(smiles)
                except Exception as exc:
                    skipped_ligands.append(ligand_id)
                    log.warning("[Pocket %s] Skip ligand %s due to 3D embedding failure: %s", self.paths.root.name, ligand_id, exc)
                    failed_map[ligand_id] = f"embed_fail: {exc}"
                    if canonical_key:
                        failed_canonical.add(canonical_key)
                    updated = True
                    continue
                try:
                    pre_path = self.paths.docking_dir / f"{ligand_id}.pre.sdf"
                    if self.overwrite_docking or not pre_path.exists():
                        _write_pre_sdf_from_mol(lig_mol, pre_path)
                except Exception:
                    pass
                lig_pdbqt = self.paths.docking_dir / f"{ligand_id}.pdbqt"
                try:
                    _write_pdbqt_from_rdkit(lig_mol, lig_pdbqt)
                    vina_score, pose_text = _run_vina_binary(
                        self.vina_exec,
                        pocket_pdbqt,
                        lig_pdbqt,
                        center=center, box_size=box_size,
                        exhaustiveness=self.exhaustiveness,
                    )
                except Exception as exc:
                    skipped_ligands.append(ligand_id)
                    log.warning("[Pocket %s] Skip ligand %s due to docking failure: %s", self.paths.root.name, ligand_id, exc)
                    _safe_unlink(lig_pdbqt)
                    failed_map[ligand_id] = f"docking_fail: {exc}"
                    if canonical_key:
                        failed_canonical.add(canonical_key)
                    updated = True
                    continue
                try:
                    sdf_block = _pose_to_sdf([pose_text], lig_mol)
                    pose_path = self.paths.docking_dir / f"{ligand_id}.sdf"
                    with pose_path.open("w") as handle:
                        handle.write(sdf_block)
                    supplier_check = Chem.SDMolSupplier(str(pose_path), removeHs=False)
                    test_mol = next(iter(supplier_check), None)
                    if test_mol is None or test_mol.GetNumAtoms() == 0:
                        raise ValueError("Parsed pose is empty")
                except Exception as exc:
                    skipped_ligands.append(ligand_id)
                    log.warning("[Pocket %s] Skip ligand %s due to invalid pose file: %s", self.paths.root.name, ligand_id, exc)
                    try:
                        _safe_unlink(pose_path)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    _safe_unlink(lig_pdbqt)
                    failed_map[ligand_id] = f"invalid_pose: {exc}"
                    if canonical_key:
                        failed_canonical.add(canonical_key)
                    updated = True
                    continue
                with meta_path.open("w") as handle:
                    json.dump(
                        {
                            "ligand_id": ligand_id,
                            "vina_score": vina_score,
                            "pose_path": str(pose_path.relative_to(self.paths.root)),
                            "pdbqt": str(lig_pdbqt.relative_to(self.paths.root)),
                            "center": center,
                            "box_size": box_size,
                        },
                        handle,
                        indent=2,
                    )
                score = vina_score
                pose_rel = str(pose_path.relative_to(self.paths.root))
                if canonical_key:
                    canonical_cache[canonical_key] = {
                        "pose_rel": pose_rel,
                        "score": score,
                        "pdbqt_rel": str(lig_pdbqt.relative_to(self.paths.root)),
                        "center": center,
                        "box_size": box_size,
                    }
                    failed_canonical.discard(canonical_key)
                updated = True
                if vina_score < best_score:
                    best_score = vina_score
                    best_pose_path = pose_path
                    best_mol = Chem.SDMolSupplier(str(pose_path), removeHs=False)[0]
            else:
                if reused_from_canonical and cached_entry and not meta_path.exists():
                    with meta_path.open("w") as handle:
                        json.dump(
                            {
                                "ligand_id": ligand_id,
                                "vina_score": score,
                                "pose_path": pose_rel,
                                "pdbqt": cached_entry.get("pdbqt_rel", ""),
                                "center": cached_entry.get("center"),
                                "box_size": cached_entry.get("box_size"),
                            },
                            handle,
                            indent=2,
                        )
                if reused_from_canonical and not pose_rel_raw:
                    updated = True
            def _scores_equal(a, b) -> bool:
                try:
                    if math.isnan(a) and math.isnan(b):
                        return True
                except Exception:
                    pass
                return a == b

            if (pose_rel != orig_pose) or (not _scores_equal(orig_score, score)):
                updated = True
            df.loc[row_index, "dock_pose"] = pose_rel
            df.loc[row_index, "dock_score"] = score
        if skipped_ligands:
            log.warning("[Pocket %s] Skipped %d ligands due to preprocessing failures: %s", self.paths.root.name, len(skipped_ligands), ",".join(map(str, skipped_ligands)))
        if failed_map:
            try:
                with (self.paths.docking_dir / "failed.json").open("w") as handle:
                    json.dump(failed_map, handle, indent=2)
            except Exception:
                pass
        if updated:
            df.to_csv(self.paths.smiles_csv, index=False)
        if ligand_atoms is None:
            if best_pose_path is None or best_mol is None:
                raise RuntimeError("Failed to generate docking poses to derive reference ligand.")
            ref_path = self.paths.root / "reference_ligand.sdf"
            Chem.MolToMolFile(best_mol, str(ref_path))
            lig_coords = np.asarray(best_mol.GetConformer().GetPositions(), dtype=np.float64)
            ligand_atoms = (lig_coords, [atom.GetSymbol() for atom in best_mol.GetAtoms()])
            pocket_coords, pocket_elems = _extract_pocket(protein_atoms, ligand_atoms, radius=self.paths.pocket_radius)
            _write_pocket_pdb(self.paths.pocket_pdb, pocket_coords, pocket_elems)
            pocket_coords, pocket_elems = _read_pdb_atoms(self.paths.pocket_pdb)

    def sample_num_nodes(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample ligand node counts from empirical distribution (single pocket)."""
        num = max(int(batch_size), 1)
        if self.size_counts_.numel() == 0 or self.size_counts_.sum() <= 0:
            idx = np.random.randint(0, len(self), size=num)
            sizes: List[int] = []
            for i in idx:
                data_i = super().__getitem__(int(i))
                if hasattr(data_i, "num_ligand_nodes"):
                    sizes.append(int(data_i.num_ligand_nodes))
                elif hasattr(data_i, "mask_ligand"):
                    sizes.append(int(data_i.mask_ligand.sum().item()))
                else:
                    sizes.append(int(data_i.num_nodes))
            out = torch.tensor(sizes, dtype=torch.long)
        else:
            probs = (self.size_counts_.double() / self.size_counts_.sum()).float()
            choice = torch.multinomial(probs, num_samples=num, replacement=True)
            out = self.size_bins_[choice]
        if device is not None:
            out = out.to(device)
        return out

    def sample_num_nodes_numpy(self, batch_size: int) -> np.ndarray:
        """Numpy variant mirroring :meth:`sample_num_nodes`."""
        num = max(int(batch_size), 1)
        if self.size_counts_.numel() == 0 or self.size_counts_.sum() <= 0:
            idx = np.random.randint(0, len(self), size=num)
            sizes: List[int] = []
            for i in idx:
                data_i = super().__getitem__(int(i))
                if hasattr(data_i, "num_ligand_nodes"):
                    sizes.append(int(data_i.num_ligand_nodes))
                elif hasattr(data_i, "mask_ligand"):
                    sizes.append(int(data_i.mask_ligand.sum().item()))
                else:
                    sizes.append(int(data_i.num_nodes))
            return np.array(sizes, dtype=np.int64)
        probs = (self.size_counts_.double() / self.size_counts_.sum()).cpu().numpy()
        bins = self.size_bins_.cpu().numpy()
        return np.random.choice(bins, size=num, p=probs)


@dataclass(frozen=True)
class _PocketSpec:
    """Internal container describing one pocket directory."""

    name: str
    root: Path
    radius: float
    context_radius: float
    context_knn: int
    tmp_dir: Optional[Path]
    protein_src: Optional[Path]
    ligand_src: Optional[Path]
    smiles_src: Optional[Path]
    lazy_prepare: bool


class MultiPocketLigandDockingDataset(InMemoryDataset):
    """
    Dataset that enumerates multiple pocket directories and returns the Cartesian
    product of ligands × pockets. Each pocket directory must follow the same
    layout required by :class:`PocketLigandDockingDataset` (i.e. contain
    ``protein.pdb``, ``pocket.pdb`` and ``smiles.csv``). During processing we
    instantiate an individual :class:`PocketLigandDockingDataset` for every
    pocket, force it to prepare/dock if necessary, then merge the resulting
    graphs while keeping a single, unified vocabulary.

    Args:
        target_root: Base directory where the aggregated dataset (processed
            tensors) should be stored. Pocket directories may live inside this
            root or elsewhere on disk.
        pocket_dirs: Iterable of pocket directories (either absolute paths or
            paths relative to ``target_root``). Each directory is treated as one
            pocket entry.
    """

    NO_BOND_IDX = PocketLigandDockingDataset.NO_BOND_IDX

    def __init__(self,
                 target_root: Path | str,
                 pocket_dirs: Sequence[str | Path],
                 split: str = "train",
                 radius: float = 10.0,
                 auto_prepare: bool = True,
                 overwrite_docking: bool = False,
                 exhaustiveness: int = 8,
                 vina_executable: Optional[Path | str] = None,
                 tmp_dir: Optional[Path | str] = None,
                 context_radius: float = 6.0,
                 context_knn: int = 24,
                 rare_atom_threshold: float = 0.01,
                 ligand_vocab_override: Optional[List[str]] = None,
                 transform=None,
                 pre_transform=None):
        if not pocket_dirs:
            raise ValueError("pocket_dirs must contain at least one directory.")
        self.root_dir = Path(target_root).resolve()
        self.split = split.lower()
        if self.split == "val":
            self.split = "valid"
        if self.split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.auto_prepare = auto_prepare
        self.overwrite_docking = overwrite_docking
        self.exhaustiveness = int(exhaustiveness)
        self.vina_exec = vina_executable
        self.default_tmp_dir = Path(tmp_dir) if tmp_dir is not None else None
        self.default_radius = float(radius)
        self.default_context_radius = float(context_radius)
        self.default_context_knn = int(context_knn)
        self.rare_atom_threshold = float(rare_atom_threshold)
        self.pocket_specs = self._normalize_pocket_specs(pocket_dirs)
        self._lazy_prepare_all_pockets()
        # 不再强制压缩 vocab；按各 pocket 自身的完整词表处理
        self.global_ligand_vocab = None
        super().__init__(self.root_dir, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        from graph.schema import assert_graph_schema, assert_no_legacy_fields

        assert_no_legacy_fields(self._data, where="MultiPocketLigandDockingDataset.load_processed(self._data)")
        assert_graph_schema(self._data, where="MultiPocketLigandDockingDataset.load_processed(self._data)", require_edge_attr=True)
        meta_path = Path(self.processed_paths[0] + ".meta")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing: {meta_path}")
        with meta_path.open("r") as handle:
            meta = json.load(handle)
        self.ligand_vocab = meta.get("ligand_vocab", [])
        self.ligand_vocab_index = {sym: i for i, sym in enumerate(self.ligand_vocab)}
        self.residue_vocab = meta.get("residue_vocab", [])
        atom_dim = len(self.ligand_vocab)
        atom_feat_dim = int(meta.get("atom_feat_dim", atom_dim + ATOM_EXTRA_DIM))
        self.atom_dim = atom_dim
        self.atom_extra_dim = int(meta.get("atom_extra_dim", max(atom_feat_dim - atom_dim, 0)))
        self.atom_feat_dim = atom_feat_dim
        self.residue_dim = len(self.residue_vocab)
        extra_names = [f"FEAT_{i}" for i in range(max(atom_feat_dim - atom_dim, 0))]
        self.atom_vocab_ = meta.get(
            "atom_vocab",
            self.ligand_vocab + extra_names + [f"RES_{r}" for r in self.residue_vocab],
        )
        self.num_node_classes_ = atom_feat_dim + len(self.residue_vocab)
        EDGE_DECODER_BASE = [
            None,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        EDGE_CLASS_NAMES_BASE = ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
        self.edge_decoder_ = list(EDGE_DECODER_BASE)
        self.num_edge_classes_ = int(meta.get("edge_attr_dim", BOND_TYPE_CLASSES + BOND_EXTRA_DIM))
        self.bond_extra_dim = int(meta.get("bond_extra_dim", max(self.num_edge_classes_ - BOND_TYPE_CLASSES, 0)))
        self.edge_classes_ = list(EDGE_CLASS_NAMES_BASE)
        periodic_table = Chem.GetPeriodicTable()
        mass_list: List[float] = []
        for sym in self.ligand_vocab:
            try:
                mass_list.append(float(periodic_table.GetAtomicWeight(periodic_table.GetAtomicNumber(sym))))
            except Exception:
                mass_list.append(0.0)
        mass_list.extend([0.0 for _ in range(max(atom_feat_dim - atom_dim, 0))])
        mass_list.extend([0.0 for _ in self.residue_vocab])
        self.atom_masses_ = torch.tensor(mass_list, dtype=torch.float32)
        self.pocket_names = meta.get("pocket_names", [])
        self.smiles_list_ = [getattr(self.get(i), "smiles", "") for i in range(len(self))]
        self.reference_smiles_is_canonical_ = False
        self.num_graphs_ = len(self)
        x_counts = torch.zeros(self.num_node_classes_, dtype=torch.float64)
        e_counts = torch.zeros(self.num_edge_classes_, dtype=torch.float64)
        ligand_sizes: List[int] = []
        for idx in range(len(self)):
            data_i = super().__getitem__(idx)
            x_counts += torch.bincount(data_i.x_idx, minlength=self.num_node_classes_).double()
            if data_i.edge_type.numel() > 0:
                e_counts += torch.bincount(data_i.edge_type, minlength=self.num_edge_classes_).double()
            if hasattr(data_i, "num_ligand_nodes"):
                ligand_sizes.append(int(data_i.num_ligand_nodes))
            elif hasattr(data_i, "mask_ligand"):
                ligand_sizes.append(int(data_i.mask_ligand.sum().item()))
            else:
                ligand_sizes.append(int(data_i.num_nodes))
        self.x_hist_total_ = x_counts.long()
        self.e_hist_total_ = e_counts.long()
        self.x_prob_ = (x_counts / x_counts.sum().clamp_min(1e-12)).float()
        self.e_prob_ = (e_counts / e_counts.sum().clamp_min(1e-12)).float()
        if ligand_sizes:
            unique_sizes, counts = torch.unique(torch.tensor(ligand_sizes, dtype=torch.long), return_counts=True)
        else:
            unique_sizes = torch.zeros(0, dtype=torch.long)
            counts = torch.zeros(0, dtype=torch.long)
        self.size_bins_ = unique_sizes
        self.size_counts_ = counts

    def _normalize_pocket_specs(self, pocket_dirs: Sequence[str | Path | dict]) -> List[_PocketSpec]:
        specs: List[_PocketSpec] = []
        for entry in pocket_dirs:
            tmp_dir_override: Optional[Path] = self.default_tmp_dir
            protein_src: Optional[Path] = None
            ligand_src: Optional[Path] = None
            smiles_src: Optional[Path] = None
            lazy_prepare = False

            if isinstance(entry, dict):
                raw_path = entry.get("root") or entry.get("path") or entry.get("dir")
                if raw_path is None:
                    raise ValueError("Pocket config dict must contain 'root'/'path'.")
                path = Path(raw_path)
                name = entry.get("name") or path.name
                radius = float(entry.get("radius", self.default_radius))
                context_radius = float(entry.get("context_radius", self.default_context_radius))
                context_knn = int(entry.get("context_knn", self.default_context_knn))
                if entry.get("tmp_dir") is not None:
                    tmp_dir_override = Path(entry["tmp_dir"])

                def _resolve_source(value: Optional[str | Path]) -> Optional[Path]:
                    if value is None:
                        return None
                    candidate = Path(value)
                    if candidate.is_absolute():
                        return candidate
                    first = (self.root_dir / candidate).resolve()
                    if first.exists():
                        return first
                    return (Path.cwd() / candidate).resolve()

                protein_src = _resolve_source(entry.get("protein") or entry.get("protein_path"))
                ligand_src = _resolve_source(entry.get("ligand") or entry.get("ligand_path"))
                smiles_src = _resolve_source(entry.get("smiles") or entry.get("smiles_csv"))
                lazy_prepare = bool(entry.get("lazy_prepare", False)) or any(
                    src is not None for src in (protein_src, ligand_src, smiles_src)
                )
            else:
                path = Path(entry)
                name = path.name
                radius = self.default_radius
                context_radius = self.default_context_radius
                context_knn = self.default_context_knn
            if not path.is_absolute():
                path = (self.root_dir / path).resolve()
            if not path.exists() and not (isinstance(entry, dict) and lazy_prepare):
                raise FileNotFoundError(f"Pocket directory not found: {path}")
            path.mkdir(parents=True, exist_ok=True)
            if tmp_dir_override is not None and not tmp_dir_override.is_absolute():
                tmp_dir = path / tmp_dir_override
            else:
                tmp_dir = tmp_dir_override
            specs.append(
                _PocketSpec(
                    name=name,
                    root=path,
                    radius=radius,
                    context_radius=context_radius,
                    context_knn=context_knn,
                    tmp_dir=tmp_dir,
                    protein_src=protein_src,
                    ligand_src=ligand_src,
                    smiles_src=smiles_src,
                    lazy_prepare=lazy_prepare,
                )
            )
        return specs

    def _lazy_prepare_all_pockets(self) -> None:
        for spec in self.pocket_specs:
            if spec.lazy_prepare:
                log.info("Lazy prepare pocket '%s' at %s", spec.name, spec.root)
                self._lazy_prepare_single_pocket(spec)

    def _build_global_ligand_vocab(self) -> List[str]:
        log.info("Global ligand vocabulary override disabled; using per-pocket vocab.")
        return []

    def _lazy_prepare_single_pocket(self, spec: _PocketSpec) -> None:
        root = spec.root
        root.mkdir(parents=True, exist_ok=True)
        (root / "docking").mkdir(exist_ok=True)

        protein_path = root / "protein.pdb"
        ligand_pdb_path = root / "reference_ligand.pdb"
        ligand_sdf_path = root / "reference_ligand.sdf"
        pocket_path = root / "pocket.pdb"
        smiles_path = root / "smiles.csv"

        def _ensure_copy(src: Optional[Path], dst: Path, desc: str) -> None:
            if dst.exists():
                log.debug("%s already exists for pocket '%s'", dst.name, spec.name)
                return
            if src is None:
                raise FileNotFoundError(f"{desc} missing for pocket '{spec.name}' and no source provided.")
            if not src.exists():
                raise FileNotFoundError(f"Source file for {desc} not found: {src}")
            log.info("Copying %s -> %s", src, dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        _ensure_copy(spec.protein_src, protein_path, "protein.pdb")
        _ensure_copy(spec.ligand_src, ligand_pdb_path, "reference_ligand.pdb")
        _ensure_copy(spec.smiles_src, smiles_path, "smiles.csv")

        if not ligand_sdf_path.exists():
            log.info("Generating reference_ligand.sdf for pocket '%s'", spec.name)
            success = False
            try:
                mol = Chem.MolFromPDBFile(str(ligand_pdb_path), removeHs=False, sanitize=False)
                if mol is None:
                    mol = Chem.MolFromPDBFile(str(ligand_pdb_path), removeHs=False, sanitize=True)
                if mol is not None:
                    Chem.MolToMolFile(mol, str(ligand_sdf_path))
                    success = True
            except Exception:
                log.exception("RDKit failed to export ligand SDF for pocket '%s'", spec.name)
                success = False
            if not success:
                ligand_atoms = _read_pdb_atoms(ligand_pdb_path)
                _write_atomarray_to_sdf(ligand_atoms, ligand_sdf_path, title=f"{spec.name}_ligand")
                log.warning("Fell back to coordinate-only SDF for pocket '%s'", spec.name)

        if not pocket_path.exists():
            log.info("Generating pocket.pdb for pocket '%s' with radius %.1f", spec.name, spec.radius)
            protein_atoms_full = _parse_protein_atoms(protein_path)
            ligand_atoms = _read_pdb_atoms(ligand_pdb_path)
            pocket_atoms = _extract_pocket_atoms(protein_atoms_full, ligand_atoms, radius=spec.radius)
            if not pocket_atoms:
                raise ValueError(f"Pocket extraction produced empty atom set for {spec.name}. Increase radius.")
            _write_pocket_pdb_from_atoms(pocket_atoms, pocket_path)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"multi_pocket_dataset_{self.split}.pt"]

    def _process(self) -> None:
        if all(Path(path).exists() for path in self.processed_paths):
            return
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.process()

    def process(self) -> None:  # noqa: D401
        data_list: List[Data] = []
        smiles_out: List[str] = []
        ligand_vocab: Optional[List[str]] = None
        residue_union: List[str] = []
        residue_map: Dict[str, int] = {}
        pocket_entries: List[dict] = []
        pocket_templates: List[Data] = []

        pocket_datasets: List[PocketLigandDockingDataset] = []
        total_specs = len(self.pocket_specs)
        for idx, spec in enumerate(self.pocket_specs, start=1):
            log.info("[MultiPocket] Prepare pocket %s (%d/%d)", spec.name, idx, total_specs)
            dataset = PocketLigandDockingDataset(
                target_root=spec.root,
                split=self.split,
                radius=spec.radius,
                auto_prepare=self.auto_prepare,
                overwrite_docking=self.overwrite_docking,
                exhaustiveness=self.exhaustiveness,
                vina_executable=self.vina_exec,
                tmp_dir=spec.tmp_dir,
                context_radius=spec.context_radius,
                context_knn=spec.context_knn,
                rare_atom_threshold=self.rare_atom_threshold,
                ligand_vocab_override=self.global_ligand_vocab,
            )
            pocket_datasets.append(dataset)
            pocket_entries.append({
                "name": spec.name,
                "root": str(spec.root),
                "residue_vocab": list(dataset.residue_vocab),
                "context_radius": dataset.context_radius,
                "context_knn": dataset.context_knn,
            })
            try:
                tpl = dataset.sample_pocket()
                tpl.pocket_index = torch.tensor([len(pocket_templates)], dtype=torch.long)
                tpl.pocket_name = spec.name
                pocket_templates.append(tpl.cpu())
            except Exception as exc:
                log.warning("Failed to load pocket template for %s: %s", spec.name, exc)
            if ligand_vocab is None:
                ligand_vocab = list(dataset.ligand_vocab)
            elif ligand_vocab != list(dataset.ligand_vocab):
                raise ValueError("All pockets must share the same ligand vocabulary.")
            for res in dataset.residue_vocab:
                key = f"{spec.name}::{res}"
                if key not in residue_map:
                    residue_map[key] = len(residue_union)
                    residue_union.append(key)

        if ligand_vocab is None:
            raise RuntimeError("Failed to collect ligand vocabulary from pockets.")

        feat_dim = len(ligand_vocab) + len(residue_union)
        for pocket_idx, (spec, dataset, meta) in enumerate(zip(self.pocket_specs, pocket_datasets, pocket_entries)):
            for idx in tqdm(range(len(dataset)), desc=f"Merge pocket {spec.name}", leave=False):
                data = dataset[idx]
                remapped = self._remap_features(
                    data,
                    ligand_vocab,
                    meta["residue_vocab"],
                    residue_map,
                    spec.name,
                    feat_dim,
                )
                remapped.pocket_index = torch.tensor([pocket_idx], dtype=torch.long)
                remapped.pocket_name = spec.name
                data_list.append(remapped)
                smiles_out.append(getattr(data, "smiles", ""))

        if not data_list:
            empty = Data(
                x=torch.zeros(0, feat_dim),
                pos=torch.zeros(0, 3),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                edge_attr=torch.zeros(0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM),
                mask_ligand=torch.zeros(0, dtype=torch.bool),
                mask_ligand_edge=torch.zeros(0, dtype=torch.bool),
                x_idx=torch.zeros(0, dtype=torch.long),
            )
            data_list.append(empty)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        meta = {
            "ligand_vocab": ligand_vocab,
            "residue_vocab": residue_union,
            "atom_vocab": ligand_vocab + residue_union,
            "pocket_names": [spec.name for spec in self.pocket_specs],
        }
        meta_path = Path(self.processed_paths[0] + ".meta")
        with meta_path.open("w") as handle:
            json.dump(meta, handle, indent=2)
        torch.save(pocket_templates, self.processed_paths[0] + ".pockets.pt")

    @staticmethod
    def _remap_features(data: Data,
                        ligand_vocab: List[str],
                        residue_vocab: List[str],
                        residue_map: Dict[str, int],
                        pocket_name: str,
                        feat_dim: int) -> Data:
        atom_dim = len(ligand_vocab)
        res_dim = len(residue_vocab)
        atom_feat_dim = atom_dim + ATOM_EXTRA_DIM
        if data.x.size(1) != atom_feat_dim + res_dim:
            raise ValueError("Feature dimension mismatch when remapping multi-pocket dataset.")
        new_x = torch.zeros(data.x.size(0), feat_dim, dtype=data.x.dtype)
        new_x[:, :atom_feat_dim] = data.x[:, :atom_feat_dim]
        residue_slice = data.x[:, atom_feat_dim:]
        for idx, res_name in enumerate(residue_vocab):
            column = residue_slice[:, idx]
            if not torch.any(column):
                continue
            key = f"{pocket_name}::{res_name}"
            global_idx = atom_feat_dim + residue_map[key]
            new_x[:, global_idx] = column
        data.x = new_x
        data.x_idx = torch.argmax(new_x, dim=-1)
        return data

    def _build_ligand_example(
        self,
        row,
        lig_mol: Chem.Mol,
        pocket_x: torch.Tensor,
        pocket_pos: torch.Tensor,
        symbol2idx: dict,
        feat_dim: int,
        atom_dim: int,
        pose_path: Path,
    ) -> Data:
        lig_conf = lig_mol.GetConformer()
        lig_coords = np.asarray(lig_conf.GetPositions(), dtype=np.float32)
        lig_pos = torch.tensor(lig_coords, dtype=torch.float32)
        lig_atoms = [atom.GetSymbol() for atom in lig_mol.GetAtoms()]
        atom_feat_dim = atom_dim + ATOM_EXTRA_DIM
        lig_feat = torch.zeros(len(lig_atoms), feat_dim, dtype=torch.float32)
        atom_indices = torch.tensor([symbol2idx[sym] for sym in lig_atoms], dtype=torch.long)
        lig_feat[torch.arange(len(lig_atoms)), atom_indices] = 1.0
        extra_dim = max(int(atom_feat_dim - atom_dim), 0)
        if extra_dim > 0:
            extra_vals = torch.zeros((len(lig_atoms), extra_dim), dtype=torch.float32)
            for idx, atom in enumerate(lig_mol.GetAtoms()):
                vec = _adjust_feature_vec(_atom_feature_vec(atom), extra_dim)
                if vec:
                    extra_vals[idx] = torch.tensor(vec, dtype=torch.float32)
            lig_feat[:, atom_dim:atom_feat_dim] = extra_vals

        # Combine pocket and ligand into single Data object
        num_pocket = pocket_x.size(0)
        num_ligand = lig_feat.size(0)
        total_nodes = num_pocket + num_ligand
        global_offset = num_pocket

        x = torch.cat([pocket_x, lig_feat], dim=0)
        pos = torch.cat([pocket_pos, lig_pos], dim=0)
        residue_dim = max(int(feat_dim - atom_feat_dim), 0)
        if residue_dim > 0:
            pocket_x_idx = pocket_x[:, atom_feat_dim:].argmax(dim=-1) + atom_feat_dim
        else:
            pocket_x_idx = torch.zeros(num_pocket, dtype=torch.long)
        x_idx = torch.cat([pocket_x_idx, atom_indices], dim=0)

        mask_context_node = torch.zeros(total_nodes, dtype=torch.bool)
        mask_context_node[:num_pocket] = True
        mask_ligand = ~mask_context_node

        # --- Ligand bonds (directed edges) ---
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_type: List[int] = []
        bond_type_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }
        no_bond_idx = 0
        edge_feat: List[List[float]] = []

        for bond in lig_mol.GetBonds():
            i = global_offset + bond.GetBeginAtomIdx()
            j = global_offset + bond.GetEndAtomIdx()
            btype_raw = bond.GetBondType()
            if btype_raw not in bond_type_map:
                log.warning("[pocket_dataset] drop molecule containing unsupported bond type: %s", str(btype_raw))
                return None
            btype = bond_type_map[btype_raw]

            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_type.extend([btype, btype])
            bfeat = _bond_feature_vec(bond)
            edge_feat.extend([bfeat, bfeat])

        # --- Context edges: ligand <-> pocket ---
        if num_pocket > 0 and num_ligand > 0:
            dist_lp = torch.cdist(lig_pos, pocket_pos)
            lp_pairs = (dist_lp <= self.context_radius).nonzero(as_tuple=False)
            for lig_idx, poc_idx in lp_pairs.tolist():
                g_lig = global_offset + lig_idx
                g_poc = poc_idx
                # ligand -> pocket
                edge_src.append(g_lig)
                edge_dst.append(g_poc)
                edge_type.append(no_bond_idx)
                edge_feat.append([0.0] * BOND_EXTRA_DIM)
                # pocket -> ligand
                edge_src.append(g_poc)
                edge_dst.append(g_lig)
                edge_type.append(no_bond_idx)
                edge_feat.append([0.0] * BOND_EXTRA_DIM)

        # --- Optional pocket-pockey edges (KNN) ---
        if self.context_knn > 0 and num_pocket > 1:
            dist_pp = torch.cdist(pocket_pos, pocket_pos)
            dist_pp.fill_diagonal_(float("inf"))
            k = min(self.context_knn, num_pocket - 1)
            knn_idx = torch.topk(dist_pp, k, largest=False).indices  # (num_pocket, k)
            for i in range(num_pocket):
                for j in knn_idx[i].tolist():
                    if not np.isfinite(dist_pp[i, j].item()):
                        continue
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_type.append(no_bond_idx)
                    edge_feat.append([0.0] * BOND_EXTRA_DIM)

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type_tensor = torch.empty((0,), dtype=torch.long)

        if edge_type_tensor.numel() > 0:
            edge_onehot = F.one_hot(edge_type_tensor.clamp(min=0), num_classes=BOND_TYPE_CLASSES).float()
            if edge_feat:
                extra = torch.tensor([_adjust_feature_vec(f, BOND_EXTRA_DIM) for f in edge_feat], dtype=edge_onehot.dtype)
            else:
                extra = torch.zeros((edge_onehot.size(0), BOND_EXTRA_DIM), dtype=edge_onehot.dtype)
            edge_attr = torch.cat([edge_onehot, extra], dim=-1)
        else:
            edge_attr = torch.zeros((0, BOND_TYPE_CLASSES + BOND_EXTRA_DIM), dtype=torch.float32)

        from graph.schema import compute_mask_ligand_edge

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            edge_attr=edge_attr,
            mask_ligand=mask_ligand,
            mask_ligand_edge=compute_mask_ligand_edge(edge_index, mask_ligand),
            x_idx=x_idx,
        )
        data.no_bond_idx = no_bond_idx
        data.smiles = getattr(row, "smiles")  # type: ignore[attr-defined]
        data.ligand_id = getattr(row, "ligand_id")  # type: ignore[attr-defined]
        dock_score_val = float(getattr(row, "dock_score", float("nan")))  # type: ignore[attr-defined]
        data.vina_score = dock_score_val
        data.dock_score = dock_score_val
        data.pose_path = str(pose_path.relative_to(self.paths.root))  # type: ignore[attr-defined]
        data.num_pocket_nodes = num_pocket
        data.num_ligand_nodes = num_ligand
        return data

    def no_bond_index(self) -> int:
        return self.NO_BOND_IDX

    def reference_smiles_for_metrics(self) -> Tuple[List[str], bool]:
        smiles: List[str] = []
        for data in self:
            smi = getattr(data, "smiles", "")
            if smi:
                smiles.append(smi)
        return smiles, False

    def sample_num_nodes(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample ligand node counts from the merged empirical distribution."""
        num = max(int(batch_size), 1)
        if self.size_counts_.numel() == 0 or self.size_counts_.sum() <= 0:
            idx = np.random.randint(0, len(self), size=num)
            sizes = []
            for i in idx:
                data_i = super().__getitem__(int(i))
                if hasattr(data_i, "num_ligand_nodes"):
                    sizes.append(int(data_i.num_ligand_nodes))
                elif hasattr(data_i, "mask_ligand"):
                    sizes.append(int(data_i.mask_ligand.sum().item()))
                else:
                    sizes.append(int(data_i.num_nodes))
            out = torch.tensor(sizes, dtype=torch.long)
        else:
            probs = (self.size_counts_.double() / self.size_counts_.sum()).float()
            choice = torch.multinomial(probs, num_samples=num, replacement=True)
            out = self.size_bins_[choice]
        if device is not None:
            out = out.to(device)
        return out

    @property
    def num_node_classes(self) -> int:
        return self.num_node_classes_

    @property
    def num_edge_classes(self) -> int:
        if hasattr(self, "edge_decoder_"):
            return int(len(self.edge_decoder_))
        return self.num_edge_classes_

    @property
    def protein_atomic_numbers(self) -> torch.Tensor:
        return self.protein_atomic_numbers_

    @property
    def max_num_aa(self) -> int:
        return int(self.max_num_aa_)

    @property
    def atom_vocab(self) -> List[str]:
        return list(getattr(self, "atom_vocab_", []))

    @property
    def edge_classes(self) -> List[str]:
        return list(getattr(self, "edge_classes_", [])) if hasattr(self, "edge_classes_") else ["NO_BOND", "SINGLE", "DOUBLE"]

    @property
    def atom_masses(self) -> torch.Tensor:
        return self.atom_masses_

    @property
    def x_prob(self) -> torch.Tensor:
        return self.x_prob_

    @property
    def e_prob(self) -> torch.Tensor:
        return self.e_prob_

    @property
    def size_hist_bins(self) -> torch.Tensor:
        return self.size_bins_

    @property
    def size_hist_counts(self) -> torch.Tensor:
        return self.size_counts_

    def sample_num_nodes(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample ligand node counts from empirical distribution."""
        num = max(int(batch_size), 1)
        if self.size_counts_.numel() == 0 or self.size_counts_.sum() <= 0:
            idx = np.random.randint(0, len(self), size=num)
            sizes = []
            for i in idx:
                data_i = super().__getitem__(int(i))
                if hasattr(data_i, "num_ligand_nodes"):
                    sizes.append(int(data_i.num_ligand_nodes))
                elif hasattr(data_i, "mask_ligand"):
                    sizes.append(int(data_i.mask_ligand.sum().item()))
                else:
                    sizes.append(int(data_i.num_nodes))
            out = torch.tensor(sizes, dtype=torch.long)
        else:
            probs = (self.size_counts_.double() / self.size_counts_.sum()).float()
            choice = torch.multinomial(probs, num_samples=num, replacement=True)
            out = self.size_bins_[choice]
        if device is not None:
            out = out.to(device)
        return out

    def sample_num_nodes_numpy(self, batch_size: int) -> np.ndarray:
        """Numpy variant mirroring :meth:`sample_num_nodes`."""
        num = max(int(batch_size), 1)
        if self.size_counts_.numel() == 0 or self.size_counts_.sum() <= 0:
            idx = np.random.randint(0, len(self), size=num)
            sizes = []
            for i in idx:
                data_i = super().__getitem__(int(i))
                if hasattr(data_i, "num_ligand_nodes"):
                    sizes.append(int(data_i.num_ligand_nodes))
                elif hasattr(data_i, "mask_ligand"):
                    sizes.append(int(data_i.mask_ligand.sum().item()))
                else:
                    sizes.append(int(data_i.num_nodes))
            return np.array(sizes, dtype=np.int64)
        probs = (self.size_counts_.double() / self.size_counts_.sum()).cpu().numpy()
        bins = self.size_bins_.cpu().numpy()
        return np.random.choice(bins, size=num, p=probs)

    def _load_pocket_templates(self) -> List[Data]:
        path = Path(self.processed_paths[0] + ".pockets.pt")
        if path.exists():
            try:
                templates = torch.load(path, weights_only=False)
                if isinstance(templates, list):
                    return templates
            except Exception:
                log.warning("Failed to load pocket templates from %s", path)
        # fallback: rebuild from pocket specs using pocket.pdb files
        templates: List[Data] = []
        atom_dim = len(self.ligand_vocab)
        atom_feat_dim = atom_dim + ATOM_EXTRA_DIM
        feat_dim = atom_feat_dim + len(self.residue_vocab)
        for idx, spec in enumerate(self.pocket_specs):
            try:
                pocket_path = Path(spec.root) / "pocket.pdb"
                pocket_res_coords, pocket_res_names = _read_pocket_residues(pocket_path)
                pocket_x = torch.zeros(len(pocket_res_names), feat_dim, dtype=torch.float32)
                for ridx, res in enumerate(pocket_res_names):
                    key = f"{spec.name}::{res.upper()}"
                    if key not in self.residue_vocab:
                        continue
                    res_id = self.residue_vocab.index(key)
                    pocket_x[ridx, atom_feat_dim + res_id] = 1.0
                pocket_pos = torch.tensor(pocket_res_coords, dtype=torch.float32)
                tpl = PocketLigandDockingDataset._make_pocket_template(
                    pocket_x=pocket_x,
                    pocket_pos=pocket_pos,
                    pocket_name=spec.name,
                    pocket_index=idx,
                )
                templates.append(tpl)
            except Exception as exc:
                log.warning("Failed to rebuild pocket template for %s: %s", spec.name, exc)
        return templates

    def _derive_templates_from_data(self) -> List[Data]:
        if len(self) == 0:
            return []
        templates: List[Data] = []
        seen: Set[int] = set()
        for idx in range(len(self)):
            data = super().__getitem__(idx)
            pocket_idx = int(getattr(data, "pocket_index", -1))
            if pocket_idx in seen:
                continue
            seen.add(pocket_idx)
            tpl = PocketLigandDockingDataset._extract_pocket_template(
                data,
                pocket_name=getattr(data, "pocket_name", f"pocket_{pocket_idx}"),
            )
            if tpl is not None:
                tpl.pocket_index = torch.tensor([pocket_idx], dtype=torch.long)
                templates.append(tpl)
            if len(seen) >= len(self.pocket_specs):
                break
        return templates

    def sample_pocket(self, device: torch.device | str | None = None) -> Data:
        if not self.pocket_templates_:
            self.pocket_templates_ = self._load_pocket_templates()
        if not self.pocket_templates_:
            self.pocket_templates_ = self._derive_templates_from_data()
        if not self.pocket_templates_:
            raise RuntimeError("Pocket templates unavailable; dataset may be empty or unprocessed.")
        choice = int(torch.randint(0, len(self.pocket_templates_), (1,)).item())
        tpl = self.pocket_templates_[choice].clone()
        if device is not None:
            tpl = tpl.to(device)
        if getattr(tpl, "batch", None) is None or tpl.batch.numel() == 0:
            tpl.batch = torch.zeros(tpl.num_nodes, dtype=torch.long, device=tpl.pos.device)
        if not hasattr(tpl, "sample_num_nodes"):
            tpl.sample_num_nodes = lambda batch_size, device=None: self.sample_num_nodes(batch_size, device)  # type: ignore[attr-defined]
        return tpl

    def no_bond_index(self) -> int:
        return int(self.NO_BOND_IDX)

    def reference_smiles_for_metrics(self) -> Tuple[List[str], bool]:
        smiles: List[str] = []
        for data in self:
            smi = getattr(data, "smiles", "")
            if smi:
                smiles.append(smi)
        return smiles, False

    @property
    def num_node_classes(self) -> int:
        return int(self.num_node_classes_)

    @property
    def num_edge_classes(self) -> int:
        return int(self.num_edge_classes_)

    @property
    def atom_vocab(self) -> List[str]:
        return list(getattr(self, "atom_vocab_", []))

    @property
    def edge_classes(self) -> List[str]:
        return list(getattr(self, "edge_classes_", [])) if hasattr(self, "edge_classes_") else ["NO_BOND", "SINGLE", "DOUBLE"]

    @property
    def atom_masses(self) -> torch.Tensor:
        return self.atom_masses_

    @property
    def x_prob(self) -> torch.Tensor:
        return self.x_prob_

    @property
    def e_prob(self) -> torch.Tensor:
        return self.e_prob_
