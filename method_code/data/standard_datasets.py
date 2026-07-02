from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from tdc.generation import MolGen  # type: ignore
    _TDC_AVAILABLE = True
except Exception as e:
    _TDC_AVAILABLE = False

from rdkit import Chem
from torch.utils.data import DataLoader, Dataset


def _read_smiles_file(path: Path) -> List[str]:
    smiles = []
    if not path.exists():
        return smiles
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token:
                smiles.append(token)
    return smiles


def _write_csv(rows: Iterable[Tuple[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "split"])
        for smi, split in rows:
            writer.writerow([smi, split])


def _collate_identity(batch):
    return batch


class _TDCMosesStreamDataset(Dataset):
    """
    Map-style dataset over TDC MOSES splits, processed in DataLoader workers.
    """

    def __init__(self, split_to_smiles: dict, canonicalize: bool):
        self.canonicalize = bool(canonicalize)
        self._parts: List[Tuple[str, List[str], int, int]] = []
        start = 0
        for split_name in ("train", "valid", "test"):
            smiles_list = split_to_smiles.get(split_name, [])
            if not smiles_list:
                continue
            end = start + len(smiles_list)
            self._parts.append((split_name, smiles_list, start, end))
            start = end
        self._length = start

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Tuple[str, str]:
        idx = int(index)
        for split_name, smiles_list, start, end in self._parts:
            if start <= idx < end:
                raw = smiles_list[idx - start]
                smi = str(raw).strip() if raw is not None else ""
                if not smi:
                    return "", split_name
                if self.canonicalize:
                    cano = _canonical_smiles(smi)
                    return (cano or ""), split_name
                return smi, split_name
        raise IndexError(f"index out of range: {index}")


def _canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True)



def _fetch_moses_from_tdc(
    csv_out: Path,
    max_rows: Optional[int] = None,
    canonicalize: bool = False,
    log_every: int = 100000,
    stream_num_workers: int = 4,
    stream_batch_size: int = 4096,
) -> str:
    if not _TDC_AVAILABLE:
        raise RuntimeError("PyTDC is required to fetch MOSES dataset automatically.")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    print("[moses_tdc] loading MOSES from TDC ...", flush=True)
    dataset = MolGen(name="MOSES")
    splits = dataset.get_split()
    split_sizes = {
        k: int(len(v["smiles"])) if isinstance(v, dict) and "smiles" in v else int(len(v))
        for k, v in splits.items()
    }
    print(f"[moses_tdc] split sizes: {split_sizes}", flush=True)
    split_to_smiles = {
        str(k).lower(): (v["smiles"] if isinstance(v, dict) and "smiles" in v else v)
        for k, v in splits.items()
    }
    stream_dataset = _TDCMosesStreamDataset(
        split_to_smiles=split_to_smiles, canonicalize=canonicalize
    )
    if len(stream_dataset) == 0:
        raise RuntimeError("TDC MOSES split is empty.")
    stream_loader = DataLoader(
        stream_dataset,
        batch_size=int(stream_batch_size),
        shuffle=False,
        num_workers=int(stream_num_workers),
        pin_memory=False,
        collate_fn=_collate_identity,
        persistent_workers=bool(int(stream_num_workers) > 0),
    )
    total_seen = 0
    total_kept = 0
    limit = None if (max_rows is None or int(max_rows) <= 0) else int(max_rows)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "split"])
        for batch in stream_loader:
            for smi_clean, split in batch:
                total_seen += 1
                if not smi_clean:
                    continue
                writer.writerow([smi_clean, split])
                total_kept += 1
                if total_kept % int(log_every) == 0:
                    print(
                        f"[moses_tdc] kept={total_kept} seen={total_seen} split={split}",
                        flush=True,
                    )
                if limit is not None and total_kept >= limit:
                    break
            if limit is not None and total_kept >= limit:
                break
    if total_kept == 0:
        raise RuntimeError("Failed to fetch any valid SMILES from TDC MOSES dataset.")
    print(
        f"[moses_tdc] wrote {total_kept} rows to {csv_out} "
        f"(canonicalize={canonicalize}, limit={limit}, workers={stream_num_workers}, batch_size={stream_batch_size})",
        flush=True,
    )
    return str(csv_out)


def prepare_guacamol_csv(root: str,
                         csv_out: str,
                         seed: int = 42,
                         ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    """
    Build a CSV with columns [smiles, split] from the GuacaMol dataset.

    Expects files named like `guacamol_v1_train.smiles`, `guacamol_v1_valid.smiles`,
    `guacamol_v1_test.smiles` (case-insensitive). If validation/test files are missing,
    they are sampled from the training set according to `ratios`.
    """
    csv_path = Path(csv_out)
    if csv_path.exists():
        return str(csv_path)

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"GuacaMol root directory not found: {root}")

    def find_file(candidates: List[str]) -> Optional[Path]:
        for name in candidates:
            cand = root_path / name
            if cand.exists():
                return cand
        return None

    train_file = find_file([
        "guacamol_v1_train.smiles", "train.smiles", "train.smi"
    ])
    if train_file is None:
        raise FileNotFoundError(
            f"GuacaMol train file not found in {root}. "
            "Expected e.g. guacamol_v1_train.smiles"
        )
    valid_file = find_file([
        "guacamol_v1_valid.smiles", "valid.smiles", "valid.smi"
    ])
    test_file = find_file([
        "guacamol_v1_test.smiles", "test.smiles", "test.smi"
    ])

    train_smiles = _read_smiles_file(train_file)
    if not train_smiles:
        raise RuntimeError(f"No valid SMILES read from {train_file}")

    splits: List[Tuple[str, str]] = []

    valid_smiles = _read_smiles_file(valid_file) if valid_file is not None else []
    test_smiles = _read_smiles_file(test_file) if test_file is not None else []

    if not valid_smiles or not test_smiles:
        remaining_train = train_smiles.copy()
        rnd = random.Random(seed)
        rnd.shuffle(remaining_train)
        train_ratio, valid_ratio, test_ratio = ratios
        total = len(remaining_train)
        n_train = int(total * train_ratio)
        n_valid = int(total * valid_ratio)
        n_test = total - n_train - n_valid
        train_subset = remaining_train[:n_train]
        valid_subset = remaining_train[n_train:n_train + n_valid]
        test_subset = remaining_train[n_train + n_valid:n_train + n_valid + n_test]
        splits.extend((smi, "train") for smi in train_subset)
        if not valid_smiles:
            splits.extend((smi, "valid") for smi in valid_subset)
        else:
            splits.extend((smi, "valid") for smi in valid_smiles)
        if not test_smiles:
            splits.extend((smi, "test") for smi in test_subset)
        else:
            splits.extend((smi, "test") for smi in test_smiles)
    else:
        splits.extend((smi, "train") for smi in train_smiles)
        splits.extend((smi, "valid") for smi in valid_smiles)
        splits.extend((smi, "test") for smi in test_smiles)

    _write_csv(splits, csv_path)
    return str(csv_path)


def prepare_moses_csv(root: str,
                      csv_out: str,
                      seed: int = 42,
                      max_rows: Optional[int] = None,
                      canonicalize: bool = False,
                      stream_num_workers: int = 4,
                      stream_batch_size: int = 4096) -> str:
    """
    Build a CSV with columns [smiles, split] from the MOSES dataset.

    Expects files such as train.smi/train.csv, valid.smi, test.smi and optionally
    test_scaffolds.smi. Falls back to sampling if some splits are missing.
    """
    csv_path = Path(csv_out)
    if csv_path.exists():
        return str(csv_path)

    if root.lower() == "tdc" or not Path(root).exists():
        return _fetch_moses_from_tdc(
            csv_path,
            max_rows=max_rows,
            canonicalize=canonicalize,
            stream_num_workers=stream_num_workers,
            stream_batch_size=stream_batch_size,
        )

    root_path = Path(root)
    if not root_path.exists():
        # fallback: try TDC if available
        if _TDC_AVAILABLE:
            return _fetch_moses_from_tdc(
                csv_path,
                max_rows=max_rows,
                canonicalize=canonicalize,
                stream_num_workers=stream_num_workers,
                stream_batch_size=stream_batch_size,
            )
        raise FileNotFoundError(f"MOSES root directory not found: {root}")

    def load_any(name_stem: str) -> List[str]:
        candidates = [
            f"{name_stem}.smi", f"{name_stem}.smiles", f"{name_stem}.txt",
            f"{name_stem}.csv"
        ]
        for cand in candidates:
            path = root_path / cand
            if path.exists():
                if path.suffix == ".csv":
                    with path.open("r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header and len(header) >= 1 and header[0].lower() != "smiles":
                            smiles_idx = 0
                        else:
                            smiles_idx = 0
                        return [row[smiles_idx] for row in reader if row]
                return _read_smiles_file(path)
        return []

    train_smiles = load_any("train")
    valid_smiles = load_any("valid")
    test_smiles = load_any("test")

    if not train_smiles:
        raise RuntimeError("MOSES train split not found or empty.")

    splits: List[Tuple[str, str]] = []
    if valid_smiles:
        splits.extend((smi, "valid") for smi in valid_smiles)
    if test_smiles:
        splits.extend((smi, "test") for smi in test_smiles)

    if not valid_smiles or not test_smiles:
        rnd = random.Random(seed)
        shuffled = train_smiles.copy()
        rnd.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * 0.8)
        n_valid = int(n_total * 0.1)
        train_subset = shuffled[:n_train]
        valid_subset = shuffled[n_train:n_train + n_valid]
        test_subset = shuffled[n_train + n_valid:]
        splits.extend((smi, "train") for smi in train_subset)
        if not valid_smiles:
            splits.extend((smi, "valid") for smi in valid_subset)
        if not test_smiles:
            splits.extend((smi, "test") for smi in test_subset)
    else:
            rows = [(smi, "train") for smi in train_smiles]
            rows.extend((smi, "valid") for smi in valid_smiles)
            rows.extend((smi, "test") for smi in test_smiles)
            _write_csv(rows, csv_path)
            return str(csv_path)

    _write_csv(splits, csv_path)
    return str(csv_path)


def load_reference_smiles(dataset: str, root: str) -> List[str]:
    """
    Load reference SMILES for evaluation (e.g., training set) for the given dataset.
    """
    dataset = dataset.lower()
    root_path = Path(root)
    if dataset == "guacamol":
        candidates = [
            root_path / "guacamol_v1_train.smiles",
            root_path / "train.smiles",
            root_path / "train.smi",
        ]
        for cand in candidates:
            if cand.exists():
                return _read_smiles_file(cand)
        return []
    if dataset == "moses":
        for stem in ["train", "train.smiles", "train.smi"]:
            path = root_path / stem
            if path.exists():
                return _read_smiles_file(path)
        # fallback attempt for csv
        path = root_path / "train.csv"
        if path.exists():
            smiles = []
            with path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                smiles_idx = 0 if not header else header.index("SMILES") if "SMILES" in header else 0
                for row in reader:
                    if row:
                        smiles.append(row[smiles_idx])
            return smiles
        return []
    raise ValueError(f"Unsupported dataset for reference loading: {dataset}")
