from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mobo.config_utils import load_config
from mobo.dataset_utils import resplit_smiles_csv
from mobo.io_utils import _pick_smiles_column
from mobo.oracle import run_oracle_docking
from mobo.smiles_utils import canonicalize_smiles_noh, compute_qed, compute_sa
from rdkit import Chem


def _section(cfg: dict, key: str) -> dict:
    val = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    return val if isinstance(val, dict) else {}


def _make_unique_ids(ids: Sequence[str]) -> list[str]:
    seen = {}
    out = []
    for raw in ids:
        base = str(raw)
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}_{seen[base]}")
    return out


def _extract_resname_to_sdf(pdb_path: Path, resname: str, out_sdf: Path) -> bool:
    resname = resname.strip().upper()
    kept = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            name = line[17:20].strip().upper()
            if name != resname:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            kept.append(line.rstrip("\n"))
    if not kept:
        return False
    pdb_block = "\n".join(kept) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    out_sdf.write_text(Chem.MolToMolBlock(mol), encoding="utf-8")
    return True


def _reference_smiles_from_sdf(path: Path) -> str | None:
    if not path.exists():
        return None
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = next(iter(supplier), None)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build oracle dataset from smiles.csv")
    ap.add_argument("--dataset-root", default=None, help="Target dataset root directory")
    ap.add_argument("--smiles", default=None, help="Path to input smiles.csv")
    ap.add_argument("--template-root", default=None, help="(deprecated) no longer used")
    ap.add_argument("--config", default="config/surrogate/config.yaml", help="Config file for docking params")
    ap.add_argument("--id-prefix", default=None, help="Ligand id prefix if ligand_id missing")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing docking results")
    ap.add_argument("--limit", type=int, default=0, help="Dock only first N ligands (0 = all)")
    ap.add_argument("--resplit", action="store_true", help="Force resplit after writing smiles.csv")
    ap.add_argument("--no-resplit", action="store_true", help="Disable resplit even if config enables it")
    ap.add_argument("--ref-resname", default=None, help="Extract reference_ligand.sdf from pocket.pdb by residue name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    general_cfg = _section(cfg, "general")
    oracle_cfg = _section(cfg, "oracle")
    dataset_cfg = _section(cfg, "dataset")
    objective_cfg = _section(cfg, "objective")
    run_cfg = _section(cfg, "run")
    oracle_3d_cfg = _section(oracle_cfg, "oracle_3d")

    dataset_root = Path(args.dataset_root or general_cfg.get("dataset_root", "dataset/6KRO")).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    smiles_src = Path(args.smiles) if args.smiles else (dataset_root / "smiles.csv")
    if not smiles_src.exists():
        print(f"[error] smiles.csv not found: {smiles_src}")
        return 1

    if args.template_root:
        print("[warn] --template-root is deprecated; ignored. Provide protein/pocket files in dataset_root.")

    protein_pdb = dataset_root / "protein.pdb"
    reference_ligand = dataset_root / "reference_ligand.sdf"
    pocket_pdb = dataset_root / "pocket.pdb"
    if not protein_pdb.exists():
        print("[error] protein.pdb is required in dataset_root to run docking.")
        return 1
    if not reference_ligand.exists():
        if args.ref_resname:
            ok = _extract_resname_to_sdf(protein_pdb, args.ref_resname, reference_ligand)
            if ok:
                print(f"[ok] extracted reference_ligand.sdf from {protein_pdb} resname={args.ref_resname}")
            else:
                print(f"[error] failed to extract resname={args.ref_resname} from {protein_pdb}")
                return 1
        else:
            print("[error] reference_ligand.sdf missing; provide it or use --ref-resname with protein.pdb.")
            return 1

    df = pd.read_csv(smiles_src)
    if df.empty:
        print("[error] smiles.csv is empty")
        return 1
    smiles_col = _pick_smiles_column(df.columns)
    if smiles_col is None:
        print("[error] smiles.csv missing SMILES column")
        return 1

    if smiles_col != "smiles":
        df["smiles"] = df[smiles_col].astype(str)

    if "ligand_id" not in df.columns:
        prefix = args.id_prefix or "LIG"
        width = max(5, len(str(len(df))))
        df["ligand_id"] = [f"{prefix}_{i:0{width}d}" for i in range(1, len(df) + 1)]
        print(f"[info] generated ligand_id with prefix={prefix}")
    else:
        df["ligand_id"] = df["ligand_id"].astype(str)

    if df["ligand_id"].duplicated().any():
        before = int(df["ligand_id"].duplicated().sum())
        df["ligand_id"] = _make_unique_ids(df["ligand_id"].tolist())
        print(f"[warn] duplicate ligand_id fixed: {before} duplicates")

    canon = []
    qed_vals = []
    sa_vals = []
    sa_clamp_min = objective_cfg.get("sa_clamp_min", None)
    sa_clamp_max = objective_cfg.get("sa_clamp_max", None)
    for smi in df["smiles"].astype(str).tolist():
        try:
            canon.append(canonicalize_smiles_noh(smi) or "")
        except Exception:
            canon.append("")
        qed_vals.append(float(compute_qed(smi)))
        sa_vals.append(float(compute_sa(smi, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max)))
    df["smiles_canonical"] = canon
    df["qed"] = qed_vals
    df["sa_score"] = sa_vals
    if "is_reference" not in df.columns:
        df["is_reference"] = 0

    ref_smi = _reference_smiles_from_sdf(reference_ligand)
    if ref_smi:
        ref_canon = canonicalize_smiles_noh(ref_smi) or ""
        if ref_canon:
            match = df["smiles_canonical"] == ref_canon
            if match.any():
                df.loc[match, "is_reference"] = 1
                existing_ids = set(df["ligand_id"].astype(str).tolist())
                ref_id = "REF_LIG"
                if ref_id in existing_ids and not df.loc[match, "ligand_id"].eq(ref_id).all():
                    idx = 2
                    while f"{ref_id}_{idx}" in existing_ids:
                        idx += 1
                    ref_id = f"{ref_id}_{idx}"
                df.loc[match, "ligand_id"] = ref_id
                print(f"[info] reference_ligand already in smiles.csv (marked is_reference=1, ligand_id={ref_id})")
            else:
                existing_ids = set(df["ligand_id"].astype(str).tolist())
                ref_id = "REF_LIG"
                if ref_id in existing_ids:
                    idx = 2
                    while f"{ref_id}_{idx}" in existing_ids:
                        idx += 1
                    ref_id = f"{ref_id}_{idx}"
                new_row = {col: np.nan for col in df.columns}
                new_row["ligand_id"] = ref_id
                new_row["smiles"] = ref_smi
                new_row["smiles_canonical"] = ref_canon
                new_row["qed"] = float(compute_qed(ref_smi))
                new_row["sa_score"] = float(compute_sa(ref_smi, clamp_min=sa_clamp_min, clamp_max=sa_clamp_max))
                new_row["is_reference"] = 1
                if "split" in df.columns:
                    new_row["split"] = "train"
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"[info] reference_ligand appended as {ref_id}")

    out_csv = dataset_root / "smiles.csv"
    df.to_csv(out_csv, index=False)
    print(f"[ok] smiles.csv written: {out_csv}")

    resplit_cfg = bool(dataset_cfg.get("resplit", False))
    if args.no_resplit:
        resplit_cfg = False
    if args.resplit:
        resplit_cfg = True
    if resplit_cfg:
        split_ratio = tuple(dataset_cfg.get("split_ratio", (0.9, 0.1, 0.0)))
        split_seed = int(dataset_cfg.get("split_seed", 42))
        resplit_smiles_csv(str(out_csv), split_ratio=split_ratio, seed=split_seed)
        print(f"[split] resplit={split_ratio} seed={split_seed}")

    vina_exec = run_cfg.get("vina_executable", None)
    print(
        "[oracle_config] "
        f"backend={oracle_cfg.get('docking_backend', 'vina')} "
        f"exhaustiveness={oracle_cfg.get('oracle_exhaustiveness', 8)} "
        f"pocket_radius={oracle_cfg.get('oracle_pocket_radius', 10.0)} "
        f"overwrite={bool(args.overwrite or oracle_cfg.get('oracle_overwrite', False))} "
        f"vina={vina_exec if vina_exec else 'vina'} "
        f"unidock_service_url={oracle_cfg.get('unidock_service_url', None)}"
    )
    print(
        "[oracle_3d] "
        f"num_confs={oracle_3d_cfg.get('num_confs', 8)} "
        f"max_attempts={oracle_3d_cfg.get('max_attempts', 3)} "
        f"max_opt_iters={oracle_3d_cfg.get('max_opt_iters', 200)} "
        f"optimize={bool(oracle_3d_cfg.get('optimize', True))} "
        f"prefer_mmff={bool(oracle_3d_cfg.get('prefer_mmff', False))}"
    )

    ligand_ids = df["ligand_id"].astype(str).tolist()
    if args.limit and args.limit > 0:
        ligand_ids = ligand_ids[: args.limit]
        print(f"[info] docking limit={len(ligand_ids)}")

    stats = run_oracle_docking(
        dataset_root=str(dataset_root),
        ligand_ids=ligand_ids,
        vina_executable=run_cfg.get("vina_executable", None),
        docking_backend=str(oracle_cfg.get("docking_backend", "vina")).strip().lower(),
        overwrite=bool(args.overwrite or oracle_cfg.get("oracle_overwrite", False)),
        exhaustiveness=int(oracle_cfg.get("oracle_exhaustiveness", 8)),
        pocket_radius=float(oracle_cfg.get("oracle_pocket_radius", 10.0)),
        confgen_max_attempts=int(oracle_3d_cfg.get("max_attempts", 3)),
        confgen_seed=int(oracle_3d_cfg.get("seed", 0)),
        confgen_num_confs=int(oracle_3d_cfg.get("num_confs", 8)),
        confgen_max_opt_iters=int(oracle_3d_cfg.get("max_opt_iters", 200)),
        confgen_optimize=bool(oracle_3d_cfg.get("optimize", True)),
        confgen_prefer_mmff=bool(oracle_3d_cfg.get("prefer_mmff", False)),
        meeko_allow_bad_res=bool(oracle_cfg.get("meeko_allow_bad_res", False)),
        meeko_default_altloc=oracle_cfg.get("meeko_default_altloc", None),
        unidock_service_url=oracle_cfg.get("unidock_service_url", None),
        unidock_service_wsl_distro=oracle_cfg.get("unidock_service_wsl_distro", None),
        unidock_scoring=oracle_cfg.get("unidock_scoring", "vina"),
        unidock_search_mode=oracle_cfg.get("unidock_search_mode", "balance"),
        unidock_num_modes=int(oracle_cfg.get("unidock_num_modes", 1)),
        unidock_timeout_sec=int(oracle_cfg.get("unidock_timeout_sec", 3600)),
        unidock_service_local_input_root=oracle_cfg.get("unidock_service_local_input_root", None),
    )
    print(f"[oracle] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
