from __future__ import annotations

import argparse
import csv
import gc
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.ligand_only_3d_dataset import (  # noqa: E402
    GLOBAL_LIGAND3D_CACHE_DIR,
    LigandOnly3DStore,
    _canonical_cache_smiles,
    ligand3d_cache_key_from_parts,
)


DEFAULT_RUN_NAMES = [
    "virtual_loop_scratch_gaussianens4_sharedpool_v4_alltargets_seed42",
    "virtual_loop_scratch_gaussianens4_sharedpool_v4_alltargets_seed43",
    "virtual_loop_scratch_gaussianens4_sharedpool_v4_alltargets_seed44",
]


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO / path).resolve()


def default_run_roots() -> list[Path]:
    run_root = REPO / "runs" / "dockstring_benchmark"
    return [run_root / name for name in DEFAULT_RUN_NAMES]


def discover_pool_files(run_roots: Iterable[Path], targets: set[str], rounds: int, include_init: bool) -> list[Path]:
    files: list[Path] = []
    for run_root in run_roots:
        if not run_root.exists():
            raise FileNotFoundError(run_root)
        target_dirs = sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "_shared").exists())
        if targets:
            target_dirs = [path for path in target_dirs if path.name in targets]
        for target_dir in target_dirs:
            shared = target_dir / "_shared"
            if include_init:
                init_path = shared / "init_set.csv"
                if not init_path.exists():
                    raise FileNotFoundError(init_path)
                files.append(init_path)
            for round_idx in range(1, int(rounds) + 1):
                path = shared / f"round{round_idx:02d}_candidate_pool.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                files.append(path)
    if not files:
        raise RuntimeError("No Dockstring shared-pool files were discovered.")
    return files


def load_shared_vocab(run_roots: Iterable[Path], targets: set[str]) -> list[str]:
    vocab_paths: list[Path] = []
    for run_root in run_roots:
        target_dirs = sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "_shared").exists())
        if targets:
            target_dirs = [path for path in target_dirs if path.name in targets]
        for target_dir in target_dirs:
            for method in ["analytic_ehvi", "analytic_pomhi", "qnehvi_mc", "qnparego_mc", "greedy_mean"]:
                path = target_dir / method / "round01_dataset" / "ligand_vocab.json"
                if path.exists():
                    vocab_paths.append(path)
                    break
    if not vocab_paths:
        raise RuntimeError("Failed to find any round01_dataset/ligand_vocab.json under the run roots.")
    seen: dict[str, Path] = {}
    for path in vocab_paths:
        vocab = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(vocab, list) or not vocab:
            raise RuntimeError(f"Invalid ligand vocab: {path}")
        key = json.dumps([str(item) for item in vocab], sort_keys=False)
        seen.setdefault(key, path)
    if len(seen) != 1:
        details = {str(path): json.loads(path.read_text(encoding="utf-8")) for path in seen.values()}
        raise RuntimeError(f"Found incompatible ligand vocabularies across run roots: {details}")
    first = next(iter(seen.values()))
    return [str(item) for item in json.loads(first.read_text(encoding="utf-8"))]


def read_unique_rows(pool_files: Iterable[Path]) -> list[dict]:
    by_smiles: dict[str, dict] = {}
    for path in pool_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RuntimeError(f"CSV has no header: {path}")
            smiles_col = "smiles_canonical" if "smiles_canonical" in reader.fieldnames else "smiles"
            if smiles_col not in reader.fieldnames:
                raise RuntimeError(f"CSV lacks smiles column: {path}")
            for row in reader:
                raw_smiles = str(row.get(smiles_col, "")).strip()
                if not raw_smiles:
                    continue
                smiles = _canonical_cache_smiles(raw_smiles)
                if smiles in by_smiles:
                    by_smiles[smiles]["source_count"] += 1
                    continue
                by_smiles[smiles] = {
                    "smiles_canonical": smiles,
                    "split": "test",
                    "dock_score": row.get("dock_score", ""),
                    "qed": row.get("qed", ""),
                    "sa_score": row.get("sa_score", ""),
                    "source_count": 1,
                    "source_example": str(path),
                }
    rows = list(by_smiles.values())
    rows.sort(key=lambda item: item["smiles_canonical"])
    for idx, row in enumerate(rows):
        row["ligand_id"] = f"PREWARM_{idx:09d}"
    return rows


def cache_key(smiles: str, args: argparse.Namespace) -> str:
    return ligand3d_cache_key_from_parts(
        smiles,
        max_attempts=int(args.max_attempts),
        num_confs=int(args.num_confs),
        max_opt_iters=int(args.max_opt_iters),
        optimize=bool(args.optimize),
        prefer_mmff=bool(args.prefer_mmff),
    )


def cache_statuses(cache_dir: Path, rows: list[dict], args: argparse.Namespace) -> dict[str, int]:
    db_path = cache_dir / "cache.sqlite3"
    counts = {"ready": 0, "failed": 0, "missing": 0, "other": 0}
    if not db_path.exists():
        counts["missing"] = len(rows)
        return counts
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    for row in rows:
        key = cache_key(str(row["smiles_canonical"]), args)
        found = cur.execute("SELECT status FROM ligand3d_cache WHERE cache_key = ?", (key,)).fetchone()
        if found is None:
            counts["missing"] += 1
        elif str(found[0]) in counts:
            counts[str(found[0])] += 1
        else:
            counts["other"] += 1
    con.close()
    return counts


def missing_rows(cache_dir: Path, rows: list[dict], args: argparse.Namespace) -> list[dict]:
    db_path = cache_dir / "cache.sqlite3"
    if not db_path.exists():
        return list(rows)
    out: list[dict] = []
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    for row in rows:
        key = cache_key(str(row["smiles_canonical"]), args)
        found = cur.execute("SELECT status FROM ligand3d_cache WHERE cache_key = ?", (key,)).fetchone()
        if found is None:
            out.append(row)
    con.close()
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prewarm ligand-only 3D graph cache for Dockstring shared pools.")
    parser.add_argument("--run-root", action="append", default=None)
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--include-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", default=str(GLOBAL_LIGAND3D_CACHE_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--build-chunk-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--num-confs", type=int, default=1)
    parser.add_argument("--max-opt-iters", type=int, default=40)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-mmff", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    run_roots = [repo_path(item) for item in args.run_root] if args.run_root else default_run_roots()
    targets = {str(item).strip() for item in (args.target or []) if str(item).strip()}
    cache_dir = repo_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_json = repo_path(args.progress_json) if args.progress_json else cache_dir / "prewarm_dockstring_sharedpool_progress.json"

    pool_files = discover_pool_files(run_roots, targets=targets, rounds=int(args.rounds), include_init=bool(args.include_init))
    vocab = load_shared_vocab(run_roots, targets=targets)
    rows = read_unique_rows(pool_files)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    before = cache_statuses(cache_dir, rows, args)
    pending = missing_rows(cache_dir, rows, args)
    summary = {
        "run_roots": [str(path) for path in run_roots],
        "targets": sorted(targets),
        "pool_file_count": len(pool_files),
        "unique_smiles": len(rows),
        "cache_dir": str(cache_dir),
        "confgen": {
            "max_attempts": int(args.max_attempts),
            "num_confs": int(args.num_confs),
            "max_opt_iters": int(args.max_opt_iters),
            "optimize": bool(args.optimize),
            "prefer_mmff": bool(args.prefer_mmff),
        },
        "before": before,
        "pending_missing": len(pending),
        "workers": int(args.workers),
        "build_chunk_size": int(args.build_chunk_size),
        "dry_run": bool(args.dry_run),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(progress_json, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run or not pending:
        return 0

    total = len(pending)
    started = time.perf_counter()
    completed = 0
    for chunk_start in range(0, total, int(args.build_chunk_size)):
        chunk = pending[chunk_start : chunk_start + int(args.build_chunk_size)]
        chunk_t0 = time.perf_counter()
        LigandOnly3DStore(
            root=REPO,
            ligand_vocab_override=list(vocab),
            cache_dir=cache_dir,
            rows=chunk,
            confgen_max_attempts=int(args.max_attempts),
            confgen_seed=0,
            confgen_num_confs=int(args.num_confs),
            confgen_max_opt_iters=int(args.max_opt_iters),
            confgen_optimize=bool(args.optimize),
            confgen_prefer_mmff=bool(args.prefer_mmff),
            build_num_workers=int(args.workers),
            build_mp_chunksize=int(args.chunksize),
        )
        completed += len(chunk)
        chunk_sec = time.perf_counter() - chunk_t0
        elapsed = time.perf_counter() - started
        progress = dict(summary)
        progress.update(
            {
                "completed_missing": completed,
                "remaining_missing": total - completed,
                "last_chunk_size": len(chunk),
                "last_chunk_seconds": chunk_sec,
                "elapsed_seconds": elapsed,
                "overall_input_mol_per_second": completed / elapsed if elapsed > 0 else None,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        write_json(progress_json, progress)
        print(json.dumps(progress, indent=2), flush=True)
        del chunk
        gc.collect()

    final = dict(summary)
    final["after"] = cache_statuses(cache_dir, rows, args)
    final["completed_missing"] = completed
    final["elapsed_seconds"] = time.perf_counter() - started
    final["overall_input_mol_per_second"] = completed / final["elapsed_seconds"] if final["elapsed_seconds"] > 0 else None
    final["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(progress_json, final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
