from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import gzip
import io
import sqlite3

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from data.pocket_dataset import (
    _adjust_feature_vec,
    _atom_feature_vec,
    _load_ligand_vocab_override,
    ATOM_EXTRA_DIM,
)
from mobo.pretrain_targets import PRETRAIN_PROPERTY_NAMES, build_pretrain_props

rdBase.DisableLog("rdApp.error")

GLOBAL_LIGAND3D_CACHE_DIR = Path(__file__).resolve().parents[1] / ".oracle_cache" / "ligand3d_cache"
CACHE_DB_NAME = "cache.sqlite3"
CACHE_WRITE_BATCH = 2560
SINGLE_MOL_TIMEOUT_SEC = 15.0
WORKER_POLL_TIMEOUT_SEC = 0.2
CACHE_PROGRESS_EVERY = 128
POOL_MAX_TASKS_PER_CHILD = 32
POOL_SHUTDOWN_TIMEOUT_SEC = 5.0


def ligand3d_cache_key_from_parts(
    canonical_smiles: str,
    *,
    max_attempts: int,
    num_confs: int,
    max_opt_iters: int,
    optimize: bool,
    prefer_mmff: bool,
) -> str:
    base = f"smiles::{str(canonical_smiles).strip()}"
    cfg_key = (
        f"attempts={int(max_attempts)};confs={int(num_confs)};"
        f"iters={int(max_opt_iters)};opt={int(bool(optimize))};mmff={int(bool(prefer_mmff))}"
    )
    return f"{base}|{cfg_key}"


def _canonical_cache_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    mol = Chem.RemoveHs(mol, sanitize=True)
    return Chem.MolToSmiles(mol, canonical=True)


def _smiles_to_fp(smiles: str, fp_dim: int, fp_radius: int = 2) -> torch.Tensor | None:
    fp_dim = int(fp_dim)
    if fp_dim <= 0:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(fp_radius), nBits=fp_dim)
    arr = torch.zeros(fp_dim, dtype=torch.float32)
    onbits = list(fp.GetOnBits())
    if onbits:
        arr[torch.tensor(onbits, dtype=torch.long)] = 1.0
    return arr


def _pick_smiles_column(columns) -> Optional[str]:
    col_map = {str(c).lower(): str(c) for c in columns}
    for key in ["smiles_canonical", "smiles", "smile", "smiles_clean", "smiles_cleaned"]:
        if key in col_map:
            return col_map[key]
    return None


def _scan_vocab(smiles_list: List[str]) -> List[str]:
    vocab = set()
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            vocab.add(atom.GetSymbol())
    return sorted(vocab)


def _serialize_template(data: Data) -> bytes:
    buf = io.BytesIO()
    torch.save(data, buf)
    return gzip.compress(buf.getvalue())


def _deserialize_template(blob: bytes) -> Data:
    payload = gzip.decompress(blob)
    return torch.load(io.BytesIO(payload), weights_only=False)


def _log_cache(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


_MP_LIGAND3D_FN = None


def _mp_init_ligand3d_pool() -> None:
    global _MP_LIGAND3D_FN
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    from mobo.graphs import smiles_to_ligand_3d
    _MP_LIGAND3D_FN = smiles_to_ligand_3d


def _mp_warmup_ligand3d_pool(token: int) -> int:
    if _MP_LIGAND3D_FN is None:
        raise RuntimeError("Ligand 3D pool initializer did not load smiles_to_ligand_3d.")
    return int(token)


def _worker_build_single_graph(smiles: str, cfg: dict) -> Data:
    fn = _MP_LIGAND3D_FN
    if fn is None:
        _mp_init_ligand3d_pool()
        fn = _MP_LIGAND3D_FN
    if fn is None:
        raise RuntimeError("Ligand 3D worker failed to initialize smiles_to_ligand_3d.")

    return fn(
        smiles,
        atom_index=cfg["atom_index"],
        atom_feat_dim=cfg["atom_feat_dim"],
        atom_extra_dim=cfg["atom_extra_dim"],
        fp_dim=cfg["fp_dim"],
        fp_radius=cfg["fp_radius"],
        max_attempts=cfg["max_attempts"],
        seed=cfg["seed"],
        num_confs=cfg["num_confs"],
        max_opt_iters=cfg["max_opt_iters"],
        optimize=cfg["optimize"],
        prefer_mmff=cfg["prefer_mmff"],
    )


class _SplitView(Dataset):
    def __init__(self, store: "LigandOnly3DStore", indices: Sequence[int]):
        super().__init__()
        self._store = store
        self._indices = list(indices)
        self.num_node_classes_ = getattr(store, "num_node_classes_", 0)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        return self._store.data_list[self._indices[idx]]


class LigandOnly3DStore:
    def __init__(
        self,
        root: str | Path,
        ligand_vocab_override: Optional[List[str]] = None,
        atom_extra_dim: Optional[int] = None,
        cache_dir: Optional[str | Path] = None,
        fp_dim: int = 0,
        fp_radius: int = 2,
        rows: Optional[Sequence[dict]] = None,
        confgen_max_attempts: int = 10,
        confgen_seed: int = 0,
        confgen_num_confs: int = 10,
        confgen_max_opt_iters: int = 200,
        confgen_optimize: bool = True,
        confgen_prefer_mmff: bool = True,
        build_num_workers: int = 1,
        build_mp_chunksize: int = 16,
    ) -> None:
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir else GLOBAL_LIGAND3D_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / CACHE_DB_NAME
        self._db = sqlite3.connect(str(self.db_path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()

        self.ligand_vocab = []
        if ligand_vocab_override:
            self.ligand_vocab = [str(x).strip() for x in ligand_vocab_override if str(x).strip()]
        else:
            vocab = _load_ligand_vocab_override(self.root)
            if vocab:
                self.ligand_vocab = list(vocab)

        self.atom_extra_dim = int(atom_extra_dim) if atom_extra_dim is not None else ATOM_EXTRA_DIM
        self.fp_dim = int(fp_dim)
        self.fp_radius = int(fp_radius)
        df = None
        if rows is None:
            smiles_path = self.root / "smiles.csv"
            if not smiles_path.exists():
                raise FileNotFoundError(smiles_path)
            try:
                df = pd.read_csv(smiles_path)
            except pd.errors.ParserError:
                df = pd.read_csv(smiles_path, engine="python", on_bad_lines="skip")
            if df.empty:
                raise RuntimeError(f"smiles.csv is empty: {smiles_path}")
            smiles_col = _pick_smiles_column(df.columns)
            if smiles_col is None:
                raise RuntimeError("No smiles column found in smiles.csv.")
            self.smiles_col = smiles_col
        else:
            cols = list(rows[0].keys()) if rows else []
            smiles_col = _pick_smiles_column(cols)
            self.smiles_col = smiles_col or "smiles"

        if not self.ligand_vocab:
            if df is not None:
                self.ligand_vocab = _scan_vocab(df[self.smiles_col].astype(str).tolist())
            else:
                smiles_list = [str(row.get(self.smiles_col, "")).strip() for row in rows or []]
                self.ligand_vocab = _scan_vocab(smiles_list)
        if not self.ligand_vocab:
            raise RuntimeError("LigandOnly3DStore requires a non-empty ligand vocabulary.")

        self.atom_dim = len(self.ligand_vocab)
        self.atom_feat_dim = self.atom_dim + max(self.atom_extra_dim, 0)
        self.num_node_classes_ = self.atom_feat_dim
        self.atom_index = {sym: i for i, sym in enumerate(self.ligand_vocab)}
        self.confgen = {
            "max_attempts": int(confgen_max_attempts),
            "seed": int(confgen_seed),
            "num_confs": int(confgen_num_confs),
            "max_opt_iters": int(confgen_max_opt_iters),
            "optimize": bool(confgen_optimize),
            "prefer_mmff": bool(confgen_prefer_mmff),
        }
        self.build_num_workers = max(int(build_num_workers), 1)
        self.build_mp_chunksize = max(int(build_mp_chunksize), 1)

        self.data_list: list[Data] = []
        self.split_indices: Dict[str, list[int]] = {"train": [], "valid": [], "test": []}
        self.id_to_idx: Dict[str, int] = {}
        self.failed_ = 0

        if rows is None:
            if "split" not in df.columns:
                df = df.copy()
                df["split"] = "train"
            self._append_rows(df.to_dict("records"))
        elif rows:
            self._append_rows(rows)

    def __del__(self):
        try:
            self._db.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS ligand3d_cache (
              cache_key TEXT PRIMARY KEY,
              canonical_smiles TEXT NOT NULL,
              status TEXT NOT NULL,
              payload BLOB,
              error_stage TEXT,
              error_message TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_ligand3d_status ON ligand3d_cache(status)")
        self._db.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _query_cache(self, cache_key: str) -> tuple[str | None, Data | None]:
        row = self._db.execute(
            "SELECT status, payload FROM ligand3d_cache WHERE cache_key = ?",
            (str(cache_key),),
        ).fetchone()
        if row is None:
            return None, None
        status = str(row["status"])
        if status != "ready":
            return status, None
        payload = row["payload"]
        if payload is None:
            return "failed", None
        try:
            return "ready", _deserialize_template(bytes(payload))
        except Exception:
            return None, None

    def _upsert_success_records(self, records: list[tuple]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO ligand3d_cache
        (cache_key, canonical_smiles, status, payload, error_stage, error_message, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
          canonical_smiles=excluded.canonical_smiles,
          status=excluded.status,
          payload=excluded.payload,
          error_stage=excluded.error_stage,
          error_message=excluded.error_message,
          updated_at=excluded.updated_at
        """
        for start in range(0, len(records), CACHE_WRITE_BATCH):
            chunk = records[start : start + CACHE_WRITE_BATCH]
            self._db.executemany(sql, chunk)
            self._db.commit()

    def _upsert_failure_records(self, records: list[tuple]) -> None:
        if not records:
            return
        sql = """
        INSERT INTO ligand3d_cache
        (cache_key, canonical_smiles, status, payload, error_stage, error_message, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
          canonical_smiles=excluded.canonical_smiles,
          status=excluded.status,
          payload=NULL,
          error_stage=excluded.error_stage,
          error_message=excluded.error_message,
          updated_at=excluded.updated_at
        """
        for start in range(0, len(records), CACHE_WRITE_BATCH):
            chunk = records[start : start + CACHE_WRITE_BATCH]
            self._db.executemany(sql, chunk)
            self._db.commit()

    def _cache_key(self, ligand_id: str, smiles: str) -> str:
        del ligand_id
        cfg = self.confgen
        return ligand3d_cache_key_from_parts(
            _canonical_cache_smiles(smiles),
            max_attempts=int(cfg["max_attempts"]),
            num_confs=int(cfg["num_confs"]),
            max_opt_iters=int(cfg["max_opt_iters"]),
            optimize=bool(cfg["optimize"]),
            prefer_mmff=bool(cfg["prefer_mmff"]),
        )

    def _dock_value(self, row: dict) -> float:
        try:
            return float(row.get("dock_score", float("nan")))
        except Exception:
            return float("nan")

    def _materialize_data(self, template: Data, smi: str, lig_id: str, row: dict) -> Data | None:
        data = deepcopy(template)
        data.smiles = smi
        data.ligand_id = lig_id
        x_cached = getattr(data, "x", None)
        if x_cached is None or x_cached.size(-1) != self.atom_feat_dim:
            return None
        if self.fp_dim > 0:
            fp = getattr(data, "fp", None)
            if fp is None or fp.size(-1) != self.fp_dim:
                fp_new = _smiles_to_fp(smi, self.fp_dim, self.fp_radius)
                if fp_new is not None:
                    data.fp = fp_new.unsqueeze(0)
        elif hasattr(data, "fp"):
            del data.fp
        props = getattr(data, "pretrain_props", None)
        if props is None or props.size(-1) != len(PRETRAIN_PROPERTY_NAMES):
            props_new = build_pretrain_props(smi, pos=getattr(data, "pos", None))
            if props_new is not None:
                data.pretrain_props = props_new.unsqueeze(0)
        dock_val = self._dock_value(row)
        data.dock_score = dock_val
        data.vina_score = dock_val
        return data

    def _append_row_data(self, data: Data, split: str, lig_id: str) -> None:
        idx = len(self.data_list)
        self.data_list.append(data)
        self.split_indices[split].append(idx)
        if lig_id:
            self.id_to_idx[lig_id] = idx

    def _append_rows(self, rows: Sequence[dict]) -> Dict[str, int]:
        added = 0
        failed = 0
        pending: list[dict] = []
        fetched_cache: dict[str, tuple[str | None, Data | None]] = {}
        for row in rows:
            smi = str(row.get(self.smiles_col, "")).strip()
            if not smi:
                failed += 1
                continue
            lig_id = str(row.get("ligand_id", "")).strip()
            if lig_id and lig_id in self.id_to_idx:
                continue
            split = str(row.get("split", "train")).strip().lower()
            if split not in self.split_indices:
                split = "train"
            cache_key = self._cache_key(lig_id, smi)
            if cache_key not in fetched_cache:
                fetched_cache[cache_key] = self._query_cache(cache_key)
            status, template = fetched_cache[cache_key]
            if status == "ready" and template is not None:
                data = self._materialize_data(template, smi, lig_id, row)
                if data is not None:
                    self._append_row_data(data, split, lig_id)
                    added += 1
                    continue
            if status == "failed":
                failed += 1
                continue
            pending.append({
                "row": row,
                "smi": smi,
                "lig_id": lig_id,
                "split": split,
                "cache_key": cache_key,
            })

        if pending:
            unique_pending: dict[str, dict] = {}
            pending_by_key: dict[str, list[dict]] = {}
            for item in pending:
                cache_key = str(item["cache_key"])
                unique_pending.setdefault(cache_key, item)
                pending_by_key.setdefault(cache_key, []).append(item)
            unique_items = list(unique_pending.values())
            total_unique = len(unique_items)
            worker_count = max(1, min(self.build_num_workers, total_unique))
            _log_cache(f"[ligand3d_cache] dispatching {total_unique} unique molecules to {worker_count} workers")

            worker_cfg = {
                "atom_index": self.atom_index,
                "atom_feat_dim": self.atom_feat_dim,
                "atom_extra_dim": self.atom_extra_dim,
                "fp_dim": self.fp_dim,
                "fp_radius": self.fp_radius,
                "max_attempts": self.confgen["max_attempts"],
                "seed": self.confgen["seed"],
                "num_confs": self.confgen["num_confs"],
                "max_opt_iters": self.confgen["max_opt_iters"],
                "optimize": self.confgen["optimize"],
                "prefer_mmff": self.confgen["prefer_mmff"],
            }
            ctx = mp.get_context("spawn")
            success_records: list[tuple] = []
            failure_records: list[tuple] = []
            completed_unique = 0
            next_task_id = 0
            pending_queue = deque(unique_items)
            active_tasks: dict[int, dict] = {}
            abandoned_tasks: dict[int, dict] = {}
            pool = None
            overall_started_at = time.monotonic()
            batch_started_at = overall_started_at
            batch_ready_unique = 0
            batch_failed_unique = 0
            batch_timeout_unique = 0
            batch_flushed_unique = 0

            def flush_buffers() -> None:
                nonlocal success_records, failure_records
                nonlocal batch_started_at, batch_ready_unique, batch_failed_unique, batch_timeout_unique, batch_flushed_unique
                if not success_records and not failure_records:
                    return
                if success_records:
                    self._upsert_success_records(success_records)
                    success_records = []
                if failure_records:
                    self._upsert_failure_records(failure_records)
                    failure_records = []
                batch_total_unique = batch_ready_unique + batch_failed_unique + batch_timeout_unique
                batch_elapsed = max(time.monotonic() - batch_started_at, 1e-6)
                overall_elapsed = max(time.monotonic() - overall_started_at, 1e-6)
                batch_rate = batch_total_unique / batch_elapsed
                overall_rate = completed_unique / overall_elapsed if completed_unique > 0 else 0.0
                batch_flushed_unique += batch_total_unique
                _log_cache(
                    "[ligand3d_cache] flush "
                    f"batch_unique={batch_total_unique} "
                    f"ready={batch_ready_unique} "
                    f"failed={batch_failed_unique} "
                    f"timeout={batch_timeout_unique} "
                    f"batch_rate={batch_rate:.2f} mol/s "
                    f"overall_rate={overall_rate:.2f} mol/s "
                    f"flushed_total={batch_flushed_unique}/{total_unique}"
                )
                batch_started_at = time.monotonic()
                batch_ready_unique = 0
                batch_failed_unique = 0
                batch_timeout_unique = 0

            def build_pool():
                pool_obj = ctx.Pool(
                    processes=worker_count,
                    initializer=_mp_init_ligand3d_pool,
                    initargs=(),
                    maxtasksperchild=POOL_MAX_TASKS_PER_CHILD,
                )
                pool_obj.map(_mp_warmup_ligand3d_pool, range(worker_count), chunksize=1)
                return pool_obj

            def close_pool_workers(cur_pool) -> None:
                workers = list(getattr(cur_pool, "_pool", []) or [])
                for worker in workers:
                    try:
                        if worker.is_alive():
                            worker.terminate()
                    except Exception as exc:
                        _log_cache(f"[ligand3d_cache] worker terminate failed pid={getattr(worker, 'pid', None)} error={type(exc).__name__}: {exc}")
                for worker in workers:
                    try:
                        worker.join(timeout=POOL_SHUTDOWN_TIMEOUT_SEC)
                    except Exception as exc:
                        _log_cache(f"[ligand3d_cache] worker join failed pid={getattr(worker, 'pid', None)} error={type(exc).__name__}: {exc}")
                for worker in workers:
                    try:
                        if worker.is_alive():
                            _log_cache(f"[ligand3d_cache] force killing stuck worker pid={getattr(worker, 'pid', None)}")
                            worker.kill()
                            worker.join(timeout=1.0)
                    except Exception as exc:
                        _log_cache(f"[ligand3d_cache] worker kill failed pid={getattr(worker, 'pid', None)} error={type(exc).__name__}: {exc}")

            def close_pool(cur_pool) -> None:
                if cur_pool is None:
                    return
                try:
                    cur_pool.terminate()
                except AssertionError as exc:
                    msg = str(exc)
                    if "result_handler not alive" not in msg:
                        raise
                    cache = getattr(cur_pool, "_cache", None)
                    cache_size = len(cache) if cache is not None else 0
                    _log_cache(
                        "[ligand3d_cache] pool terminate hit multiprocessing cache assertion; "
                        f"clearing {cache_size} abandoned async results before retry"
                    )
                    if cache is not None:
                        cache.clear()
                    try:
                        cur_pool.terminate()
                    except AssertionError:
                        pass
                finally:
                    close_pool_workers(cur_pool)

            def mark_failure(item: dict, stage: str, message: str) -> None:
                nonlocal failed, completed_unique, batch_failed_unique, batch_timeout_unique
                now = self._now()
                cache_key = str(item["cache_key"])
                rows_for_key = pending_by_key.get(cache_key, [])
                failure_records.append((cache_key, _canonical_cache_smiles(str(item["smi"])), "failed", None, stage, message, now, now))
                failed += len(rows_for_key)
                completed_unique += 1
                if str(stage) == "timeout":
                    batch_timeout_unique += 1
                else:
                    batch_failed_unique += 1

            def mark_success(item: dict, template: Data) -> None:
                nonlocal added, failed, completed_unique, batch_ready_unique
                now = self._now()
                cache_key = str(item["cache_key"])
                success_records.append((cache_key, _canonical_cache_smiles(str(item["smi"])), "ready", sqlite3.Binary(_serialize_template(template)), None, None, now, now))
                rows_for_key = pending_by_key.get(cache_key, [])
                for row_item in rows_for_key:
                    data = self._materialize_data(template, str(row_item["smi"]), str(row_item["lig_id"]), row_item["row"])
                    if data is None:
                        failed += 1
                        continue
                    self._append_row_data(data, str(row_item["split"]), str(row_item["lig_id"]))
                    added += 1
                completed_unique += 1
                batch_ready_unique += 1

            def dispatch(cur_pool) -> None:
                nonlocal next_task_id
                while (len(active_tasks) + len(abandoned_tasks)) < worker_count and pending_queue:
                    item = pending_queue.popleft()
                    task_id = next_task_id
                    next_task_id += 1
                    async_result = cur_pool.apply_async(_worker_build_single_graph, (str(item["smi"]), worker_cfg))
                    active_tasks[task_id] = {
                        "item": item,
                        "async_result": async_result,
                        "started_at": time.monotonic(),
                    }

            def reap_abandoned_tasks() -> None:
                for task_id, task in list(abandoned_tasks.items()):
                    async_result = task["async_result"]
                    if not async_result.ready():
                        continue
                    try:
                        async_result.get()
                    except Exception:
                        pass
                    abandoned_tasks.pop(task_id, None)

            pool = build_pool()
            try:
                while completed_unique < total_unique:
                    reap_abandoned_tasks()
                    dispatch(pool)
                    finished_task_ids: list[int] = []
                    timeout_task_id = None
                    now_mono = time.monotonic()
                    for task_id, task in list(active_tasks.items()):
                        async_result = task["async_result"]
                        if async_result.ready():
                            try:
                                payload = async_result.get()
                                mark_success(task["item"], payload)
                            except Exception as exc:
                                mark_failure(task["item"], "graph_build", f"{type(exc).__name__}: {exc}")
                            finished_task_ids.append(task_id)
                            if (completed_unique % CACHE_PROGRESS_EVERY) == 0 or completed_unique == total_unique:
                                _log_cache(f"[ligand3d_cache] progress {completed_unique}/{total_unique} ready={added} failed={failed}")
                            if (len(success_records) + len(failure_records)) >= CACHE_WRITE_BATCH:
                                flush_buffers()
                            continue
                        if (now_mono - float(task["started_at"])) > SINGLE_MOL_TIMEOUT_SEC:
                            timeout_task_id = task_id
                            break

                    for task_id in finished_task_ids:
                        active_tasks.pop(task_id, None)

                    if timeout_task_id is not None:
                        timeout_task = active_tasks.pop(timeout_task_id)
                        timeout_item = timeout_task["item"]
                        _log_cache(
                            f"[ligand3d_cache] timeout cache_key={timeout_item['cache_key']} "
                            f"after {SINGLE_MOL_TIMEOUT_SEC:.1f}s; marking failure without pool restart"
                        )
                        mark_failure(timeout_item, "timeout", f"timeout>{SINGLE_MOL_TIMEOUT_SEC:.1f}s")
                        abandoned_tasks[timeout_task_id] = timeout_task
                        if (len(success_records) + len(failure_records)) >= CACHE_WRITE_BATCH:
                            flush_buffers()
                        if (completed_unique % CACHE_PROGRESS_EVERY) == 0 or completed_unique == total_unique:
                            _log_cache(f"[ligand3d_cache] progress {completed_unique}/{total_unique} ready={added} failed={failed}")
                        if pending_queue and not active_tasks and len(abandoned_tasks) >= worker_count:
                            _log_cache("[ligand3d_cache] all workers occupied by timed-out tasks; restarting pool")
                            abandoned_tasks.clear()
                            close_pool(pool)
                            pool = build_pool()
                        continue

                    if completed_unique >= total_unique:
                        break
                    time.sleep(WORKER_POLL_TIMEOUT_SEC)
            finally:
                flush_buffers()
                close_pool(pool)

        self.failed_ += failed
        return {"added": added, "failed": failed}

    def append_from_csv(self, csv_path: str | Path, ligand_ids: Sequence[str]) -> Dict[str, int]:
        if not ligand_ids:
            return {"added": 0, "failed": 0}
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return {"added": 0, "failed": len(ligand_ids)}
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.ParserError:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        if df.empty or "ligand_id" not in df.columns:
            return {"added": 0, "failed": len(ligand_ids)}
        id_set = {str(x).strip() for x in ligand_ids if str(x).strip()}
        if not id_set:
            return {"added": 0, "failed": 0}
        rows = df[df["ligand_id"].astype(str).isin(id_set)].to_dict("records")
        return self._append_rows(rows)

    def append_rows(self, rows: Sequence[dict]) -> Dict[str, int]:
        if not rows:
            return {"added": 0, "failed": 0}
        return self._append_rows(rows)

    def set_split_indices(self, split_indices: Dict[str, Sequence[int]]) -> None:
        self.split_indices = {
            "train": list(split_indices.get("train", [])),
            "valid": list(split_indices.get("valid", [])),
            "test": list(split_indices.get("test", [])),
        }

    def get_dataset_from_indices(self, indices: Sequence[int]) -> Dataset:
        return _SplitView(self, indices)

    def get_split_dataset(self, split: str) -> Dataset:
        split = str(split).strip().lower()
        indices = self.split_indices.get(split, [])
        return _SplitView(self, indices)


class LigandOnly3DDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        ligand_vocab_override: Optional[List[str]] = None,
        atom_extra_dim: Optional[int] = None,
        cache_dir: Optional[str | Path] = None,
        fp_dim: int = 0,
        fp_radius: int = 2,
        confgen_max_attempts: int = 10,
        confgen_seed: int = 0,
        confgen_num_confs: int = 10,
        confgen_max_opt_iters: int = 200,
        confgen_optimize: bool = True,
        confgen_prefer_mmff: bool = True,
        build_num_workers: int = 1,
        build_mp_chunksize: int = 16,
    ):
        super().__init__()
        self.split = str(split).lower()
        self._store = LigandOnly3DStore(
            root=root,
            ligand_vocab_override=ligand_vocab_override,
            atom_extra_dim=atom_extra_dim,
            cache_dir=cache_dir,
            fp_dim=fp_dim,
            fp_radius=fp_radius,
            confgen_max_attempts=confgen_max_attempts,
            confgen_seed=confgen_seed,
            confgen_num_confs=confgen_num_confs,
            confgen_max_opt_iters=confgen_max_opt_iters,
            confgen_optimize=confgen_optimize,
            confgen_prefer_mmff=confgen_prefer_mmff,
            build_num_workers=build_num_workers,
            build_mp_chunksize=build_mp_chunksize,
        )
        self.num_node_classes_ = getattr(self._store, "num_node_classes_", 0)
        self.failed_ = getattr(self._store, "failed_", 0)
        indices = list(self._store.split_indices.get(self.split, []))
        self.data_list = [self._store.data_list[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]

    @staticmethod
    def _pick_smiles_column(columns) -> Optional[str]:
        return _pick_smiles_column(columns)

    @staticmethod
    def _scan_vocab(smiles_list: List[str]) -> List[str]:
        return _scan_vocab(smiles_list)

