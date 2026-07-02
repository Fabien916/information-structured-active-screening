#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")
warnings.simplefilter(action="ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data.ligand_only_3d_dataset import GLOBAL_LIGAND3D_CACHE_DIR
from mobo.config_utils import load_config
from mobo.dataset_utils import clear_processed
from mobo.init_selection import load_latent_library_csv
from mobo.metrics import evaluate_oracle_accuracy_from_csv
from mobo.oracle import run_oracle_docking
from mobo.retrospective import build_objective_tensor, compute_hypervolume
from mobo.smiles_utils import canonicalize_smiles_noh
from mobo.surrogate import load_surrogate, train_surrogate_from_scratch
from mobo_qpmhi import _fill_extra_metrics_in_csv, append_candidates_to_smiles_csv
from run_mobo_main_experiment import (
    _assign_new_rows_random_predictor_split,
    _batch_hv_gain,
    _build_or_load_downstream_protocol,
    _build_oracle_call_kwargs,
    _candidate_frame,
    _default_init_library_candidates,
    _dynamic_ref_point_from_labeled,
    _evaluate_holdout_surrogate,
    _export_selected_batch_with_oracle,
    _load_holdout_df,
    _load_observed_df,
    _log,
    _method_seed,
    _normalize_ratio_config,
    _prepare_init_dataset,
    _require_dict,
    _resolve_append_global_vocab,
    _resolve_cli_or_config,
    _resolve_init_pretrain_ligand_vocab,
    _resolve_or_train_init_encoder_ckpt,
    _resolve_oracle_asset_root,
    _score_with_predictions_only,
    _section,
    _validate_mobo_config,
)


def _janus_get_selfies_chars(selfies_text: str) -> list[str]:
    return list(sf.split_selfies(selfies_text))


def _janus_fp_scores(smiles_list: list[str], target_smiles: str) -> list[float]:
    target = Chem.MolFromSmiles(target_smiles)
    if target is None:
        raise RuntimeError(f"Invalid target SMILES for JANUS similarity scoring: {target_smiles}")
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    scores: list[float] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"Invalid SMILES for JANUS similarity scoring: {smi}")
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        scores.append(float(TanimotoSimilarity(fp_mol, fp_target)))
    return scores


def _janus_mutate_sf(
    sf_chars: list[str],
    alphabet: list[str],
    num_sample_frags: int,
    base_alphabet: list[str] | None = None,
) -> str:
    if base_alphabet is None:
        base_alphabet = list(sf.get_semantic_robust_alphabet())
    random_char_idx = random.choice(range(len(sf_chars)))
    mut_choice = random.choice([1, 2, 3])
    if alphabet:
        sampled = random.sample(alphabet, min(int(num_sample_frags), len(alphabet)))
        full_alphabet = sampled + list(base_alphabet)
    else:
        full_alphabet = list(base_alphabet)
    if mut_choice == 1:
        random_char = random.choice(full_alphabet)
        out = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx + 1 :]
    elif mut_choice == 2:
        random_char = random.choice(full_alphabet)
        out = sf_chars[0:random_char_idx] + [random_char] + sf_chars[random_char_idx:]
    else:
        out = sf_chars if len(sf_chars) == 1 else sf_chars[0:random_char_idx] + sf_chars[random_char_idx + 1 :]
    return "".join(out)


def _janus_mutate_smiles(
    smile: str,
    *,
    alphabet: list[str],
    num_random_samples: int,
    num_mutations: int,
    num_sample_frags: int,
) -> list[str]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return []
    Chem.Kekulize(mol, clearAromaticFlags=True)
    randomized_smiles: list[str] = []
    for _ in range(max(1, int(num_random_samples))):
        randomized_smiles.append(
            Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
        )
    selfies_ls = [sf.encoder(x) for x in randomized_smiles]
    selfies_chars = [_janus_get_selfies_chars(x) for x in selfies_ls if x]
    mutated_sf: list[str] = []
    for chars in selfies_chars:
        latest_chars = chars
        for mut_idx in range(max(1, int(num_mutations))):
            if mut_idx == 0:
                mutated = _janus_mutate_sf(chars, alphabet=alphabet, num_sample_frags=int(num_sample_frags))
            else:
                mutated = _janus_mutate_sf(latest_chars, alphabet=alphabet, num_sample_frags=int(num_sample_frags))
            latest_chars = _janus_get_selfies_chars(mutated)
            mutated_sf.append(mutated)
    out: list[str] = []
    for item in mutated_sf:
        decoded = sf.decoder(item)
        if not decoded:
            continue
        mol_dec = Chem.MolFromSmiles(decoded, sanitize=True)
        if mol_dec is None:
            continue
        canon = Chem.MolToSmiles(mol_dec, isomericSmiles=False, canonical=True)
        if canon:
            out.append(canon)
    return list(dict.fromkeys(out))


def _janus_obtain_path(starting_smile: str, target_smile: str) -> list[str]:
    starting_selfie = sf.encoder(starting_smile)
    target_selfie = sf.encoder(target_smile)
    if not starting_selfie or not target_selfie:
        return []
    starting_chars = _janus_get_selfies_chars(starting_selfie)
    target_chars = _janus_get_selfies_chars(target_selfie)
    if len(starting_chars) < len(target_chars):
        starting_chars.extend([" "] * (len(target_chars) - len(starting_chars)))
    elif len(target_chars) < len(starting_chars):
        target_chars.extend([" "] * (len(starting_chars) - len(target_chars)))
    diff = [i for i in range(len(starting_chars)) if starting_chars[i] != target_chars[i]]
    path = {0: list(starting_chars)}
    for step_idx in range(len(diff)):
        idx = random.choice(diff)
        diff.remove(idx)
        path_member = list(path[step_idx])
        path_member[idx] = target_chars[idx]
        path[step_idx + 1] = list(path_member)
    selfies_path = ["".join(path[i]).replace(" ", "") for i in range(len(path))]
    return [sf.decoder(x) for x in selfies_path if sf.decoder(x)]


def _janus_joint_similarity(all_smiles: list[str], starting_smile: str, target_smile: str) -> np.ndarray:
    scores_start = np.asarray(_janus_fp_scores(all_smiles, starting_smile), dtype=np.float64)
    scores_target = np.asarray(_janus_fp_scores(all_smiles, target_smile), dtype=np.float64)
    data = np.vstack([scores_target, scores_start])
    avg_score = np.average(data, axis=0)
    better = avg_score - np.abs(data[0] - data[1])
    return ((1.0 / 9.0) * np.power(better, 3)) - ((7.0 / 9.0) * np.power(better, 2)) + ((19.0 / 12.0) * better)


def _janus_crossover_smiles(smiles_join: str, crossover_num_random_samples: int) -> list[str]:
    smi_a, smi_b = smiles_join.split("xxx")
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return []
    Chem.Kekulize(mol_a, clearAromaticFlags=True)
    Chem.Kekulize(mol_b, clearAromaticFlags=True)
    random_a: list[str] = []
    random_b: list[str] = []
    for _ in range(max(1, int(crossover_num_random_samples))):
        random_a.append(Chem.MolToSmiles(mol_a, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True))
        random_b.append(Chem.MolToSmiles(mol_b, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True))
    collected: list[str] = []
    for sa in random_a:
        for sb in random_b:
            collected.extend(_janus_obtain_path(sa, sb))
    canonical: list[str] = []
    for item in collected:
        mol = Chem.MolFromSmiles(item, sanitize=True)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        if canon:
            canonical.append(canon)
    canonical = list(dict.fromkeys(canonical))
    if not canonical:
        return []
    order = np.argsort(_janus_joint_similarity(canonical, smi_a, smi_b))[::-1]
    return [canonical[int(i)] for i in order]


def _janus_get_fragments(smiles_text: str, radius: int) -> list[str]:
    mol = Chem.MolFromSmiles(smiles_text, sanitize=True)
    if mol is None:
        return []
    frags: list[str] = []
    for atom_idx in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), atom_idx)
        amap: dict[int, int] = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag = Chem.MolToSmiles(submol, isomericSmiles=False, canonical=True)
        if frag:
            frags.append(frag)
    return list(dict.fromkeys(frags))


def _janus_form_fragments(smiles_text: str) -> list[str]:
    out: list[str] = []
    for frag in _janus_get_fragments(smiles_text, radius=3):
        try:
            frag_sf = sf.encoder(frag)
        except Exception:
            continue
        if not frag_sf:
            continue
        decoded = sf.decoder(frag_sf)
        if not decoded:
            continue
        mol = Chem.MolFromSmiles(decoded)
        if mol is None:
            continue
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            dearom_smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False, kekuleSmiles=True)
        except Exception:
            continue
        try:
            encoded = sf.encoder(dearom_smiles)
        except Exception:
            continue
        if encoded:
            out.append(encoded)
    return list(dict.fromkeys(out))


@dataclass
class ObjectiveScalarizer:
    dock_sign: float
    qed_sign: float
    sa_sign: float
    weights: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray

    @classmethod
    def from_init_observed(
        cls,
        *,
        init_df: pd.DataFrame,
        dock_sign: float,
        qed_sign: float,
        sa_sign: float,
        weights: list[float],
    ) -> "ObjectiveScalarizer":
        if len(weights) != 3:
            raise RuntimeError(f"Population baselines currently require exactly 3 weights (dock/qed/sa), got {weights!r}")
        obj = build_objective_tensor(init_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
        if obj.numel() == 0:
            raise RuntimeError("Cannot build scalarization bounds from an empty init set.")
        mins = obj.min(dim=0).values.detach().cpu().numpy().astype(np.float64, copy=False)
        maxs = obj.max(dim=0).values.detach().cpu().numpy().astype(np.float64, copy=False)
        return cls(
            dock_sign=float(dock_sign),
            qed_sign=float(qed_sign),
            sa_sign=float(sa_sign),
            weights=np.asarray(weights, dtype=np.float64),
            mins=mins,
            maxs=maxs,
        )

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        dock_obj = self.dock_sign * pd.to_numeric(work["pred_dock_mean"], errors="coerce").to_numpy(dtype=np.float64)
        qed_obj = self.qed_sign * pd.to_numeric(work["qed"], errors="coerce").to_numpy(dtype=np.float64)
        sa_obj = self.sa_sign * pd.to_numeric(work["sa_score"], errors="coerce").to_numpy(dtype=np.float64)
        vals = np.column_stack([dock_obj, qed_obj, sa_obj])
        span = np.maximum(self.maxs - self.mins, 1.0e-8)
        norm = np.clip((vals - self.mins[None, :]) / span[None, :], 0.0, 1.0)
        score = norm @ self.weights
        work["obj_dock"] = vals[:, 0]
        work["obj_qed"] = vals[:, 1]
        work["obj_sa"] = vals[:, 2]
        work["norm_obj_dock"] = norm[:, 0]
        work["norm_obj_qed"] = norm[:, 1]
        work["norm_obj_sa"] = norm[:, 2]
        work["score_scalar"] = score
        return work


class SurrogateScalarScorer:
    def __init__(
        self,
        *,
        scalarizer: ObjectiveScalarizer,
        model: torch.nn.Module,
        mean: float,
        std: float,
        device: torch.device,
        global_vocab: list[str],
        surrogate_kind: str,
        model_node_dim: int,
        model_edge_dim: int,
        model_atom_extra_dim: int,
        model_bond_extra_dim: int,
        model_fp_dim: int,
        model_fp_radius: int,
        surrogate_meta: dict,
        pred_batch_size: int,
        candidate_3d_cfg: dict,
    ):
        self.scalarizer = scalarizer
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        self.device = device
        self.global_vocab = list(global_vocab)
        self.surrogate_kind = str(surrogate_kind)
        self.model_node_dim = int(model_node_dim)
        self.model_edge_dim = int(model_edge_dim)
        self.model_atom_extra_dim = int(model_atom_extra_dim)
        self.model_bond_extra_dim = int(model_bond_extra_dim)
        self.model_fp_dim = int(model_fp_dim)
        self.model_fp_radius = int(model_fp_radius)
        self.surrogate_meta = dict(surrogate_meta)
        self.pred_batch_size = int(pred_batch_size)
        self.candidate_3d_cfg = dict(candidate_3d_cfg)
        self._cache: dict[str, dict] = {}

    def score_smiles(self, smiles_list: list[str]) -> pd.DataFrame:
        canonical_seen: set[str] = set()
        ordered_unique: list[str] = []
        for smi in smiles_list:
            try:
                canon = canonicalize_smiles_noh(str(smi).strip())
            except Exception:
                continue
            if not canon or canon in canonical_seen:
                continue
            canonical_seen.add(canon)
            ordered_unique.append(canon)
        if not ordered_unique:
            return pd.DataFrame()
        missing = [smi for smi in ordered_unique if smi not in self._cache]
        if missing:
            candidate_frames: list[pd.DataFrame] = []
            for smi in missing:
                try:
                    candidate_frames.append(_candidate_frame([smi]))
                except Exception:
                    continue
            if not candidate_frames:
                return pd.DataFrame([dict(self._cache[s]) for s in ordered_unique if s in self._cache])

            def _score_candidate_block(block_df: pd.DataFrame) -> pd.DataFrame:
                return _score_with_predictions_only(
                    block_df,
                    model=self.model,
                    mean=self.mean,
                    std=self.std,
                    device=self.device,
                    ligand_vocab=self.global_vocab,
                    surrogate_kind=self.surrogate_kind,
                    model_node_dim=self.model_node_dim,
                    model_edge_dim=self.model_edge_dim,
                    model_atom_extra_dim=self.model_atom_extra_dim,
                    model_bond_extra_dim=self.model_bond_extra_dim,
                    model_fp_dim=self.model_fp_dim,
                    model_fp_radius=self.model_fp_radius,
                    surrogate_meta=self.surrogate_meta,
                    pred_batch_size=self.pred_batch_size,
                    candidate_3d_cfg=self.candidate_3d_cfg,
                )

            def _is_recoverable_scoring_error(exc: Exception) -> bool:
                msg = str(exc)
                return (
                    "Length of values" in msg
                    or "expected a non-empty list of Tensors" in msg
                )

            chunk_size = max(1, min(512, int(self.pred_batch_size) * 8))
            scored_blocks: list[pd.DataFrame] = []
            fallback_single_n = 0
            for offset in range(0, len(candidate_frames), chunk_size):
                block_df = pd.concat(candidate_frames[offset : offset + chunk_size], ignore_index=True)
                try:
                    block_scored = _score_candidate_block(block_df)
                    if not block_scored.empty:
                        scored_blocks.append(block_scored)
                except (ValueError, RuntimeError) as exc:
                    if not _is_recoverable_scoring_error(exc):
                        raise
                    _log(
                        "population baseline scorer fallback | "
                        f"chunk_start={offset} chunk_n={block_df.shape[0]} reason={str(exc)}"
                    )
                    for single_df in candidate_frames[offset : offset + chunk_size]:
                        try:
                            single_scored = _score_candidate_block(single_df)
                            if single_scored.empty:
                                continue
                            scored_blocks.append(single_scored)
                            fallback_single_n += 1
                        except (ValueError, RuntimeError) as single_exc:
                            if not _is_recoverable_scoring_error(single_exc):
                                raise
                            continue
            if not scored_blocks:
                return pd.DataFrame([dict(self._cache[s]) for s in ordered_unique if s in self._cache])
            scored = pd.concat(scored_blocks, ignore_index=True)
            if fallback_single_n > 0:
                _log(f"population baseline scorer fallback done | scored_single_n={fallback_single_n}")
            scored = self.scalarizer.transform_frame(scored)
            for row in scored.to_dict("records"):
                self._cache[str(row["smiles_canonical"])] = row
        rows = [dict(self._cache[s]) for s in ordered_unique if s in self._cache]
        return pd.DataFrame(rows)

    def score_map(self, smiles_list: list[str]) -> dict[str, float]:
        scored = self.score_smiles(smiles_list)
        if scored.empty:
            return {}
        return {str(row["smiles_canonical"]): float(row["score_scalar"]) for row in scored.to_dict("records")}


def _seed_population(current_smiles: list[str], population_size: int, seed: int) -> list[str]:
    if not current_smiles:
        raise RuntimeError("Cannot seed a population from an empty observed set.")
    rng = random.Random(int(seed))
    if len(current_smiles) >= int(population_size):
        return rng.sample(current_smiles, int(population_size))
    return [rng.choice(current_smiles) for _ in range(int(population_size))]


def _top_unique_by_score(scored: pd.DataFrame, *, exclude: set[str], top_k: int) -> pd.DataFrame:
    work = scored.loc[~scored["smiles_canonical"].isin(list(exclude))].copy()
    if work.empty:
        return work
    work = work.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).drop_duplicates(subset=["smiles_canonical"])
    return work.head(int(top_k)).reset_index(drop=True)


def _predicted_objective_matrix(scored: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    pred_dock = pd.to_numeric(scored.get("pred_dock_mean"), errors="coerce")
    qed = pd.to_numeric(scored.get("qed"), errors="coerce")
    sa = pd.to_numeric(scored.get("sa_score", scored.get("sa")), errors="coerce")
    valid_mask = pred_dock.notna() & qed.notna() & sa.notna()
    work = scored.loc[valid_mask].copy().reset_index(drop=True)
    if work.empty:
        return work, np.zeros((0, 3), dtype=np.float64)
    points = np.column_stack(
        [
            -pd.to_numeric(work["pred_dock_mean"], errors="raise").to_numpy(dtype=np.float64),
            pd.to_numeric(work["qed"], errors="raise").to_numpy(dtype=np.float64),
            -pd.to_numeric(work.get("sa_score", work.get("sa")), errors="raise").to_numpy(dtype=np.float64),
        ]
    )
    return work, points


def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise RuntimeError(f"Expected a 2D objective array, got shape={points.shape}.")
    n = int(points.shape[0])
    keep = np.ones(n, dtype=bool)
    for idx in range(n):
        if not keep[idx]:
            continue
        dominates_idx = np.all(points >= points[idx], axis=1) & np.any(points > points[idx], axis=1)
        if np.any(dominates_idx):
            keep[idx] = False
    return keep


def _select_predicted_frontier_batch(
    scored: pd.DataFrame,
    *,
    exclude: set[str],
    top_k: int,
    fallback_scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if int(top_k) <= 0:
        raise RuntimeError(f"top_k must be positive, got {top_k}.")
    work = scored.loc[~scored["smiles_canonical"].isin(list(exclude))].copy()
    if work.empty:
        return work
    work = work.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
    work_valid, points = _predicted_objective_matrix(work)
    if work_valid.empty:
        raise RuntimeError("Predicted frontier selection received no rows with valid predicted objectives.")
    front = work_valid.loc[_non_dominated_mask(points)].copy()
    front = front.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).reset_index(drop=True)
    if int(front.shape[0]) >= int(top_k):
        return front.head(int(top_k)).reset_index(drop=True)
    selected = front.copy()
    seen = set(selected["smiles_canonical"].astype(str).tolist())
    remaining = work.loc[~work["smiles_canonical"].isin(list(seen))].copy()
    if not remaining.empty and int(selected.shape[0]) < int(top_k):
        supplement = remaining.head(int(top_k) - int(selected.shape[0])).copy().reset_index(drop=True)
        selected = pd.concat([selected, supplement], ignore_index=True)
        seen.update(supplement["smiles_canonical"].astype(str).tolist())
    if fallback_scored is not None and int(selected.shape[0]) < int(top_k):
        fallback = fallback_scored.loc[
            ~fallback_scored["smiles_canonical"].isin(list(exclude | seen))
        ].copy()
        if not fallback.empty:
            fallback = (
                fallback.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True])
                .drop_duplicates(subset=["smiles_canonical"])
                .head(int(top_k) - int(selected.shape[0]))
                .reset_index(drop=True)
            )
            if not fallback.empty:
                selected = pd.concat([selected, fallback], ignore_index=True)
    return selected.reset_index(drop=True)


def _build_next_population(
    snapshot_smiles: list[str],
    *,
    observed_smiles: list[str],
    population_size: int,
    seed: int,
) -> list[str]:
    ordered = list(dict.fromkeys(str(s) for s in snapshot_smiles if str(s)))
    if len(ordered) >= int(population_size):
        return ordered[: int(population_size)]
    pool = ordered + [str(s) for s in observed_smiles if str(s) and str(s) not in set(ordered)]
    if len(pool) < int(population_size):
        raise RuntimeError(
            f"Cannot build next population: pool={len(pool)} population_size={int(population_size)}"
        )
    rng = random.Random(int(seed))
    fill = list(pool[len(ordered) :])
    rng.shuffle(fill)
    return ordered + fill[: int(population_size) - len(ordered)]


def _prepare_closed_loop_start_population(
    *,
    method: str,
    start_population: list[str],
    observed_smiles: list[str],
    population_cfg: dict,
    population_size: int,
    seed: int,
    scorer: SurrogateScalarScorer | None = None,
) -> list[str]:
    if method != "janus":
        if len(start_population) >= int(population_size):
            return list(start_population[: int(population_size)])
        return _build_next_population(
            list(start_population),
            observed_smiles=observed_smiles,
            population_size=int(population_size),
            seed=int(seed),
        )
    max_smiles_len = int(population_cfg.get("max_smiles_len", 140))
    prepared: list[str] = []
    for smi in list(start_population) + list(observed_smiles):
        try:
            canon = canonicalize_smiles_noh(str(smi).strip())
        except Exception:
            continue
        if not canon or len(canon) > int(max_smiles_len):
            continue
        prepared.append(canon)
    if not prepared:
        raise RuntimeError("JANUS closed-loop could not build any valid start molecules after filtering.")
    if scorer is not None:
        score_map = scorer.score_map(prepared)
        prepared = [s for s in prepared if s in score_map]
        if not prepared:
            raise RuntimeError("JANUS closed-loop found no start molecules that the surrogate can score.")
    if len(prepared) >= int(population_size):
        return prepared[: int(population_size)]
    rng = random.Random(int(seed))
    out = list(prepared)
    while len(out) < int(population_size):
        out.append(rng.choice(prepared))
    return out


def _write_population_search_artifacts(
    work_dir: Path,
    *,
    history_rows: list[dict],
    generation_rows: list[dict],
    progress: dict,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(work_dir / "search_history.csv", index=False)
    if generation_rows:
        pd.DataFrame(generation_rows).to_csv(work_dir / "generation_population_scores.csv", index=False)
    progress_payload = dict(progress)
    progress_payload["history_rows"] = int(len(history_rows))
    progress_payload["generation_rows"] = int(len(generation_rows))
    progress_payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (work_dir / "search_progress.json").write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")


def _get_good_bad_smiles(fitness: list[float], population: list[str], generation_size: int) -> tuple[list[str], list[str]]:
    fit = np.asarray(fitness, dtype=np.float64)
    idx_sort = fit.argsort()[::-1]
    keep_ratio = 0.2
    keep_idx = max(1, int(len(idx_sort) * keep_ratio))
    try:
        f50_val = fit[idx_sort[keep_idx]]
        f25_val = np.asarray([x for x in (fit - f50_val) if x < 0.0], dtype=np.float64) + f50_val
        if f25_val.size == 0:
            raise RuntimeError("No lower-quartile fitness values available.")
        f25_val = f25_val[np.argsort(f25_val)[::-1][0]]
        denom = max(1.0e-8, float(f50_val - f25_val))
        prob = 1.0 / (3.0 ** ((f50_val - fit) / denom) + 1.0)
        prob = prob / prob.sum()
        to_keep = np.random.choice(generation_size, keep_idx, p=prob)
        to_replace = [i for i in range(generation_size) if i not in set(int(x) for x in to_keep)][0 : generation_size - len(to_keep)]
        keep_smiles = [population[int(i)] for i in to_keep]
        replace_smiles = [population[int(i)] for i in to_replace]
        best_smi = population[int(idx_sort[0])]
        if best_smi not in keep_smiles:
            keep_smiles.append(best_smi)
            if best_smi in replace_smiles:
                replace_smiles.remove(best_smi)
        if not keep_smiles or not replace_smiles:
            raise RuntimeError("Population split failed.")
        return keep_smiles, replace_smiles
    except Exception:
        keep_smiles = [population[int(i)] for i in idx_sort[:keep_idx]]
        replace_smiles = [population[int(i)] for i in idx_sort[keep_idx:]]
        return keep_smiles, replace_smiles


def _graph_mol_ok(mol: Chem.Mol, average_size: int, size_stdev: int) -> bool:
    try:
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if test_mol is None:
            return False
        target_size = float(size_stdev) * np.random.randn() + float(average_size)
        return 5 < mol.GetNumAtoms() < target_size
    except Exception:
        return False


def _graph_ring_ok(mol: Chem.Mol) -> bool:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True
    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max(len(j) for j in cycle_list) if cycle_list else 0
    macro_cycle = max_cycle_length > 6
    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))
    return (not ring_allene) and (not macro_cycle) and (not double_bond_in_small_ring)


def _graph_cut(mol: Chem.Mol):
    patt = Chem.MolFromSmarts("[*]-;!@[*]")
    if not mol.HasSubstructMatch(patt):
        return None
    bis = random.choice(mol.GetSubstructMatches(patt))
    bond_idx = mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()
    fragments_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True, dummyLabels=[(1, 1)])
    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True)
    except Exception:
        return None


def _graph_cut_ring(mol: Chem.Mol):
    for _ in range(10):
        if random.random() < 0.5:
            patt = Chem.MolFromSmarts("[R]@[R]@[R]@[R]")
            if not mol.HasSubstructMatch(patt):
                return None
            bis = random.choice(mol.GetSubstructMatches(patt))
            bond_pairs = ((bis[0], bis[1]), (bis[2], bis[3]))
        else:
            patt = Chem.MolFromSmarts("[R]@[R;!D2]@[R]")
            if not mol.HasSubstructMatch(patt):
                return None
            bis = random.choice(mol.GetSubstructMatches(patt))
            bond_pairs = ((bis[0], bis[1]), (bis[1], bis[2]))
        bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bond_pairs]
        fragments_mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True, dummyLabels=[(1, 1), (1, 1)])
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except Exception:
            return None
        if len(fragments) == 2:
            return fragments
    return None


def _graph_crossover_non_ring(parent_a: Chem.Mol, parent_b: Chem.Mol, average_size: int, size_stdev: int):
    for _ in range(10):
        fragments_a = _graph_cut(parent_a)
        fragments_b = _graph_cut(parent_b)
        if fragments_a is None or fragments_b is None:
            return None
        rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
        new_mols: list[Chem.Mol] = []
        for fa in fragments_a:
            for fb in fragments_b:
                trials = rxn.RunReactants((fa, fb))
                for prod in trials:
                    mol = prod[0]
                    if _graph_mol_ok(mol, average_size, size_stdev):
                        new_mols.append(mol)
        if new_mols:
            return random.choice(new_mols)
    return None


def _graph_crossover_ring(parent_a: Chem.Mol, parent_b: Chem.Mol, average_size: int, size_stdev: int):
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_a.HasSubstructMatch(ring_smarts) and not parent_b.HasSubstructMatch(ring_smarts):
        return None
    rxn_smarts1 = ["[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]", "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]"]
    rxn_smarts2 = ["([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]", "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]"]
    for _ in range(10):
        fragments_a = _graph_cut_ring(parent_a)
        fragments_b = _graph_cut_ring(parent_b)
        if fragments_a is None or fragments_b is None:
            return None
        new_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            for fa in fragments_a:
                for fb in fragments_b:
                    new_trial.extend(rxn1.RunReactants((fa, fb)))
        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for item in new_trial:
                mol = item[0]
                if _graph_mol_ok(mol, average_size, size_stdev):
                    new_mols.extend(rxn2.RunReactants((mol,)))
        valid = []
        for item in new_mols:
            mol = item[0]
            if _graph_mol_ok(mol, average_size, size_stdev) and _graph_ring_ok(mol):
                valid.append(mol)
        if valid:
            return random.choice(valid)
    return None


def _graph_crossover(parent_a: Chem.Mol, parent_b: Chem.Mol, average_size: int, size_stdev: int):
    parent_smiles = {Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)}
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except Exception:
        pass
    for _ in range(10):
        if random.random() <= 0.5:
            new_mol = _graph_crossover_non_ring(parent_a, parent_b, average_size, size_stdev)
        else:
            new_mol = _graph_crossover_ring(parent_a, parent_b, average_size, size_stdev)
        if new_mol is None:
            continue
        new_smiles = Chem.MolToSmiles(new_mol)
        if new_smiles not in parent_smiles:
            return new_mol
    return None


def _graph_mutate(mol: Chem.Mol, mutation_rate: float, average_size: int, size_stdev: int):
    if random.random() > float(mutation_rate):
        return mol
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rxn_smarts_list = [
        lambda m: "[*:1]~[*:2]>>[*:1]{}[*:2]".format(random.choice(["C", "N", "O", "S"])),
        lambda m: random.choice(
            [
                "[*:1]!-[*:2]>>[*:1]-[*:2]",
                "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
                "[*:1]#[*:2]>>[*:1]=[*:2]",
                "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
            ]
        ),
        lambda m: "[*:1]@[*:2]>>([*:1].[*:2])",
        lambda m: random.choice(
            [
                "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
                "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
                "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
            ]
        ),
        lambda m: random.choice(
            [
                "[*:1]~[D1]>>[*:1]",
                "[*:1]~[D2]~[*:2]>>[*:1]-[*:2]",
                "[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]",
            ]
        ),
        lambda m: "[X:1]>>[Y:1]".replace(
            "X",
            random.choice(["#6", "#7", "#8", "#9", "#16", "#17", "#35"]),
        ).replace(
            "Y",
            random.choice(["#6", "#7", "#8", "#9", "#16", "#17", "#35"]),
        ),
        lambda m: random.choice(
            [
                "[*;!H0:1]>>[*:1]-{}".format(random.choice(["C", "N", "O", "F", "S", "Cl", "Br"])),
                "[*;!H0;!H1:1]>>[*:1]={}".format(random.choice(["C", "N", "O"])),
                "[*;H3:1]>>[*:1]#{}".format(random.choice(["C", "N"])),
            ]
        ),
    ]
    probs = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(10):
        rxn_smarts = np.random.choice([fn(mol) for fn in rxn_smarts_list], p=probs)
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        try:
            trials = rxn.RunReactants((mol,))
        except Exception:
            continue
        new_mols = []
        for item in trials:
            cand = item[0]
            if _graph_mol_ok(cand, average_size, size_stdev) and _graph_ring_ok(cand):
                new_mols.append(cand)
        if new_mols:
            return random.choice(new_mols)
    return None


def _normalize_fitness(scores: list[float]) -> list[float]:
    pos = [max(float(s), 0.0) for s in scores]
    ssum = float(sum(pos))
    if ssum <= 0.0:
        return [1.0 / len(pos)] * len(pos)
    return [x / ssum for x in pos]


def _make_mating_pool(population: list[str], fitness: list[float], pool_size: int, rng: random.Random) -> list[str]:
    cumulative: list[float] = []
    acc = 0.0
    for w in fitness:
        acc += float(w)
        cumulative.append(acc)
    out: list[str] = []
    for _ in range(int(pool_size)):
        r = rng.random() * acc
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        out.append(population[lo])
    return out


def _run_janus_search(
    *,
    scorer: SurrogateScalarScorer,
    start_population: list[str],
    seed: int,
    generations: int,
    generation_size: int,
    max_smiles_len: int,
    num_sample_frags: int,
    explr_num_random_samples: int,
    explr_num_mutations: int,
    crossover_num_random_samples: int,
    exploit_num_random_samples: int,
    exploit_num_mutations: int,
    top_mols: int,
    num_exchanges: int,
    max_filter_rounds: int,
    score_flush_size: int,
    work_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def custom_filter(smi: str) -> bool:
        try:
            canon = canonicalize_smiles_noh(str(smi).strip())
        except Exception:
            return False
        return bool(canon) and len(canon) <= int(max_smiles_len)

    start_population = [canonicalize_smiles_noh(str(s)) for s in start_population if custom_filter(str(s))]
    if len(start_population) < int(generation_size):
        raise RuntimeError(f"JANUS start_population smaller than generation_size: {len(start_population)} < {generation_size}")
    frag_alphabet: list[str] = []
    for smi in start_population:
        frag_alphabet.extend(_janus_form_fragments(smi))
    frag_alphabet = list(dict.fromkeys(str(x) for x in frag_alphabet if str(x)))
    init_scores = scorer.score_map(start_population)
    population = sorted(
        [s for s in start_population if s in init_scores],
        key=lambda smi: float(init_scores[smi]),
        reverse=True,
    )[: int(generation_size)]
    if len(population) < int(generation_size):
        raise RuntimeError("JANUS failed to build a full initial population with valid scores.")
    fitness = [float(init_scores[s]) for s in population]
    smiles_collector: dict[str, list[float | int]] = {}
    for smi, score in zip(population, fitness):
        smiles_collector[smi] = [float(score), 1]
    rng = random.Random(int(seed))
    history_rows: list[dict] = []
    generation_rows: list[dict] = []
    score_flush_size = max(1, int(score_flush_size))
    _write_population_search_artifacts(
        work_dir,
        history_rows=history_rows,
        generation_rows=generation_rows,
        progress={
            "method": "janus",
            "status": "running",
            "generation_completed": 0,
            "generations_total": int(generations),
            "collector_n": int(len(smiles_collector)),
            "population_n": int(len(population)),
        },
    )

    def score_valid_smiles(smiles_list: list[str]) -> tuple[list[str], dict[str, float]]:
        unique = list(dict.fromkeys(str(x) for x in smiles_list if str(x)))
        if not unique:
            return [], {}
        scored = scorer.score_smiles(unique)
        if scored.empty:
            return [], {}
        score_map = {str(r["smiles_canonical"]): float(r["score_scalar"]) for r in scored.to_dict("records")}
        valid = [s for s in unique if s in score_map]
        return valid, score_map

    def mutate_smi_list(smi_list: list[str], *, space: str) -> list[str]:
        if space == "local":
            num_random_samples = int(exploit_num_random_samples)
            num_mutations = int(exploit_num_mutations)
        elif space == "explore":
            num_random_samples = int(explr_num_random_samples)
            num_mutations = int(explr_num_mutations)
        else:
            raise ValueError(space)
        out: list[str] = []
        for smi in smi_list * max(1, int(num_random_samples)):
            out.extend(
                _janus_mutate_smiles(
                    smi,
                    alphabet=frag_alphabet,
                    num_random_samples=1,
                    num_mutations=int(num_mutations),
                    num_sample_frags=int(num_sample_frags),
                )
            )
        return [s for s in dict.fromkeys(str(x) for x in out if custom_filter(str(x)))]

    def crossover_smi_list(joined: list[str]) -> list[str]:
        out: list[str] = []
        for item in joined:
            out.extend(_janus_crossover_smiles(item, crossover_num_random_samples=int(crossover_num_random_samples)))
        return [s for s in dict.fromkeys(str(x) for x in out if custom_filter(str(x)))]

    for gen_idx in range(int(generations)):
        keep_smiles, replace_smiles = _get_good_bad_smiles(fitness, population, int(generation_size))
        replace_smiles = list(dict.fromkeys(replace_smiles))
        explore_target = int(generation_size) - len(keep_smiles)
        explr_smiles: list[str] = []
        explr_seen: set[str] = set()
        explr_valid: list[str] = []
        explr_valid_seen: set[str] = set()
        explr_score_map: dict[str, float] = {}
        pending_explr: list[str] = []
        rounds = 0
        while len(explr_valid) < explore_target:
            rounds += 1
            if rounds > int(max_filter_rounds):
                raise RuntimeError(
                    f"JANUS exploration stalled at generation {gen_idx}: "
                    f"target_valid={explore_target} valid={len(explr_valid)} raw={len(explr_smiles)}"
                )
            mut_explr = mutate_smi_list(replace_smiles[0 : max(1, len(replace_smiles) // 2)], space="explore")
            smiles_join = [f"{item}xxx{rng.choice(keep_smiles)}" for item in replace_smiles[len(replace_smiles) // 2 :]]
            cross_explr = crossover_smi_list(smiles_join)
            for smi in list(dict.fromkeys(mut_explr + cross_explr)):
                if smi in smiles_collector or smi in explr_seen:
                    continue
                explr_seen.add(smi)
                explr_smiles.append(smi)
                pending_explr.append(smi)
            if pending_explr and (
                len(pending_explr) >= score_flush_size
                or len(explr_smiles) >= max(explore_target, score_flush_size)
                or rounds == int(max_filter_rounds)
            ):
                batch_valid, batch_scores = score_valid_smiles(pending_explr)
                pending_explr = []
                if batch_scores:
                    explr_score_map.update(batch_scores)
                if batch_valid:
                    for smi in batch_valid:
                        if smi in explr_valid_seen:
                            continue
                        explr_valid_seen.add(smi)
                        explr_valid.append(smi)
        replaced_pop = rng.sample(explr_valid, explore_target)
        population = keep_smiles + replaced_pop
        score_map = {}
        for smi in population:
            if smi in smiles_collector:
                score_map[smi] = float(smiles_collector[smi][0])
            elif smi in explr_score_map:
                score_map[smi] = float(explr_score_map[smi])
        fitness = []
        for smi in population:
            if smi not in score_map:
                raise RuntimeError(f"JANUS scorer dropped a population molecule: {smi}")
            score = float(score_map[smi])
            fitness.append(score)
            if smi in smiles_collector:
                smiles_collector[smi][1] = int(smiles_collector[smi][1]) + 1
            else:
                smiles_collector[smi] = [score, 1]
        pop_sort_idx = np.argsort(np.asarray(fitness, dtype=np.float64))[::-1]
        population_sort = [population[int(i)] for i in pop_sort_idx]
        history_rows.append({"generation": int(gen_idx), "phase": "explore", "top_smiles": population_sort[0], "top_score": float(fitness[int(pop_sort_idx[0])]), "population_n": len(population_sort)})

        exploit_smiles: list[str] = []
        exploit_seen: set[str] = set()
        exploit_valid: list[str] = []
        exploit_valid_seen: set[str] = set()
        local_score_map: dict[str, float] = {}
        pending_exploit: list[str] = []
        rounds = 0
        while len(exploit_valid) < int(generation_size):
            rounds += 1
            if rounds > int(max_filter_rounds):
                raise RuntimeError(
                    f"JANUS exploitation stalled at generation {gen_idx}: "
                    f"target_valid={generation_size} valid={len(exploit_valid)} raw={len(exploit_smiles)}"
                )
            local_seed = population_sort[0 : max(1, int(top_mols))]
            for smi in mutate_smi_list(local_seed, space="local"):
                if smi in smiles_collector or smi in exploit_seen:
                    continue
                exploit_seen.add(smi)
                exploit_smiles.append(smi)
                pending_exploit.append(smi)
            if pending_exploit and (
                len(pending_exploit) >= score_flush_size
                or len(exploit_smiles) >= max(int(generation_size), score_flush_size)
                or rounds == int(max_filter_rounds)
            ):
                batch_valid, batch_scores = score_valid_smiles(pending_exploit)
                pending_exploit = []
                if batch_scores:
                    local_score_map.update(batch_scores)
                if batch_valid:
                    for smi in batch_valid:
                        if smi in exploit_valid_seen:
                            continue
                        exploit_valid_seen.add(smi)
                        exploit_valid.append(smi)
        fp_scores = _janus_fp_scores(exploit_valid, population_sort[0])
        fp_order = np.argsort(np.asarray(fp_scores, dtype=np.float64))[::-1][: int(generation_size)]
        population_loc = [exploit_valid[int(i)] for i in fp_order]
        fitness_loc = []
        for smi in population_loc:
            if smi not in local_score_map:
                raise RuntimeError(f"JANUS scorer dropped a local-search molecule: {smi}")
            score = float(local_score_map[smi])
            fitness_loc.append(score)
            smiles_collector[smi] = [score, 1]
        loc_sort_idx = np.argsort(np.asarray(fitness_loc, dtype=np.float64))[::-1]
        population_loc_sort = [population_loc[int(i)] for i in loc_sort_idx]
        fitness_loc_sort = [float(fitness_loc[int(i)]) for i in loc_sort_idx]
        history_rows.append({"generation": int(gen_idx), "phase": "local", "top_smiles": population_loc_sort[0], "top_score": float(fitness_loc_sort[0]), "population_n": len(population_loc_sort)})

        worst_indices = list(np.argsort(np.asarray(fitness, dtype=np.float64))[::-1][-int(num_exchanges) :])
        for repl_idx, pop_idx in enumerate(worst_indices):
            if repl_idx >= len(population_loc_sort):
                break
            population[int(pop_idx)] = population_loc_sort[repl_idx]
            fitness[int(pop_idx)] = fitness_loc_sort[repl_idx]
        generation_frame = scorer.score_smiles(population)
        if generation_frame.empty:
            raise RuntimeError(f"JANUS generation {gen_idx + 1} produced no scorable population snapshot.")
        generation_frame = generation_frame.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
        generation_frame["generation"] = int(gen_idx + 1)
        generation_rows.extend(generation_frame.to_dict("records"))
        best_row = generation_frame.iloc[0]
        _log(
            f"janus generation {gen_idx + 1}/{int(generations)} | "
            f"explore_valid={len(explr_valid)}/{explore_target} raw={len(explr_smiles)} "
            f"local_valid={len(exploit_valid)}/{int(generation_size)} raw={len(exploit_smiles)} "
            f"collector_n={len(smiles_collector)} best_score={float(best_row['score_scalar']):.4f} "
            f"best_smiles={str(best_row['smiles_canonical'])}"
        )
        _write_population_search_artifacts(
            work_dir,
            history_rows=history_rows,
            generation_rows=generation_rows,
            progress={
                "method": "janus",
                "status": "running",
                "generation_completed": int(gen_idx + 1),
                "generations_total": int(generations),
                "collector_n": int(len(smiles_collector)),
                "population_n": int(generation_frame.shape[0]),
                "explore_target": int(explore_target),
                "explore_valid_n": int(len(explr_valid)),
                "explore_raw_n": int(len(explr_smiles)),
                "local_target": int(generation_size),
                "local_valid_n": int(len(exploit_valid)),
                "local_raw_n": int(len(exploit_smiles)),
                "best_smiles": str(best_row["smiles_canonical"]),
                "best_score": float(best_row["score_scalar"]),
            },
        )

    final_scored = scorer.score_smiles(list(smiles_collector.keys()))
    final_scored = final_scored.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).reset_index(drop=True)
    final_scored["search_method"] = "janus"
    final_scored["visit_count"] = final_scored["smiles_canonical"].map(lambda s: int(smiles_collector.get(str(s), [0.0, 0])[1]))
    final_scored.to_csv(work_dir / "janus_pool.csv", index=False)
    _write_population_search_artifacts(
        work_dir,
        history_rows=history_rows,
        generation_rows=generation_rows,
        progress={
            "method": "janus",
            "status": "completed",
            "generation_completed": int(generations),
            "generations_total": int(generations),
            "collector_n": int(len(smiles_collector)),
            "population_n": int(final_scored.shape[0]),
            "best_smiles": str(final_scored.iloc[0]["smiles_canonical"]) if not final_scored.empty else "",
            "best_score": float(final_scored.iloc[0]["score_scalar"]) if not final_scored.empty else float('nan'),
        },
    )
    return final_scored, pd.DataFrame(history_rows), pd.DataFrame(generation_rows)


def _run_graph_ga_search(
    *,
    scorer: SurrogateScalarScorer,
    start_population: list[str],
    seed: int,
    generations: int,
    population_size: int,
    mating_pool_size: int,
    mutation_rate: float,
    prune_population: bool,
    max_attempts_per_child: int,
    work_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_start = [s for s in start_population if Chem.MolFromSmiles(str(s)) is not None]
    if len(valid_start) < int(population_size):
        raise RuntimeError(f"Graph-GA start population smaller than popsize: {len(valid_start)} < {population_size}")
    rng = random.Random(int(seed))
    population = [rng.choice(valid_start) for _ in range(int(population_size))]
    atom_counts = [Chem.MolFromSmiles(s).GetNumAtoms() for s in valid_start if Chem.MolFromSmiles(s) is not None]
    if not atom_counts:
        raise RuntimeError("Graph-GA could not infer atom-count statistics from the start population.")
    average_size = int(round(float(np.mean(atom_counts))))
    size_stdev = max(1, int(round(float(np.std(atom_counts)))))
    scored = scorer.score_smiles(population)
    if scored.empty:
        raise RuntimeError("Graph-GA initial scoring returned no valid molecules.")
    score_map = {str(r["smiles_canonical"]): float(r["score_scalar"]) for r in scored.to_dict("records")}
    pop_tuples = [(score_map[s], s) for s in population if s in score_map]
    pop_tuples.sort(key=lambda t: t[0], reverse=True)
    if prune_population:
        deduped: list[tuple[float, str]] = []
        seen: set[str] = set()
        for item in pop_tuples:
            if item[1] in seen:
                continue
            seen.add(item[1])
            deduped.append(item)
        pop_tuples = deduped
    pop_tuples = pop_tuples[: int(population_size)]
    population = [t[1] for t in pop_tuples]
    scores = [float(t[0]) for t in pop_tuples]
    visited: set[str] = set(population)
    history_rows: list[dict] = [{"generation": 0, "top_smiles": population[0], "top_score": float(scores[0]), "population_n": len(population)}]
    generation_rows: list[dict] = []
    generation_frame = scorer.score_smiles(population)
    if generation_frame.empty:
        raise RuntimeError("Graph-GA initial population snapshot returned no scorable molecules.")
    generation_frame = generation_frame.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
    generation_frame["generation"] = 0
    generation_rows.extend(generation_frame.to_dict("records"))
    _write_population_search_artifacts(
        work_dir,
        history_rows=history_rows,
        generation_rows=generation_rows,
        progress={
            "method": "graph_ga",
            "status": "running",
            "generation_completed": 0,
            "generations_total": int(generations),
            "population_n": int(generation_frame.shape[0]),
            "visited_n": int(len(visited)),
            "best_smiles": str(generation_frame.iloc[0]["smiles_canonical"]),
            "best_score": float(generation_frame.iloc[0]["score_scalar"]),
        },
    )

    def make_child(parent_a: str, parent_b: str) -> str | None:
        mol_a = Chem.MolFromSmiles(parent_a)
        mol_b = Chem.MolFromSmiles(parent_b)
        if mol_a is None or mol_b is None:
            return None
        for _ in range(int(max_attempts_per_child)):
            child = _graph_crossover(mol_a, mol_b, average_size, size_stdev)
            if child is None:
                continue
            child = _graph_mutate(child, float(mutation_rate), average_size, size_stdev)
            if child is None:
                continue
            smi = Chem.MolToSmiles(child, isomericSmiles=False, canonical=True)
            if smi:
                return smi
        return None

    for gen_idx in range(1, int(generations) + 1):
        fitness = _normalize_fitness(scores)
        mating_pool = _make_mating_pool(population, fitness, int(mating_pool_size), rng)
        children: list[str] = []
        for _ in range(int(population_size)):
            child = make_child(rng.choice(mating_pool), rng.choice(mating_pool))
            if child is not None:
                children.append(child)
        while len(children) < int(population_size):
            children.append(rng.choice(mating_pool))
        child_scored = scorer.score_smiles(children)
        child_score_map = {str(r["smiles_canonical"]): float(r["score_scalar"]) for r in child_scored.to_dict("records")}
        merged = [(float(score), smi) for score, smi in zip(scores, population)]
        merged.extend((child_score_map[s], s) for s in children if s in child_score_map)
        merged.sort(key=lambda t: t[0], reverse=True)
        if prune_population:
            deduped: list[tuple[float, str]] = []
            seen: set[str] = set()
            for item in merged:
                if item[1] in seen:
                    continue
                seen.add(item[1])
                deduped.append(item)
            merged = deduped
        merged = merged[: int(population_size)]
        population = [t[1] for t in merged]
        scores = [float(t[0]) for t in merged]
        visited.update(population)
        history_rows.append({"generation": int(gen_idx), "top_smiles": population[0], "top_score": float(scores[0]), "population_n": len(population)})
        generation_frame = scorer.score_smiles(population)
        if generation_frame.empty:
            raise RuntimeError(f"Graph-GA generation {gen_idx} produced no scorable population snapshot.")
        generation_frame = generation_frame.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).drop_duplicates(subset=["smiles_canonical"]).reset_index(drop=True)
        generation_frame["generation"] = int(gen_idx)
        generation_rows.extend(generation_frame.to_dict("records"))
        best_row = generation_frame.iloc[0]
        _log(
            f"graph_ga generation {gen_idx}/{int(generations)} | "
            f"children={len(children)} visited_n={len(visited)} pop_n={generation_frame.shape[0]} "
            f"best_score={float(best_row['score_scalar']):.4f} best_smiles={str(best_row['smiles_canonical'])}"
        )
        _write_population_search_artifacts(
            work_dir,
            history_rows=history_rows,
            generation_rows=generation_rows,
            progress={
                "method": "graph_ga",
                "status": "running",
                "generation_completed": int(gen_idx),
                "generations_total": int(generations),
                "population_n": int(generation_frame.shape[0]),
                "visited_n": int(len(visited)),
                "children_n": int(len(children)),
                "best_smiles": str(best_row["smiles_canonical"]),
                "best_score": float(best_row["score_scalar"]),
            },
        )

    final_scored = scorer.score_smiles(list(visited))
    final_scored = final_scored.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).reset_index(drop=True)
    final_scored["search_method"] = "graph_ga"
    final_scored.to_csv(work_dir / "graph_ga_pool.csv", index=False)
    _write_population_search_artifacts(
        work_dir,
        history_rows=history_rows,
        generation_rows=generation_rows,
        progress={
            "method": "graph_ga",
            "status": "completed",
            "generation_completed": int(generations),
            "generations_total": int(generations),
            "population_n": int(final_scored.shape[0]),
            "visited_n": int(len(visited)),
            "best_smiles": str(final_scored.iloc[0]["smiles_canonical"]) if not final_scored.empty else "",
            "best_score": float(final_scored.iloc[0]["score_scalar"]) if not final_scored.empty else float('nan'),
        },
    )
    return final_scored, pd.DataFrame(history_rows), pd.DataFrame(generation_rows)


def _run_population_search(
    *,
    method: str,
    scorer: SurrogateScalarScorer,
    start_population: list[str],
    seed: int,
    baseline_cfg: dict,
    work_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = str(method).strip().lower()
    if method == "janus":
        return _run_janus_search(
            scorer=scorer,
            start_population=start_population,
            seed=seed,
            generations=int(baseline_cfg.get("generations", 20)),
            generation_size=int(baseline_cfg.get("population_size", 64)),
            max_smiles_len=int(baseline_cfg.get("max_smiles_len", 140)),
            num_sample_frags=int(baseline_cfg.get("num_sample_frags", 8)),
            explr_num_random_samples=int(baseline_cfg.get("explr_num_random_samples", 4)),
            explr_num_mutations=int(baseline_cfg.get("explr_num_mutations", 4)),
            crossover_num_random_samples=int(baseline_cfg.get("crossover_num_random_samples", 1)),
            exploit_num_random_samples=int(baseline_cfg.get("exploit_num_random_samples", 4)),
            exploit_num_mutations=int(baseline_cfg.get("exploit_num_mutations", 2)),
            top_mols=int(baseline_cfg.get("top_mols", 4)),
            num_exchanges=int(baseline_cfg.get("num_exchanges", 4)),
            max_filter_rounds=int(baseline_cfg.get("max_filter_rounds", 120)),
            score_flush_size=int(baseline_cfg.get("score_flush_size", 256)),
            work_dir=work_dir,
        )
    if method == "graph_ga":
        return _run_graph_ga_search(
            scorer=scorer,
            start_population=start_population,
            seed=seed,
            generations=int(baseline_cfg.get("generations", 20)),
            population_size=int(baseline_cfg.get("population_size", 128)),
            mating_pool_size=int(baseline_cfg.get("mating_pool_size", max(128, 2 * int(baseline_cfg.get("population_size", 128))))),
            mutation_rate=float(baseline_cfg.get("mutation_rate", 1.0)),
            prune_population=bool(baseline_cfg.get("prune_population", True)),
            max_attempts_per_child=int(baseline_cfg.get("max_attempts_per_child", 24)),
            work_dir=work_dir,
        )
    raise RuntimeError(f"Unsupported population baseline method: {method}")


def _run_smoke_test(method: str) -> int:
    seed = 42
    work_dir = REPO / "runs" / f"{method}_smoke_test"
    work_dir.mkdir(parents=True, exist_ok=True)
    base_smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC", "CCS", "CCCl"]
    candidate_df = _candidate_frame(base_smiles)
    candidate_df["pred_dock_mean"] = np.linspace(-8.0, -5.0, num=int(candidate_df.shape[0]), dtype=np.float64)
    scalarizer = ObjectiveScalarizer(
        dock_sign=-1.0,
        qed_sign=1.0,
        sa_sign=-1.0,
        weights=np.asarray([0.7, 0.1, 0.2], dtype=np.float64),
        mins=np.asarray([5.0, 0.0, -6.0], dtype=np.float64),
        maxs=np.asarray([9.0, 1.0, -1.0], dtype=np.float64),
    )

    class _FakeScorer:
        def __init__(self, frame: pd.DataFrame, scalarizer_obj: ObjectiveScalarizer) -> None:
            self.scalarizer = scalarizer_obj
            self.rows = {
                str(row["smiles_canonical"]): dict(row)
                for row in scalarizer_obj.transform_frame(frame).to_dict("records")
            }

        @staticmethod
        def _canon(smi: str) -> str | None:
            mol = Chem.MolFromSmiles(str(smi).strip())
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

        def _materialize_missing(self, smiles_list: list[str]) -> None:
            missing: list[str] = []
            seen: set[str] = set()
            for item in smiles_list:
                canon = self._canon(item)
                if canon is None or canon in self.rows or canon in seen:
                    continue
                seen.add(canon)
                missing.append(canon)
            if not missing:
                return
            rows: list[dict] = []
            for smi in missing:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                atom_n = float(mol.GetNumAtoms())
                rows.append(
                    {
                        "smiles": smi,
                        "smiles_canonical": smi,
                        "qed": float(np.clip(0.2 + 0.05 * atom_n, 0.0, 1.0)),
                        "sa_score": float(1.0 + 0.35 * atom_n),
                        "pred_dock_mean": float(-4.0 - 0.15 * atom_n),
                    }
                )
            frame = pd.DataFrame(rows)
            if frame.empty:
                return
            for row in self.scalarizer.transform_frame(frame).to_dict("records"):
                self.rows[str(row["smiles_canonical"])] = dict(row)

        def score_smiles(self, smiles_list: list[str]) -> pd.DataFrame:
            self._materialize_missing(smiles_list)
            rows = []
            for item in smiles_list:
                canon = self._canon(item)
                if canon is None or canon not in self.rows:
                    continue
                rows.append(dict(self.rows[canon]))
            return pd.DataFrame(rows)

        def score_map(self, smiles_list: list[str]) -> dict[str, float]:
            return {
                str(row["smiles_canonical"]): float(row["score_scalar"])
                for row in self.score_smiles(smiles_list).to_dict("records")
            }

    scorer = _FakeScorer(candidate_df, scalarizer)
    if method == "janus":
        baseline_cfg = {
            "generations": 3,
            "population_size": 6,
            "max_smiles_len": 80,
            "num_sample_frags": 4,
            "explr_num_random_samples": 2,
            "explr_num_mutations": 2,
            "crossover_num_random_samples": 1,
            "exploit_num_random_samples": 2,
            "exploit_num_mutations": 1,
            "top_mols": 2,
            "num_exchanges": 2,
            "max_filter_rounds": 40,
        }
    elif method == "graph_ga":
        baseline_cfg = {
            "generations": 3,
            "population_size": 8,
            "mating_pool_size": 12,
            "mutation_rate": 1.0,
            "prune_population": True,
            "max_attempts_per_child": 12,
        }
    else:
        raise RuntimeError(f"Unsupported smoke-test method: {method}")
    scored, history, generation_populations = _run_population_search(
        method=method,
        scorer=scorer,  # type: ignore[arg-type]
        start_population=base_smiles,
        seed=seed,
        baseline_cfg=baseline_cfg,
        work_dir=work_dir,
    )
    if scored.empty:
        raise RuntimeError(f"{method} smoke test produced an empty scored pool.")
    if history.empty:
        raise RuntimeError(f"{method} smoke test produced an empty search history.")
    if generation_populations.empty:
        raise RuntimeError(f"{method} smoke test produced an empty generation population trace.")
    _log(
        f"{method} smoke test passed | "
        f"pool_n={scored.shape[0]} "
        f"history_n={history.shape[0]} "
        f"best_smiles={str(scored.iloc[0]['smiles_canonical'])} "
        f"best_score={float(scored.iloc[0]['score_scalar']):.4f}"
    )
    return 0


def _fit_closed_loop_surrogate(
    *,
    method: str,
    iter_idx: int,
    out_root: Path,
    iter_dir: Path,
    dataset_root: Path,
    holdout_df: pd.DataFrame,
    dock_valid_max: float | None,
    surrogate_cfg: dict,
    model_cfg: dict,
    candidate_3d_cfg: dict,
    backbone: str,
    device: torch.device,
    resolved_seed: int,
    init_encoder_ckpt: str | None,
    shared_ligand3d_cache: Path,
    global_vocab: list[str],
    scalarizer: ObjectiveScalarizer,
) -> tuple[SurrogateScalarScorer, dict, Path]:
    clear_processed(str(dataset_root))
    ckpt_path = out_root / "checkpoints" / f"surrogate_iter{int(iter_idx):03d}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    train_started = time.perf_counter()
    labeled_now = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
    _log(f"{method} iter {iter_idx} train_surrogate start | labeled_n={labeled_now.shape[0]}")
    train_surrogate_from_scratch(
        root=str(dataset_root),
        save_path=str(ckpt_path),
        epochs=int(surrogate_cfg.get("retrain_epochs", 100)),
        batch_size=int(surrogate_cfg.get("retrain_batch_size", 48)),
        lr=float(surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3))),
        encoder_lr=float(surrogate_cfg.get("retrain_encoder_lr", surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3)))),
        weight_decay=float(surrogate_cfg.get("retrain_weight_decay", 1e-6)),
        hidden_dim=int(surrogate_cfg.get("retrain_hidden_dim", 48)),
        num_layers=int(surrogate_cfg.get("retrain_num_layers", 3)),
        dropout=float(surrogate_cfg.get("retrain_dropout", 0.0)),
        use_edge_attr=bool(surrogate_cfg.get("retrain_use_edge_attr", True)),
        use_ligand_mask=bool(surrogate_cfg.get("retrain_use_ligand_mask", True)),
        standardize=bool(surrogate_cfg.get("retrain_standardize", False)),
        fp_dim=int(surrogate_cfg.get("retrain_fp_dim", 0)),
        fp_radius=int(surrogate_cfg.get("retrain_fp_radius", 2)),
        eval_samples=int(surrogate_cfg.get("retrain_eval_samples", 8)),
        scheduler=str(surrogate_cfg.get("retrain_scheduler", "cosine")),
        warmup_epochs=int(surrogate_cfg.get("retrain_warmup_epochs", 15)),
        min_lr=float(surrogate_cfg.get("retrain_min_lr", 1e-5)),
        early_stop_patience=int(surrogate_cfg.get("retrain_early_stop_patience", 100)),
        early_stop_min_delta=float(surrogate_cfg.get("retrain_early_stop_min_delta", 0.0)),
        device=device,
        dock_valid_max=dock_valid_max,
        backbone=backbone,
        uncertainty_mode=str(surrogate_cfg.get("retrain_uncertainty_mode", "nig" if backbone == "tensornet" else "gaussian")),
        torchmd_cfg=_section(model_cfg, "torchmd"),
        ensemble_heads=int(surrogate_cfg.get("retrain_ensemble_heads", 1 if backbone == "tensornet" else 1)),
        freeze_backbone=bool(surrogate_cfg.get("retrain_freeze_backbone", False)),
        pretrained_encoder_ckpt=str(init_encoder_ckpt) if init_encoder_ckpt else None,
        ensemble_scheme=str(surrogate_cfg.get("retrain_ensemble_scheme", "full")),
        ensemble_bootstrap=bool(surrogate_cfg.get("retrain_ensemble_bootstrap", True)),
        random_seed=int(resolved_seed),
        ligand3d_cache_dir=str(shared_ligand3d_cache),
        ligand_vocab_override=global_vocab,
        confgen_max_attempts=int(candidate_3d_cfg.get("max_attempts", 3)),
        confgen_seed=int(candidate_3d_cfg.get("seed", resolved_seed)),
        confgen_num_confs=int(candidate_3d_cfg.get("num_confs", 4)),
        confgen_max_opt_iters=int(candidate_3d_cfg.get("max_opt_iters", 100)),
        confgen_optimize=bool(candidate_3d_cfg.get("optimize", True)),
        confgen_prefer_mmff=bool(candidate_3d_cfg.get("prefer_mmff", False)),
        ligand3d_num_workers=int(candidate_3d_cfg.get("workers", 8)),
        ligand3d_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
        train_log_csv=str(iter_dir / "surrogate_train_log.csv"),
        train_summary_json=str(iter_dir / "surrogate_summary.json"),
    )
    (
        model,
        mean,
        std,
        model_node_dim,
        model_edge_dim,
        model_fp_dim,
        model_fp_radius,
        model_atom_extra_dim,
        model_bond_extra_dim,
        _model_pocket_graph,
        surrogate_kind,
        surrogate_meta,
    ) = load_surrogate(str(ckpt_path), device)
    _log(
        f"{method} iter {iter_idx} train_surrogate done | "
        f"elapsed={time.perf_counter() - train_started:.1f}s ckpt={ckpt_path.name}"
    )
    holdout_eval = _evaluate_holdout_surrogate(
        holdout_df=holdout_df,
        model=model,
        mean=mean,
        std=std,
        device=device,
        global_vocab=global_vocab,
        surrogate_kind=surrogate_kind,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=int(surrogate_cfg.get("pred_batch_size", 32)),
        candidate_3d_cfg=candidate_3d_cfg,
        iter_idx=int(iter_idx),
        method_dir=iter_dir,
    )
    scorer = SurrogateScalarScorer(
        scalarizer=scalarizer,
        model=model,
        mean=mean,
        std=std,
        device=device,
        global_vocab=global_vocab,
        surrogate_kind=surrogate_kind,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=int(surrogate_cfg.get("pred_batch_size", 32)),
        candidate_3d_cfg=candidate_3d_cfg,
    )
    return scorer, holdout_eval, ckpt_path


def _run_population_baseline_closed_loop(
    *,
    method: str,
    out_root: Path,
    dataset_root: Path,
    labeled_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    init_stats: dict,
    fixed_ref_point: np.ndarray,
    acq_ref_point: np.ndarray,
    batch_size: int,
    population_cfg: dict,
    scalarizer: ObjectiveScalarizer,
    dock_sign: float,
    qed_sign: float,
    sa_sign: float,
    dock_valid_max: float | None,
    weights: np.ndarray,
    online_oracle_kwargs: dict,
    oracle_cfg: dict,
    objective_cfg: dict,
    predictor_valid_ratio: float,
    predictor_valid_seed: int,
    surrogate_cfg: dict,
    model_cfg: dict,
    candidate_3d_cfg: dict,
    backbone: str,
    device: torch.device,
    resolved_seed: int,
    resolved_rounds: int,
    global_vocab: list[str],
    init_encoder_ckpt: str | None,
    shared_ligand3d_cache: Path,
) -> None:
    population_size = int(population_cfg.get("population_size", batch_size))
    target_batch_n = int(population_cfg.get("oracle_topk_per_generation", batch_size))
    if target_batch_n <= 0:
        target_batch_n = int(batch_size)
    search_generations = max(1, int(population_cfg.get("generations", 1)))
    search_root = out_root / "search"
    search_root.mkdir(parents=True, exist_ok=True)
    iter_rows: list[dict] = [{
        "iter": 0,
        "execution_mode": "closed_loop_generation_oracle",
        "labeled_n": int(labeled_df.shape[0]),
        "labeled_n_final": int(labeled_df.shape[0]),
        "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
        "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
        "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
        "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
        "best_observed_dock": float(labeled_df["dock_score"].min()),
        "best_observed_dock_final": float(labeled_df["dock_score"].min()),
        "hv": float(compute_hypervolume(
            build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
            ref_point=fixed_ref_point,
        )),
        "hv_final": float(compute_hypervolume(
            build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
            ref_point=fixed_ref_point,
        )),
        "hv_gain_total": 0.0,
        "acq_ref_point_dock": float(acq_ref_point[0]),
        "acq_ref_point_qed": float(acq_ref_point[1]),
        "acq_ref_point_sa": float(acq_ref_point[2]),
        "holdout_pred_n": 0,
        "holdout_pred_rmse": np.nan,
        "holdout_pred_mae": np.nan,
        "holdout_pred_spearman": np.nan,
        "holdout_pred_kendall": np.nan,
        "holdout_pred_calibration_gap": np.nan,
        "start_pool_n": int(population_size),
        "search_generations": int(search_generations),
        "search_pool_n": 0,
        "candidate_pool_n": 0,
        "selected_n": 0,
        "selected_true_dock_mean": np.nan,
        "selected_true_dock_best": np.nan,
        "selected_oracle_hv_gain": 0.0,
        "oracle_attempted": 0,
        "oracle_docked": 0,
        "oracle_failed": 0,
        "selected_pred_n": 0,
        "selected_pred_skipped": 0,
        "selected_pred_invalid_dock": 0,
        "selected_pred_invalid_pred": 0,
        "selected_pred_rmse": np.nan,
        "selected_pred_mae": np.nan,
        "selected_pred_spearman": np.nan,
        "selected_pred_kendall": np.nan,
        "utility_weight_dock": float(weights[0]),
        "utility_weight_qed": float(weights[1]),
        "utility_weight_sa": float(weights[2]),
    }]
    hv_init = float(iter_rows[0]["hv"])
    hv_running = hv_init
    selected_batches_oracle: list[pd.DataFrame] = []
    holdout_rows: list[dict] = []
    search_history_rows: list[dict] = []
    generation_rows: list[dict] = []
    current_population = _prepare_closed_loop_start_population(
        method=method,
        start_population=_seed_population(
            labeled_df["smiles_canonical"].astype(str).tolist(),
            population_size=population_size,
            seed=_method_seed(int(resolved_seed), method, 0),
        ),
        observed_smiles=labeled_df["smiles_canonical"].astype(str).tolist(),
        population_cfg=population_cfg,
        population_size=population_size,
        seed=_method_seed(int(resolved_seed), method, 0),
    )

    for iter_idx in range(1, int(resolved_rounds) + 1):
        iter_dir = out_root / f"iter{int(iter_idx):03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        scorer, holdout_eval, _ckpt_path = _fit_closed_loop_surrogate(
            method=method,
            iter_idx=int(iter_idx),
            out_root=out_root,
            iter_dir=iter_dir,
            dataset_root=dataset_root,
            holdout_df=holdout_df,
            dock_valid_max=dock_valid_max,
            surrogate_cfg=surrogate_cfg,
            model_cfg=model_cfg,
            candidate_3d_cfg=candidate_3d_cfg,
            backbone=backbone,
            device=device,
            resolved_seed=int(resolved_seed),
            init_encoder_ckpt=init_encoder_ckpt,
            shared_ligand3d_cache=shared_ligand3d_cache,
            global_vocab=global_vocab,
            scalarizer=scalarizer,
        )
        holdout_row = dict(holdout_eval)
        holdout_row["iter"] = int(iter_idx)
        holdout_rows.append(holdout_row)
        pd.DataFrame(holdout_rows).to_csv(out_root / "holdout_eval_metrics.csv", index=False)

        search_dir = iter_dir / "search"
        search_dir.mkdir(parents=True, exist_ok=True)
        search_cfg = dict(population_cfg)
        search_cfg["generations"] = int(search_generations)
        search_started = time.perf_counter()
        current_population = _prepare_closed_loop_start_population(
            method=method,
            start_population=current_population,
            observed_smiles=_load_observed_df(dataset_root, dock_valid_max=dock_valid_max)["smiles_canonical"].astype(str).tolist(),
            population_cfg=population_cfg,
            population_size=population_size,
            seed=_method_seed(int(resolved_seed), method, int(iter_idx)),
            scorer=scorer,
        )
        search_pool, search_history, generation_populations = _run_population_search(
            method=method,
            scorer=scorer,
            start_population=current_population,
            seed=_method_seed(int(resolved_seed), method, int(iter_idx)),
            baseline_cfg=search_cfg,
            work_dir=search_dir,
        )
        if search_history.empty:
            raise RuntimeError(f"{method} iter {iter_idx} returned an empty search history.")
        if generation_populations.empty:
            raise RuntimeError(f"{method} iter {iter_idx} returned an empty generation population trace.")
        search_history_step = search_history.copy()
        search_history_step["outer_iter"] = int(iter_idx)
        generation_rows_step = generation_populations.copy()
        generation_rows_step["outer_iter"] = int(iter_idx)
        search_history_rows.extend(search_history_step.to_dict("records"))
        generation_rows.extend(generation_rows_step.to_dict("records"))
        pd.DataFrame(search_history_rows).to_csv(search_root / "search_history.csv", index=False)
        pd.DataFrame(generation_rows).to_csv(search_root / "generation_population_scores.csv", index=False)

        generation_series = pd.to_numeric(generation_populations["generation"], errors="raise").astype(int)
        generation_idx = int(generation_series.max())
        generation_df = generation_populations.loc[generation_series == generation_idx].copy().reset_index(drop=True)
        if generation_df.empty:
            raise RuntimeError(f"{method} iter {iter_idx} produced an empty latest generation snapshot.")
        dataset_smiles_df = pd.read_csv(dataset_root / "smiles.csv")
        if "smiles_canonical" in dataset_smiles_df.columns:
            existing_smiles = set(dataset_smiles_df["smiles_canonical"].astype(str).tolist())
        else:
            existing_smiles = set(dataset_smiles_df["smiles"].astype(str).tolist())
        selected_iter = _select_predicted_frontier_batch(
            generation_df,
            exclude=existing_smiles,
            top_k=int(target_batch_n),
            fallback_scored=search_pool,
        )
        if int(selected_iter.shape[0]) < int(target_batch_n):
            raise RuntimeError(
                f"{method} iter {iter_idx} produced too few novel oracle candidates: "
                f"selected={selected_iter.shape[0]} required={target_batch_n}"
            )
        selected_iter["selected_iter"] = int(iter_idx)
        selected_iter.to_csv(iter_dir / "selected_batch.csv", index=False)
        _log(
            f"{method} iter {iter_idx} search done | elapsed={time.perf_counter() - search_started:.1f}s "
            f"current_pop_n={len(current_population)} search_pool_n={search_pool.shape[0]} "
            f"candidate_pool_n={generation_df.shape[0]} selected_n={selected_iter.shape[0]}"
        )
        selected_smiles = selected_iter["smiles_canonical"].astype(str).tolist()
        selected_pred_added = [None if pd.isna(v) else float(v) for v in selected_iter["pred_dock_mean"].tolist()]
        added, new_ids, kept_indices = append_candidates_to_smiles_csv(
            str(dataset_root / "smiles.csv"),
            selected_smiles,
            id_prefix=str(oracle_cfg.get("oracle_id_prefix", "GEN")),
            sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
            sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
            split_ratio=None,
            added_iter=int(iter_idx),
            molecule_origin=method,
        )
        if added != len(new_ids) or added != len(kept_indices):
            raise RuntimeError(
                f"Append bookkeeping mismatch for method {method} iter {iter_idx}: "
                f"added={added} new_ids={len(new_ids)} kept={len(kept_indices)}"
            )
        if kept_indices != list(range(selected_iter.shape[0])):
            selected_iter = selected_iter.iloc[kept_indices].copy().reset_index(drop=True)
            selected_pred_added = [selected_pred_added[i] for i in kept_indices]
        if int(selected_iter.shape[0]) < int(target_batch_n):
            raise RuntimeError(
                f"{method} iter {iter_idx} append dedup reduced oracle batch below target: "
                f"selected={selected_iter.shape[0]} required={target_batch_n}"
            )
        dock_stats = run_oracle_docking(str(dataset_root), new_ids, **online_oracle_kwargs)
        _fill_extra_metrics_in_csv(dataset_root, objective_cfg, prior_state=None, force=False)
        new_train_n, new_valid_n = _assign_new_rows_random_predictor_split(
            dataset_root,
            new_ids,
            valid_ratio=float(predictor_valid_ratio),
            seed=int(predictor_valid_seed + iter_idx),
        )
        _log(
            f"{method} iter {iter_idx} predictor_split_new_rows | "
            f"train={new_train_n} valid={new_valid_n} holdout_excluded=1"
        )
        merged_iter = _export_selected_batch_with_oracle(
            dataset_root,
            selected_iter,
            new_ids,
            iter_dir / "selected_batch_with_oracle.csv",
        )
        selected_batches_oracle.append(merged_iter)

        smiles_all = pd.read_csv(dataset_root / "smiles.csv")
        smiles_all["ligand_id"] = smiles_all["ligand_id"].astype(str)
        new_df = smiles_all.loc[smiles_all["ligand_id"].isin([str(x) for x in new_ids])].copy().reset_index(drop=True)
        valid_mask = []
        for val in new_df.get("dock_score", pd.Series([], dtype=float)).tolist():
            try:
                num = float(val)
            except Exception:
                num = None
            valid_mask.append(num is not None and (dock_valid_max is None or num <= dock_valid_max))
        valid_new_df = new_df.loc[np.asarray(valid_mask, dtype=bool)].copy().reset_index(drop=True) if len(valid_mask) else new_df.head(0).copy()
        oracle_eval = evaluate_oracle_accuracy_from_csv(
            str(dataset_root / "smiles.csv"),
            new_ids,
            selected_pred_added,
            dock_valid_max=dock_valid_max,
        )
        labeled_before = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
        labeled_before = labeled_before.loc[~labeled_before["ligand_id"].astype(str).isin([str(x) for x in new_ids])].copy().reset_index(drop=True)
        labeled_after = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
        hv_after = compute_hypervolume(
            build_objective_tensor(labeled_after, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
            ref_point=fixed_ref_point,
        )
        if valid_new_df.empty:
            selected_true_dock_mean = np.nan
            selected_true_dock_best = np.nan
            selected_oracle_hv_gain = 0.0
        else:
            dock_vals = pd.to_numeric(valid_new_df["dock_score"], errors="coerce")
            selected_true_dock_mean = float(dock_vals.mean())
            selected_true_dock_best = float(dock_vals.min())
            selected_oracle_hv_gain = float(_batch_hv_gain(labeled_before, valid_new_df, dock_sign, qed_sign, sa_sign, fixed_ref_point))
        iter_rows.append({
            "iter": int(iter_idx),
            "execution_mode": "closed_loop_generation_oracle",
            "labeled_n": int(labeled_before.shape[0]),
            "labeled_n_final": int(labeled_after.shape[0]),
            "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
            "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
            "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
            "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
            "best_observed_dock": float(labeled_before["dock_score"].min()),
            "best_observed_dock_final": float(labeled_after["dock_score"].min()),
            "hv": float(hv_running),
            "hv_final": float(hv_after),
            "hv_gain_total": float(hv_after - hv_init),
            "acq_ref_point_dock": float(acq_ref_point[0]),
            "acq_ref_point_qed": float(acq_ref_point[1]),
            "acq_ref_point_sa": float(acq_ref_point[2]),
            "holdout_pred_n": int(holdout_eval.get("n", 0)),
            "holdout_pred_rmse": float(holdout_eval.get("rmse", np.nan)),
            "holdout_pred_mae": float(holdout_eval.get("mae", np.nan)),
            "holdout_pred_spearman": float(holdout_eval.get("spearman", np.nan)),
            "holdout_pred_kendall": float(holdout_eval.get("kendall", np.nan)),
            "holdout_pred_calibration_gap": float(holdout_eval.get("calibration_gap", np.nan)),
            "start_pool_n": int(len(current_population)),
            "search_generations": int(search_generations),
            "search_pool_n": int(search_pool.shape[0]),
            "candidate_pool_n": int(generation_df.shape[0]),
            "selected_n": int(selected_iter.shape[0]),
            "selected_true_dock_mean": selected_true_dock_mean,
            "selected_true_dock_best": selected_true_dock_best,
            "selected_oracle_hv_gain": selected_oracle_hv_gain,
            "oracle_attempted": int(dock_stats.get("attempted", 0)),
            "oracle_docked": int(dock_stats.get("docked", 0)),
            "oracle_failed": int(dock_stats.get("failed", 0)),
            "selected_pred_n": int(oracle_eval.get("matched", 0)),
            "selected_pred_skipped": int(oracle_eval.get("skipped", 0)),
            "selected_pred_invalid_dock": int(oracle_eval.get("invalid_dock", 0)),
            "selected_pred_invalid_pred": int(oracle_eval.get("invalid_pred", 0)),
            "selected_pred_rmse": float(oracle_eval.get("rmse", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_mae": float(oracle_eval.get("mae", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_spearman": float(oracle_eval.get("spearman", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_kendall": float(oracle_eval.get("kendall", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "utility_weight_dock": float(weights[0]),
            "utility_weight_qed": float(weights[1]),
            "utility_weight_sa": float(weights[2]),
        })
        pd.DataFrame(iter_rows).to_csv(out_root / "iter_metrics.csv", index=False)
        if selected_batches_oracle:
            pd.concat(selected_batches_oracle, ignore_index=True).to_csv(out_root / "selected_batches_with_oracle.csv", index=False)
        hv_running = float(hv_after)
        current_population = _prepare_closed_loop_start_population(
            method=method,
            start_population=_build_next_population(
                generation_df["smiles_canonical"].astype(str).tolist(),
                observed_smiles=labeled_after["smiles_canonical"].astype(str).tolist(),
                population_size=population_size,
                seed=_method_seed(int(resolved_seed), method, int(iter_idx) + 1000),
            ),
            observed_smiles=labeled_after["smiles_canonical"].astype(str).tolist(),
            population_cfg=population_cfg,
            population_size=population_size,
            seed=_method_seed(int(resolved_seed), method, int(iter_idx) + 1000),
            scorer=scorer,
        )

    final_row = iter_rows[-1]
    _log(
        f"{method} done | hv_init={iter_rows[0]['hv']:.4f} hv_final={final_row['hv_final']:.4f} "
        f"best_observed_dock_final={final_row['best_observed_dock_final']:.4f}"
    )


def main(method_override: str | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run JANUS or Graph-GA baseline under the main 8UN4 protocol.")
    ap.add_argument("--method", choices=["janus", "graph_ga"], default=None)
    ap.add_argument("--config", default="config/surrogate/config.yaml")
    ap.add_argument("--mobo-config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--reuse-init-from", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-library-csv", default=None)
    ap.add_argument("--init-library-smiles-col", default=None)
    ap.add_argument("--init-encoder-ckpt", default=None)
    ap.add_argument("--init-training-dataset-root", default=None)
    ap.add_argument("--init-ligand-vocab-file", default=None)
    ap.add_argument("--init-library-max-rows", type=int, default=None)
    ap.add_argument("--butina-cutoff", type=float, default=None)
    ap.add_argument("--fp-radius", type=int, default=None)
    ap.add_argument("--fp-bits", type=int, default=None)
    ap.add_argument("--init-train-valid-total", type=int, default=None)
    ap.add_argument("--holdout-test-total", type=int, default=None)
    ap.add_argument("--selection-seed", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    method = str(method_override or args.method or "").strip().lower()
    if not method:
        raise RuntimeError("Population baseline requires --method or a method-specific wrapper.")
    if method not in {"janus", "graph_ga"}:
        raise RuntimeError(f"Unsupported population baseline method: {method}")
    if args.smoke_test:
        return _run_smoke_test(method)

    default_mobo_cfg = REPO / "config" / "mobo" / f"{method}_baseline.yaml"
    mobo_cfg_path = args.mobo_config or str(default_mobo_cfg)
    cfg = load_config(args.config)
    mobo_cfg = load_config(mobo_cfg_path)
    mobo_validate_cfg = dict(mobo_cfg)
    mobo_validate_cfg.pop("population_baseline", None)
    _validate_mobo_config(mobo_validate_cfg)
    mobo_paths_cfg = _require_dict(mobo_cfg.get("paths"), "mobo.paths")
    mobo_init_cfg = _require_dict(mobo_cfg.get("init_selection"), "mobo.init_selection")
    mobo_experiment_cfg = _require_dict(mobo_cfg.get("experiment"), "mobo.experiment")
    population_cfg = _require_dict(mobo_cfg.get("population_baseline"), "mobo.population_baseline")

    cfg_method = str(population_cfg.get("method", method)).strip().lower()
    if cfg_method != method:
        raise RuntimeError(f"Method/config mismatch: cli={method} config={cfg_method}")

    resolved = argparse.Namespace(
        init_library_csv=args.init_library_csv if args.init_library_csv is not None else mobo_paths_cfg.get("init_library_csv"),
        init_library_smiles_col=_resolve_cli_or_config(args.init_library_smiles_col, mobo_paths_cfg.get("init_library_smiles_col"), "paths.init_library_smiles_col"),
        init_encoder_ckpt=args.init_encoder_ckpt if args.init_encoder_ckpt is not None else mobo_paths_cfg.get("init_encoder_ckpt"),
        init_training_dataset_root=_resolve_cli_or_config(args.init_training_dataset_root, mobo_paths_cfg.get("init_training_dataset_root"), "paths.init_training_dataset_root"),
        init_ligand_vocab_file=args.init_ligand_vocab_file if args.init_ligand_vocab_file is not None else mobo_paths_cfg.get("init_ligand_vocab_file"),
        output_dir=_resolve_cli_or_config(args.output_dir, mobo_paths_cfg.get("output_dir"), "paths.output_dir"),
        init_library_max_rows=int(_resolve_cli_or_config(args.init_library_max_rows, mobo_init_cfg.get("library_max_rows", 0), "init_selection.library_max_rows")),
        butina_cutoff=float(_resolve_cli_or_config(args.butina_cutoff, mobo_init_cfg.get("butina_cutoff", 0.6), "init_selection.butina_cutoff")),
        fp_radius=int(_resolve_cli_or_config(args.fp_radius, mobo_init_cfg.get("fp_radius", 2), "init_selection.fp_radius")),
        fp_bits=int(_resolve_cli_or_config(args.fp_bits, mobo_init_cfg.get("fp_bits", 2048), "init_selection.fp_bits")),
        init_train_valid_total=int(_resolve_cli_or_config(args.init_train_valid_total, mobo_init_cfg.get("init_train_valid_total", 480), "init_selection.init_train_valid_total")),
        holdout_test_total=int(_resolve_cli_or_config(args.holdout_test_total, mobo_init_cfg.get("holdout_test_total", 120), "init_selection.holdout_test_total")),
        selection_seed=int(_resolve_cli_or_config(args.selection_seed, mobo_init_cfg.get("selection_seed", mobo_experiment_cfg.get("seed")), "init_selection.selection_seed")),
        rounds=int(_resolve_cli_or_config(args.rounds, mobo_experiment_cfg.get("rounds"), "experiment.rounds")),
        seed=int(_resolve_cli_or_config(args.seed, mobo_experiment_cfg.get("seed"), "experiment.seed")),
    )

    general_cfg = _section(cfg, "general")
    dataset_cfg = _section(cfg, "dataset")
    surrogate_cfg = _section(cfg, "surrogate")
    model_cfg = _section(cfg, "model")
    selection_cfg = _section(cfg, "selection")
    objective_cfg = _section(cfg, "objective")
    oracle_cfg = _section(cfg, "oracle")
    run_cfg = _section(cfg, "run")
    candidate_cfg = _section(cfg, "candidate")
    candidate_3d_cfg = dict(_section(candidate_cfg, "candidate_3d"))
    candidate_3d_cfg["seed"] = int(resolved.seed)

    if objective_cfg.get("extra_objectives"):
        raise RuntimeError("Population baselines currently support only dock/QED/SA objectives.")
    if not bool(objective_cfg.get("use_sa", True)):
        raise RuntimeError("Population baselines require dock/QED/SA objectives.")

    weights = [float(x) for x in selection_cfg.get("weights", [0.7, 0.1, 0.2])]
    if len(weights) != 3:
        raise RuntimeError(f"selection.weights must have length 3 for dock/QED/SA, got {weights}")
    batch_size = int(selection_cfg.get("batch_size", 20))
    dock_sign = float(objective_cfg.get("dock_sign", -1.0))
    qed_sign = float(objective_cfg.get("qed_sign", 1.0))
    sa_sign = float(objective_cfg.get("sa_sign", -1.0))
    fixed_ref_point = [float(x) for x in objective_cfg.get("ref_point", [0.0, 0.0, -20.0])]
    dock_valid_max = objective_cfg.get("dock_valid_max", 0.0)
    dock_valid_max = None if dock_valid_max is None else float(dock_valid_max)
    pred_batch_size = int(surrogate_cfg.get("pred_batch_size", surrogate_cfg.get("retrain_batch_size", 48)))
    train_epochs = int(surrogate_cfg.get("retrain_epochs", 80))
    backbone = str(model_cfg.get("surrogate_backbone", "tensornet")).strip().lower()
    if backbone != "tensornet":
        raise RuntimeError(f"Population baselines currently expect surrogate_backbone=tensornet, got {backbone}")

    device = torch.device("cuda" if str(general_cfg.get("device", "auto")).lower() != "cpu" and torch.cuda.is_available() else "cpu")
    gpu_total_gib = None
    gpu_free_gib = None
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_total_gib = round(float(total_b) / float(1024 ** 3), 3)
        gpu_free_gib = round(float(free_b) / float(1024 ** 3), 3)

    out_root = Path(str(resolved.output_dir)) / method
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_root = out_root / "dataset"
    pretrain_split_ratio = _normalize_ratio_config(
        dataset_cfg.get("pretrain_split_ratio", dataset_cfg.get("split_ratio", [0.8, 0.1, 0.1])),
        default=(0.8, 0.1, 0.1),
        expected_len=3,
    )
    pretrain_split_seed = int(dataset_cfg.get("pretrain_split_seed", dataset_cfg.get("split_seed", resolved.seed)))
    pretrain_use_existing_split = bool(dataset_cfg.get("pretrain_use_existing_split", False))
    downstream_pool_ratio = _normalize_ratio_config(
        dataset_cfg.get("downstream_pool_split_ratio", [0.8, 0.2]),
        default=(0.8, 0.2),
        expected_len=2,
    )
    downstream_pool_seed = int(dataset_cfg.get("downstream_pool_split_seed", resolved.seed))
    scaffold_include_chirality = bool(dataset_cfg.get("scaffold_include_chirality", False))
    predictor_valid_ratio = float(surrogate_cfg.get("predictor_valid_ratio", dataset_cfg.get("predictor_valid_ratio", 0.2)))
    predictor_valid_seed = int(surrogate_cfg.get("predictor_valid_seed", dataset_cfg.get("split_seed", resolved.seed)))

    shared_ligand3d_cache = GLOBAL_LIGAND3D_CACHE_DIR
    shared_ligand3d_cache.mkdir(parents=True, exist_ok=True)
    candidate_3d_cfg["cache_dir"] = str(shared_ligand3d_cache)
    oracle_asset_root = _resolve_oracle_asset_root(oracle_cfg)
    online_oracle_kwargs = _build_oracle_call_kwargs(
        run_cfg=run_cfg,
        oracle_cfg=oracle_cfg,
        evaluate_reference=False,
        dock_valid_max=dock_valid_max,
    )
    _log(
        f"{method} oracle config | "
        f"backend={online_oracle_kwargs['docking_backend']} "
        f"exhaustiveness={oracle_cfg.get('oracle_exhaustiveness', 32)} "
        f"pocket_radius={oracle_cfg.get('oracle_pocket_radius', 10.0)} "
        f"vina_cuda_thread={online_oracle_kwargs['vina_cuda_thread']} "
        f"vina_cuda_search_depth={online_oracle_kwargs['vina_cuda_search_depth']} "
        f"unidock_scoring={online_oracle_kwargs['unidock_scoring']} "
        f"unidock_search_mode={online_oracle_kwargs['unidock_search_mode']} "
        f"unidock_num_modes={online_oracle_kwargs['unidock_num_modes']} "
        f"unidock_timeout_sec={online_oracle_kwargs['unidock_timeout_sec']}"
    )

    if args.reuse_init_from:
        init_root = Path(str(args.reuse_init_from)).resolve()
        init_set_path = init_root / "init_set.csv"
        if not init_set_path.exists():
            raise FileNotFoundError(f"reuse-init-from missing init_set.csv: {init_set_path}")
        init_df = pd.read_csv(init_set_path)
        if init_df.empty:
            raise RuntimeError(f"reuse-init-from init_set is empty: {init_set_path}")
        meta_path = init_root / "experiment_meta.json"
        existing_methods: list[str] = []
        existing_meta: dict | None = None
        if not meta_path.exists():
            raise FileNotFoundError(f"reuse-init-from missing experiment_meta.json: {meta_path}")
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(existing_meta, dict):
            raise RuntimeError(f"experiment_meta.json must contain an object: {meta_path}")
        existing_methods = [str(x).strip() for x in existing_meta.get("methods", []) if str(x).strip()]
        global_vocab = _resolve_append_global_vocab(
            out_root=init_root,
            existing_methods=existing_methods,
            init_df=init_df,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
        )
        existing_ckpt = existing_meta.get("init_encoder_ckpt")
        if not existing_ckpt:
            raise RuntimeError(
                f"reuse-init-from requires init_encoder_ckpt in experiment_meta.json: {meta_path}"
            )
        init_encoder_ckpt = Path(str(existing_ckpt)).resolve()
        if not init_encoder_ckpt.exists():
            raise FileNotFoundError(init_encoder_ckpt)
        _log(
            f"{method} reuse_init | from={init_root} init_n={init_df.shape[0]} "
            f"init_encoder_ckpt={init_encoder_ckpt}"
        )
    else:
        default_init_library_candidates = _default_init_library_candidates(resolved.init_training_dataset_root, oracle_asset_root)
        if resolved.init_library_csv:
            init_library_csv = Path(str(resolved.init_library_csv)).resolve()
        else:
            init_library_csv = None
            for candidate in default_init_library_candidates:
                if candidate.exists():
                    init_library_csv = candidate.resolve()
                    break
            if init_library_csv is None:
                raise RuntimeError(f"Failed to resolve init library for {method}.")
        init_library_raw_df, init_library_smiles_col = load_latent_library_csv(
            init_library_csv,
            smiles_col=None if not resolved.init_library_smiles_col else str(resolved.init_library_smiles_col),
        )
        if int(resolved.init_library_max_rows) > 0:
            init_library_raw_df = init_library_raw_df.head(int(resolved.init_library_max_rows)).copy().reset_index(drop=True)
        init_encoder_ckpt = _resolve_or_train_init_encoder_ckpt(
            args=resolved,
            cfg_path=args.config,
            out_root=out_root,
            init_library_df=init_library_raw_df,
            shared_ligand3d_cache=shared_ligand3d_cache,
            candidate_3d_cfg=candidate_3d_cfg,
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            source_csv_path=init_library_csv,
            pretrain_split_ratio=pretrain_split_ratio,
            pretrain_split_seed=pretrain_split_seed,
            pretrain_use_existing_split=pretrain_use_existing_split,
        )
        protocol = _build_or_load_downstream_protocol(
            out_root=out_root,
            downstream_raw_df=init_library_raw_df,
            smiles_col=init_library_smiles_col,
            source_csv_path=init_library_csv,
            downstream_pool_ratio=downstream_pool_ratio,
            downstream_pool_seed=downstream_pool_seed,
            scaffold_include_chirality=scaffold_include_chirality,
            butina_cutoff=float(resolved.butina_cutoff),
            fp_radius=int(resolved.fp_radius),
            fp_bits=int(resolved.fp_bits),
            selection_seed=int(resolved.selection_seed),
            init_train_valid_total=int(resolved.init_train_valid_total),
            holdout_test_total=int(resolved.holdout_test_total),
            predictor_valid_ratio=float(predictor_valid_ratio),
            predictor_valid_seed=int(predictor_valid_seed),
            append_mode=False,
        )
        init_df = protocol["protocol_init_df"].copy().reset_index(drop=True)
        global_vocab = _resolve_init_pretrain_ligand_vocab(
            init_training_dataset_root=resolved.init_training_dataset_root,
            init_ligand_vocab_file=resolved.init_ligand_vocab_file,
            init_library_df=init_library_raw_df,
        )
        _log(
            f"{method} init protocol | "
            f"init_train_valid_total={int((init_df['split_role'].astype(str) == 'init_pool').sum())} "
            f"holdout_test_total={int((init_df['split_role'].astype(str) == 'holdout_test').sum())} "
            "holdout_excluded=1"
        )

    init_stats = _prepare_init_dataset(
        oracle_asset_root,
        dataset_root,
        schema_df=init_df,
        init_df=init_df,
        oracle_cfg=oracle_cfg,
        objective_cfg=objective_cfg,
        run_cfg=run_cfg,
        global_vocab=global_vocab,
    )
    init_observed_df = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
    scalarizer = ObjectiveScalarizer.from_init_observed(
        init_df=init_observed_df,
        dock_sign=dock_sign,
        qed_sign=qed_sign,
        sa_sign=sa_sign,
        weights=weights,
    )
    meta = {
        "method": method,
        "methods": [method],
        "rounds": int(resolved.rounds),
        "batch_size": int(batch_size),
        "population_baseline": dict(population_cfg),
        "pretrain_split_ratio": [float(x) for x in pretrain_split_ratio],
        "pretrain_split_seed": int(pretrain_split_seed),
        "downstream_pool_split_ratio": [float(x) for x in downstream_pool_ratio],
        "downstream_pool_split_seed": int(downstream_pool_seed),
        "scaffold_include_chirality": bool(scaffold_include_chirality),
        "butina_cutoff": float(resolved.butina_cutoff),
        "fp_radius": int(resolved.fp_radius),
        "fp_bits": int(resolved.fp_bits),
        "init_train_valid_total": int(resolved.init_train_valid_total),
        "holdout_test_total": int(resolved.holdout_test_total),
        "selection_seed": int(resolved.selection_seed),
        "predictor_valid_ratio": float(predictor_valid_ratio),
        "predictor_valid_seed": int(predictor_valid_seed),
        "surrogate_backbone": backbone,
        "surrogate_pred_batch_size": int(pred_batch_size),
        "surrogate_train_epochs": int(train_epochs),
        "device": str(device),
        "gpu_total_gib": gpu_total_gib,
        "gpu_free_gib": gpu_free_gib,
        "init_encoder_ckpt": None if init_encoder_ckpt is None else str(Path(init_encoder_ckpt).resolve()),
        "reuse_init_from": str(Path(args.reuse_init_from).resolve()) if args.reuse_init_from else None,
        "oracle_config": {
            "docking_backend": str(online_oracle_kwargs["docking_backend"]),
            "oracle_exhaustiveness": int(oracle_cfg.get("oracle_exhaustiveness", 32)),
            "oracle_pocket_radius": float(oracle_cfg.get("oracle_pocket_radius", 10.0)),
            "vina_cuda_thread": int(online_oracle_kwargs["vina_cuda_thread"]),
            "vina_cuda_search_depth": int(online_oracle_kwargs["vina_cuda_search_depth"]),
            "vina_cuda_rilc_bfgs": int(online_oracle_kwargs["vina_cuda_rilc_bfgs"]),
            "unidock_service_url": online_oracle_kwargs["unidock_service_url"],
            "unidock_scoring": str(online_oracle_kwargs["unidock_scoring"]),
            "unidock_search_mode": str(online_oracle_kwargs["unidock_search_mode"]),
            "unidock_num_modes": int(online_oracle_kwargs["unidock_num_modes"]),
            "unidock_timeout_sec": int(online_oracle_kwargs["unidock_timeout_sec"]),
        },
    }
    (out_root / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(
        f"{method} start | "
        f"out_root={out_root} rounds={resolved.rounds} batch_size={batch_size} "
        f"population_size={int(population_cfg.get('population_size', 0))} "
        f"generations={int(population_cfg.get('generations', 0))} "
        f"pred_batch_size={pred_batch_size} gpu_free_gib={gpu_free_gib} gpu_total_gib={gpu_total_gib}"
    )

    labeled_df = init_observed_df.copy().reset_index(drop=True)
    hv_init = compute_hypervolume(
        build_objective_tensor(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
        ref_point=fixed_ref_point,
    )
    acq_ref_point = _dynamic_ref_point_from_labeled(labeled_df, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign)
    oracle_topk_per_generation = int(population_cfg.get("oracle_topk_per_generation", 0))
    if oracle_topk_per_generation > 0:
        _run_population_baseline_closed_loop(
            method=method,
            out_root=out_root,
            dataset_root=dataset_root,
            labeled_df=labeled_df,
            holdout_df=_load_holdout_df(dataset_root, dock_valid_max=dock_valid_max),
            init_stats=init_stats,
            fixed_ref_point=fixed_ref_point,
            acq_ref_point=acq_ref_point,
            batch_size=int(batch_size),
            population_cfg=population_cfg,
            scalarizer=scalarizer,
            dock_sign=float(dock_sign),
            qed_sign=float(qed_sign),
            sa_sign=float(sa_sign),
            dock_valid_max=dock_valid_max,
            weights=weights,
            online_oracle_kwargs=online_oracle_kwargs,
            oracle_cfg=oracle_cfg,
            objective_cfg=objective_cfg,
            predictor_valid_ratio=float(predictor_valid_ratio),
            predictor_valid_seed=int(predictor_valid_seed),
            surrogate_cfg=surrogate_cfg,
            model_cfg=model_cfg,
            candidate_3d_cfg=candidate_3d_cfg,
            backbone=backbone,
            device=device,
            resolved_seed=int(resolved.seed),
            resolved_rounds=int(resolved.rounds),
            global_vocab=global_vocab,
            init_encoder_ckpt=None if init_encoder_ckpt is None else str(init_encoder_ckpt),
            shared_ligand3d_cache=shared_ligand3d_cache,
        )
        return 0

    clear_processed(str(dataset_root))
    ckpt_path = out_root / "checkpoints" / "surrogate_init.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    train_started = time.perf_counter()
    _log(f"{method} train_surrogate start | labeled_n={labeled_df.shape[0]}")
    train_surrogate_from_scratch(
        root=str(dataset_root),
        save_path=str(ckpt_path),
        epochs=int(train_epochs),
        batch_size=int(surrogate_cfg.get("retrain_batch_size", 48)),
        lr=float(surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3))),
        encoder_lr=float(surrogate_cfg.get("retrain_encoder_lr", surrogate_cfg.get("retrain_head_lr", surrogate_cfg.get("retrain_lr", 1e-3)))),
        weight_decay=float(surrogate_cfg.get("retrain_weight_decay", 1e-6)),
        hidden_dim=int(surrogate_cfg.get("retrain_hidden_dim", 48)),
        num_layers=int(surrogate_cfg.get("retrain_num_layers", 3)),
        dropout=float(surrogate_cfg.get("retrain_dropout", 0.0)),
        use_edge_attr=bool(surrogate_cfg.get("retrain_use_edge_attr", True)),
        use_ligand_mask=bool(surrogate_cfg.get("retrain_use_ligand_mask", True)),
        standardize=bool(surrogate_cfg.get("retrain_standardize", False)),
        fp_dim=int(surrogate_cfg.get("retrain_fp_dim", 0)),
        fp_radius=int(surrogate_cfg.get("retrain_fp_radius", 2)),
        eval_samples=int(surrogate_cfg.get("retrain_eval_samples", 8)),
        scheduler=str(surrogate_cfg.get("retrain_scheduler", "cosine")),
        warmup_epochs=int(surrogate_cfg.get("retrain_warmup_epochs", 15)),
        min_lr=float(surrogate_cfg.get("retrain_min_lr", 1e-5)),
        early_stop_patience=int(surrogate_cfg.get("retrain_early_stop_patience", 100)),
        early_stop_min_delta=float(surrogate_cfg.get("retrain_early_stop_min_delta", 0.0)),
        device=device,
        dock_valid_max=dock_valid_max,
        backbone=backbone,
        uncertainty_mode=str(surrogate_cfg.get("retrain_uncertainty_mode", "nig" if backbone == "tensornet" else "gaussian")),
        torchmd_cfg=_section(model_cfg, "torchmd"),
        ensemble_heads=int(surrogate_cfg.get("retrain_ensemble_heads", 1 if backbone == "tensornet" else 1)),
        freeze_backbone=bool(surrogate_cfg.get("retrain_freeze_backbone", False)),
        pretrained_encoder_ckpt=str(init_encoder_ckpt) if init_encoder_ckpt else None,
        ensemble_scheme=str(surrogate_cfg.get("retrain_ensemble_scheme", "full")),
        ensemble_bootstrap=bool(surrogate_cfg.get("retrain_ensemble_bootstrap", True)),
        random_seed=int(resolved.seed),
        ligand3d_cache_dir=str(shared_ligand3d_cache),
        ligand_vocab_override=global_vocab,
        confgen_max_attempts=int(candidate_3d_cfg.get("max_attempts", 3)),
        confgen_seed=int(candidate_3d_cfg.get("seed", resolved.seed)),
        confgen_num_confs=int(candidate_3d_cfg.get("num_confs", 4)),
        confgen_max_opt_iters=int(candidate_3d_cfg.get("max_opt_iters", 100)),
        confgen_optimize=bool(candidate_3d_cfg.get("optimize", True)),
        confgen_prefer_mmff=bool(candidate_3d_cfg.get("prefer_mmff", False)),
        ligand3d_num_workers=int(candidate_3d_cfg.get("workers", 8)),
        ligand3d_mp_chunksize=int(candidate_3d_cfg.get("chunksize", 16)),
        train_log_csv=str(out_root / "surrogate_train_log.csv"),
        train_summary_json=str(out_root / "surrogate_summary.json"),
    )
    (
        model,
        mean,
        std,
        model_node_dim,
        model_edge_dim,
        model_fp_dim,
        model_fp_radius,
        model_atom_extra_dim,
        model_bond_extra_dim,
        _model_pocket_graph,
        surrogate_kind,
        surrogate_meta,
    ) = load_surrogate(str(ckpt_path), device)
    _log(f"{method} train_surrogate done | elapsed={time.perf_counter() - train_started:.1f}s ckpt={ckpt_path.name}")

    holdout_df = _load_holdout_df(dataset_root, dock_valid_max=dock_valid_max)
    holdout_eval = _evaluate_holdout_surrogate(
        holdout_df=holdout_df,
        model=model,
        mean=mean,
        std=std,
        device=device,
        global_vocab=global_vocab,
        surrogate_kind=surrogate_kind,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
        iter_idx=0,
        method_dir=out_root,
    )
    pd.DataFrame([holdout_eval]).to_csv(out_root / "holdout_eval_metrics.csv", index=False)

    scorer = SurrogateScalarScorer(
        scalarizer=scalarizer,
        model=model,
        mean=mean,
        std=std,
        device=device,
        global_vocab=global_vocab,
        surrogate_kind=surrogate_kind,
        model_node_dim=model_node_dim,
        model_edge_dim=model_edge_dim,
        model_atom_extra_dim=model_atom_extra_dim,
        model_bond_extra_dim=model_bond_extra_dim,
        model_fp_dim=model_fp_dim,
        model_fp_radius=model_fp_radius,
        surrogate_meta=surrogate_meta,
        pred_batch_size=pred_batch_size,
        candidate_3d_cfg=candidate_3d_cfg,
    )

    start_pool_size = int(population_cfg.get("start_pool_size", 1000))
    seed_pool = list(dict.fromkeys(labeled_df["smiles_canonical"].astype(str).tolist()))
    rng = random.Random(_method_seed(int(resolved.seed), method, 0))
    if start_pool_size > 0 and len(seed_pool) > start_pool_size:
        start_population = rng.sample(seed_pool, start_pool_size)
    else:
        start_population = list(seed_pool)
    search_cfg = dict(population_cfg)
    search_cfg["generations"] = int(resolved.rounds)
    search_dir = out_root / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    search_started = time.perf_counter()
    search_pool, search_history, generation_populations = _run_population_search(
        method=method,
        scorer=scorer,
        start_population=start_population,
        seed=_method_seed(int(resolved.seed), method, 0),
        baseline_cfg=search_cfg,
        work_dir=search_dir,
    )
    if search_history.empty:
        raise RuntimeError(f"{method} returned an empty search history.")
    search_history.to_csv(search_dir / "search_history.csv", index=False)
    if generation_populations.empty:
        raise RuntimeError(f"{method} returned an empty generation population trace.")
    generation_populations.to_csv(search_dir / "generation_population_scores.csv", index=False)
    candidate_keep_topk = int(population_cfg.get("candidate_keep_topk", max(batch_size, 4 * batch_size)))
    candidate_df = _top_unique_by_score(
        search_pool,
        exclude=set(labeled_df["smiles_canonical"].astype(str).tolist()),
        top_k=candidate_keep_topk,
    )
    if int(candidate_df.shape[0]) < int(batch_size):
        raise RuntimeError(
            f"{method} produced too few novel candidates: candidate_df={candidate_df.shape[0]} batch_size={batch_size}"
        )
    candidate_df.to_csv(search_dir / "candidate_scores.csv", index=False)
    selected = candidate_df.sort_values(["score_scalar", "pred_dock_mean"], ascending=[False, True]).head(int(batch_size)).copy().reset_index(drop=True)
    selected["selected_iter"] = 1
    selected.to_csv(search_dir / "selected_batch.csv", index=False)
    _log(
        f"{method} search done | elapsed={time.perf_counter() - search_started:.1f}s "
        f"start_pool_n={len(start_population)} search_pool_n={search_pool.shape[0]} "
        f"candidate_pool_n={candidate_df.shape[0]} selected_n={selected.shape[0]}"
    )

    oracle_topk_per_generation = int(population_cfg.get("oracle_topk_per_generation", 0))
    if oracle_topk_per_generation > 0:
        iter_rows: list[dict] = [{
            "iter": 0,
            "execution_mode": "single_loop_generation_oracle",
            "labeled_n": int(labeled_df.shape[0]),
            "labeled_n_final": int(labeled_df.shape[0]),
            "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
            "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
            "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
            "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
            "best_observed_dock": float(labeled_df["dock_score"].min()),
            "best_observed_dock_final": float(labeled_df["dock_score"].min()),
            "hv": float(hv_init),
            "hv_final": float(hv_init),
            "hv_gain_total": 0.0,
            "acq_ref_point_dock": float(acq_ref_point[0]),
            "acq_ref_point_qed": float(acq_ref_point[1]),
            "acq_ref_point_sa": float(acq_ref_point[2]),
            "holdout_pred_n": int(holdout_eval.get("n", 0)),
            "holdout_pred_rmse": float(holdout_eval.get("rmse", np.nan)),
            "holdout_pred_mae": float(holdout_eval.get("mae", np.nan)),
            "holdout_pred_spearman": float(holdout_eval.get("spearman", np.nan)),
            "holdout_pred_kendall": float(holdout_eval.get("kendall", np.nan)),
            "holdout_pred_calibration_gap": float(holdout_eval.get("calibration_gap", np.nan)),
            "start_pool_n": int(len(start_population)),
            "search_generations": int(search_cfg["generations"]),
            "search_pool_n": int(search_pool.shape[0]),
            "candidate_pool_n": int(candidate_df.shape[0]),
            "selected_n": 0,
            "selected_true_dock_mean": np.nan,
            "selected_true_dock_best": np.nan,
            "selected_oracle_hv_gain": 0.0,
            "oracle_attempted": 0,
            "oracle_docked": 0,
            "oracle_failed": 0,
            "selected_pred_n": 0,
            "selected_pred_skipped": 0,
            "selected_pred_invalid_dock": 0,
            "selected_pred_invalid_pred": 0,
            "selected_pred_rmse": np.nan,
            "selected_pred_mae": np.nan,
            "selected_pred_spearman": np.nan,
            "selected_pred_kendall": np.nan,
            "utility_weight_dock": float(weights[0]),
            "utility_weight_qed": float(weights[1]),
            "utility_weight_sa": float(weights[2]),
        }]
        selected_batches_oracle: list[pd.DataFrame] = []
        labeled_running = labeled_df.copy()
        hv_running = float(hv_init)
        generation_series = pd.to_numeric(generation_populations["generation"], errors="raise").astype(int)
        observed_smiles = set(labeled_running["smiles_canonical"].astype(str).tolist())
        generation_indices = sorted(generation_series.unique().tolist())
        if method == "graph_ga":
            if 0 in generation_indices:
                _log(
                    "graph_ga generation-oracle selection skips generation 0 because it is the "
                    "initial labeled start population snapshot."
                )
            generation_indices = [int(g) for g in generation_indices if int(g) > 0]
            if not generation_indices:
                raise RuntimeError("graph_ga produced no post-initial generations for oracle selection.")
        for generation_idx in generation_indices:
            generation_df = generation_populations.loc[generation_series == int(generation_idx)].copy().reset_index(drop=True)
            if generation_df.empty:
                raise RuntimeError(f"Missing population snapshot for generation {generation_idx}.")
            iter_dir = out_root / f"iter{int(generation_idx):03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            selected_iter = _top_unique_by_score(
                generation_df,
                exclude=observed_smiles,
                top_k=oracle_topk_per_generation,
            )
            if int(selected_iter.shape[0]) < int(oracle_topk_per_generation):
                raise RuntimeError(
                    f"{method} generation {generation_idx} produced too few novel oracle candidates: "
                    f"selected={selected_iter.shape[0]} required={oracle_topk_per_generation}"
                )
            selected_iter["selected_iter"] = int(generation_idx)
            selected_iter.to_csv(iter_dir / "selected_batch.csv", index=False)
            selected_smiles = selected_iter["smiles_canonical"].astype(str).tolist()
            selected_pred_added = [None if pd.isna(v) else float(v) for v in selected_iter["pred_dock_mean"].tolist()]
            added, new_ids, kept_indices = append_candidates_to_smiles_csv(
                str(dataset_root / "smiles.csv"),
                selected_smiles,
                id_prefix=str(oracle_cfg.get("oracle_id_prefix", "GEN")),
                sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
                sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
                split_ratio=None,
                added_iter=int(generation_idx),
                molecule_origin=method,
            )
            if added != len(new_ids) or added != len(kept_indices):
                raise RuntimeError(
                    f"Append bookkeeping mismatch for method {method} generation {generation_idx}: "
                    f"added={added} new_ids={len(new_ids)} kept={len(kept_indices)}"
                )
            if kept_indices != list(range(selected_iter.shape[0])):
                selected_iter = selected_iter.iloc[kept_indices].copy().reset_index(drop=True)
                selected_pred_added = [selected_pred_added[i] for i in kept_indices]
            if int(selected_iter.shape[0]) < int(oracle_topk_per_generation):
                raise RuntimeError(
                    f"{method} generation {generation_idx} append dedup reduced oracle batch below target: "
                    f"selected={selected_iter.shape[0]} required={oracle_topk_per_generation}"
                )
            dock_stats = run_oracle_docking(str(dataset_root), new_ids, **online_oracle_kwargs)
            _fill_extra_metrics_in_csv(dataset_root, objective_cfg, prior_state=None, force=False)
            new_train_n, new_valid_n = _assign_new_rows_random_predictor_split(
                dataset_root,
                new_ids,
                valid_ratio=float(predictor_valid_ratio),
                seed=int(predictor_valid_seed + generation_idx),
            )
            _log(
                f"{method} iter {generation_idx} predictor_split_new_rows | "
                f"train={new_train_n} valid={new_valid_n} holdout_excluded=1"
            )
            merged_iter = _export_selected_batch_with_oracle(dataset_root, selected_iter, new_ids, iter_dir / "selected_batch_with_oracle.csv")
            selected_batches_oracle.append(merged_iter)

            smiles_all = pd.read_csv(dataset_root / "smiles.csv")
            smiles_all["ligand_id"] = smiles_all["ligand_id"].astype(str)
            new_df = smiles_all.loc[smiles_all["ligand_id"].isin([str(x) for x in new_ids])].copy().reset_index(drop=True)
            valid_mask = []
            for val in new_df.get("dock_score", pd.Series([], dtype=float)).tolist():
                try:
                    num = float(val)
                except Exception:
                    num = None
                valid_mask.append(num is not None and (dock_valid_max is None or num <= dock_valid_max))
            valid_new_df = new_df.loc[np.asarray(valid_mask, dtype=bool)].copy().reset_index(drop=True) if len(valid_mask) else new_df.head(0).copy()
            oracle_eval = evaluate_oracle_accuracy_from_csv(
                str(dataset_root / "smiles.csv"),
                new_ids,
                selected_pred_added,
                dock_valid_max=dock_valid_max,
            )
            labeled_after = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
            hv_after = compute_hypervolume(
                build_objective_tensor(labeled_after, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
                ref_point=fixed_ref_point,
            )
            if valid_new_df.empty:
                selected_true_dock_mean = np.nan
                selected_true_dock_best = np.nan
                selected_oracle_hv_gain = 0.0
            else:
                dock_vals = pd.to_numeric(valid_new_df["dock_score"], errors="coerce")
                selected_true_dock_mean = float(dock_vals.mean())
                selected_true_dock_best = float(dock_vals.min())
                selected_oracle_hv_gain = float(_batch_hv_gain(labeled_running, valid_new_df, dock_sign, qed_sign, sa_sign, fixed_ref_point))
            iter_rows.append({
                "iter": int(generation_idx),
                "execution_mode": "single_loop_generation_oracle",
                "labeled_n": int(labeled_running.shape[0]),
                "labeled_n_final": int(labeled_after.shape[0]),
                "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
                "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
                "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
                "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
                "best_observed_dock": float(labeled_running["dock_score"].min()),
                "best_observed_dock_final": float(labeled_after["dock_score"].min()),
                "hv": float(hv_running),
                "hv_final": float(hv_after),
                "hv_gain_total": float(hv_after - hv_init),
                "acq_ref_point_dock": float(acq_ref_point[0]),
                "acq_ref_point_qed": float(acq_ref_point[1]),
                "acq_ref_point_sa": float(acq_ref_point[2]),
                "holdout_pred_n": int(holdout_eval.get("n", 0)),
                "holdout_pred_rmse": float(holdout_eval.get("rmse", np.nan)),
                "holdout_pred_mae": float(holdout_eval.get("mae", np.nan)),
                "holdout_pred_spearman": float(holdout_eval.get("spearman", np.nan)),
                "holdout_pred_kendall": float(holdout_eval.get("kendall", np.nan)),
                "holdout_pred_calibration_gap": float(holdout_eval.get("calibration_gap", np.nan)),
                "start_pool_n": int(len(start_population)),
                "search_generations": int(search_cfg["generations"]),
                "search_pool_n": int(search_pool.shape[0]),
                "candidate_pool_n": int(generation_df.shape[0]),
                "selected_n": int(selected_iter.shape[0]),
                "selected_true_dock_mean": selected_true_dock_mean,
                "selected_true_dock_best": selected_true_dock_best,
                "selected_oracle_hv_gain": selected_oracle_hv_gain,
                "oracle_attempted": int(dock_stats.get("attempted", 0)),
                "oracle_docked": int(dock_stats.get("docked", 0)),
                "oracle_failed": int(dock_stats.get("failed", 0)),
                "selected_pred_n": int(oracle_eval.get("matched", 0)),
                "selected_pred_skipped": int(oracle_eval.get("skipped", 0)),
                "selected_pred_invalid_dock": int(oracle_eval.get("invalid_dock", 0)),
                "selected_pred_invalid_pred": int(oracle_eval.get("invalid_pred", 0)),
                "selected_pred_rmse": float(oracle_eval.get("rmse", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                "selected_pred_mae": float(oracle_eval.get("mae", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                "selected_pred_spearman": float(oracle_eval.get("spearman", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                "selected_pred_kendall": float(oracle_eval.get("kendall", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
                "utility_weight_dock": float(weights[0]),
                "utility_weight_qed": float(weights[1]),
                "utility_weight_sa": float(weights[2]),
            })
            observed_smiles.update(selected_iter["smiles_canonical"].astype(str).tolist())
            labeled_running = labeled_after
            hv_running = float(hv_after)
        pd.DataFrame(iter_rows).to_csv(out_root / "iter_metrics.csv", index=False)
        if selected_batches_oracle:
            pd.concat(selected_batches_oracle, ignore_index=True).to_csv(out_root / "selected_batches_with_oracle.csv", index=False)
        final_row = iter_rows[-1]
        _log(
            f"{method} done | hv_init={iter_rows[0]['hv']:.4f} hv_final={final_row['hv_final']:.4f} "
            f"best_observed_dock_final={final_row['best_observed_dock_final']:.4f}"
        )
    else:
        selected_smiles = selected["smiles_canonical"].astype(str).tolist()
        selected_pred_added = [None if pd.isna(v) else float(v) for v in selected["pred_dock_mean"].tolist()]
        added, new_ids, kept_indices = append_candidates_to_smiles_csv(
            str(dataset_root / "smiles.csv"),
            selected_smiles,
            id_prefix=str(oracle_cfg.get("oracle_id_prefix", "GEN")),
            sa_clamp_min=float(objective_cfg.get("sa_clamp_min", -10.0)),
            sa_clamp_max=float(objective_cfg.get("sa_clamp_max", 20.0)),
            split_ratio=None,
            added_iter=1,
            molecule_origin=method,
        )
        if added != len(new_ids) or added != len(kept_indices):
            raise RuntimeError(
                f"Append bookkeeping mismatch for method {method}: added={added} new_ids={len(new_ids)} kept={len(kept_indices)}"
            )
        if kept_indices != list(range(selected.shape[0])):
            selected = selected.iloc[kept_indices].copy().reset_index(drop=True)
            selected_pred_added = [selected_pred_added[i] for i in kept_indices]

        dock_stats = run_oracle_docking(str(dataset_root), new_ids, **online_oracle_kwargs)
        _fill_extra_metrics_in_csv(dataset_root, objective_cfg, prior_state=None, force=False)
        new_train_n, new_valid_n = _assign_new_rows_random_predictor_split(
            dataset_root,
            new_ids,
            valid_ratio=float(predictor_valid_ratio),
            seed=int(predictor_valid_seed + 1),
        )
        _log(
            f"{method} predictor_split_new_rows | train={new_train_n} valid={new_valid_n} holdout_excluded=1"
        )
        _export_selected_batch_with_oracle(dataset_root, selected, new_ids, search_dir / "selected_batch_with_oracle.csv")

        smiles_all = pd.read_csv(dataset_root / "smiles.csv")
        smiles_all["ligand_id"] = smiles_all["ligand_id"].astype(str)
        new_df = smiles_all.loc[smiles_all["ligand_id"].isin([str(x) for x in new_ids])].copy().reset_index(drop=True)
        valid_mask = []
        for val in new_df.get("dock_score", pd.Series([], dtype=float)).tolist():
            try:
                num = float(val)
            except Exception:
                num = None
            valid_mask.append(num is not None and (dock_valid_max is None or num <= dock_valid_max))
        valid_new_df = new_df.loc[np.asarray(valid_mask, dtype=bool)].copy().reset_index(drop=True) if len(valid_mask) else new_df.head(0).copy()
        oracle_eval = evaluate_oracle_accuracy_from_csv(
            str(dataset_root / "smiles.csv"),
            new_ids,
            selected_pred_added,
            dock_valid_max=dock_valid_max,
        )
        labeled_df_after = _load_observed_df(dataset_root, dock_valid_max=dock_valid_max)
        hv_final = compute_hypervolume(
            build_objective_tensor(labeled_df_after, dock_sign=dock_sign, qed_sign=qed_sign, sa_sign=sa_sign),
            ref_point=fixed_ref_point,
        )

        if valid_new_df.empty:
            selected_true_dock_mean = np.nan
            selected_true_dock_best = np.nan
            selected_oracle_hv_gain = 0.0
        else:
            dock_vals = pd.to_numeric(valid_new_df["dock_score"], errors="coerce")
            selected_true_dock_mean = float(dock_vals.mean())
            selected_true_dock_best = float(dock_vals.min())
            selected_oracle_hv_gain = float(_batch_hv_gain(labeled_df, valid_new_df, dock_sign, qed_sign, sa_sign, fixed_ref_point))

        metrics_row = {
            "iter": 0,
            "execution_mode": "single_loop_ref_style",
            "labeled_n": int(labeled_df.shape[0]),
            "labeled_n_final": int(labeled_df_after.shape[0]),
            "init_n": int(init_stats.get("init_n", labeled_df.shape[0])),
            "init_valid_dock_n": int(init_stats.get("valid_init_dock_n", labeled_df.shape[0])),
            "init_best_dock": float(init_stats.get("init_best_dock", labeled_df["dock_score"].min())),
            "init_mean_dock": float(init_stats.get("init_mean_dock", pd.to_numeric(labeled_df["dock_score"], errors="coerce").mean())),
            "best_observed_dock": float(labeled_df["dock_score"].min()),
            "best_observed_dock_final": float(labeled_df_after["dock_score"].min()),
            "hv": float(hv_init),
            "hv_final": float(hv_final),
            "hv_gain_total": float(hv_final - hv_init),
            "acq_ref_point_dock": float(acq_ref_point[0]),
            "acq_ref_point_qed": float(acq_ref_point[1]),
            "acq_ref_point_sa": float(acq_ref_point[2]),
            "holdout_pred_n": int(holdout_eval.get("n", 0)),
            "holdout_pred_rmse": float(holdout_eval.get("rmse", np.nan)),
            "holdout_pred_mae": float(holdout_eval.get("mae", np.nan)),
            "holdout_pred_spearman": float(holdout_eval.get("spearman", np.nan)),
            "holdout_pred_kendall": float(holdout_eval.get("kendall", np.nan)),
            "holdout_pred_calibration_gap": float(holdout_eval.get("calibration_gap", np.nan)),
            "start_pool_n": int(len(start_population)),
            "search_generations": int(search_cfg["generations"]),
            "search_pool_n": int(search_pool.shape[0]),
            "candidate_pool_n": int(candidate_df.shape[0]),
            "selected_n": int(selected.shape[0]),
            "selected_true_dock_mean": selected_true_dock_mean,
            "selected_true_dock_best": selected_true_dock_best,
            "selected_oracle_hv_gain": selected_oracle_hv_gain,
            "oracle_attempted": int(dock_stats.get("attempted", 0)),
            "oracle_docked": int(dock_stats.get("docked", 0)),
            "oracle_failed": int(dock_stats.get("failed", 0)),
            "selected_pred_n": int(oracle_eval.get("matched", 0)),
            "selected_pred_skipped": int(oracle_eval.get("skipped", 0)),
            "selected_pred_invalid_dock": int(oracle_eval.get("invalid_dock", 0)),
            "selected_pred_invalid_pred": int(oracle_eval.get("invalid_pred", 0)),
            "selected_pred_rmse": float(oracle_eval.get("rmse", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_mae": float(oracle_eval.get("mae", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_spearman": float(oracle_eval.get("spearman", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "selected_pred_kendall": float(oracle_eval.get("kendall", np.nan)) if oracle_eval.get("matched", 0) > 0 else np.nan,
            "utility_weight_dock": float(weights[0]),
            "utility_weight_qed": float(weights[1]),
            "utility_weight_sa": float(weights[2]),
        }
        pd.DataFrame([metrics_row]).to_csv(out_root / "iter_metrics.csv", index=False)
        _log(
            f"{method} done | hv_init={metrics_row['hv']:.4f} hv_final={metrics_row['hv_final']:.4f} "
            f"best_observed_dock_final={metrics_row['best_observed_dock_final']:.4f}"
        )
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

