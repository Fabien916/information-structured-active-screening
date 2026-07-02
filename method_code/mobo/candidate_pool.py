from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from smiles_vae_utils import sample_sequences as sample_vae_sequences
from mobo.smiles_utils import canonicalize_smiles_noh


def _sample_from_legacy_vae(
    model,
    vocab,
    token_type: str,
    latent_dim: int,
    batch_n: int,
    max_len: int,
    temperature: float,
    top_k: int,
    device,
) -> list[str]:
    samples_out = sample_vae_sequences(
        model,
        vocab,
        int(latent_dim),
        int(batch_n),
        int(max_len),
        float(temperature),
        int(top_k),
        device,
        str(token_type),
    )
    return [str(smi) for _, smi in samples_out]


def build_candidate_pool(*args, **kwargs):
    """
    Backward-compatible candidate-pool sampler.

    New style:
        build_candidate_pool(generator, target_valid, sample_batch, temperature, top_k,
                             exclude_smiles=None, log_prefix="", return_stats=False)

    Legacy VAE style:
        build_candidate_pool(model, vocab, token_type, latent_dim, max_len,
                             target_valid, sample_batch, temperature, top_k, device,
                             exclude_smiles=None, log_prefix="", return_stats=False)
    """
    return_stats = bool(kwargs.get("return_stats", False))
    if len(args) >= 10:
        model, vocab, token_type, latent_dim, max_len, target_valid, sample_batch, temperature, top_k, device = args[:10]
        exclude_smiles = kwargs.get("exclude_smiles")
        log_prefix = kwargs.get("log_prefix", "")

        def sampler(n: int) -> list[str]:
            return _sample_from_legacy_vae(model, vocab, token_type, latent_dim, n, max_len, temperature, top_k, device)

    else:
        generator, target_valid, sample_batch, temperature, top_k = args[:5]
        exclude_smiles = kwargs.get("exclude_smiles")
        log_prefix = kwargs.get("log_prefix", "")

        def sampler(n: int) -> list[str]:
            return generator.sample_smiles(int(n), temperature=float(temperature), top_k=int(top_k))

    valid: list[str] = []
    seen: set[str] = set()
    invalid = 0
    dup = 0
    excluded = 0
    total = 0
    rounds = 0
    stagnant_rounds = 0
    round_rows: list[dict[str, Any]] = []
    max_stagnant_rounds = 60
    desc = f"{log_prefix}collecting"
    with tqdm(total=target_valid, desc=desc, unit="mol") as progress:
        while len(valid) < target_valid:
            rounds += 1
            batch_n = min(int(sample_batch), int(target_valid) - len(valid))
            samples_out = sampler(batch_n)
            batch_total = len(samples_out)
            total += batch_total
            before = len(valid)
            batch_invalid = 0
            batch_dup = 0
            batch_excluded = 0
            for smi in samples_out:
                try:
                    canon = canonicalize_smiles_noh(smi)
                except Exception:
                    invalid += 1
                    batch_invalid += 1
                    continue
                if not canon:
                    invalid += 1
                    batch_invalid += 1
                    continue
                if canon in seen:
                    dup += 1
                    batch_dup += 1
                    continue
                if exclude_smiles and canon in exclude_smiles:
                    excluded += 1
                    batch_excluded += 1
                    continue
                seen.add(canon)
                valid.append(canon)
            batch_valid = len(valid) - before
            if len(valid) == before:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            progress.update(batch_valid)
            progress.set_postfix(total=total, invalid=invalid, dup=dup, excluded=excluded)
            round_rows.append({
                "round": int(rounds),
                "requested": int(batch_n),
                "sampled": int(batch_total),
                "accepted": int(batch_valid),
                "invalid": int(batch_invalid),
                "duplicate": int(batch_dup),
                "excluded": int(batch_excluded),
                "accepted_cum": int(len(valid)),
                "invalid_cum": int(invalid),
                "duplicate_cum": int(dup),
                "excluded_cum": int(excluded),
                "total_sampled_cum": int(total),
                "stagnant_rounds": int(stagnant_rounds),
            })
            if stagnant_rounds >= max_stagnant_rounds:
                raise RuntimeError(
                    f"{log_prefix}candidate pool stalled for {stagnant_rounds} rounds "
                    f"(rounds={rounds}, total={total}, valid={len(valid)}, invalid={invalid}, dup={dup}, excluded={excluded}). "
                    "Check generator validity / sampling params (temperature, top_k)."
                )
    stats = {
        "target_valid": int(target_valid),
        "sample_batch": int(sample_batch),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "rounds": int(rounds),
        "total_sampled": int(total),
        "valid": int(len(valid)),
        "invalid": int(invalid),
        "duplicate": int(dup),
        "excluded": int(excluded),
        "validity_rate": float(len(valid) / total) if total > 0 else 0.0,
        "invalid_rate": float(invalid / total) if total > 0 else 0.0,
        "duplicate_rate": float(dup / total) if total > 0 else 0.0,
        "excluded_rate": float(excluded / total) if total > 0 else 0.0,
        "round_rows": round_rows,
    }
    print(
        f"{log_prefix}pool_total={total} valid={len(valid)} "
        f"invalid={invalid} dup={dup} excluded={excluded} validity_rate={stats['validity_rate']:.4f}"
    )
    if return_stats:
        return valid, stats
    return valid

