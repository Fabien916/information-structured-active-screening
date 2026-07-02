from __future__ import annotations

from typing import Sequence, Tuple

import torch


def log_section(title: str) -> None:
    print(f"[{title}]")


def log_kv(key: str, value) -> None:
    print(f"  {key}: {value}")


def log_kv_table(title: str, items: Sequence[Tuple[str, object]], cols: int = 3) -> None:
    log_section(title)
    if not items:
        log_kv("empty", "")
        return
    cells = [f"{k}: {v}" for k, v in items]
    width = max(len(c) for c in cells)
    for i in range(0, len(cells), max(cols, 1)):
        row = cells[i : i + cols]
        line = "  " + " | ".join(c.ljust(width) for c in row)
        print(line)


def log_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    log_section(title)
    if not rows:
        log_kv("empty", "")
        return
    str_rows = [[str(x) for x in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "  " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  " + "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    for row in str_rows:
        print("  " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        print(f"{name}: empty")
        return
    t = tensor.detach().cpu().float()
    print(
        f"{name}: mean={t.mean().item():.4f} std={t.std(unbiased=False).item():.4f} "
        f"min={t.min().item():.4f} max={t.max().item():.4f}"
    )


def log_model_param_report(model: torch.nn.Module) -> None:
    total = 0
    trainable = 0
    groups: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        key = name.split(".", 1)[0] if "." in name else "root"
        group = groups.setdefault(key, {"total": 0, "trainable": 0})
        group["total"] += count
        if param.requires_grad:
            group["trainable"] += count
    log_kv_table(
        "model_params",
        [
            ("total_params", total),
            ("trainable_params", trainable),
            ("modules", len(groups)),
        ],
        cols=3,
    )
    rows = []
    for key, stats in sorted(groups.items(), key=lambda kv: kv[1]["total"], reverse=True):
        pct = (stats["total"] / total * 100.0) if total else 0.0
        rows.append([key, stats["total"], f"{pct:.2f}%"])
    log_table("model_param_share", ["module", "params", "pct"], rows)
