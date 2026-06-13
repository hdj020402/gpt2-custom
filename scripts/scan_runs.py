#!/usr/bin/env python3
"""Scan training runs and emit a CSV of hyperparameters + results for comparison.

Default output is CSV (all columns auto-discovered from yaml + logs).
Use ``--table`` for a terminal-friendly subset.

Usage
-----
.. code-block:: bash

    # CSV (all columns) — open in Excel / pandas
    python scripts/scan_runs.py > runs.csv

    # Filter + sort, still CSV
    python scripts/scan_runs.py --filter inchi_3M --sort best_eval_loss > runs.csv

    # Terminal table (compact subset)
    python scripts/scan_runs.py --table --filter 1p0kcal --last 10

Columns in CSV
--------------
- **meta**  — run_dir, jobtype, timestamp
- **param** — every key from model_parameters.yml (original names, flattened)
- **metric** — best_eval_loss, final_eval_loss, train_loss, train_runtime_s,
  and per-property mae_* values extracted from the training log
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# ═══════════════════════════════════════════════════════════════════════════════
# YAML parser — handles our flat + shallow-nested configs without PyYAML
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_yaml_flat(text: str) -> dict[str, Any]:
    """Parse a YAML file into a flat dict.

    Top-level scalars keep their key.  Nested blocks like::

        early_stopping:
          patience: 400
          threshold: 0

    become ``early_stopping.patience``, ``early_stopping.threshold``.
    """
    result: dict[str, Any] = OrderedDict()
    current_section: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            continue

        # nested key?
        if line.startswith("  ") or line.startswith("\t"):
            if current_section is None:
                continue
            m = re.match(r"^\s+(\S+):\s*(.*?)\s*$", line)
            if m:
                key, val = m.group(1), m.group(2)
                result[f"{current_section}.{key}"] = _coerce_yaml_val(val)
            continue

        # top-level key
        m = re.match(r"^(\S+):\s*(.*?)\s*$", stripped)
        if not m:
            continue
        key, val = m.group(1), m.group(2)

        if val == "":
            # next lines may be nested — peek ahead
            current_section = key
            continue
        else:
            current_section = None
            result[key] = _coerce_yaml_val(val)

    return result


def _coerce_yaml_val(raw: str) -> Any:
    raw = raw.strip().strip("'\"")
    if raw in ("null", "~", ""):
        return None
    if raw.lower() in ("true", "yes", "on"):
        return True
    if raw.lower() in ("false", "no", "off"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# log parsing
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_METRIC_RE = re.compile(r"'([^']+)':\s*'([^']+)'")


def _parse_training_log(log_path: Path) -> dict[str, Any]:
    """Extract training summary and all eval metrics from a log file."""
    with open(log_path) as f:
        content = f.read()

    result: dict[str, Any] = {}
    eval_losses: list[float] = []
    eval_metrics: list[dict] = []

    for line in content.splitlines():
        if "'train_runtime'" in line:
            m = re.search(r"'train_runtime':\s*'([^']+)'", line)
            if m:
                result["train_runtime_s"] = float(m.group(1))
        if "'train_loss'" in line:
            m = re.search(r"'train_loss':\s*'([^']+)'", line)
            if m:
                result["train_loss"] = float(m.group(1))

        # eval line — capture all key: 'value' pairs
        if "'eval_loss'" in line:
            metrics = {}
            for m in EVAL_METRIC_RE.finditer(line):
                k, v = m.group(1), m.group(2)
                try:
                    metrics[k] = float(v)
                except ValueError:
                    metrics[k] = v
            if "eval_loss" in metrics:
                eval_losses.append(metrics["eval_loss"])
                eval_metrics.append(metrics)

    if eval_losses:
        result["final_eval_loss"] = eval_losses[-1]
        result["final_epoch"] = eval_metrics[-1].get("epoch") if eval_metrics else None

        best_idx = min(range(len(eval_losses)), key=lambda i: eval_losses[i])
        best = eval_metrics[best_idx]
        for k, v in best.items():
            if k in ("eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
                continue  # too noisy for comparison
            result[f"best_{k}"] = v

        # also add best per-property MAE as top-level convenience keys
        # (already covered by best_eval_mae_*, but keep for readability)
        for k, v in best.items():
            if k.startswith("eval_mae_") and k != "eval_mae_avg":
                result[f"best_{k}"] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# run extraction
# ═══════════════════════════════════════════════════════════════════════════════

SKIP_PARAMS = {
    # Paths that point to data files — vary by machine but don't carry
    # useful information for run comparison.  Module paths like
    # custom_tokenize / custom_metrics are intentionally kept.
    "train_file", "val_file", "test_file", "corpus_file", "custom_tokens_file",
    "pretrained_model",
    "output_dir", "resume_from_checkpoint", "time",
    "mode",
}


def _extract_run(run_dir: Path) -> dict | None:
    """Extract meta + params + metrics from one training run directory."""
    try:
        yml_path = run_dir / "model_parameters.yml"
        if not yml_path.exists():
            return None
        with open(yml_path) as f:
            params = _parse_yaml_flat(f.read())

        log_files = sorted(run_dir.glob("training_*.log"))
        metrics = _parse_training_log(log_files[0]) if log_files else {}

        jobtype = params.pop("jobtype", run_dir.parent.name)

        # clean up unwanted params
        for k in SKIP_PARAMS:
            params.pop(k, None)

        rec = OrderedDict()
        rec["run_dir"] = str(run_dir)
        rec["jobtype"] = jobtype
        rec["timestamp"] = run_dir.name
        for k in sorted(params):
            rec[k] = params[k]
        for k in sorted(metrics):
            rec[k] = metrics[k]

        return rec

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# table (terminal)
# ═══════════════════════════════════════════════════════════════════════════════

# Default subset for terminal display — full yaml names, no abbreviations.
# Every key here must exist in the CSV columns (or be silently skipped).
TABLE_COLS = [
    "jobtype",
    "timestamp",
    "custom_tokenize",
    "learning_rate",
    "dropout",
    "weight_decay",
    "num_train_epochs",
    "best_eval_loss",
    "best_eval_mae_Energy",
    "train_runtime_s",
]


def _print_table(runs: list[dict], col_keys: list[str]) -> None:
    # build headers
    headers = [k for k in col_keys]  # use raw key as header
    widths = [max(len(h), 10) for h in headers]

    # format values, track widths
    rows: list[list[str]] = []
    for r in runs:
        cells: list[str] = []
        for i, k in enumerate(col_keys):
            v = r.get(k)
            if v is None:
                s = "—"
            elif isinstance(v, bool):
                s = "Y" if v else "N"
            elif isinstance(v, float):
                s = f"{v:.4g}"
            elif k == "train_runtime_s" and isinstance(v, (int, float)):
                s = _fmt_time(float(v))
            else:
                s = str(v)
                # for module-path params, show only basename in table
                if k in ("custom_tokenize", "custom_metrics") and "/" in s:
                    s = Path(s).name
            widths[i] = max(widths[i], len(s))
            cells.append(s)
        rows.append(cells)

    sep = "─┼─".join("─" * w for w in widths)
    header_line = " │ ".join(f"{h:^{widths[i]}}" for i, h in enumerate(headers))
    total_w = sum(widths) + 3 * (len(widths) - 1)

    print(f"\n{'=' * total_w}")
    print(f"  Training Run Summary  ({len(runs)} runs)")
    print(f"{'=' * total_w}")
    print(header_line)
    print(sep)
    for cells in rows:
        print(" │ ".join(f"{c:^{widths[i]}}" for i, c in enumerate(cells)))
    print(sep)
    print(f"{len(runs)} runs\n")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan training runs — default CSV with all columns auto-discovered")
    parser.add_argument("--filter", "-f", type=str,
                        help="Substring filter on jobtype or timestamp")
    parser.add_argument("--sort", "-s", type=str, default="timestamp",
                        help="Sort by column name (default: timestamp)")
    parser.add_argument("--last", "-n", type=int, default=0,
                        help="Show only the most recent N runs")
    parser.add_argument("--table", "-t", action="store_true",
                        help="Terminal table mode (compact subset); default is CSV")
    parser.add_argument("--cols", type=str,
                        help="Custom columns for --table mode, comma-separated")
    parser.add_argument("--dir", type=str, default="outputs/training",
                        help="Root training output directory")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    # ── discover runs ──
    runs: list[dict] = []
    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        for run_dir in sorted(job_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            rec = _extract_run(run_dir)
            if rec:
                runs.append(rec)

    if not runs:
        print("No training runs found.", file=sys.stderr)
        sys.exit(0)

    # ── filter ──
    if args.filter:
        f = args.filter.lower()
        runs = [r for r in runs
                if f in str(r.get("jobtype", "")).lower()
                or f in str(r.get("timestamp", "")).lower()]

    # ── sort ──
    sort_key = args.sort
    def _sort_key(r: dict) -> tuple:
        v = r.get(sort_key)
        # None sorts last
        return (v is None, v if v is not None else "")
    runs.sort(key=_sort_key)

    if args.last > 0:
        runs = runs[-args.last:]

    # ── collect union of all column keys (preserving order) ──
    all_keys: list[str] = []
    seen = set()
    for r in runs:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # ── output ──
    if args.table:
        if args.cols:
            table_keys = [k.strip() for k in args.cols.split(",")]
        else:
            table_keys = [k for k in TABLE_COLS if k in seen]
        _print_table(runs, table_keys)
        return

    # CSV (default)
    writer = csv.writer(sys.stdout)
    writer.writerow(all_keys)
    for r in runs:
        writer.writerow([r.get(k, "") for k in all_keys])


if __name__ == "__main__":
    main()
