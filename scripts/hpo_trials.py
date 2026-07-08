#!/usr/bin/env python3
"""Print a trial summary from an Optuna study database (SQLite).

Usage:
  python scripts/hpo_trials.py <db_or_dir>              # terminal table
  python scripts/hpo_trials.py <db_or_dir> --csv        # CSV to stdout
"""
import csv
import io
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_mode = "--csv" in sys.argv

    target = Path(args[0]) if args else Path.cwd()

    if target.suffix == ".db":
        db_file = target
    else:
        dbs = list(target.glob("*.db"))
        if not dbs:
            sys.exit(f"No .db file found in {target}")
        db_file = dbs[0]

    db = sqlite3.connect(str(db_file))

    trials = db.execute(
        "SELECT t.number, t.state, tv.value "
        "FROM trials t LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id "
        "ORDER BY t.number"
    ).fetchall()

    params = db.execute(
        "SELECT t.number, tp.param_name, tp.param_value, tp.distribution_json "
        "FROM trials t JOIN trial_params tp ON t.trial_id = tp.trial_id "
        "ORDER BY t.number, tp.param_name"
    ).fetchall()

    # Collect all param names (sorted, for consistent columns)
    param_names = sorted({p[1] for p in params})

    # Build per-trial dict with actual values (resolve categorical indices)
    trial_p = defaultdict(dict)
    for num, name, val, dist in params:
        d = json.loads(dist)
        choices = d["attributes"].get("choices")
        trial_p[num][name] = choices[int(val)] if choices else val

    # Build rows
    header = ["#", "objective"] + param_names
    rows = []
    for num, _state, value in trials:
        p = trial_p.get(num, {})
        row = [num]
        if csv_mode:
            row.append(f"{value:.6f}" if value is not None else "")
        else:
            row.append(f"{value:.4f}" if value is not None else "N/A")
        row.extend([p.get(name, "") for name in param_names])
        rows.append(row)

    if csv_mode:
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(header)
        w.writerows(rows)
        print(out.getvalue(), end="")
    else:
        # Compute column widths
        widths = [len(str(h)) for h in header]
        for r in rows:
            for i, v in enumerate(r):
                widths[i] = max(widths[i], len(str(v)))

        fmt = "  ".join(f"{{:>{w}}}" for w in widths)
        print(fmt.format(*[str(h) for h in header]))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for r in rows:
            print(fmt.format(*[str(v) for v in r]))


if __name__ == "__main__":
    main()
