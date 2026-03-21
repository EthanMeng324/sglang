#!/usr/bin/env python3
"""Build a concise profile matrix summary for no/old/new/phase offload runs."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


DEFAULT_PROFILES = {
    "no": Path("offload_profiling/results/profiles/no_offload_nsys.nsys-rep"),
    "old": Path("offload_profiling/results/profiles/old_offload_nsys.nsys-rep"),
    "new": Path("offload_profiling/results/profiles/new_offload_nsys.nsys-rep"),
    "phase": Path("offload_profiling/results/profiles/phase_offload_nsys.nsys-rep"),
}

LABELS = {
    "no": "No Offload",
    "old": "Old",
    "new": "New",
    "phase": "Phase",
}

STEP_GLOB = "SGL_DENOISING_STEP_*"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, path in DEFAULT_PROFILES.items():
        parser.add_argument(f"--{key}", type=Path, default=path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("offload_profiling/results/profiles/analysis/analysis_summary.md"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("offload_profiling/results/profiles/analysis/profile_matrix.csv"),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("offload_profiling/results/profiles/analysis/sqlite_cache"),
    )
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def is_valid_sqlite(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(path)
        row = conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        conn.close()
        return bool(row and row[0] > 0)
    except sqlite3.Error:
        return False


def export_sqlite(rep_path: Path, sqlite_path: Path) -> Path:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    cmd = [
        "nsys",
        "export",
        "--type=sqlite",
        f"--output={sqlite_path}",
        str(rep_path),
    ]
    log(f"Export SQLite: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if is_valid_sqlite(sqlite_path):
        return sqlite_path
    alt = sqlite_path.with_suffix("")
    if is_valid_sqlite(alt):
        return alt
    raise RuntimeError(f"Exported sqlite is invalid: {sqlite_path}")


def ensure_sqlite(input_path: Path, export_dir: Path, force_export: bool) -> Path:
    if is_valid_sqlite(input_path):
        return input_path
    if input_path.suffix != ".nsys-rep":
        raise RuntimeError(f"Unsupported input type: {input_path}")
    export_name = input_path.name.replace(".nsys-rep", "_summary_export.sqlite")
    export_path = export_dir / export_name
    if (
        not force_export
        and is_valid_sqlite(export_path)
        and export_path.stat().st_mtime >= input_path.stat().st_mtime
    ):
        return export_path
    return export_sqlite(input_path, export_path)


def get_step_time_from_trace(sqlite_path: Path) -> float | None:
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        """
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE text GLOB ? AND end IS NOT NULL
        ORDER BY start
        """,
        (STEP_GLOB,),
    ).fetchall()
    conn.close()
    by_step: dict[int, list[tuple[int, int]]] = {}
    for text, start, end in rows:
        try:
            step = int(text.split("_")[-1])
        except ValueError:
            continue
        by_step.setdefault(step, []).append((start, end))
    steady = []
    for step, intervals in sorted(by_step.items()):
        if step < 1:
            continue
        dur_s = sum((end - start) / 1e9 for start, end in intervals) / len(intervals)
        steady.append(dur_s)
    if not steady:
        return None
    return sum(steady) / len(steady)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def ratio(base: float | None, other: float | None) -> float | None:
    if base in (None, 0) or other is None:
        return None
    return other / base


def delta(base: float | None, other: float | None) -> float | None:
    if base is None or other is None:
        return None
    return other - base


def build_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    export_dir = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    for key in ["no", "old", "new", "phase"]:
        path: Path = getattr(args, key)
        step_override = env_float(f"{key.upper()}_STEP_TIME_S")
        peak_override = env_float(f"{key.upper()}_PEAK_MEMORY_MB")
        step_time_s = step_override
        step_source = "env_override" if step_override is not None else None
        trace_status = "not_checked"
        sqlite_path = None
        if step_time_s is None:
            try:
                sqlite_path = ensure_sqlite(path, export_dir, args.force_export)
                step_time_s = get_step_time_from_trace(sqlite_path)
                step_source = f"trace:{sqlite_path.name}" if step_time_s is not None else "trace_missing_step"
                trace_status = "ok"
            except Exception as exc:
                trace_status = f"unreadable: {exc}"
        else:
            trace_status = "override_only"
        rows.append(
            {
                "run": key,
                "label": LABELS[key],
                "path": str(path),
                "trace_status": trace_status,
                "step_time_s": step_time_s,
                "step_source": step_source or "unavailable",
                "peak_memory_mb": peak_override,
                "peak_source": "env_override" if peak_override is not None else "unavailable",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "label",
                "path",
                "trace_status",
                "step_time_s",
                "step_source",
                "peak_memory_mb",
                "peak_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_run = {row["run"]: row for row in rows}
    base_step = by_run["no"]["step_time_s"]
    base_mem = by_run["no"]["peak_memory_mb"]

    lines = [
        "# Offload Profile Summary",
        "",
        "## Step Time",
        "",
        "| Profile | Step Time (s) | Delta vs No (s) | Ratio vs No | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {step} | {delta} | {ratio} | {source} |".format(
                label=row["label"],
                step=fmt(row["step_time_s"], 3),
                delta=fmt(delta(base_step, row["step_time_s"]), 3),
                ratio=fmt(ratio(base_step, row["step_time_s"]), 2),
                source=row["step_source"],
            )
        )

    lines += [
        "",
        "## Peak Memory",
        "",
        "| Profile | Peak Memory (MB) | Delta vs No (MB) | Ratio vs No | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {mem} | {delta} | {ratio} | {source} |".format(
                label=row["label"],
                mem=fmt(row["peak_memory_mb"], 0),
                delta=fmt(delta(base_mem, row["peak_memory_mb"]), 0),
                ratio=fmt(ratio(base_mem, row["peak_memory_mb"]), 2),
                source=row["peak_source"],
            )
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(rows, args.csv_output)
    write_markdown(rows, args.output)


if __name__ == "__main__":
    main()
