#!/usr/bin/env python3
"""Build a concise profile matrix summary for no/old/new/phase offload runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import statistics
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


def load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_perf_json(run: str, trace_path: Path) -> Path | None:
    exact_names = {
        "new": ["perf_new_offload_profiled.json", "perf_new_offload.json"],
        "old": ["perf_old_offload_profiled.json", "perf_old_offload.json"],
        "no": ["perf_no_offload_profiled.json", "perf_no_offload.json"],
        "phase": ["perf_phase_offload_profiled.json", "perf_phase_offload.json"],
    }
    profiles_dir = trace_path.parent
    if trace_path.suffix == ".sqlite" and profiles_dir.name == "sqlite_cache":
        profiles_dir = profiles_dir.parent.parent
    results_dir = profiles_dir.parent
    trace_stem = trace_path.name.replace(".nsys-rep", "")
    derived_names = []
    if trace_stem.endswith("_nsys"):
        derived_names.append(f"perf_{trace_stem[:-5]}_profiled.json")
    for root in (profiles_dir, results_dir):
        for name in derived_names:
            path = root / name
            if path.exists():
                return path
        for name in exact_names[run]:
            path = root / name
            if path.exists():
                return path
    for root in (profiles_dir, results_dir):
        matches = sorted(root.glob(f"perf_{run}*.json"))
        if matches:
            return matches[-1]
    return None


def extract_peak_memory_pair(perf: dict) -> tuple[float | None, float | None]:
    peak_reserved_mb = None
    peak_allocated_mb = None
    checkpoints = perf.get("memory_checkpoints")
    if isinstance(checkpoints, dict):
        for key in ("mem_analysis", "after_forward", "before_forward"):
            value = checkpoints.get(key)
            if not isinstance(value, dict):
                continue
            peak_reserved = value.get("peak_reserved_mb")
            if peak_reserved_mb is None and isinstance(peak_reserved, (int, float)):
                peak_reserved_mb = float(peak_reserved)
            peak_allocated = value.get("peak_allocated_mb")
            if peak_allocated_mb is None and isinstance(peak_allocated, (int, float)):
                peak_allocated_mb = float(peak_allocated)
    peak_memory = perf.get("peak_memory_mb")
    if peak_reserved_mb is None and isinstance(peak_memory, (int, float)):
        peak_reserved_mb = float(peak_memory)
    return peak_reserved_mb, peak_allocated_mb


def extract_total_duration_ms(perf: dict) -> float | None:
    total_duration_ms = perf.get("total_duration_ms")
    if isinstance(total_duration_ms, (int, float)):
        return float(total_duration_ms)
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


def get_step_durations_from_trace(sqlite_path: Path) -> dict[int, float]:
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
    durations: dict[int, float] = {}
    for step, intervals in sorted(by_step.items()):
        dur_s = sum((end - start) / 1e9 for start, end in intervals) / len(intervals)
        durations[step] = dur_s
    return durations


def detect_switch_step(step_durations: dict[int, float]) -> tuple[int, float] | None:
    steady = sorted((step, dur) for step, dur in step_durations.items() if step >= 1)
    if len(steady) < 4:
        return None
    values = [dur for _, dur in steady]
    median = statistics.median(values)
    if median <= 0:
        return None
    step_idx, max_value = max(steady, key=lambda item: item[1])
    if max_value >= median * 1.15:
        return step_idx, max_value
    return None


def summarize_step_trace(sqlite_path: Path) -> dict[str, float | int | None]:
    step_durations = get_step_durations_from_trace(sqlite_path)
    steady = sorted((step, dur) for step, dur in step_durations.items() if step >= 1)
    if not steady:
        return {
            "step_time_s": None,
            "step_time_raw_s": None,
            "switch_step_idx": None,
            "switch_step_time_s": None,
            "steady_step_count": 0,
        }
    raw_mean = sum(dur for _, dur in steady) / len(steady)
    switch = detect_switch_step(step_durations)
    filtered = steady
    if switch is not None:
        switch_idx, _ = switch
        filtered = [(step, dur) for step, dur in steady if step != switch_idx]
    if not filtered:
        return None
    adjusted_mean = sum(dur for _, dur in filtered) / len(filtered)
    return {
        "step_time_s": adjusted_mean,
        "step_time_raw_s": raw_mean,
        "switch_step_idx": switch[0] if switch is not None else None,
        "switch_step_time_s": switch[1] if switch is not None else None,
        "steady_step_count": len(filtered),
    }


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
        peak_reserved_override = env_float(f"{key.upper()}_PEAK_RESERVED_MB")
        if peak_reserved_override is None:
            peak_reserved_override = env_float(f"{key.upper()}_PEAK_MEMORY_MB")
        peak_allocated_override = env_float(f"{key.upper()}_PEAK_ALLOCATED_MB")
        perf_path = find_perf_json(key, path)
        perf_json = load_json(perf_path)
        perf_peak_reserved_mb, perf_peak_allocated_mb = extract_peak_memory_pair(
            perf_json
        )
        perf_total_duration_ms = extract_total_duration_ms(perf_json)
        step_time_s = step_override
        step_source = "env_override" if step_override is not None else None
        trace_status = "not_checked"
        sqlite_path = None
        step_time_raw_s = None
        switch_step_idx = None
        switch_step_time_s = None
        steady_step_count = None
        if step_time_s is None:
            try:
                sqlite_path = ensure_sqlite(path, export_dir, args.force_export)
                step_summary = summarize_step_trace(sqlite_path)
                step_time_s = step_summary["step_time_s"]
                step_time_raw_s = step_summary["step_time_raw_s"]
                switch_step_idx = step_summary["switch_step_idx"]
                switch_step_time_s = step_summary["switch_step_time_s"]
                steady_step_count = step_summary["steady_step_count"]
                if step_time_s is not None:
                    if switch_step_idx is None:
                        step_source = f"trace:{sqlite_path.name}"
                    else:
                        step_source = (
                            f"trace:{sqlite_path.name}; exclude_step={switch_step_idx}"
                        )
                else:
                    step_source = "trace_missing_step"
                trace_status = "ok"
            except Exception as exc:
                trace_status = f"unreadable: {exc}"
        else:
            trace_status = "override_only"
        peak_reserved_mb = peak_reserved_override
        if peak_reserved_mb is not None:
            peak_reserved_source = "env_override"
        else:
            peak_reserved_mb = perf_peak_reserved_mb
            peak_reserved_source = (
                f"perf_json:{perf_path.name}"
                if peak_reserved_mb is not None and perf_path is not None
                else "unavailable"
            )

        peak_allocated_mb = peak_allocated_override
        if peak_allocated_mb is not None:
            peak_allocated_source = "env_override"
        else:
            peak_allocated_mb = perf_peak_allocated_mb
            peak_allocated_source = (
                f"perf_json:{perf_path.name}"
                if peak_allocated_mb is not None and perf_path is not None
                else "unavailable"
            )
        total_duration_source = (
            f"perf_json:{perf_path.name}"
            if perf_total_duration_ms is not None and perf_path is not None
            else "unavailable"
        )
        rows.append(
            {
                "run": key,
                "label": LABELS[key],
                "path": str(path),
                "trace_status": trace_status,
                "step_time_s": step_time_s,
                "step_time_raw_s": step_time_raw_s,
                "step_source": step_source or "unavailable",
                "switch_step_idx": switch_step_idx,
                "switch_step_time_s": switch_step_time_s,
                "steady_step_count": steady_step_count,
                "total_duration_ms": perf_total_duration_ms,
                "total_duration_source": total_duration_source,
                "peak_reserved_mb": peak_reserved_mb,
                "peak_reserved_source": peak_reserved_source,
                "peak_allocated_mb": peak_allocated_mb,
                "peak_allocated_source": peak_allocated_source,
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
                "step_time_raw_s",
                "step_source",
                "switch_step_idx",
                "switch_step_time_s",
                "steady_step_count",
                "total_duration_ms",
                "total_duration_source",
                "peak_reserved_mb",
                "peak_reserved_source",
                "peak_allocated_mb",
                "peak_allocated_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_run = {row["run"]: row for row in rows}
    base_step = by_run["no"]["step_time_s"]
    base_total_duration = by_run["no"]["total_duration_ms"]
    base_peak_reserved = by_run["no"]["peak_reserved_mb"]
    base_peak_allocated = by_run["no"]["peak_allocated_mb"]

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
        "Step Time above excludes `step 0` and, when detected, one switch-step outlier.",
        "",
        "## Switch Step Outlier",
        "",
        "| Profile | Switch Step | Step Time (s) | Raw Mean incl. Switch (s) | Used Steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {step_idx} | {switch_time} | {raw_mean} | {count} |".format(
                label=row["label"],
                step_idx=row["switch_step_idx"] if row["switch_step_idx"] is not None else "N/A",
                switch_time=fmt(row["switch_step_time_s"], 3),
                raw_mean=fmt(row["step_time_raw_s"], 3),
                count=fmt(row["steady_step_count"], 0),
            )
        )

    lines += [
        "",
        "## Total Duration",
        "",
        "| Profile | Total Duration (ms) | Delta vs No (ms) | Ratio vs No | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {duration} | {delta} | {ratio} | {source} |".format(
                label=row["label"],
                duration=fmt(row["total_duration_ms"], 3),
                delta=fmt(delta(base_total_duration, row["total_duration_ms"]), 3),
                ratio=fmt(ratio(base_total_duration, row["total_duration_ms"]), 2),
                source=row["total_duration_source"],
            )
        )

    lines += [
        "",
        "## Peak Reserved Memory",
        "",
        "| Profile | Peak Reserved (MB) | Delta vs No (MB) | Ratio vs No | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {mem} | {delta} | {ratio} | {source} |".format(
                label=row["label"],
                mem=fmt(row["peak_reserved_mb"], 0),
                delta=fmt(delta(base_peak_reserved, row["peak_reserved_mb"]), 0),
                ratio=fmt(ratio(base_peak_reserved, row["peak_reserved_mb"]), 2),
                source=row["peak_reserved_source"],
            )
        )

    lines += [
        "",
        "## Peak Allocated Memory",
        "",
        "| Profile | Peak Allocated (MB) | Delta vs No (MB) | Ratio vs No | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for key in ["no", "old", "new", "phase"]:
        row = by_run[key]
        lines.append(
            "| {label} | {mem} | {delta} | {ratio} | {source} |".format(
                label=row["label"],
                mem=fmt(row["peak_allocated_mb"], 0),
                delta=fmt(delta(base_peak_allocated, row["peak_allocated_mb"]), 0),
                ratio=fmt(ratio(base_peak_allocated, row["peak_allocated_mb"]), 2),
                source=row["peak_allocated_source"],
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
