#!/usr/bin/env python3
"""Summarize per-step NCCL/H2D metrics for one or more explicit NSYS trace files."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


STEP_GLOBS = ("SGL_DENOISING_STEP_*", "denoising_step_*")
PREFETCH_TEXTS = (
    "SGL_PREFETCH_H2D",
    "SGL_PREFETCH_H2D_BG",
    "SGL_PREFETCH_H2D_SYNC",
    "SGL_PREFETCH_H2D_CHUNK",
)
WAIT_COMM_BG = "SGL_PREFETCH_WAIT_COMM_BG"
NCCL_PATTERN = "%ncclDevKernel_SendRecv%"


@dataclass
class TraceMetrics:
    trace_name: str
    step_avg_ms: float
    step_count: int
    device_count: int
    prefetch_streams: list[int]
    nccl_total_ms: float
    nccl_h2d_overlap_ms: float
    uncovered_h2d_ms: float
    wait_comm_bg_ms: float


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=Path,
        help="One or more explicit .nsys-rep/.sqlite files to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to <common_parent>/analysis_summary.md.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Directory for exported sqlite files. Defaults to <common_parent>/sqlite_cache_steps.",
    )
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument(
        "--prefetch-min-mb",
        type=float,
        default=1.0,
        help="Minimum H2D memcpy size considered part of prefetch traffic.",
    )
    parser.add_argument(
        "--prefetch-marker-slack-ms",
        type=float,
        default=1.0,
        help="Allowed launch slack between prefetch NVTX range and actual memcpy start.",
    )
    return parser.parse_args()


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


def collect_inputs(paths: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"Directory input is no longer supported; pass explicit file paths instead: {path}"
            )
        if not (is_valid_sqlite(path) or path.suffix == ".nsys-rep"):
            raise RuntimeError(f"Unsupported input type: {path}")
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        inputs.append(path)
    return inputs


def common_parent(paths: list[Path]) -> Path:
    resolved_parents = [str(path.resolve().parent) for path in paths]
    if not resolved_parents:
        return Path.cwd()
    return Path(os.path.commonpath(resolved_parents))


def parse_step_index(text: str) -> int | None:
    try:
        return int(text.split("_")[-1])
    except (ValueError, IndexError):
        return None


def get_step_windows(conn: sqlite3.Connection) -> tuple[list[tuple[int, int]], float]:
    rows: list[tuple[str, int, int]] = []
    for glob in STEP_GLOBS:
        rows.extend(
            conn.execute(
                """
                SELECT text, start, end
                FROM NVTX_EVENTS
                WHERE text GLOB ? AND end IS NOT NULL
                ORDER BY start
                """,
                (glob,),
            ).fetchall()
        )
    by_step: dict[int, list[tuple[int, int]]] = {}
    for text, start, end in rows:
        step = parse_step_index(text)
        if step is None:
            continue
        by_step.setdefault(step, []).append((start, end))

    windows: list[tuple[int, int]] = []
    step_avgs_ms: list[float] = []
    for step in sorted(by_step):
        ints = by_step[step]
        dur_ms = [float(end - start) / 1e6 for start, end in ints]
        avg_ms = sum(dur_ms) / len(dur_ms)
        if step >= 1:
            windows.append((min(start for start, _ in ints), max(end for _, end in ints)))
            step_avgs_ms.append(avg_ms)

    if not windows:
        raise RuntimeError("No steady-state denoising step windows found")

    return windows, sum(step_avgs_ms) / len(step_avgs_ms)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def total_len(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersect_total(
    lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]
) -> int:
    i = 0
    j = 0
    total = 0
    while i < len(lhs) and j < len(rhs):
        start = max(lhs[i][0], rhs[j][0])
        end = min(lhs[i][1], rhs[j][1])
        if end > start:
            total += end - start
        if lhs[i][1] <= rhs[j][1]:
            i += 1
        else:
            j += 1
    return total


def subtract_total(
    lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]
) -> int:
    return total_len(lhs) - intersect_total(lhs, rhs)


def clip_and_merge(
    intervals: list[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    out = []
    for item_start, item_end in intervals:
        if item_end <= start:
            continue
        if item_start >= end:
            break
        out.append((max(item_start, start), min(item_end, end)))
    return merge_intervals(out)


def detect_prefetch_streams(
    conn: sqlite3.Connection,
    global_start: int,
    global_end: int,
    prefetch_min_bytes: int,
    prefetch_marker_slack_ns: int,
) -> list[int]:
    marker_placeholders = ",".join("?" for _ in PREFETCH_TEXTS)
    query = f"""
        SELECT m.streamId, COUNT(*), SUM(m.bytes)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        WHERE m.copyKind = 1
          AND m.bytes >= ?
          AND m.start < ?
          AND m.end > ?
          AND EXISTS (
              SELECT 1
              FROM NVTX_EVENTS n
              WHERE n.text IN ({marker_placeholders})
                AND n.end IS NOT NULL
                AND (
                    (n.start < m.end AND n.end > m.start)
                    OR (m.start >= n.start AND m.start <= n.end + ?)
                )
          )
        GROUP BY m.streamId
        ORDER BY COUNT(*) DESC, SUM(m.bytes) DESC, m.streamId
    """
    params = [
        prefetch_min_bytes,
        global_end,
        global_start,
        *PREFETCH_TEXTS,
        prefetch_marker_slack_ns,
    ]
    rows = conn.execute(query, params).fetchall()
    if rows:
        return [int(row[0]) for row in rows]

    fallback = conn.execute(
        """
        SELECT streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 1
          AND bytes >= ?
          AND start < ?
          AND end > ?
        GROUP BY streamId
        ORDER BY COUNT(*) DESC, SUM(bytes) DESC, streamId
        LIMIT 1
        """,
        (prefetch_min_bytes, global_end, global_start),
    ).fetchone()
    return [int(fallback[0])] if fallback else []


def sum_marker_overlap(
    conn: sqlite3.Connection,
    marker_text: str,
    step_windows: list[tuple[int, int]],
) -> int:
    markers = conn.execute(
        """
        SELECT start, end
        FROM NVTX_EVENTS
        WHERE text = ? AND end IS NOT NULL
        ORDER BY start
        """,
        (marker_text,),
    ).fetchall()
    total = 0
    for marker in markers:
        total += intersect_total([marker], step_windows)
    return total


def compute_metrics_for_trace(
    trace_path: Path,
    sqlite_path: Path,
    prefetch_min_bytes: int,
    prefetch_marker_slack_ns: int,
) -> TraceMetrics:
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    step_windows, step_avg_ms = get_step_windows(conn)
    global_start = min(start for start, _ in step_windows)
    global_end = max(end for _, end in step_windows)
    device_ids = [
        int(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start < ? AND end > ?
              AND deviceId IS NOT NULL AND deviceId >= 0
            ORDER BY deviceId
            """,
            (global_end, global_start),
        )
    ]
    if not device_ids:
        raise RuntimeError("No CUDA device kernels found in steady-state step windows")

    prefetch_streams = detect_prefetch_streams(
        conn,
        global_start,
        global_end,
        prefetch_min_bytes,
        prefetch_marker_slack_ns,
    )

    per_device_nccl_ms: list[float] = []
    per_device_overlap_ms: list[float] = []
    per_device_uncovered_ms: list[float] = []

    stream_placeholders = ",".join("?" for _ in prefetch_streams)
    for device_id in device_ids:
        nccl_intervals = merge_intervals(
            [
                (int(start), int(end))
                for start, end in conn.execute(
                    """
                    SELECT k.start, k.end
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k
                    JOIN StringIds si ON k.demangledName = si.id
                    WHERE k.deviceId = ?
                      AND k.start < ?
                      AND k.end > ?
                      AND si.value LIKE ?
                    ORDER BY k.start
                    """,
                    (device_id, global_end, global_start, NCCL_PATTERN),
                )
            ]
        )
        all_kernel_intervals = merge_intervals(
            [
                (int(start), int(end))
                for start, end in conn.execute(
                    """
                    SELECT start, end
                    FROM CUPTI_ACTIVITY_KIND_KERNEL
                    WHERE deviceId = ?
                      AND start < ?
                      AND end > ?
                    ORDER BY start
                    """,
                    (device_id, global_end, global_start),
                )
            ]
        )
        if prefetch_streams:
            h2d_query = f"""
                SELECT start, end
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE deviceId = ?
                  AND copyKind = 1
                  AND streamId IN ({stream_placeholders})
                  AND start < ?
                  AND end > ?
                ORDER BY start
            """
            h2d_params = (device_id, *prefetch_streams, global_end, global_start)
            h2d_intervals = merge_intervals(
                [(int(start), int(end)) for start, end in conn.execute(h2d_query, h2d_params)]
            )
        else:
            h2d_intervals = []

        nccl_total_ns = 0
        overlap_ns = 0
        uncovered_ns = 0
        for step_start, step_end in step_windows:
            nccl_step = clip_and_merge(nccl_intervals, step_start, step_end)
            kernel_step = clip_and_merge(all_kernel_intervals, step_start, step_end)
            h2d_step = clip_and_merge(h2d_intervals, step_start, step_end)
            nccl_total_ns += total_len(nccl_step)
            overlap_ns += intersect_total(nccl_step, h2d_step)
            uncovered_ns += subtract_total(h2d_step, kernel_step)

        step_count = len(step_windows)
        per_device_nccl_ms.append(nccl_total_ns / 1e6 / step_count)
        per_device_overlap_ms.append(overlap_ns / 1e6 / step_count)
        per_device_uncovered_ms.append(uncovered_ns / 1e6 / step_count)

    wait_bg_ns = sum_marker_overlap(conn, WAIT_COMM_BG, step_windows)
    conn.close()

    device_count = len(device_ids)
    step_count = len(step_windows)
    trace_label = trace_path.stem.replace(".nsys", "")

    return TraceMetrics(
        trace_name=trace_label,
        step_avg_ms=step_avg_ms,
        step_count=step_count,
        device_count=device_count,
        prefetch_streams=prefetch_streams,
        nccl_total_ms=sum(per_device_nccl_ms) / device_count,
        nccl_h2d_overlap_ms=sum(per_device_overlap_ms) / device_count,
        uncovered_h2d_ms=sum(per_device_uncovered_ms) / device_count,
        wait_comm_bg_ms=wait_bg_ns / 1e6 / (step_count * device_count),
    )


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def build_markdown(
    input_path: Path,
    metrics: list[TraceMetrics],
    errors: list[tuple[str, str]],
) -> str:
    lines = [
        "# Analysis Summary",
        "",
        f"- Input: `{input_path}`",
        "- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.",
        "- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.",
        "",
        "| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in metrics:
        stream_text = ", ".join(str(s) for s in item.prefetch_streams) if item.prefetch_streams else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.trace_name,
                    fmt_float(item.step_avg_ms),
                    fmt_float(item.nccl_total_ms),
                    fmt_float(item.nccl_h2d_overlap_ms),
                    fmt_float(item.uncovered_h2d_ms),
                    fmt_float(item.wait_comm_bg_ms),
                    str(item.step_count),
                    str(item.device_count),
                    stream_text,
                ]
            )
            + " |"
        )

    if errors:
        lines.extend(["", "## Errors", ""])
        for trace_name, error in errors:
            lines.append(f"- `{trace_name}`: {error}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_paths = collect_inputs(args.input_paths)
    root_dir = common_parent(input_paths)
    output_path = args.output or (root_dir / "analysis_summary.md")
    export_dir = args.export_dir or (root_dir / "sqlite_cache_steps")
    export_dir.mkdir(parents=True, exist_ok=True)

    prefetch_min_bytes = max(1, int(args.prefetch_min_mb * 1024 * 1024))
    prefetch_marker_slack_ns = int(args.prefetch_marker_slack_ms * 1e6)

    metrics: list[TraceMetrics] = []
    errors: list[tuple[str, str]] = []

    for path in input_paths:
        trace_label = path.stem.replace(".nsys", "")
        try:
            sqlite_path = ensure_sqlite(path, export_dir, args.force_export)
            metrics.append(
                compute_metrics_for_trace(
                    trace_path=path,
                    sqlite_path=sqlite_path,
                    prefetch_min_bytes=prefetch_min_bytes,
                    prefetch_marker_slack_ns=prefetch_marker_slack_ns,
                )
            )
            log(f"Analyzed {path.name}")
        except Exception as exc:  # noqa: BLE001
            errors.append((trace_label, str(exc)))
            log(f"Failed {path.name}: {exc}")

    metrics.sort(key=lambda item: item.trace_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(args.input_path, metrics, errors))
    log(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()
