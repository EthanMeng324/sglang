#!/usr/bin/env python3
"""Summarize nsys_query CSV outputs into a readable Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def safe_pct_change(new_val: float | None, base_val: float | None) -> float | None:
    if new_val is None or base_val is None or base_val == 0:
        return None
    return (new_val / base_val - 1.0) * 100.0


def find_context(rows: list[dict[str, str]], context: str) -> dict[str, str]:
    for row in rows:
        if row.get("context") == context:
            return row
    return {}


def find_by_key(
    rows: list[dict[str, str]], key: str, value: str
) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def find_run(rows: list[dict[str, str]], run: str) -> dict[str, str]:
    for row in rows:
        if row.get("run") == run:
            return row
    return {}


def aggregate_marker_metrics(
    rows: list[dict[str, str]], markers: list[str]
) -> tuple[float | None, float | None, float | None]:
    matched = [r for r in rows if r.get("marker") in markers]
    if not matched:
        return None, None, None
    total_ms = sum(to_float(r.get("total_ms")) or 0.0 for r in matched)
    total_count = sum(to_float(r.get("num_ranges")) or 0.0 for r in matched)
    avg_ms = None
    if total_count > 0:
        avg_ms = total_ms / total_count
    return total_ms, avg_ms, total_count


def one_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def load_denoise_steps_from_perf_json(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    steps = data.get("denoise_steps_ms")
    if not isinstance(steps, list):
        return {}
    out: dict[int, float] = {}
    for item in steps:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        dur = item.get("duration_ms")
        if isinstance(step, int) and isinstance(dur, (int, float)):
            out[step] = float(dur)
    return out


def normalize_step_components(
    total_ms: float | None, comm_ms: float | None, ensure_ms: float | None
) -> tuple[float | None, float | None, float | None, float | None]:
    if total_ms is None:
        return None, None, None, None
    total = max(0.0, total_ms)
    comm = max(0.0, comm_ms or 0.0)
    if comm > total:
        comm = total
    ensure = max(0.0, ensure_ms or 0.0)
    max_ensure = max(0.0, total - comm)
    if ensure > max_ensure:
        ensure = max_ensure
    compute = total - comm - ensure
    return total, compute, comm, ensure


def build_summary(analysis_dir: Path) -> str:
    marker_status = load_csv(analysis_dir / "marker_status.csv")

    prefetch_stats_new = one_row(load_csv(analysis_dir / "prefetch_h2d_stats_new.csv"))
    prefetch_stats_old = one_row(load_csv(analysis_dir / "prefetch_h2d_stats_old.csv"))
    prefetch_mode_new = load_csv(analysis_dir / "prefetch_mode_breakdown_new.csv")
    prefetch_mode_old = load_csv(analysis_dir / "prefetch_mode_breakdown_old.csv")
    prefetch_ensure_mem_new = one_row(
        load_csv(analysis_dir / "prefetch_ensure_memory_ratio_new.csv")
    )
    prefetch_ensure_mem_old = one_row(
        load_csv(analysis_dir / "prefetch_ensure_memory_ratio_old.csv")
    )
    prefetch_wait_new = load_csv(analysis_dir / "prefetch_wait_breakdown_new.csv")
    prefetch_wait_old = load_csv(analysis_dir / "prefetch_wait_breakdown_old.csv")
    mock_timing_new = load_csv(analysis_dir / "mock_comm_timing_new.csv")
    mock_timing_old = load_csv(analysis_dir / "mock_comm_timing_old.csv")

    h2d_new = one_row(load_csv(analysis_dir / "h2d_new.csv"))
    h2d_old = one_row(load_csv(analysis_dir / "h2d_old.csv"))
    d2h_new = one_row(load_csv(analysis_dir / "d2h_new.csv"))
    d2h_old = one_row(load_csv(analysis_dir / "d2h_old.csv"))

    h2d_buckets_new = load_csv(analysis_dir / "h2d_buckets_new.csv")
    h2d_buckets_old = load_csv(analysis_dir / "h2d_buckets_old.csv")

    mock_overlap_new = load_csv(analysis_dir / "mock_d2h_overlap_new.csv")
    mock_overlap_old = load_csv(analysis_dir / "mock_d2h_overlap_old.csv")
    mock_overlap_ratio_new = one_row(load_csv(analysis_dir / "mock_d2h_overlap_ratio_new.csv"))
    mock_overlap_ratio_old = one_row(load_csv(analysis_dir / "mock_d2h_overlap_ratio_old.csv"))
    mock_order_new = one_row(load_csv(analysis_dir / "mock_d2h_prefetch_order_new.csv"))
    mock_order_old = one_row(load_csv(analysis_dir / "mock_d2h_prefetch_order_old.csv"))
    denoise_step_new = load_csv(analysis_dir / "denoising_step_duration_new.csv")
    denoise_step_old = load_csv(analysis_dir / "denoising_step_duration_old.csv")
    denoise_comp_new = load_csv(analysis_dir / "denoising_step_components_new.csv")
    denoise_comp_old = load_csv(analysis_dir / "denoising_step_components_old.csv")

    idle_new = one_row(load_csv(analysis_dir / "idle_gaps_new.csv"))
    idle_old = one_row(load_csv(analysis_dir / "idle_gaps_old.csv"))
    idle_break_new = one_row(load_csv(analysis_dir / "idle_gap_breakdown_new.csv"))
    idle_break_old = one_row(load_csv(analysis_dir / "idle_gap_breakdown_old.csv"))

    lg_new = one_row(load_csv(analysis_dir / "large_gaps_new.csv"))
    lg_old = one_row(load_csv(analysis_dir / "large_gaps_old.csv"))

    marker_new = find_run(marker_status, "new")
    marker_old = find_run(marker_status, "old")

    m_new_denoise = to_float(marker_new.get("denoise_windows")) if marker_new else None
    m_old_denoise = to_float(marker_old.get("denoise_windows")) if marker_old else None
    m_new_prefetch = to_float(marker_new.get("prefetch_nvtx_ranges")) if marker_new else None
    m_old_prefetch = to_float(marker_old.get("prefetch_nvtx_ranges")) if marker_old else None
    m_new_mock = to_float(marker_new.get("mock_nvtx_ranges")) if marker_new else None
    m_old_mock = to_float(marker_old.get("mock_nvtx_ranges")) if marker_old else None

    pre_cnt_new = to_float(prefetch_stats_new.get("prefetch_h2d_count"))
    pre_cnt_old = to_float(prefetch_stats_old.get("prefetch_h2d_count"))
    pre_size_new = to_float(prefetch_stats_new.get("prefetch_h2d_avg_size_mb"))
    pre_size_old = to_float(prefetch_stats_old.get("prefetch_h2d_avg_size_mb"))
    pre_dur_new = to_float(prefetch_stats_new.get("prefetch_h2d_avg_dur_ms"))
    pre_dur_old = to_float(prefetch_stats_old.get("prefetch_h2d_avg_dur_ms"))
    pre_bw_new = to_float(prefetch_stats_new.get("prefetch_h2d_avg_bw_gbps"))
    pre_bw_old = to_float(prefetch_stats_old.get("prefetch_h2d_avg_bw_gbps"))
    pre_small_new = to_float(prefetch_stats_new.get("small_lt4mb_ratio_pct"))
    pre_small_old = to_float(prefetch_stats_old.get("small_lt4mb_ratio_pct"))
    pre_large_new = to_float(prefetch_stats_new.get("large_gt64mb_ratio_pct"))
    pre_large_old = to_float(prefetch_stats_old.get("large_gt64mb_ratio_pct"))

    pre_cnt_ratio = safe_ratio(pre_cnt_new, pre_cnt_old)
    pre_size_ratio = safe_ratio(pre_size_new, pre_size_old)

    mode_new_sync = find_by_key(prefetch_mode_new, "prefetch_mode", "sync")
    mode_new_bg = find_by_key(prefetch_mode_new, "prefetch_mode", "background")
    mode_new_unlabeled = find_by_key(prefetch_mode_new, "prefetch_mode", "unlabeled")
    mode_old_sync = find_by_key(prefetch_mode_old, "prefetch_mode", "sync")
    mode_old_bg = find_by_key(prefetch_mode_old, "prefetch_mode", "background")
    mode_old_unlabeled = find_by_key(prefetch_mode_old, "prefetch_mode", "unlabeled")

    mode_new_sync_gb = to_float(mode_new_sync.get("total_gb")) or 0.0
    mode_new_bg_gb = (to_float(mode_new_bg.get("total_gb")) or 0.0) + (
        to_float(mode_new_unlabeled.get("total_gb")) or 0.0
    )
    mode_old_sync_gb = to_float(mode_old_sync.get("total_gb")) or 0.0
    mode_old_bg_gb = (to_float(mode_old_bg.get("total_gb")) or 0.0) + (
        to_float(mode_old_unlabeled.get("total_gb")) or 0.0
    )

    mode_new_total_gb = mode_new_sync_gb + mode_new_bg_gb
    mode_old_total_gb = mode_old_sync_gb + mode_old_bg_gb
    mode_new_sync_ratio = (
        None if mode_new_total_gb <= 0 else mode_new_sync_gb / mode_new_total_gb
    )
    mode_old_sync_ratio = (
        None if mode_old_total_gb <= 0 else mode_old_sync_gb / mode_old_total_gb
    )

    mode_new_sync_time_s = to_float(mode_new_sync.get("total_time_s")) or 0.0
    mode_new_bg_time_s = (to_float(mode_new_bg.get("total_time_s")) or 0.0) + (
        to_float(mode_new_unlabeled.get("total_time_s")) or 0.0
    )
    mode_old_sync_time_s = to_float(mode_old_sync.get("total_time_s")) or 0.0
    mode_old_bg_time_s = (to_float(mode_old_bg.get("total_time_s")) or 0.0) + (
        to_float(mode_old_unlabeled.get("total_time_s")) or 0.0
    )

    mode_new_sync_bw = (
        None if mode_new_sync_time_s <= 0 else mode_new_sync_gb / mode_new_sync_time_s
    )
    mode_new_bg_bw = (
        None if mode_new_bg_time_s <= 0 else mode_new_bg_gb / mode_new_bg_time_s
    )
    mode_old_sync_bw = (
        None if mode_old_sync_time_s <= 0 else mode_old_sync_gb / mode_old_sync_time_s
    )
    mode_old_bg_bw = (
        None if mode_old_bg_time_s <= 0 else mode_old_bg_gb / mode_old_bg_time_s
    )
    ensure_prefetch_ratio_new = to_float(prefetch_ensure_mem_new.get("ensure_prefetch_ratio_pct"))
    ensure_prefetch_ratio_old = to_float(prefetch_ensure_mem_old.get("ensure_prefetch_ratio_pct"))

    wait_comm_markers = [
        "SGL_PREFETCH_WAIT_COMM",
        "SGL_PREFETCH_WAIT_COMM_BG",
        "SGL_PREFETCH_WAIT_COMM_SYNC",
    ]
    wait_lock_markers = [
        "SGL_PREFETCH_WAIT_LOCK_BG",
        "SGL_PREFETCH_WAIT_LOCK_SYNC",
    ]
    ensure_markers = ["SGL_PREFETCH_ENSURE_READY"]

    (
        wait_new_comm_ms,
        wait_new_comm_avg_ms,
        wait_new_comm_count,
    ) = aggregate_marker_metrics(prefetch_wait_new, wait_comm_markers)
    (
        wait_old_comm_ms,
        wait_old_comm_avg_ms,
        wait_old_comm_count,
    ) = aggregate_marker_metrics(prefetch_wait_old, wait_comm_markers)
    (
        wait_new_lock_ms,
        wait_new_lock_avg_ms,
        wait_new_lock_count,
    ) = aggregate_marker_metrics(prefetch_wait_new, wait_lock_markers)
    (
        wait_old_lock_ms,
        wait_old_lock_avg_ms,
        wait_old_lock_count,
    ) = aggregate_marker_metrics(prefetch_wait_old, wait_lock_markers)
    (
        wait_new_ensure_ms,
        wait_new_ensure_avg_ms,
        wait_new_ensure_count,
    ) = aggregate_marker_metrics(prefetch_wait_new, ensure_markers)
    (
        wait_old_ensure_ms,
        wait_old_ensure_avg_ms,
        wait_old_ensure_count,
    ) = aggregate_marker_metrics(prefetch_wait_old, ensure_markers)

    mock_new_pcie = find_by_key(mock_timing_new, "marker", "SGL_MOCK_COMM_PCIE")
    mock_old_pcie = find_by_key(mock_timing_old, "marker", "SGL_MOCK_COMM_PCIE")
    mock_new_active = find_by_key(mock_timing_new, "marker", "SGL_MOCK_COMM_ACTIVE")
    mock_old_active = find_by_key(mock_timing_old, "marker", "SGL_MOCK_COMM_ACTIVE")
    mock_new_pcie_avg_ms = to_float(mock_new_pcie.get("avg_ms"))
    mock_old_pcie_avg_ms = to_float(mock_old_pcie.get("avg_ms"))
    mock_new_active_avg_ms = to_float(mock_new_active.get("avg_ms"))
    mock_old_active_avg_ms = to_float(mock_old_active.get("avg_ms"))
    mock_new_active_total_ms = to_float(mock_new_active.get("total_ms"))
    mock_old_active_total_ms = to_float(mock_old_active.get("total_ms"))

    overlap_new_during = find_context(mock_overlap_new, "during_prefetch_h2d")
    overlap_new_no = find_context(mock_overlap_new, "without_prefetch_h2d")
    overlap_old_during = find_context(mock_overlap_old, "during_prefetch_h2d")
    overlap_old_no = find_context(mock_overlap_old, "without_prefetch_h2d")

    new_during_bw = to_float(overlap_new_during.get("avg_mock_d2h_bw_mbps"))
    new_no_bw = to_float(overlap_new_no.get("avg_mock_d2h_bw_mbps"))
    old_during_bw = to_float(overlap_old_during.get("avg_mock_d2h_bw_mbps"))
    old_no_bw = to_float(overlap_old_no.get("avg_mock_d2h_bw_mbps"))

    new_during_dur = to_float(overlap_new_during.get("avg_mock_d2h_dur_ms"))
    new_no_dur = to_float(overlap_new_no.get("avg_mock_d2h_dur_ms"))
    old_during_dur = to_float(overlap_old_during.get("avg_mock_d2h_dur_ms"))
    old_no_dur = to_float(overlap_old_no.get("avg_mock_d2h_dur_ms"))

    new_wait_vs_d2h = safe_ratio(wait_new_comm_avg_ms, new_no_dur)
    old_wait_vs_d2h = safe_ratio(wait_old_comm_avg_ms, old_no_dur)
    new_active_vs_d2h = safe_ratio(mock_new_active_avg_ms, new_no_dur)
    old_active_vs_d2h = safe_ratio(mock_old_active_avg_ms, old_no_dur)

    new_bw_drop_pct = None
    if new_during_bw is not None and new_no_bw is not None and new_no_bw > 0:
        new_bw_drop_pct = (1.0 - new_during_bw / new_no_bw) * 100.0
    old_bw_drop_pct = None
    if old_during_bw is not None and old_no_bw is not None and old_no_bw > 0:
        old_bw_drop_pct = (1.0 - old_during_bw / old_no_bw) * 100.0

    new_dur_increase_pct = safe_pct_change(new_during_dur, new_no_dur)
    old_dur_increase_pct = safe_pct_change(old_during_dur, old_no_dur)

    new_overlap_ratio = to_float(mock_overlap_ratio_new.get("overlapped_mock_d2h_ratio_pct"))
    old_overlap_ratio = to_float(mock_overlap_ratio_old.get("overlapped_mock_d2h_ratio_pct"))
    new_overlap_count = to_float(mock_overlap_ratio_new.get("overlapped_mock_d2h_count"))
    old_overlap_count = to_float(mock_overlap_ratio_old.get("overlapped_mock_d2h_count"))
    new_overlap_total = to_float(mock_overlap_ratio_new.get("total_mock_d2h_count"))
    old_overlap_total = to_float(mock_overlap_ratio_old.get("total_mock_d2h_count"))

    new_order_total = to_float(mock_order_new.get("total_mock_d2h_count"))
    old_order_total = to_float(mock_order_old.get("total_mock_d2h_count"))
    new_prev_lt1 = to_float(mock_order_new.get("prev_gap_lt1ms_count"))
    old_prev_lt1 = to_float(mock_order_old.get("prev_gap_lt1ms_count"))
    new_next_lt1 = to_float(mock_order_new.get("next_gap_lt1ms_count"))
    old_next_lt1 = to_float(mock_order_old.get("next_gap_lt1ms_count"))
    new_prev_lt1_ratio = safe_ratio(new_prev_lt1, new_order_total)
    old_prev_lt1_ratio = safe_ratio(old_prev_lt1, old_order_total)
    new_next_lt1_ratio = safe_ratio(new_next_lt1, new_order_total)
    old_next_lt1_ratio = safe_ratio(old_next_lt1, old_order_total)
    new_avg_gap_prev_ms = to_float(mock_order_new.get("avg_gap_from_prev_prefetch_end_ms"))
    old_avg_gap_prev_ms = to_float(mock_order_old.get("avg_gap_from_prev_prefetch_end_ms"))
    new_avg_gap_next_ms = to_float(mock_order_new.get("avg_gap_to_next_prefetch_start_ms"))
    old_avg_gap_next_ms = to_float(mock_order_old.get("avg_gap_to_next_prefetch_start_ms"))

    step_new_total_map: dict[int, float] = {}
    for row in denoise_step_new:
        idx = to_float(row.get("step_idx"))
        dur = to_float(row.get("avg_ms"))
        if idx is None or dur is None:
            continue
        step_new_total_map[int(idx)] = dur

    step_old_total_map: dict[int, float] = {}
    for row in denoise_step_old:
        idx = to_float(row.get("step_idx"))
        dur = to_float(row.get("avg_ms"))
        if idx is None or dur is None:
            continue
        step_old_total_map[int(idx)] = dur

    step_comp_new_map: dict[int, dict[str, float]] = {}
    for row in denoise_comp_new:
        idx = to_float(row.get("step_idx"))
        if idx is None:
            continue
        total = to_float(row.get("step_total_ms"))
        if total is None:
            continue
        comm = max(0.0, to_float(row.get("comm_ms")) or 0.0)
        ensure = max(0.0, to_float(row.get("ensure_ms")) or 0.0)
        comm = min(comm, total)
        ensure = min(ensure, total)
        kernel_ms = to_float(row.get("kernel_ms"))
        if kernel_ms is None:
            _, compute, _, _ = normalize_step_components(total, comm, ensure)
            compute = 0.0 if compute is None else compute
        else:
            compute = max(0.0, kernel_ms)
        step_comp_new_map[int(idx)] = {
            "total": total,
            "compute": compute,
            "comm": comm,
            "ensure": ensure,
            "residual": total - comm - ensure,
            "closure": total - compute - comm - ensure,
        }

    step_comp_old_map: dict[int, dict[str, float]] = {}
    for row in denoise_comp_old:
        idx = to_float(row.get("step_idx"))
        if idx is None:
            continue
        total = to_float(row.get("step_total_ms"))
        if total is None:
            continue
        comm = max(0.0, to_float(row.get("comm_ms")) or 0.0)
        ensure = max(0.0, to_float(row.get("ensure_ms")) or 0.0)
        comm = min(comm, total)
        ensure = min(ensure, total)
        kernel_ms = to_float(row.get("kernel_ms"))
        if kernel_ms is None:
            _, compute, _, _ = normalize_step_components(total, comm, ensure)
            compute = 0.0 if compute is None else compute
        else:
            compute = max(0.0, kernel_ms)
        step_comp_old_map[int(idx)] = {
            "total": total,
            "compute": compute,
            "comm": comm,
            "ensure": ensure,
            "residual": total - comm - ensure,
            "closure": total - compute - comm - ensure,
        }

    step_source_new = "nsys_components"
    step_source_old = "nsys_components"
    results_dir = analysis_dir.parent.parent

    # Fallback path: only per-step total exists, then decompose as compute=total.
    if not step_comp_new_map and step_new_total_map:
        for idx, total in step_new_total_map.items():
            t, compute, comm, ensure = normalize_step_components(total, 0.0, 0.0)
            if t is None or compute is None or comm is None or ensure is None:
                continue
            step_comp_new_map[idx] = {
                "total": t,
                "compute": compute,
                "comm": comm,
                "ensure": ensure,
            }
        step_source_new = "nsys_total_only"

    if not step_comp_old_map and step_old_total_map:
        for idx, total in step_old_total_map.items():
            t, compute, comm, ensure = normalize_step_components(total, 0.0, 0.0)
            if t is None or compute is None or comm is None or ensure is None:
                continue
            step_comp_old_map[idx] = {
                "total": t,
                "compute": compute,
                "comm": comm,
                "ensure": ensure,
            }
        step_source_old = "nsys_total_only"

    if not step_comp_new_map:
        perf_step_new = load_denoise_steps_from_perf_json(
            results_dir / "perf_new_offload_profiled.json"
        )
        if perf_step_new:
            for idx, total in perf_step_new.items():
                t, compute, comm, ensure = normalize_step_components(total, 0.0, 0.0)
                if t is None or compute is None or comm is None or ensure is None:
                    continue
                step_comp_new_map[idx] = {
                    "total": t,
                    "compute": compute,
                    "comm": comm,
                    "ensure": ensure,
                }
            step_source_new = "perf_json_total_only"

    if not step_comp_old_map:
        perf_step_old = load_denoise_steps_from_perf_json(
            results_dir / "perf_old_offload_profiled.json"
        )
        if perf_step_old:
            for idx, total in perf_step_old.items():
                t, compute, comm, ensure = normalize_step_components(total, 0.0, 0.0)
                if t is None or compute is None or comm is None or ensure is None:
                    continue
                step_comp_old_map[idx] = {
                    "total": t,
                    "compute": compute,
                    "comm": comm,
                    "ensure": ensure,
                }
            step_source_old = "perf_json_total_only"

    all_step_indices = sorted(set(step_comp_new_map) | set(step_comp_old_map))
    step_new_total_sum = (
        sum(v["total"] for v in step_comp_new_map.values()) if step_comp_new_map else None
    )
    step_old_total_sum = (
        sum(v["total"] for v in step_comp_old_map.values()) if step_comp_old_map else None
    )
    step_new_ensure_sum = (
        sum(v["ensure"] for v in step_comp_new_map.values()) if step_comp_new_map else None
    )
    step_old_ensure_sum = (
        sum(v["ensure"] for v in step_comp_old_map.values()) if step_comp_old_map else None
    )
    ensure_wait_ratio_new = safe_ratio(step_new_ensure_sum, step_new_total_sum)
    ensure_wait_ratio_old = safe_ratio(step_old_ensure_sum, step_old_total_sum)

    idle_total_new = to_float(idle_new.get("total_gap_ms"))
    idle_total_old = to_float(idle_old.get("total_gap_ms"))

    idle_prefetch_new = to_float(idle_break_new.get("gap_with_prefetch_h2d_ms"))
    idle_prefetch_old = to_float(idle_break_old.get("gap_with_prefetch_h2d_ms"))
    idle_mock_new = to_float(idle_break_new.get("gap_with_mock_d2h_ms"))
    idle_mock_old = to_float(idle_break_old.get("gap_with_mock_d2h_ms"))
    idle_none_new = to_float(idle_break_new.get("gap_without_memcpy_ms"))
    idle_none_old = to_float(idle_break_old.get("gap_without_memcpy_ms"))

    idle_prefetch_new_pct = safe_ratio(idle_prefetch_new, idle_total_new)
    idle_prefetch_old_pct = safe_ratio(idle_prefetch_old, idle_total_old)
    idle_mock_new_pct = safe_ratio(idle_mock_new, idle_total_new)
    idle_mock_old_pct = safe_ratio(idle_mock_old, idle_total_old)
    idle_none_new_pct = safe_ratio(idle_none_new, idle_total_new)
    idle_none_old_pct = safe_ratio(idle_none_old, idle_total_old)

    h2d_bw_new = to_float(h2d_new.get("avg_bw_gbps"))
    h2d_bw_old = to_float(h2d_old.get("avg_bw_gbps"))
    d2h_bw_new = to_float(d2h_new.get("avg_bw_gbps"))
    d2h_bw_old = to_float(d2h_old.get("avg_bw_gbps"))

    lines: list[str] = []
    lines.append("# NSYS 分析摘要")
    lines.append("")
    lines.append(f"- 生成时间(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 分析目录: `{analysis_dir}`")
    lines.append("")

    lines.append("## 0) Marker 覆盖度与含义")
    lines.append("")
    lines.append("| Marker | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(f"| SGL_DENOISING_LOOP | {fmt_num(m_new_denoise, 0)} | {fmt_num(m_old_denoise, 0)} |")
    lines.append(f"| SGL_PREFETCH_H2D* | {fmt_num(m_new_prefetch, 0)} | {fmt_num(m_old_prefetch, 0)} |")
    lines.append(f"| SGL_MOCK_COMM_PCIE | {fmt_num(m_new_mock, 0)} | {fmt_num(m_old_mock, 0)} |")
    lines.append("")
    lines.append("- `SGL_PREFETCH_H2D*` 统计包含 `H2D/H2D_BG/H2D_SYNC/H2D_CHUNK`，是 copy launch 次数，不是传输总量。")
    lines.append("- chunk-wise 会把一次整层拷贝拆成很多次小拷贝，因此该计数会显著更大。")
    lines.append("")

    marker_warnings: list[str] = []
    if m_new_denoise is not None and m_new_denoise <= 0:
        marker_warnings.append("New 缺少 `SGL_DENOISING_LOOP`，无法限定 denoising critical path。")
    if m_old_denoise is not None and m_old_denoise <= 0:
        marker_warnings.append("Old 缺少 `SGL_DENOISING_LOOP`，无法限定 denoising critical path。")
    if m_new_mock is not None and m_new_mock <= 0:
        marker_warnings.append("New 缺少 `SGL_MOCK_COMM_PCIE`，mock comm 只能按包大小启发式识别。")
    if m_old_mock is not None and m_old_mock <= 0:
        marker_warnings.append("Old 缺少 `SGL_MOCK_COMM_PCIE`，mock comm 只能按包大小启发式识别。")
    if marker_warnings:
        lines.append("**Marker 风险提示**")
        lines.append("")
        for w in marker_warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## 1) Prefetch H2D 执行画像（定位 chunk 开销）")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(f"| prefetch copy 次数 | {fmt_num(pre_cnt_new, 0)} | {fmt_num(pre_cnt_old, 0)} |")
    lines.append(f"| 平均单次 copy 大小 (MB) | {fmt_num(pre_size_new, 2)} | {fmt_num(pre_size_old, 2)} |")
    lines.append(f"| 平均单次 copy 时长 (ms) | {fmt_num(pre_dur_new, 3)} | {fmt_num(pre_dur_old, 3)} |")
    lines.append(f"| prefetch H2D 平均带宽 (GB/s) | {fmt_num(pre_bw_new, 2)} | {fmt_num(pre_bw_old, 2)} |")
    lines.append(f"| 小包占比 <4MB (%) | {fmt_num(pre_small_new, 2)} | {fmt_num(pre_small_old, 2)} |")
    lines.append(f"| 大包占比 >64MB (%) | {fmt_num(pre_large_new, 2)} | {fmt_num(pre_large_old, 2)} |")
    lines.append("")
    lines.append(f"- copy 次数倍率 New/Old: {fmt_num(pre_cnt_ratio, 2)}x")
    lines.append(f"- 平均 copy 大小 New/Old: {fmt_num(pre_size_ratio, 3)}x")
    lines.append("")

    lines.append("## 1b) Prefetch 等待/就绪开销（NVTX时长 + ensure字节占比）")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| forward窗口外 ensure等待占比(按step时长) (%) | "
        f"{fmt_num(None if ensure_wait_ratio_new is None else ensure_wait_ratio_new * 100, 2)} | "
        f"{fmt_num(None if ensure_wait_ratio_old is None else ensure_wait_ratio_old * 100, 2)} |"
    )
    lines.append(
        f"| ensure窗口内 prefetch字节占比(%) | "
        f"{fmt_num(ensure_prefetch_ratio_new, 2)} | "
        f"{fmt_num(ensure_prefetch_ratio_old, 2)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_COMM` 总时长 (ms) | {fmt_num(wait_new_comm_ms, 2)} | {fmt_num(wait_old_comm_ms, 2)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_COMM` 平均时长 (ms) | {fmt_num(wait_new_comm_avg_ms, 3)} | {fmt_num(wait_old_comm_avg_ms, 3)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_COMM` 次数 | {fmt_num(wait_new_comm_count, 0)} | {fmt_num(wait_old_comm_count, 0)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_LOCK_*` 总时长 (ms) | {fmt_num(wait_new_lock_ms, 2)} | {fmt_num(wait_old_lock_ms, 2)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_LOCK_*` 平均时长 (ms) | {fmt_num(wait_new_lock_avg_ms, 3)} | {fmt_num(wait_old_lock_avg_ms, 3)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_LOCK_*` 次数 | {fmt_num(wait_new_lock_count, 0)} | {fmt_num(wait_old_lock_count, 0)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_ENSURE_READY` 总时长 (ms) | {fmt_num(wait_new_ensure_ms, 2)} | {fmt_num(wait_old_ensure_ms, 2)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_ENSURE_READY` 平均时长 (ms) | {fmt_num(wait_new_ensure_avg_ms, 3)} | {fmt_num(wait_old_ensure_avg_ms, 3)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_ENSURE_READY` 次数 | {fmt_num(wait_new_ensure_count, 0)} | {fmt_num(wait_old_ensure_count, 0)} |"
    )
    lines.append("")
    lines.append("- `forward窗口外 ensure等待占比` 由每步 `ensure_ms / step_total_ms` 聚合得到。")
    lines.append("- `ensure窗口内 prefetch字节占比` 由 H2D prefetch 与 `SGL_PREFETCH_ENSURE_READY` 时间窗重叠的字节比例计算。")
    lines.append("- `unlabeled` prefetch copy 已并入 `background`。")
    lines.append("")

    lines.append("## 1c) New: mock 实际时间 vs comm-wait 时间")
    lines.append("")
    lines.append("| 指标 | New |")
    lines.append("|---|---:|")
    lines.append(
        f"| `SGL_MOCK_COMM_ACTIVE` 平均时长 (ms) | {fmt_num(mock_new_active_avg_ms, 3)} |"
    )
    lines.append(
        f"| `SGL_MOCK_COMM_ACTIVE` 总时长 (ms) | {fmt_num(mock_new_active_total_ms, 2)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_COMM` 平均时长 (ms) | {fmt_num(wait_new_comm_avg_ms, 3)} |"
    )
    lines.append(
        f"| `SGL_PREFETCH_WAIT_COMM` 总时长 (ms) | {fmt_num(wait_new_comm_ms, 2)} |"
    )
    lines.append(
        f"| `WAIT_COMM(avg)` / `MOCK_ACTIVE(avg)` | {fmt_num(safe_ratio(wait_new_comm_avg_ms, mock_new_active_avg_ms), 2)}x |"
    )
    lines.append("")

    lines.append("## 2) mock D2H 与 prefetch H2D 重叠（核心目标）")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| 重叠比例 (%) | {fmt_num(new_overlap_ratio, 2)} | {fmt_num(old_overlap_ratio, 2)} |"
    )
    lines.append(
        f"| 重叠事件数 / 总事件数 | {fmt_num(new_overlap_count, 0)} / {fmt_num(new_overlap_total, 0)} | {fmt_num(old_overlap_count, 0)} / {fmt_num(old_overlap_total, 0)} |"
    )
    lines.append("")

    lines.append("## 2a) mock 与 prefetch 先后顺序校验")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 总 mock D2H 数 | {fmt_num(new_order_total, 0)} | {fmt_num(old_order_total, 0)} |")
    lines.append(
        f"| mock 前 1ms 内有 prefetch 结束占比 (%) | "
        f"{fmt_num(None if new_prev_lt1_ratio is None else new_prev_lt1_ratio * 100, 2)} | "
        f"{fmt_num(None if old_prev_lt1_ratio is None else old_prev_lt1_ratio * 100, 2)} |"
    )
    lines.append(
        f"| mock 后 1ms 内有 prefetch 开始占比 (%) | "
        f"{fmt_num(None if new_next_lt1_ratio is None else new_next_lt1_ratio * 100, 2)} | "
        f"{fmt_num(None if old_next_lt1_ratio is None else old_next_lt1_ratio * 100, 2)} |"
    )
    lines.append(
        f"| 与上一个 prefetch 结束平均间隔 (ms) | {fmt_num(new_avg_gap_prev_ms, 3)} | {fmt_num(old_avg_gap_prev_ms, 3)} |"
    )
    lines.append(
        f"| 到下一个 prefetch 开始平均间隔 (ms) | {fmt_num(new_avg_gap_next_ms, 3)} | {fmt_num(old_avg_gap_next_ms, 3)} |"
    )
    lines.append("")

    lines.append("## 2b) Denoising 每步时长 (ms)")
    lines.append("")
    if all_step_indices:
        lines.append(
            f"- 数据来源: New={step_source_new}, Old={step_source_old}"
        )
        lines.append("- `计算` 列优先使用 `kernel_ms`（GPU kernel 与 step 窗口重叠时长），无该列时回退为残差。")
        lines.append("- `闭合差 = step总时长 - (计算 + mock_comm + ensure_wait)`，用于检查三项是否可加和。")
        lines.append("")
        lines.append(
            "| step | New总时长 | New计算 | New mock comm | New ensure_wait | "
            "Old总时长 | Old计算 | Old mock comm | Old ensure_wait |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for step_idx in all_step_indices:
            new_comp = step_comp_new_map.get(step_idx)
            old_comp = step_comp_old_map.get(step_idx)
            lines.append(
                f"| {step_idx} | "
                f"{fmt_num(new_comp['total'] if new_comp else None, 3)} | "
                f"{fmt_num(new_comp['compute'] if new_comp else None, 3)} | "
                f"{fmt_num(new_comp['comm'] if new_comp else None, 3)} | "
                f"{fmt_num(new_comp['ensure'] if new_comp else None, 3)} | "
                f"{fmt_num(old_comp['total'] if old_comp else None, 3)} | "
                f"{fmt_num(old_comp['compute'] if old_comp else None, 3)} | "
                f"{fmt_num(old_comp['comm'] if old_comp else None, 3)} | "
                f"{fmt_num(old_comp['ensure'] if old_comp else None, 3)} |"
            )

        def avg_comp(
            comp_map: dict[int, dict[str, float]], key: str
        ) -> float | None:
            if not comp_map:
                return None
            vals = [v[key] for v in comp_map.values() if key in v]
            if not vals:
                return None
            return sum(vals) / len(vals)

        def sum_comp(
            comp_map: dict[int, dict[str, float]], key: str
        ) -> float | None:
            if not comp_map:
                return None
            vals = [v[key] for v in comp_map.values() if key in v]
            if not vals:
                return None
            return sum(vals)

        lines.append(
            f"| avg | "
            f"{fmt_num(avg_comp(step_comp_new_map, 'total'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_new_map, 'compute'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_new_map, 'comm'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_new_map, 'ensure'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_old_map, 'total'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_old_map, 'compute'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_old_map, 'comm'), 3)} | "
            f"{fmt_num(avg_comp(step_comp_old_map, 'ensure'), 3)} |"
        )
        lines.append(
            f"| sum | "
            f"{fmt_num(sum_comp(step_comp_new_map, 'total'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_new_map, 'compute'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_new_map, 'comm'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_new_map, 'ensure'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_old_map, 'total'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_old_map, 'compute'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_old_map, 'comm'), 3)} | "
            f"{fmt_num(sum_comp(step_comp_old_map, 'ensure'), 3)} |"
        )
        lines.append(
            f"| closure(sum) | "
            f"{fmt_num(sum_comp(step_comp_new_map, 'closure'), 3)} | - | - | - | "
            f"{fmt_num(sum_comp(step_comp_old_map, 'closure'), 3)} | - | - | - |"
        )
        lines.append(
            f"| closure(avg) | "
            f"{fmt_num(avg_comp(step_comp_new_map, 'closure'), 3)} | - | - | - | "
            f"{fmt_num(avg_comp(step_comp_old_map, 'closure'), 3)} | - | - | - |"
        )
    else:
        lines.append("- 未找到 `SGL_DENOISING_STEP_*` NVTX marker，且缺少 perf_dump JSON。")
    lines.append("")

    lines.append("## 3) Idle 拆分（定位等待来源）")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 总 idle gap (ms) | {fmt_num(idle_total_new, 2)} | {fmt_num(idle_total_old, 2)} |")
    lines.append(f"| idle 中与 prefetch H2D 重叠 (ms) | {fmt_num(idle_prefetch_new, 2)} | {fmt_num(idle_prefetch_old, 2)} |")
    lines.append(
        f"| idle 中与 prefetch H2D 重叠占比 (%) | "
        f"{fmt_num(None if idle_prefetch_new_pct is None else idle_prefetch_new_pct * 100, 2)} | "
        f"{fmt_num(None if idle_prefetch_old_pct is None else idle_prefetch_old_pct * 100, 2)} |"
    )
    lines.append(f"| idle 中与 mock D2H 重叠 (ms) | {fmt_num(idle_mock_new, 2)} | {fmt_num(idle_mock_old, 2)} |")
    lines.append(
        f"| idle 中与 mock D2H 重叠占比 (%) | "
        f"{fmt_num(None if idle_mock_new_pct is None else idle_mock_new_pct * 100, 2)} | "
        f"{fmt_num(None if idle_mock_old_pct is None else idle_mock_old_pct * 100, 2)} |"
    )
    lines.append(f"| idle 中无 memcpy 重叠 (ms) | {fmt_num(idle_none_new, 2)} | {fmt_num(idle_none_old, 2)} |")
    lines.append(
        f"| idle 中无 memcpy 重叠占比 (%) | "
        f"{fmt_num(None if idle_none_new_pct is None else idle_none_new_pct * 100, 2)} | "
        f"{fmt_num(None if idle_none_old_pct is None else idle_none_old_pct * 100, 2)} |"
    )
    lines.append("")

    lines.append("## 4) 最大空洞 (large_gaps)")
    lines.append("")
    lines.append(f"- New 最大 gap: {lg_new.get('gap_ms', 'N/A')} ms")
    lines.append(f"- Old 最大 gap: {lg_old.get('gap_ms', 'N/A')} ms")
    if lg_new:
        lines.append(
            f"- New gap 前后 kernel: `{lg_new.get('before_kernel', '')}` -> `{lg_new.get('after_kernel', '')}`"
        )
    if lg_old:
        lines.append(
            f"- Old gap 前后 kernel: `{lg_old.get('before_kernel', '')}` -> `{lg_old.get('after_kernel', '')}`"
        )
    lines.append("")

    lines.append("## 5) 诊断结论（用于下一步优化）")
    lines.append("")

    findings: list[str] = []

    if pre_cnt_ratio is not None and pre_size_ratio is not None and pre_cnt_ratio >= 4 and pre_size_ratio <= 0.35:
        findings.append(
            "new 的 prefetch 被明显碎片化（copy 次数大增且单次更小），优先怀疑 chunk 过细带来的 launch/调度开销。"
        )

    if new_overlap_ratio is not None and new_overlap_ratio > 5:
        findings.append("new 的 overlap 仍偏高，距离目标（趋近 0%）还有差距。")

    if idle_none_new is not None and idle_none_old is not None and idle_none_old > 0 and idle_none_new / idle_none_old >= 1.5:
        findings.append(
            "new 的非 memcpy idle 明显更高，存在额外调度/同步/控制流开销（不完全是带宽问题）。"
        )

    if (
        wait_new_comm_avg_ms is not None
        and mock_new_active_avg_ms is not None
        and mock_new_active_avg_ms > 0
        and (wait_new_comm_avg_ms / mock_new_active_avg_ms) > 1.2
    ):
        findings.append(
            "new 的 WAIT_COMM 平均时长高于 mock active 平均时长，comm 感知窗口可能偏宽。"
        )

    if (
        mode_new_sync_ratio is not None
        and mode_old_sync_ratio is not None
        and mode_new_sync_ratio > mode_old_sync_ratio + 0.10
    ):
        findings.append(
            "new 的 sync-catchup 传输占比明显高于 old，主要问题是 prefetch 截止前完成率不足。"
        )

    if (
        mode_new_sync_ratio is not None
        and mode_new_sync_ratio < 0.01
        and wait_new_ensure_ms is not None
        and wait_new_ensure_ms > 1000.0
    ):
        findings.append(
            "当前 `sync-catchup=0` 仅代表主线程几乎没有发起同步 copy；`ensure_ready` 仍可因等待后台 prefetch 完成而很高。"
        )

    if not findings:
        findings.append("当前数据没有显示单一瓶颈，建议继续按 chunk 大小和调度策略做 A/B 细分。")

    for item in findings:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("### 附加参考")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    lines.append(f"| 全量 H2D 平均带宽 (GB/s) | {fmt_num(h2d_bw_new, 2)} | {fmt_num(h2d_bw_old, 2)} |")
    lines.append(f"| 全量 D2H 平均带宽 (GB/s) | {fmt_num(d2h_bw_new, 2)} | {fmt_num(d2h_bw_old, 2)} |")
    lines.append(f"| prefetch 分桶条目数 | {fmt_num(float(len(h2d_buckets_new)), 0)} | {fmt_num(float(len(h2d_buckets_old)), 0)} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize nsys analysis CSVs to Markdown")
    parser.add_argument("--analysis-dir", default="results/profiles/analysis")
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output = Path(args.output) if args.output else analysis_dir / "analysis_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build_summary(analysis_dir)
    output.write_text(report)
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
