#!/usr/bin/env python3
"""Summarize the core offload profiling results into a compact Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def to_float(value: str | int | float | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None



def fmt_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"



def one_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def first_existing_row(analysis_dir: Path, names: list[str]) -> dict[str, str]:
    for name in names:
        row = one_row(load_csv(analysis_dir / name))
        if row:
            return row
    return {}



def find_perf_json(analysis_dir: Path, run: str) -> Path | None:
    profiles_dir = analysis_dir.parent
    results_dir = profiles_dir.parent
    exact_names = {
        "new": ["perf_new_offload_profiled.json", "perf_new_offload.json"],
        "old": ["perf_old_offload_profiled.json", "perf_old_offload.json"],
    }
    for root in (profiles_dir, results_dir):
        for name in exact_names[run]:
            path = root / name
            if path.exists():
                return path
    for root in (profiles_dir, results_dir):
        matches = sorted(root.glob(f"perf_{run}*.json"))
        if matches:
            return matches[-1]
    return None



def load_step_totals(analysis_dir: Path, run: str) -> tuple[dict[int, float], str]:
    perf_path = find_perf_json(analysis_dir, run)
    perf = load_json(perf_path)
    totals: dict[int, float] = {}
    for item in perf.get("denoise_steps_ms", []) or []:
        step = item.get("step")
        dur_ms = item.get("duration_ms")
        if isinstance(step, int) and isinstance(dur_ms, (int, float)):
            totals[step] = float(dur_ms)
    if totals:
        return totals, f"perf_json:{perf_path.name}"

    for row in load_csv(analysis_dir / f"denoising_step_duration_{run}.csv"):
        step = row.get("step_idx")
        avg_ms = to_float(row.get("avg_ms"))
        if step is None or avg_ms is None:
            continue
        totals[int(step)] = avg_ms
    return totals, "nsys_step_duration_csv"



def load_step_comm(analysis_dir: Path, run: str) -> dict[int, float]:
    comm: dict[int, float] = {}
    for row in load_csv(analysis_dir / f"denoising_step_components_{run}.csv"):
        step = row.get("step_idx")
        comm_ms = to_float(row.get("comm_ms"))
        if step is None or comm_ms is None:
            continue
        comm[int(step)] = comm_ms
    return comm



def load_prefetch_cp_wait(
    analysis_dir: Path, run: str
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float], str]:
    perf_path = find_perf_json(analysis_dir, run)
    perf = load_json(perf_path)
    offload_profile = perf.get("extra", {}).get("offload_profile", {})

    wait_ms = offload_profile.get("prefetch_critical_path_wait_ms")
    wait_layers = offload_profile.get("prefetch_critical_path_wait_layers")
    waited_bytes = offload_profile.get("prefetch_critical_path_waited_bytes")
    total_bytes = offload_profile.get("prefetch_critical_path_total_bytes")
    wait_map: dict[int, float] = {}
    layer_map: dict[int, float] = {}
    waited_bytes_map: dict[int, float] = {}
    total_bytes_map: dict[int, float] = {}

    if isinstance(wait_ms, list):
        for idx, value in enumerate(wait_ms):
            fv = to_float(value)
            if fv is not None:
                wait_map[idx] = fv
    if isinstance(wait_layers, list):
        for idx, value in enumerate(wait_layers):
            fv = to_float(value)
            if fv is not None:
                layer_map[idx] = fv
    if isinstance(waited_bytes, list):
        for idx, value in enumerate(waited_bytes):
            fv = to_float(value)
            if fv is not None:
                waited_bytes_map[idx] = fv
    if isinstance(total_bytes, list):
        for idx, value in enumerate(total_bytes):
            fv = to_float(value)
            if fv is not None:
                total_bytes_map[idx] = fv

    source = "runtime_cuda_events"
    if perf_path is not None:
        source += f":{perf_path.name}"
    return wait_map, layer_map, waited_bytes_map, total_bytes_map, source


def fmt_waited_over_total(waited_bytes: float | None, total_bytes: float | None) -> str:
    if waited_bytes is None or total_bytes is None or total_bytes <= 0:
        return "N/A"
    waited_mb = waited_bytes / (1024.0 * 1024.0)
    total_mb = total_bytes / (1024.0 * 1024.0)
    ratio_pct = 100.0 * waited_bytes / total_bytes
    return f"{waited_mb:.1f}/{total_mb:.1f} ({ratio_pct:.1f}%)"



def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)



def build_summary(analysis_dir: Path, comm_mode: str) -> str:
    overlap_ratio_new = first_existing_row(
        analysis_dir, ["comm_overlap_ratio_new.csv", "mock_d2h_overlap_ratio_new.csv"]
    )
    overlap_ratio_old = first_existing_row(
        analysis_dir, ["comm_overlap_ratio_old.csv", "mock_d2h_overlap_ratio_old.csv"]
    )

    totals_new, total_source_new = load_step_totals(analysis_dir, "new")
    totals_old, total_source_old = load_step_totals(analysis_dir, "old")
    comm_new = load_step_comm(analysis_dir, "new")
    comm_old = load_step_comm(analysis_dir, "old")
    cp_wait_new, cp_wait_layers_new, waited_bytes_new, total_bytes_new, cp_wait_source_new = load_prefetch_cp_wait(
        analysis_dir, "new"
    )
    cp_wait_old, cp_wait_layers_old, waited_bytes_old, total_bytes_old, cp_wait_source_old = load_prefetch_cp_wait(
        analysis_dir, "old"
    )

    all_steps = sorted(
        set(totals_new)
        | set(totals_old)
        | set(comm_new)
        | set(comm_old)
        | set(cp_wait_new)
        | set(cp_wait_old)
        | set(waited_bytes_new)
        | set(total_bytes_new)
        | set(waited_bytes_old)
        | set(total_bytes_old)
    )

    lines: list[str] = []
    lines.append("# NSYS 分析摘要")
    lines.append("")
    lines.append(
        f"- 生成时间(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"- 分析目录: `{analysis_dir}`")
    lines.append(f"- 通信模式: `{comm_mode}`")
    lines.append("")

    if comm_mode == "mock":
        lines.append("## 2) mock D2H 与 prefetch H2D 重叠（核心目标）")
    else:
        lines.append("## 2) NCCL 通信 与 prefetch H2D 重叠（核心目标）")
    lines.append("")
    lines.append("| 指标 | New | Old |")
    lines.append("|---|---:|---:|")
    total_key = "total_mock_d2h_count" if comm_mode == "mock" else "total_comm_count"
    overlap_key = (
        "overlapped_mock_d2h_count" if comm_mode == "mock" else "overlapped_comm_count"
    )
    ratio_key = (
        "overlapped_mock_d2h_ratio_pct"
        if comm_mode == "mock"
        else "overlapped_comm_ratio_pct"
    )
    name_label = "mock D2H" if comm_mode == "mock" else "NCCL 通信"
    lines.append(
        f"| {name_label} 总事件数 | {fmt_num(to_float(overlap_ratio_new.get(total_key)), 0)} | {fmt_num(to_float(overlap_ratio_old.get(total_key)), 0)} |"
    )
    lines.append(
        f"| 与 prefetch H2D 重叠事件数 | {fmt_num(to_float(overlap_ratio_new.get(overlap_key)), 0)} | {fmt_num(to_float(overlap_ratio_old.get(overlap_key)), 0)} |"
    )
    lines.append(
        f"| 重叠比例 (%) | {fmt_num(to_float(overlap_ratio_new.get(ratio_key)), 2)} | {fmt_num(to_float(overlap_ratio_old.get(ratio_key)), 2)} |"
    )
    lines.append(
        f"| 重叠时 {name_label} 平均时长 (ms) | {fmt_num(to_float(overlap_ratio_new.get('avg_dur_overlapped_ms')), 3)} | {fmt_num(to_float(overlap_ratio_old.get('avg_dur_overlapped_ms')), 3)} |"
    )
    lines.append(
        f"| 非重叠时 {name_label} 平均时长 (ms) | {fmt_num(to_float(overlap_ratio_new.get('avg_dur_non_overlapped_ms')), 3)} | {fmt_num(to_float(overlap_ratio_old.get('avg_dur_non_overlapped_ms')), 3)} |"
    )
    if comm_mode == "mock":
        lines.append(
            f"| 重叠时 mock D2H 平均带宽 (MB/s) | {fmt_num(to_float(overlap_ratio_new.get('avg_bw_overlapped_mbps')), 2)} | {fmt_num(to_float(overlap_ratio_old.get('avg_bw_overlapped_mbps')), 2)} |"
        )
        lines.append(
            f"| 非重叠时 mock D2H 平均带宽 (MB/s) | {fmt_num(to_float(overlap_ratio_new.get('avg_bw_non_overlapped_mbps')), 2)} | {fmt_num(to_float(overlap_ratio_old.get('avg_bw_non_overlapped_mbps')), 2)} |"
        )
    lines.append("")
    lines.append("- `重叠比例` 直接回答通信是否被 prefetch H2D 干扰。")
    lines.append("- `重叠/非重叠时 通信平均时长` 直接反映 communication critical path 是否被拖慢。")
    lines.append("")

    lines.append("## 2b) Denoising 每步时长与 prefetch critical-path wait (ms)")
    lines.append("")
    lines.append(f"- step总时长来源: New={total_source_new}, Old={total_source_old}")
    if comm_mode == "mock":
        lines.append("- 通信时长来源: `denoising_step_components_*.csv` 中的 `comm_ms`（mock D2H）。")
    else:
        lines.append("- 通信时长来源: `denoising_step_components_*.csv` 中的 `comm_ms`（按 NCCL kernels 统计）。")
    lines.append(
        f"- prefetch critical-path wait 来源: New={cp_wait_source_new}, Old={cp_wait_source_old}。"
    )
    lines.append(
        "- `prefetch critical-path wait` 是在每层 pre-hook 上用一对 CUDA event 记录的 current-stream 真实等待时间，只统计因为当前层权重未 ready 而落到 compute stream critical path 上的额外等待。"
    )
    lines.append(
        "- `waited prefetch / total` 是在层进入时统计的：当前层还未 ready、因此必须等待的 prefetch 字节数 / 当前层总预取字节数。"
    )
    lines.append("")
    lines.append(
        "| step | New总时长 | New通信 | New prefetch CP wait | New waited prefetch / total | Old总时长 | Old通信 | Old prefetch CP wait | Old waited prefetch / total |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    new_total_vals: list[float] = []
    new_comm_vals: list[float] = []
    new_wait_vals: list[float] = []
    new_waited_bytes_vals: list[float] = []
    new_total_bytes_vals: list[float] = []
    old_total_vals: list[float] = []
    old_comm_vals: list[float] = []
    old_wait_vals: list[float] = []
    old_waited_bytes_vals: list[float] = []
    old_total_bytes_vals: list[float] = []

    for step_idx in all_steps:
        nt = totals_new.get(step_idx)
        nc = comm_new.get(step_idx)
        nw = cp_wait_new.get(step_idx)
        nwb = waited_bytes_new.get(step_idx)
        ntb = total_bytes_new.get(step_idx)
        ot = totals_old.get(step_idx)
        oc = comm_old.get(step_idx)
        ow = cp_wait_old.get(step_idx)
        owb = waited_bytes_old.get(step_idx)
        otb = total_bytes_old.get(step_idx)

        if nt is not None:
            new_total_vals.append(nt)
        if nc is not None:
            new_comm_vals.append(nc)
        if nw is not None:
            new_wait_vals.append(nw)
        if nwb is not None:
            new_waited_bytes_vals.append(nwb)
        if ntb is not None:
            new_total_bytes_vals.append(ntb)
        if ot is not None:
            old_total_vals.append(ot)
        if oc is not None:
            old_comm_vals.append(oc)
        if ow is not None:
            old_wait_vals.append(ow)
        if owb is not None:
            old_waited_bytes_vals.append(owb)
        if otb is not None:
            old_total_bytes_vals.append(otb)

        lines.append(
            f"| {step_idx} | "
            f"{fmt_num(nt, 3)} | "
            f"{fmt_num(nc, 3)} | "
            f"{fmt_num(nw, 3)} | "
            f"{fmt_waited_over_total(nwb, ntb)} | "
            f"{fmt_num(ot, 3)} | "
            f"{fmt_num(oc, 3)} | "
            f"{fmt_num(ow, 3)} | "
            f"{fmt_waited_over_total(owb, otb)} |"
        )

    lines.append(
        f"| avg | "
        f"{fmt_num(avg(new_total_vals), 3)} | "
        f"{fmt_num(avg(new_comm_vals), 3)} | "
        f"{fmt_num(avg(new_wait_vals), 3)} | "
        f"{fmt_waited_over_total(avg(new_waited_bytes_vals), avg(new_total_bytes_vals))} | "
        f"{fmt_num(avg(old_total_vals), 3)} | "
        f"{fmt_num(avg(old_comm_vals), 3)} | "
        f"{fmt_num(avg(old_wait_vals), 3)} | "
        f"{fmt_waited_over_total(avg(old_waited_bytes_vals), avg(old_total_bytes_vals))} |"
    )
    lines.append(
        f"| sum | "
        f"{fmt_num(sum(new_total_vals) if new_total_vals else None, 3)} | "
        f"{fmt_num(sum(new_comm_vals) if new_comm_vals else None, 3)} | "
        f"{fmt_num(sum(new_wait_vals) if new_wait_vals else None, 3)} | "
        f"{fmt_waited_over_total(sum(new_waited_bytes_vals) if new_waited_bytes_vals else None, sum(new_total_bytes_vals) if new_total_bytes_vals else None)} | "
        f"{fmt_num(sum(old_total_vals) if old_total_vals else None, 3)} | "
        f"{fmt_num(sum(old_comm_vals) if old_comm_vals else None, 3)} | "
        f"{fmt_num(sum(old_wait_vals) if old_wait_vals else None, 3)} | "
        f"{fmt_waited_over_total(sum(old_waited_bytes_vals) if old_waited_bytes_vals else None, sum(old_total_bytes_vals) if old_total_bytes_vals else None)} |"
    )
    lines.append("")
    lines.append("- `waited prefetch / total` 越小，表示越多参数在进入该层前就已经 prefetch ready。")

    return "\n".join(lines) + "\n"



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--comm-mode",
        choices=["mock", "real"],
        default="mock",
        help="Interpret communication as mock D2H or real NCCL kernels.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.analysis_dir, args.comm_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary)
    print(f"Saved summary: {args.output}")


if __name__ == "__main__":
    main()
