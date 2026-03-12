#!/usr/bin/env python3
"""Summarize the core offload profiling results into a compact Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_SPECS = [
    ("new", "New"),
    ("old", "Old"),
    ("no", "No Offload"),
]


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


def overlap_context_count_key(comm_mode: str) -> str:
    return "mock_d2h_count" if comm_mode == "mock" else "comm_count"


def overlap_context_total_ms_key(comm_mode: str) -> str:
    return "total_mock_d2h_ms" if comm_mode == "mock" else "total_comm_ms"


def overlap_context_avg_ms_key(comm_mode: str) -> str:
    return "avg_mock_d2h_dur_ms" if comm_mode == "mock" else "avg_comm_dur_ms"


def overlap_context_total_ms(row: dict[str, str], comm_mode: str) -> float | None:
    total_ms = to_float(row.get(overlap_context_total_ms_key(comm_mode)))
    if total_ms is not None:
        return total_ms
    count = to_float(row.get(overlap_context_count_key(comm_mode)))
    if count == 0:
        return 0.0
    avg_ms = to_float(row.get(overlap_context_avg_ms_key(comm_mode)))
    if count is None or avg_ms is None:
        return None
    return count * avg_ms



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


def first_existing_rows(analysis_dir: Path, names: list[str]) -> list[dict[str, str]]:
    for name in names:
        rows = load_csv(analysis_dir / name)
        if rows:
            return rows
    return []


def row_with_context(rows: list[dict[str, str]], context: str) -> dict[str, str]:
    for row in rows:
        if row.get("context") == context:
            return row
    return {}



def find_perf_json(analysis_dir: Path, run: str) -> Path | None:
    profiles_dir = analysis_dir.parent
    results_dir = profiles_dir.parent
    exact_names = {
        "new": ["perf_new_offload_profiled.json", "perf_new_offload.json"],
        "old": ["perf_old_offload_profiled.json", "perf_old_offload.json"],
        "no": ["perf_no_offload_profiled.json", "perf_no_offload.json"],
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
) -> tuple[dict[int, float], str]:
    perf_path = find_perf_json(analysis_dir, run)
    perf = load_json(perf_path)
    offload_profile = perf.get("extra", {}).get("offload_profile", {})

    wait_ms = offload_profile.get("prefetch_critical_path_wait_ms")
    wait_map: dict[int, float] = {}

    if isinstance(wait_ms, list):
        for idx, value in enumerate(wait_ms):
            fv = to_float(value)
            if fv is not None:
                wait_map[idx] = fv

    source = "runtime_cuda_events"
    if perf_path is not None:
        source += f":{perf_path.name}"
    return wait_map, source



def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)



def build_summary(analysis_dir: Path, comm_mode: str) -> str:
    run_data: dict[str, dict[str, Any]] = {}
    all_steps: set[int] = set()
    for run_id, label in RUN_SPECS:
        overlap_rows = first_existing_rows(
            analysis_dir,
            [f"comm_overlap_{run_id}.csv", f"mock_d2h_overlap_{run_id}.csv"],
        )
        overlap_ratio = first_existing_row(
            analysis_dir,
            [f"comm_overlap_ratio_{run_id}.csv", f"mock_d2h_overlap_ratio_{run_id}.csv"],
        )
        totals, total_source = load_step_totals(analysis_dir, run_id)
        comm = load_step_comm(analysis_dir, run_id)
        cp_wait, cp_wait_source = load_prefetch_cp_wait(analysis_dir, run_id)
        during = row_with_context(overlap_rows, "during_prefetch_h2d")
        without = row_with_context(overlap_rows, "without_prefetch_h2d")
        during_total = overlap_context_total_ms(during, comm_mode)
        without_total = overlap_context_total_ms(without, comm_mode)
        total_comm = None
        if during_total is not None or without_total is not None:
            total_comm = (during_total or 0.0) + (without_total or 0.0)
        step_indices = set(totals) | set(comm) | set(cp_wait)
        step_count = len(step_indices)

        run_data[run_id] = {
            "label": label,
            "overlap_rows": overlap_rows,
            "overlap_ratio": overlap_ratio,
            "totals": totals,
            "total_source": total_source,
            "comm": comm,
            "cp_wait": cp_wait,
            "cp_wait_source": cp_wait_source,
            "during_total": during_total,
            "without_total": without_total,
            "total_comm": total_comm,
            "step_count": step_count,
        }
        all_steps |= step_indices

    all_steps = sorted(all_steps)

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
    lines.append("| 指标 | " + " | ".join(label for _, label in RUN_SPECS) + " |")
    lines.append("|---|" + "---:|" * len(RUN_SPECS))
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

    def metric_row(metric_label: str, value_fn) -> None:
        values = [fmt_num(value_fn(run_id), 3) for run_id, _ in RUN_SPECS]
        lines.append(f"| {metric_label} | " + " | ".join(values) + " |")

    lines.append(
        f"| {name_label} 总事件数 | "
        + " | ".join(
            fmt_num(
                to_float(run_data[run_id]["overlap_ratio"].get(total_key)),
                0,
            )
            for run_id, _ in RUN_SPECS
        )
        + " |"
    )
    lines.append(
        f"| 与 prefetch H2D 重叠事件数 | "
        + " | ".join(
            fmt_num(
                to_float(run_data[run_id]["overlap_ratio"].get(overlap_key)),
                0,
            )
            for run_id, _ in RUN_SPECS
        )
        + " |"
    )
    lines.append(
        f"| 重叠比例 (%) | "
        + " | ".join(
            fmt_num(
                to_float(run_data[run_id]["overlap_ratio"].get(ratio_key)),
                2,
            )
            for run_id, _ in RUN_SPECS
        )
        + " |"
    )
    metric_row(f"{name_label} 总时间 (ms)", lambda run_id: run_data[run_id]["total_comm"])
    metric_row(
        f"重叠时 {name_label} 总时间 (ms)",
        lambda run_id: run_data[run_id]["during_total"],
    )
    metric_row(
        f"非重叠时 {name_label} 总时间 (ms)",
        lambda run_id: run_data[run_id]["without_total"],
    )
    metric_row(
        f"{name_label} 平均时长/step (ms)",
        lambda run_id: (
            None
            if not run_data[run_id]["step_count"]
            or run_data[run_id]["total_comm"] is None
            else run_data[run_id]["total_comm"] / run_data[run_id]["step_count"]
        ),
    )
    metric_row(
        f"重叠时 {name_label} 平均时长/step (ms)",
        lambda run_id: (
            None
            if not run_data[run_id]["step_count"]
            or run_data[run_id]["during_total"] is None
            else run_data[run_id]["during_total"] / run_data[run_id]["step_count"]
        ),
    )
    metric_row(
        f"非重叠时 {name_label} 平均时长/step (ms)",
        lambda run_id: (
            None
            if not run_data[run_id]["step_count"]
            or run_data[run_id]["without_total"] is None
            else run_data[run_id]["without_total"] / run_data[run_id]["step_count"]
        ),
    )
    lines.append("")
    lines.append("- `重叠比例` 直接回答通信是否被 prefetch H2D 干扰。")
    lines.append("- `重叠/非重叠时 通信平均时长/step` 是把 denoising 窗口内对应通信总时长除以 denoising step 数，不再按单个事件平均。")
    lines.append("")

    lines.append("## 2b) Denoising 每步时长与 prefetch critical-path wait (ms)")
    lines.append("")
    lines.append(
        "- step总时长来源: "
        + ", ".join(
            f"{label}={run_data[run_id]['total_source']}"
            for run_id, label in RUN_SPECS
        )
    )
    if comm_mode == "mock":
        lines.append("- 通信时长来源: `denoising_step_components_*.csv` 中的 `comm_ms`（mock D2H）。")
    else:
        lines.append("- 通信时长来源: `denoising_step_components_*.csv` 中的 `comm_ms`（按 NCCL kernels 统计）。")
    lines.append(
        "- prefetch critical-path wait 来源: "
        + ", ".join(
            f"{label}={run_data[run_id]['cp_wait_source']}"
            for run_id, label in RUN_SPECS
        )
        + "。"
    )
    lines.append(
        "- `prefetch critical-path wait` 是在每层 pre-hook 上用一对 CUDA event 记录的 current-stream 真实等待时间，只统计因为当前层权重未 ready 而落到 compute stream critical path 上的额外等待。"
    )
    step_header = ["step"]
    for _, label in RUN_SPECS:
        step_header.extend(
            [
                f"{label}总时长",
                f"{label}通信",
                f"{label} prefetch CP wait",
            ]
        )
    lines.append("| " + " | ".join(step_header) + " |")
    lines.append("|" + "---:|" * len(step_header))

    agg: dict[str, dict[str, list[float]]] = {
        run_id: {
            "totals": [],
            "comm": [],
            "wait": [],
        }
        for run_id, _ in RUN_SPECS
    }

    for step_idx in all_steps:
        row_values = [str(step_idx)]
        for run_id, _ in RUN_SPECS:
            totals = run_data[run_id]["totals"]
            comm = run_data[run_id]["comm"]
            cp_wait = run_data[run_id]["cp_wait"]
            total_v = totals.get(step_idx)
            comm_v = comm.get(step_idx)
            wait_v = cp_wait.get(step_idx)

            if total_v is not None:
                agg[run_id]["totals"].append(total_v)
            if comm_v is not None:
                agg[run_id]["comm"].append(comm_v)
            if wait_v is not None:
                agg[run_id]["wait"].append(wait_v)

            row_values.extend(
                [
                    fmt_num(total_v, 3),
                    fmt_num(comm_v, 3),
                    fmt_num(wait_v, 3),
                ]
            )
        lines.append("| " + " | ".join(row_values) + " |")

    for agg_name in ("avg", "sum"):
        row_values = [agg_name]
        for run_id, _ in RUN_SPECS:
            totals_vals = agg[run_id]["totals"]
            comm_vals = agg[run_id]["comm"]
            wait_vals = agg[run_id]["wait"]
            if agg_name == "avg":
                total_v = avg(totals_vals)
                comm_v = avg(comm_vals)
                wait_v = avg(wait_vals)
            else:
                total_v = sum(totals_vals) if totals_vals else None
                comm_v = sum(comm_vals) if comm_vals else None
                wait_v = sum(wait_vals) if wait_vals else None
            row_values.extend(
                [
                    fmt_num(total_v, 3),
                    fmt_num(comm_v, 3),
                    fmt_num(wait_v, 3),
                ]
            )
        lines.append("| " + " | ".join(row_values) + " |")

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
