#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


STAGE_LABELS = [
    "Warmup Dry Run",
    "Comm-Aware Offload",
    "Phase-Aware Offload",
    "No Offload",
    "Old Offload",
    "Analyze NSYS",
]

REASON_FIELDS = [
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.sync_boost",
]

DENSE_UTIL_THRESHOLD = 90.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize GPU clock-drop telemetry by run_profile stage."
    )
    parser.add_argument("--timeline", required=True, help="timeline_*.log from run_profile_dmon")
    parser.add_argument("--telemetry", required=True, help="telemetry_*.csv from nvidia-smi --query-gpu")
    parser.add_argument("--output-md", required=True, help="Markdown summary output path")
    parser.add_argument("--output-csv", required=True, help="CSV summary output path")
    return parser.parse_args()


def parse_epoch(value: str) -> float:
    return float(value)


def parse_timeline(path: Path):
    stages = []
    starts = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row["event"]
            label = row["label"]
            epoch = parse_epoch(row["epoch_s"])
            if event == "step_start" and label in STAGE_LABELS:
                starts[label] = epoch
            elif event == "step_end" and label in STAGE_LABELS and label in starts:
                stages.append((label, starts.pop(label), epoch))
    return stages


def parse_number(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "" or value == "[N/A]":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_reason_active(value: str) -> int:
    if value is None:
        return 0
    value = value.strip()
    if value == "" or value == "[N/A]":
        return 0
    if value.lower().startswith("0x"):
        try:
            return int(value, 16)
        except ValueError:
            return 0
    return 0


def parse_bool_field(value: str) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "active"


def parse_telemetry_timestamp(value: str) -> float:
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(
        tzinfo=timezone.utc
    ).timestamp()


def load_telemetry(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = parse_telemetry_timestamp(row["timestamp"])
                gpu = int(float(row["index"]))
            except Exception:
                continue

            parsed = {
                "timestamp": ts,
                "gpu": gpu,
                "name": row.get("name", "").strip(),
                "pstate": row.get("pstate", "").strip(),
                "pclk": parse_number(row.get("clocks.current.graphics [MHz]")),
                "pclk_max": parse_number(row.get("clocks.max.graphics [MHz]")),
                "gtemp": parse_number(row.get("temperature.gpu")),
                "mtemp": parse_number(row.get("temperature.memory")),
                "power": parse_number(row.get("power.draw [W]")),
                "power_limit": parse_number(row.get("power.limit [W]")),
                "enforced_power_limit": parse_number(row.get("enforced.power.limit [W]")),
                "util_gpu": parse_number(row.get("utilization.gpu [%]")),
                "util_mem": parse_number(row.get("utilization.memory [%]")),
                "mem_used": parse_number(row.get("memory.used [MiB]")),
                "active_mask": parse_reason_active(row.get("clocks_event_reasons.active")),
            }
            for reason in REASON_FIELDS:
                parsed[reason] = parse_bool_field(row.get(reason))
            rows.append(parsed)
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def percentile_50(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return median(values)


def ratio_true(rows, key):
    if not rows:
        return None
    return sum(1 for row in rows if row.get(key)) / len(rows)


def ratio_pred(rows, pred):
    if not rows:
        return None
    return sum(1 for row in rows if pred(row)) / len(rows)


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def summarize(stages, telemetry_rows):
    results = []
    by_stage = defaultdict(list)
    for label, start, end in stages:
        for row in telemetry_rows:
            if start <= row["timestamp"] <= end:
                by_stage[label].append(row)

    for label, _, _ in stages:
        rows = by_stage[label]
        gpus = sorted({row["gpu"] for row in rows})
        for gpu in gpus:
            gpu_rows = [row for row in rows if row["gpu"] == gpu]
            dense_rows = [
                row for row in gpu_rows if (row["util_gpu"] is not None and row["util_gpu"] >= DENSE_UTIL_THRESHOLD)
            ]
            result = {
                "stage": label,
                "gpu": gpu,
                "samples": len(gpu_rows),
                "dense_samples": len(dense_rows),
                "name": gpu_rows[0]["name"] if gpu_rows else "",
                "pstate_mode": dominant_value([row["pstate"] for row in dense_rows]),
                "pclk_avg_all_mhz": mean([row["pclk"] for row in gpu_rows]),
                "pclk_avg_dense_mhz": mean([row["pclk"] for row in dense_rows]),
                "pclk_p50_dense_mhz": percentile_50([row["pclk"] for row in dense_rows]),
                "pclk_lt_1200_frac_dense": ratio_pred(dense_rows, lambda row: (row["pclk"] or 0) < 1200),
                "gtemp_avg_dense_c": mean([row["gtemp"] for row in dense_rows]),
                "mtemp_avg_dense_c": mean([row["mtemp"] for row in dense_rows]),
                "power_avg_dense_w": mean([row["power"] for row in dense_rows]),
                "power_limit_w": mean([row["power_limit"] for row in dense_rows]),
                "enforced_power_limit_w": mean([row["enforced_power_limit"] for row in dense_rows]),
                "fb_used_avg_dense_mb": mean([row["mem_used"] for row in dense_rows]),
                "util_gpu_avg_dense": mean([row["util_gpu"] for row in dense_rows]),
                "util_mem_avg_dense": mean([row["util_mem"] for row in dense_rows]),
                "active_mask_non_idle_frac_dense": ratio_pred(
                    dense_rows, lambda row: (row["active_mask"] or 0) not in (0, 0x1)
                ),
            }
            for reason in REASON_FIELDS:
                result[f"{reason}_frac_dense"] = ratio_true(dense_rows, reason)
            results.append(result)
    return results


def dominant_value(values):
    counts = defaultdict(int)
    for value in values:
        if value:
            counts[value] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def write_csv(path: Path, rows):
    fieldnames = [
        "stage",
        "gpu",
        "name",
        "samples",
        "dense_samples",
        "pstate_mode",
        "pclk_avg_all_mhz",
        "pclk_avg_dense_mhz",
        "pclk_p50_dense_mhz",
        "pclk_lt_1200_frac_dense",
        "gtemp_avg_dense_c",
        "mtemp_avg_dense_c",
        "power_avg_dense_w",
        "power_limit_w",
        "enforced_power_limit_w",
        "fb_used_avg_dense_mb",
        "util_gpu_avg_dense",
        "util_mem_avg_dense",
        "active_mask_non_idle_frac_dense",
    ] + [f"{reason}_frac_dense" for reason in REASON_FIELDS]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows):
    lines = []
    lines.append("# Clock Drop Telemetry Summary")
    lines.append("")
    lines.append(f"Dense samples are defined as `utilization.gpu >= {int(DENSE_UTIL_THRESHOLD)}%`.")
    lines.append("")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stage"]].append(row)

    for stage in STAGE_LABELS:
        stage_rows = sorted(grouped.get(stage, []), key=lambda row: row["gpu"])
        if not stage_rows:
            continue
        lines.append(f"## {stage}")
        lines.append("")
        lines.append(
            "| GPU | Dense Samples | Pstate | pclk avg/p50 (MHz) | Power (W) | GPU Temp (C) | Mem Temp (C) | FB Used (MB) | pclk<1200 | SW Power Cap | SW Thermal | HW Thermal | HW Power Brake | HW Slowdown | Sync Boost |"
        )
        lines.append(
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in stage_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["gpu"]),
                        str(row["dense_samples"]),
                        row["pstate_mode"] or "N/A",
                        f"{fmt(row['pclk_avg_dense_mhz'])} / {fmt(row['pclk_p50_dense_mhz'])}",
                        fmt(row["power_avg_dense_w"]),
                        fmt(row["gtemp_avg_dense_c"]),
                        fmt(row["mtemp_avg_dense_c"]),
                        fmt(row["fb_used_avg_dense_mb"]),
                        fmt(100.0 * row["pclk_lt_1200_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.sw_power_cap_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.sw_thermal_slowdown_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.hw_thermal_slowdown_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.hw_power_brake_slowdown_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.hw_slowdown_frac_dense"]),
                        fmt(100.0 * row["clocks_event_reasons.sync_boost_frac_dense"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    timeline_path = Path(args.timeline)
    telemetry_path = Path(args.telemetry)
    output_md = Path(args.output_md)
    output_csv = Path(args.output_csv)

    stages = parse_timeline(timeline_path)
    telemetry_rows = load_telemetry(telemetry_path)
    summary_rows = summarize(stages, telemetry_rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, summary_rows)
    write_markdown(output_md, summary_rows)


if __name__ == "__main__":
    main()
