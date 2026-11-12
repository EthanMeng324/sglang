# Diagnosis of the two H100 gate anomalies (1x2 H100 NVLink, SP=2), 2026-11-11

Setup for every run below: Wan2.2-TI2V-5B, 704x1280, 10 steps, torch.compile on (COMPILE=1), CHUNK_MB=16 unless
stated, NNODES=1 GPUS_PER_NODE=2, `run_mn_arm_diag.sh` (copy of run_mn_arm.sh; changes: compile only when COMPILE=1,
--no-save-output removed, --seed 42 added; --seed exists and its default is already 42). Videos are saved as
`<OUT>/<arm>.out.mp4` (kept in $WS/results, not committed). PSNR: diag/psnr.py (imageio/ffmpeg decode, per-frame
PSNR on uint8 RGB, mean over frames; identical frames reported separately). Clock/power: `nvidia-smi dmon -s pc -d 1
-o T` per arm, averaged over steps 1-9 by diag/dmon_avg.py (window ends at the '[DenoisingStage] finished' log
timestamp, 1 s resolution).
Diagnosis job 22746435 (gh002, commit 864c48c0f): A -> D -> B; C skipped by decision.

## Headline
**The gate arm produces wrong output: its videos are all-black (every pixel 0) at 41 and 161 frames, and green
noise at 41 frames / 32 MB chunks.** All earlier gate step times (summary.md, 11-11) are therefore invalid.
The "gate faster than no" effect (anomaly 2) is the speed of a wrong computation. The release fix in b5ed1a60e
does not change this (see the last section). stock output is bit-identical to no.

## A. Is anomaly 2 reproducible (f161, interleaved no/gate/no/gate)

| run | steps 0-9 ms | mean 1-8 | mean 1-9 | pclk MHz gpu0/gpu1 | power W gpu0/gpu1 |
|---|---|---|---|---|---|
| no d1   | 48772, 1868, 1874, 1869, 1869, 1872, 1868, 1869, 1870, 1871 | 1870 | 1870 | 1859 / 1899 | 684 / 681 |
| gate d1 | 31651, 1848, 1751, 1707, 1708, 1706, 1705, 1706, 1709, 1714 | 1730 | 1728 | 1972 / 1972 | 524 / 521 |
| no d2   | 36441, 1863, 1875, 1877, 1871, 1879, 1875, 1871, 1876, 1877 | 1873 | 1874 | 1868 / 1887 | 674 / 667 |
| gate d2 | 31622, 1846, 1803, 1710, 1706, 1707, 1713, 1706, 1708, 1705 | 1737 | 1734 | 1969 / 1970 | 521 / 517 |

PSNR (165 frames each): no_d1 vs no_d2 = identical (all 165 frames, inf: seed fixed, deterministic);
no_d1 vs gate_d1 = 4.601 dB; no_d2 vs gate_d2 = 4.601 dB. gate videos: mean pixel 0.0, max 0 (all black).
Reading: the speed-up reproduces (7%), but gate computes something different (PSNR 4.6 dB vs an identical-output
baseline). The no arm runs near the power limit (~680 W) at lower SM clock (~1870 MHz); gate draws ~520 W and holds
~1970 MHz -> consistent with a wrong/degenerate computation, not with a real speed-up.

## D. Anomaly 1, chunk size (f41)

| run | steps 0-9 ms | mean 1-8 | mean 1-9 | video |
|---|---|---|---|---|
| gate 16 MB (11-11 sweep) | 29106, 476, 470, 473, 475, 473, 474, 477, 547, 581 | 484 | 494 | not saved |
| gate 32 MB | 27670, 448, 443, 444, 439, 444, 446, 437, 439, 441 | 442 | 442 | green noise (frame-mean 3-153) |
| gate 64 MB | 31334, 547, 551, 562, 627, 662, 639, 696, 675, 637 | 620 | 622 | all black |
| stock (11-11 sweep) | | 414 | 415 | |
| no (11-11 sweep) | | 384 | 385 | |

run.txt env lines show SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB=32 / 64. Reading: no monotonic return to stock with
bigger chunks; but since every gate output is wrong, these timings do not test the host-rate hypothesis.

## B. nsys (f161, TAG=h100n) — recorded only, not analysed (by decision)
Both arms hit ARM_TIMEOUT=1800 under nsys (rc=124; no: wall 1801 s, gate: 1800 s), so the .nsys-rep files are
incomplete/absent and analyze_nsys_step_metrics.py produced no sqlite export (kernels_step5.md empty).
Step times (perf json, written before the timeout): no 1878 (1-8 mean), gate 1735 — same as group A.

## C. eager — not run (decision).

## Release fix check (job 22758167, gh014, commit b5ed1a60e, TAG=h100fix)
`grep -n IMMEDIATE_RELEASE layerwise_offload_chunkwise.py` -> line 160:
`self._use_deferred_buffer_release = not _env_bool("SGLANG_DIT_OFFLOAD_IMMEDIATE_RELEASE", False)` (deferred on by default).

| run | video | PSNR vs no | steps 0-9 ms | mean 1-8 | mean 1-9 | pclk MHz | power W |
|---|---|---|---|---|---|---|---|
| no 41 (new, as baseline) | normal | - | 42452, 397, 387, 393, 386, 397, 394, 383, 390, 383 | 391 | 390 | 1980 / 1980 | 465 / 472 |
| gate 41  | **all black** | 5.592 dB (vs no 41 fix) | 29605, 478, 487, 477, 477, 477, 483, 479, 479, 481 | 480 | 480 | 1980 / 1980 | 374 / 371 |
| gate 161 | **all black** | 4.601 dB (vs no161 d1) | 34381, 1702, 1701, 1696, 1695, 1700, 1701, 1699, 1702, 1700 | 1700 | 1700 | 1980 / 1980 | 477 / 482 |
| stock 161 | normal, **identical to no** (165/165 frames) | inf | 33513, 1864, 1872, 1872, 1872, 1870, 1869, 1871, 1872, 1872 | 1870 | 1870 | 1892 / 1887 | 663 / 665 |

Criterion (gate PSNR at the no-vs-no baseline, step time not below no): **not met.** gate is still all black and still
faster than no at 161 frames (1700 vs 1870). stock is validated as a correct baseline (bit-identical to no, same speed).
gate.log lines requested (f161; f41 identical apart from timestamps):
```
[11-11 20:06:48] [comm-gate] stream memops gate ready on cuda:0 (flag at 0xf27fa00200)
[11-11 20:06:48] Communication-aware chunk-wise offload is enabled (chunk=16MB, patch_torch_dist=False, comm_window=gate).
[11-11 20:06:48] [comm-gate] stream memops gate ready on cuda:1 (flag at 0xf27fa00200)
[ChunkwiseStep] step=3 wall_s=0.429 bg_chunks=580 bg_mb=9052.36 bg_copy_s=0.239 sync_chunks=0 sync_mb=0.00 ... wait_comm_bg_s=0.000 ensure_ready_s=0.001 unique_loaded_layers=29 unique_loaded_mb=9052.36 copy_dup_ratio=1.000
```
No WARNING-level lines; only torch._dynamo UserWarnings (functools.lru_cache wrapper; cannot trace datetime.now).
No log line mentions deferred release, so the log alone cannot show whether the new path ran.
Both ranks report the same flag address 0xf27fa00200 (separate processes; noted, not interpreted).

## Files
diag/: DIAG.md, psnr.py, dmon_avg.py, kernels.py, diag_driver.sh, fix_driver.sh, run_mn_arm_diag.sh, diag_log.txt,
fix_log.txt, dmon_A.txt, dmon_fix.txt; per run directory logs/ (run.txt, <arm>.log, dmon_<arm>.txt) and
profiles/perf_*.json. Not committed: videos (*.mp4) and nsys output, in $WS/results.
