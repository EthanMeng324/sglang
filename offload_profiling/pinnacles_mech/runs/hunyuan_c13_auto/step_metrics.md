# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/hunyuan_c13_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1710.346 | 587.355 | 377.765 | 106.958 | 62.417 | 9 | 2 | 81, 85, 7 |
| gpu_nsys | 1749.567 | 457.642 | 39.687 | 267.231 | 400.762 | 9 | 2 | 81, 85, 7 |
| launch_nsys | 1729.360 | 550.106 | 375.307 | 135.190 | 65.412 | 9 | 2 | 81, 85, 7 |
| no_nsys | 1393.239 | 388.952 | 0.000 | 0.176 | 0.000 | 9 | 2 | 7 |
