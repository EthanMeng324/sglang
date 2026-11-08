# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c41_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1360.746 | 648.317 | 429.396 | 9.730 | 82.385 | 9 | 2 | 73 |
| gpu_nsys | 1562.409 | 668.893 | 188.233 | 189.370 | 476.259 | 9 | 1 | 73 |
| launch_nsys | 1342.142 | 645.343 | 476.755 | 3.348 | 203.065 | 9 | 2 | 73 |
| no_nsys | 1144.919 | 455.951 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
