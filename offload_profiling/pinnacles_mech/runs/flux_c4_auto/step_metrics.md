# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/flux_c4_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c4_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c4_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c4_auto/profiles/no_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c4_auto/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1749.413 | 749.188 | 485.954 | 56.681 | 0.000 | 9 | 2 | 81, 85 |
| gpu_nsys | 1896.185 | 720.619 | 118.880 | 214.720 | 535.826 | 9 | 2 | 81, 85 |
| launch_nsys | 1773.166 | 718.620 | 477.697 | 69.969 | 110.463 | 9 | 2 | 81, 85 |
| no_nsys | 1485.210 | 517.170 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
| stock_nsys | 1706.582 | 728.484 | 464.642 | 2.705 | 0.000 | 9 | 2 | 81, 85 |
