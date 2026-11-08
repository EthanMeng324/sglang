# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/flux_c8_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c8_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c8_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c8_auto/profiles/old_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c8_auto/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 3067.593 | 1132.367 | 193.655 | 10.868 | 442.884 | 9 | 2 | 81, 85 |
| gpu_nsys | 3024.711 | 1101.164 | 99.944 | 16.785 | 550.885 | 9 | 2 | 81, 85 |
| launch_nsys | 3221.324 | 1239.351 | 472.801 | 1.332 | 162.972 | 9 | 2 | 81, 85 |
| old_nsys | 3467.486 | 1160.800 | 0.000 | 324.383 | 0.000 | 9 | 2 | 81, 85 |
| stock_nsys | 3285.987 | 1307.489 | 411.935 | 1.442 | 0.000 | 9 | 2 | 81, 85 |
