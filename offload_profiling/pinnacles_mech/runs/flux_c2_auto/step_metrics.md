# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/flux_c2_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c2_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c2_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c2_auto/profiles/no_nsys.nsys-rep, /data/hanmeng/chunkflow/results/flux_c2_auto/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1234.316 | 529.682 | 425.727 | 218.905 | 0.000 | 9 | 2 | 81, 85 |
| gpu_nsys | 1621.667 | 546.611 | 96.361 | 537.769 | 456.042 | 9 | 2 | 81, 85 |
| launch_nsys | 1268.619 | 443.600 | 336.928 | 333.869 | 110.602 | 9 | 2 | 81, 85 |
| no_nsys | 811.141 | 352.380 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
| stock_nsys | 1155.218 | 448.328 | 438.400 | 248.372 | 0.000 | 9 | 2 | 81, 85 |
