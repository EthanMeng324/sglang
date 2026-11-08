# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c41/profiles/kernel_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41/profiles/no_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41/profiles/none_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41/profiles/old_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kernel_nsys | 2112.221 | 763.840 | 0.000 | 582.025 | 1087.160 | 9 | 2 | 73 |
| launch_nsys | 1347.789 | 643.854 | 471.838 | 3.263 | 203.492 | 9 | 2 | 73 |
| no_nsys | 1146.217 | 455.260 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
| none_nsys | 1360.604 | 667.379 | 477.875 | 12.142 | 0.000 | 9 | 2 | 73 |
| old_nsys | 1819.671 | 453.742 | 0.000 | 668.378 | 0.000 | 9 | 2 | 73 |
| stock_nsys | 1393.517 | 695.446 | 606.226 | 4.048 | 0.000 | 9 | 2 | 73 |
