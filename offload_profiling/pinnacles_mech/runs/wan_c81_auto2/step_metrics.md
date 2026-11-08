# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c81_auto2/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto2/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto2/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto2/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 2493.780 | 947.401 | 316.837 | 4.311 | 231.155 | 9 | 2 | 73 |
| gpu_nsys | 2429.238 | 858.514 | 129.615 | 5.778 | 457.664 | 9 | 2 | 73 |
| launch_nsys | 2563.554 | 990.148 | 465.414 | 1.796 | 255.920 | 9 | 2 | 73 |
| no_nsys | 2394.819 | 805.756 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
