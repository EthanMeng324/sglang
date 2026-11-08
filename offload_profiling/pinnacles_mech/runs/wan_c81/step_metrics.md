# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c81/profiles/kernel_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81/profiles/no_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81/profiles/none_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81/profiles/old_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kernel_nsys | 2989.614 | 997.000 | 0.000 | 366.122 | 1946.720 | 9 | 1 | 73 |
| launch_nsys | 2551.983 | 981.078 | 464.611 | 1.742 | 251.356 | 9 | 2 | 73 |
| no_nsys | 2378.127 | 795.468 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
| none_nsys | 2544.733 | 1001.102 | 455.671 | 8.235 | 0.000 | 9 | 2 | 73 |
| old_nsys | 2918.683 | 790.689 | 0.000 | 549.743 | 0.000 | 9 | 2 | 73 |
| stock_nsys | 2618.092 | 1048.019 | 626.381 | 2.004 | 0.000 | 9 | 2 | 73 |
