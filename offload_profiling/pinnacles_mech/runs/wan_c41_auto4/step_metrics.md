# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c41_auto4/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto4/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto4/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c41_auto4/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1373.087 | 666.005 | 476.301 | 9.876 | 0.000 | 9 | 2 | 73 |
| gpu_nsys | 1570.134 | 663.673 | 183.738 | 189.347 | 472.297 | 9 | 2 | 73 |
| launch_nsys | 1339.303 | 647.958 | 478.808 | 3.451 | 201.968 | 9 | 2 | 73 |
| no_nsys | 1143.071 | 459.456 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
