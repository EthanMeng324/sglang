# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c81_auto/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 2490.263 | 943.093 | 324.604 | 5.855 | 216.363 | 9 | 2 | 73 |
| gpu_nsys | 2424.659 | 855.391 | 126.324 | 6.518 | 456.159 | 9 | 2 | 73 |
| launch_nsys | 2562.752 | 987.725 | 471.209 | 1.803 | 259.822 | 9 | 2 | 73 |
| no_nsys | 2396.199 | 826.602 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
