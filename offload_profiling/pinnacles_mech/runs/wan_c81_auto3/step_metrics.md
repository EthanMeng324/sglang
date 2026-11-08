# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/wan_c81_auto3/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto3/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto3/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/wan_c81_auto3/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 2405.865 | 855.951 | 158.818 | 3.740 | 412.936 | 9 | 1 | 73 |
| gpu_nsys | 2431.635 | 860.942 | 127.877 | 4.573 | 462.580 | 9 | 2 | 73 |
| launch_nsys | 2571.048 | 1002.292 | 464.664 | 1.730 | 256.536 | 9 | 2 | 73 |
| no_nsys | 2398.629 | 823.018 | 0.000 | 0.000 | 0.000 | 9 | 2 | - |
