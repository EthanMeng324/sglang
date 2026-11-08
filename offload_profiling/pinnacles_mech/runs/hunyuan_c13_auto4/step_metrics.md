# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/hunyuan_c13_auto4/profiles/auto_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto4/profiles/gpu_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto4/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13_auto4/profiles/no_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| auto_nsys | 1685.159 | 574.892 | 406.664 | 106.701 | 0.000 | 9 | 2 | 81, 85, 7 |
| gpu_nsys | 1790.567 | 514.931 | 39.873 | 265.008 | 454.752 | 9 | 2 | 81, 85, 7 |
| launch_nsys | 1705.427 | 550.378 | 384.761 | 126.464 | 60.275 | 9 | 2 | 81, 85, 7 |
| no_nsys | 1392.247 | 389.018 | 0.000 | 0.170 | 0.000 | 9 | 2 | 7 |
