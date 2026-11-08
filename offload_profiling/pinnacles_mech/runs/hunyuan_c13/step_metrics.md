# Analysis Summary

- Input: `/data/hanmeng/chunkflow/results/hunyuan_c13/profiles/launch_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13/profiles/no_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13/profiles/none_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13/profiles/old_nsys.nsys-rep, /data/hanmeng/chunkflow/results/hunyuan_c13/profiles/stock_nsys.nsys-rep`
- All numeric columns below are steady-state per-rank-step averages over denoising steps with `step >= 1`.
- `NCCL/H2D overlap` and `uncovered H2D` use dynamically detected prefetch H2D streams from NVTX-marked prefetch traffic.

| Trace | Step Avg (ms) | NCCL Total (ms) | NCCL/H2D Overlap (ms) | Uncovered H2D (ms) | WAIT_COMM_BG (ms) | Steps | Devices | Prefetch Streams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| launch_nsys | 1706.642 | 536.072 | 370.590 | 138.301 | 61.105 | 9 | 2 | 81, 85, 7 |
| no_nsys | 1402.114 | 408.047 | 0.000 | 0.170 | 0.000 | 9 | 2 | 7 |
| none_nsys | 1722.224 | 584.958 | 411.543 | 112.287 | 0.000 | 9 | 2 | 81, 85, 7 |
| old_nsys | 2106.497 | 390.314 | 0.000 | 705.721 | 0.000 | 9 | 2 | 81, 85, 7 |
| stock_nsys | 1687.067 | 597.853 | 520.819 | 79.968 | 0.000 | 9 | 1 | 81, 85 |
