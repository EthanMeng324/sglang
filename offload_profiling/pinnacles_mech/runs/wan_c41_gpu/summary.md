# wan_c41_gpu

| arm | step ms | vs no | NCCL ms | NCCL/H2D overlap ms | uncovered H2D ms | WAIT_COMM_BG ms | peak reserved GB | peak alloc GB | e2e s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no | 1142.2 | +0.0% | 458.3 | 0.0 | 0.0 | 0.0 | 29.20 | 22.57 | 37.40 |
| launch | 1339.8 | +17.3% | 648.6 | 478.4 | 3.5 | 204.1 | 17.35 | 13.72 | 37.26 |
| gpu | 1557.5 | +36.4% | 657.8 | 185.7 | 184.2 | 473.6 | 18.66 | 13.75 | 41.44 |
