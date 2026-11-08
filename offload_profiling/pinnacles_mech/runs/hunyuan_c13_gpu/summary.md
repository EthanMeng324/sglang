# hunyuan_c13_gpu

| arm | step ms | vs no | NCCL ms | NCCL/H2D overlap ms | uncovered H2D ms | WAIT_COMM_BG ms | peak reserved GB | peak alloc GB | e2e s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no | 1389.3 | +0.0% | 389.8 | 0.0 | 0.2 | 0.0 | 39.27 | 38.85 | 55.56 |
| launch | 1703.7 | +22.6% | 537.0 | 373.2 | 136.9 | 60.0 | 18.06 | 16.54 | 53.29 |
| gpu | 1751.0 | +26.0% | 463.6 | 42.3 | 265.3 | 404.4 | 19.18 | 16.58 | 55.88 |
