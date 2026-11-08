# ChunkFlow 通信机制研究 — 阶段记录（Pinnacles 2×L40S，2026-11-06/07）

硬件：UC Merced Pinnacles，gnode015/018/020（2×L40S 46 GB，PCIe Gen4 ×16，无 NVLink，两卡经同一 NUMA 的 PCIe host bridge），Ulysses SP=2，10 步，论文设置（Wan 704×1280 / Hunyuan 720×1280 / Flux 1024²，chunk 16 MB，FlashAttention via sgl_kernel）。代码：EthanMeng324/sglang `v0.5.9-shard`（= 论文代码）+ 分支 `pinnacles-mech` 上的补丁。

## 臂的定义

| 臂 | 含义 |
|---|---|
| no | 全驻留 |
| old | 论文的 "Layerwise"：整层 prefetch + **每次集合通信前 `quiesce`（compute stream 等 copy stream 全部完成）** |
| stock | 整层 prefetch，去掉 quiesce（`SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0`）= upstream SGLang 的行为 |
| none | 切块（16 MB），从不让路 |
| launch | 论文的 ChunkFlow：窗口 = Python wrapper 生命周期（≈ launch 调用的 0.2 ms） |
| kernel | 窗口从 host 调用开始、到 NCCL Work 完成 |
| gpu | 窗口从 compute stream 上的 start event 完成（GPU 真正到达集合通信）到 NCCL Work 完成 |
| auto | gpu 窗口 + 按余量决定：剩余 prefetch 字节/带宽 ≤ 本 block 剩余纯计算时间×(1−margin) 才让路，否则共享 |

## 步时（ms，steady 平均，括号 = 相对 no）

| 配置 | no | old | stock | none | launch | kernel | gpu | auto(v1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan f41 | 1146 | 1820 (+59%) | 1394 (+22%) | 1361 (+19%) | 1348 (+18%) | 2112 (+84%) | 1557 (+36%) | 1361 (+19%) |
| Wan f81 | 2378 | 2919 (+23%) | 2618 (+10%) | 2545 (+7%) | 2552 (+7%) | 2990 (+26%) | 2439 (+1.8%) | 2490 (+3.9%) |
| Hunyuan f13 | 1402 | 2107 (+50%) | 1687 (+20%) | 1722 (+23%) | 1707 (+22%) | （bug 已修，未重跑） | 1751 (+26%) | 1710 (+23%, v2) |
| Flux bs2 | 811 | （论文 LW 1456） | 1155 (+42%) | — | 1269 (+56%) | — | 1622 (+100%) | 1234 (+52%, v4 share) |
| Flux bs4 | 1485 | （论文 LW 2046） | 1707 (+15%) | — | 1773 (+19%) | — | 1896 (+28%) | 1749 (+18%, v4 share) |

同作业重复（no/launch/gpu/auto）：Wan f41 ×3：no 1142–1145，launch 1340–1349，gpu 1557–1569，auto v1 1361 / v3 1464（v3 启动期误判 yield）；Wan f81 ×2：no 2395–2397，launch 2562–2564，gpu 2425–2439，auto v1 2490 / v2 2494。

论文 L40S 表（No/LW/CF）：Wan f41 1.111/1.772/1.347，f81 2.333/2.876/2.388，Hunyuan f13 1.376/2.088/1.704 —— no/old/launch 三臂复现在 3% 内。重复性：launch 三次 1348/1340/1342，gpu 两次 1557/1562。

## nsys 分解（每 rank 每步，ms）

| 配置/臂 | NCCL 总 | NCCL∩H2D 重叠 | 暴露 H2D | worker 等 comm |
|---|---:|---:|---:|---:|
| Wan f41 no / old / stock | 455 / 454 / 695 | 0 / 0 / 606 | 0 / 668 / 4 | — |
| Wan f41 launch / none / kernel / gpu | 644 / 667 / 764 / 658 | 472 / 478 / 0 / 186 | 3 / 12 / 582 / 184 | 204 / 0 / 1087 / 474 |
| Wan f81 no / stock / launch / gpu / auto | 795 / 1048 / 981 / 866 / 943 | 0 / 626 / 465 / 128 / 325 | 0 / 2 / 2 / 4 / 6 | — / 0 / 251 / 456 / 216 |
| Hunyuan f13 no / stock / launch / gpu | 390 / 598 / 536 / 464 | 0 / 521 / 371 / 42 | 0 / 80 / 137 / 265 | — / 0 / 60 / 404 |

每个 NCCL kernel 漏进的 chunk 数（Wan f41）：launch 1.63 个、kernel 覆盖 70%；gpu 0.55 个，漏的那个是窗口打开时在飞的 chunk（起点中位数 −0.06 ms）。Wan 每 block 16 个 NCCL kernel（中位 1.3 ms，f81 2.2 ms），Hunyuan 每 block 2 个（3.4 ms）。

## 结论

1. **论文基线被 quiesce 拖慢。** `quiesce_prefetch_for_comm` 在三个模型的 forward 里都挂进 forward context，`usp.py` 每次 all-to-all 前调用，基线也走这条路（`layerwise_offload_original.py`）。upstream 没有它。去掉后（stock）整层 prefetch 只比 no 慢 10–22%；ChunkFlow(launch) 对 stock 的收益：Wan 3%，Hunyuan 0。论文的 1.2–1.3× 和 Fig. 1b 的 21% "contention stall" 是这个强制串行化产生的暴露 H2D。
2. **launch ≈ none。** 让路窗口只覆盖 launch 调用，论文的 ChunkFlow 实际是"只切块不让路"；切块的 3% 来自更细的 PCIe 仲裁粒度（NCCL 被拉长得少一点）。
3. **kernel 模式的失败是 host 领先 GPU 一个队列深度。** active 从 host 调用起算，prefetch 在 GPU 到达集合通信前几百毫秒就被停掉。
4. **gpu 模式（正确的让路）F* 以上赢、以下输。** f81：+1.8% vs launch +6.9%、stock +10%；NCCL 回到 no 的水平。f41 / Hunyuan f13：+36% / +26%，因为没有余量：每次让路付"排干在飞 chunk + 恢复延迟"≈ 0.5–1 ms，Wan 每步 480 次就是几百毫秒链路空转。
5. **auto（有余量才让路）**：v1/v2 按集合通信逐个判断：f41 ≈ launch，Hunyuan f13 ≈ launch，f81 +3.9–4.1%（launch +7.0%、gpu +1.2%）只拿到一半收益。原因是两个 rank 独立判断会失配（一边让路一边共享，集合通信照样被拖慢，让路的一边白等并落后，于是交替失配）。v3 改为每 forward 一个决策（在线 F* 测试：T_pref=字节/空闲链路带宽 vs 实测纯计算时间）但启动测量被编译污染、yield 下的 stall 又抬高"纯计算时间"，f41 变成 +28%。v4：预热 2 个 forward → share 下测一次做 F* 测试 → 试一个 forward 的 yield → 留快的，周期复测（失败则周期翻倍）。**v4 实测 Wan f81：auto 2406 ms（+0.3%），launch 2571（+7.2%），gpu 2432（+1.4%），no 2399**；决策序列 warm,warm → share 测得 ratio 0.59 → trial 1198 < 1352 ms → 保持 yield。f41/Hunyuan/Flux 的 v4 待出。
7. **prefetch 字节量**：manager 统计 Wan 每块 327 MB（30 块 9.145 GiB），Hunyuan double 680 / single 283 MB（60 块均 415 MB）。论文附录 C 的 B_pref（Wan 520、Hunyuan 675、Flux 465）是真实值的 ~1.6 倍，F* 的预测靠 η_pref 补偿；实测链路带宽 20–22 GB/s（论文 L40S 假设 25.2）。模型一节要用真实字节和实测带宽重做。
6. 两个 ModuleList 的模型（Flux、Hunyuan）原实现每个 manager 一个 tracker、wrapper 只看第一个，single blocks 从不让路；已改共用。Ulysses 路径的 kernel/gpu 窗口原来 `get_sp_group` 未 import，直接崩；已修。

Flux bs4 每 forward：NCCL 730–760 ms（43%），prefetch 23.7 GB（double 663 MB × 19 + single 283 MB × 38），need ≈ 1000–1240 ms vs 纯计算 ≈ 1000 ms → ratio 0.95–1.22，正压在 F* 上；auto 判 share。stock 比 launch 快 4%：切块版暴露 H2D 70 ms vs 整层 2.7 ms（切块税，链路近饱和时显现）。

Flux bs2：深在 F* 以下，stock 比 launch 快 10%（暴露 H2D 248 vs 334 ms）。切块税的来源：`_copy_one_chunk` 是同步拷贝（host 线程每 16 MB 往返一次），DMA 之间有空隙（Flux bs4 每步 220 ms 空隙、p90 0.57 ms），整层拷贝一次发出 DMA 连续。改进：chunk 异步发出 + 发出处门控 + 队列深度 1–2。

## 对论文的含义（待 auto2 / Flux 数据后定稿）

- §2.3 / Fig. 1b 的 contention stall 必须改口径：对 stock 基线，PCIe 共享的代价是 NCCL 被拉长（Wan f41 455→695 ms），不是 668 ms 的暴露 H2D。
- §3.2 机制改写为 GPU 定时窗口 + 余量判据；"bounded by one chunk" 保留（实测 0.4–0.8 chunk/kernel）。
- §4 主表基线换成 stock；收益口径变成"F* 以上接近 no（vs stock +10%）、F* 以下与 stock 持平（residency 是那一段的手段）"。
- 回复 PShp 的机制问题：in-flight ≤ 1 由同步 chunk 拷贝保证；窗口起止绑 GPU 事件/NCCL Work；trace 有数。
