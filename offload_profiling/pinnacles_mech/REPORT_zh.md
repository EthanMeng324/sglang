# ChunkFlow 通信机制研究：结论与数据（Pinnacles 2×L40S，2026-11-06～07）

## 一句话

（数据齐全：Wan f41/f81、Hunyuan f13、Flux bs2/4/8，每格 4–8 臂，同作业对照。）论文里的 ChunkFlow 收益主要来自基线被人为串行化（quiesce），而不是让路；让路机制本身在论文的实现里没有生效（窗口只覆盖 launch 调用）。把窗口正确地绑到 GPU 上的集合通信、并按余量决定让不让路之后，F* 以上 offload 的代价降到 ≈0（Wan f81：对 no +0.3%，stock 基线 +10%），F* 以下退回共享（与 stock 持平或略差）。切块实现本身有一个"每 chunk 同步拷贝"的税，Flux 在 F* 以下因此比整层慢 4–10%，需要改成异步发出。

## 设置

- 2×L40S（46 GB，PCIe Gen4 ×16，无 NVLink，同 NUMA），Ulysses SP=2，10 步，论文分辨率/步数/prompt，chunk 16 MB，FlashAttention（sgl_kernel）。
- 代码：EthanMeng324/sglang `v0.5.9-shard`（= 论文代码）+ `pinnacles-mech` 分支补丁（stock 开关、none/gpu/auto 窗口模式、共用 tracker、Ulysses 路径 import 修复）。
- 论文 L40S 表的 No/LW/CF 三臂在所有配置上复现在 3% 内；同作业重复差 <1%。

## 臂

| 臂 | 含义 |
|---|---|
| no | 全驻留 |
| old | 论文 "Layerwise"：整层 prefetch + 每次集合通信前 quiesce（compute stream 等 copy stream 完成，upstream 没有） |
| stock | 整层 prefetch，无 quiesce = upstream SGLang |
| none | 16 MB 切块，不让路 |
| launch | 论文 ChunkFlow：窗口 = wrapper 调用（≈0.2 ms） |
| kernel | 窗口 host 调用 → NCCL Work 完成 |
| gpu | 窗口 GPU start event → NCCL Work 完成 |
| auto | gpu 窗口 + 测量比较法选模式：预热 2 forward → share 下测 T_pref/T_comp（在线 F* 测试）→ 试一个 forward 的 yield → 留快的 → 周期复测 |

## 步时（ms；括号为相对 no）

| 配置 | no | old（论文基线） | stock | none | launch（论文 CF） | kernel | gpu | auto v4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan f41 | 1146 | 1820 (+59%) | 1394 (+22%) | 1361 (+19%) | 1348 (+18%) | 2112 (+84%) | 1557 (+36%) | 1373 (+20%, share) |
| Wan f81 | 2378 | 2919 (+23%) | 2618 (+10%) | 2545 (+7%) | 2552 (+7%) | 2990 (+26%) | 2439 (+1.8%) | **2406 (+0.3%)** |
| Hunyuan f13 | 1402 | 2107 (+50%) | 1687 (+20%) | 1722 (+23%) | 1707 (+22%) | — | 1751 (+26%) | 1685 (+21%, share) |
| Flux bs2 | 811 | （论文 1456） | 1155 (+42%) | — | 1269 (+56%) | — | 1622 (+100%) | 1234 (+52%, share) |
| Flux bs4 | 1485 | （论文 2046） | 1707 (+15%) | — | 1773 (+19%) | — | 1896 (+28%) | 1749 (+18%, share) |
| Flux bs8 | OOM（解码阶段） | 3468 | 3286 | — | 3221 | — | **3025** | **3068**（稳态 ≈3000） |

论文 L40S 表对应值：Wan f41 1.111/1.772/1.347，f81 2.333/2.876/2.388；Hunyuan f13 1.376/2.088/1.704；Flux bs2 0.751/1.456/1.255，bs4 1.511/2.046/1.742，bs8 OOM/3.389/3.103。

## nsys 分解（每 rank 每步 ms：NCCL 总 / NCCL∩H2D 重叠 / 暴露 H2D / worker 等 comm）

| 配置 | no | old | stock | launch | gpu | auto v4 |
|---|---|---|---|---|---|---|
| Wan f41 | 455/0/0/– | 454/0/668/– | 695/606/4/– | 644/472/3/204 | 658/186/184/474 | 666/476/10/0 |
| Wan f81 | 795/0/0/– | 791/0/550/– | 1048/626/2/– | 981/465/2/251 | 866/128/4/456 | 856/159/4/413 |
| Hunyuan f13 | 390/0/0/– | 390/0/706/– | 598/521/80/– | 536/371/137/60 | 464/42/265/404 | 575/407/107/0 |
| Flux bs4 | 517/0/0/– | — | 729/465/3/– | 719/478/70/111 | 721/119/215/536 | 749/486/57/0 |
| Flux bs8 | — | 1161/0/324/– | 1308/412/1/– | 1239/473/1/163 | 1101/100/17/551 | 1132/194/11/443 |

每个 NCCL kernel 漏进的 chunk 数：launch 1.6 个（70% 时间重叠）；gpu 0.4–0.8 个（窗口打开时在飞的那个，起点中位数 −0.06 ms）。

## 机制结论

1. **基线**：quiesce 让论文基线多付 550–700 ms/步的暴露 H2D；去掉后整层 prefetch 只慢 no 10–22%（Flux bs2 42%），ChunkFlow(launch) 对 stock：Wan +3%、Hunyuan 0、Flux −4～−10%。Fig. 1b 的 "contention stall 21%" 是这个串行化。
2. **launch ≈ none**：论文的让路没有发生；PCIe 共享的真实代价是 NCCL 被拉长 40–50%（Wan f41 455→695 ms），prefetch 几乎全部藏住。
3. **host 领先 GPU**：kernel 模式从 host 调用起算 active，prefetch 多停一倍时间（f41 +84%）。窗口必须绑 GPU 事件。
4. **gpu 模式**：F* 以上赢（Wan f81 +1.8%），F* 以下输（f41 +36%、Hunyuan f13 +26%、Flux bs2/4 +100%/+28%）。原因：Ulysses 每 block 16 个 ~1.3 ms 的小 kernel，每次让路付 0.5–1 ms（排干在飞 chunk + 轮询恢复），F* 以下没有这个余量。
5. **auto v4**：F* 以上 Wan f81 +0.3%、Flux bs8 比 stock 快 7%；F* 以下与共享持平（Wan f41 +20% vs launch +17%，Hunyuan f13 +21% vs stock +20%，Flux bs2/4 介于 stock 与 launch 之间）。逐集合通信判断（v1/v2）在两个 rank 间失配，只拿到一半收益；v3 的每 forward 在线 F* 测试被启动期和 stall 自强化干扰；v4 直接比较两种模式各一个 forward 的时长，决策两 rank 一致。
6. **切块税**：`_copy_one_chunk` 同步拷贝，DMA 间有空隙（Flux bs4 每步 220 ms），链路近饱和时暴露 H2D 多 30–90 ms；整层拷贝一次发出、DMA 连续。两者忙时带宽相同（~21 GB/s）。改法：异步发出 + 发出处门控 + 在飞队列深度 1–2。
7. **多 ModuleList bug**：Flux/Hunyuan 原来只有第一个 manager 的 tracker 被 wrapper 看到，single blocks 从不让路（已修）。Ulysses 路径的精确窗口 `get_sp_group` 未 import（已修）。
8. **Flux 的异常**（bs8 验证：过 F* 后 gpu 3025 < launch 3221 < stock 3286，与 Wan f81 同理）：通信占步时 43%（Wan 19–35%）、每 forward prefetch 23.7 GB，bs4 正好 T_pref≈T_comp（ratio 0.95–1.2）；共享模式下 NCCL 被拉长的绝对量大、切块税也大，所以 old→new 收益小、new 不逼近 no。bs8 进入让路有收益区。
9. **B_pref**：实际每块 Wan 327 MB、Hunyuan 双流 680/单流 283（均 415）、Flux 双流 663/单流 283；论文附录 C 的 520/675/465 高 ~1.6×，靠 η_pref 补偿；实测链路带宽 20–22 GB/s。

## 对论文

- §2.3/Fig.1b：contention 的代价改为 NCCL 拉长（量化：Wan f41 +53%、f81 +32%、Hunyuan +53%、Flux bs4 +41%），删掉 quiesce 产生的 "stall"。
- §3.2：机制改写为 GPU 定时窗口 + 余量判据（测量比较法）；"bounded by one chunk" 保留（实测 0.4–0.8）。实现改成异步发出。
- §4：基线换 stock；主表收益口径："F* 以上 offload 接近免费（Wan f81 +0.3% vs stock +10%），F* 以下与 stock 持平，residency 是那一段的手段"。Flux bs2/4 现状是负收益，要么修切块税，要么如实报。
- 附录 C：用真实字节和实测带宽重做 F*。
- 论文表里 Flux bs6/8、Hunyuan f21/29 的 "No Offload OOM" 发生在 VAE 解码阶段，denoising 步时是可量的；harness 要改成失败的 run 也保留 nsys trace。
- PShp 的机制问题：in-flight ≤1 由同步拷贝保证（改异步后 ≤队列深度）；窗口起止绑 GPU event / NCCL Work；trace 数字见上表。

## 下一步选项

A. 实现异步 chunk 发出（2b），重跑 Flux bs2/4、Wan f41（预期把切块税消掉，F* 以下与 stock 持平或略好）。
B. 重复实验 n=3 出误差（现有同作业重复 <1%，可少跑）。
C. 4×L40S SP=4（gnode015/016 有 4 卡，等空）。
D. PShp 要的"延后到最后一个 all-to-all 再发整层"的静态基线。
E. 把 pinnacles-mech 的代码和本文档带回本地 chunkflow 分支（bundle）。
