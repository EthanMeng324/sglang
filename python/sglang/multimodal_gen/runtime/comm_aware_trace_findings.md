# Comm-Aware Offload Trace 结论

这份笔记汇总了当前 `old / new / no offload` 的 trace 级分析，重点回答三个问题：

- 为什么 `new` 会比 `old` 快
- 为什么 `new` 仍然会比 `no offload` 慢
- 当前 `new` 路径里，H2D 是否真的已经在 communication 前完全 yield

## 1. 如何理解 `old > new`

`new` 相比 `old` 的核心收益，仍然是我们之前分析的那条：

- `old` 的 H2D prefetch 不会在通信前 yield
- 所以 forward 里的 collective communication 会被往后推
- 最终拉长 step 的 critical path

在 trace 里，当前最有用的量化 proxy 是：

- `uncovered H2D`
  也就是“没有被 kernel 覆盖掉的 H2D 时间”

它不是一个完美的“通信推迟量”定义，因为它包含任何没被 kernel hide 掉的 H2D；但在当前这些 trace 里，它和上述机制是高度一致的，原因是：

- `old` 基本没有 `H2D / NCCL` overlap
- 所以 `old` 里的 `uncovered H2D`，大部分就是还留在 forward critical path 上、会把后续 comm 往后推的 copy 时间

因此目前可以用下面这个近似来理解：

`old -> new` 的收益 ≈ `uncovered H2D` 的减少量 - `new` 额外带来的 NCCL inflation

换句话说：

- `old` 比 `new` 慢，根因仍然是“old 的 H2D 不 yield，导致 forward 通信被推迟”
- trace 里把这个 effect 量化出来时，`uncovered H2D` 是目前最好用的指标

## 2. 各模型的 step time 现象

| 模型 / 设置 | `new - old` | `new - no` | 现象 |
|---|---:|---:|---|
| Wan5B `frame_41` | `-0.196 s` | `+0.442 s` | `new` 明显优于 `old`，但离 `no` 还有一段距离 |
| Wan5B `frame_81` | `-0.403 s` | `+0.181 s` | 随 workload 增大，`new` 明显更接近 `no` |
| HunyuanVideo `frame_21` | `-0.378 s` | `+0.262 s` | 和预期一致 |
| Flux `bs=8` | `-0.192 s` | `+0.291 s` | `new` 优于 `old`，但没有明显逼近 `no` |
| Flux `bs=16` | `-0.052 s` | `+0.254 s` | `old` 开始追上 `new`，而 `new` 并没有进一步明显逼近 `no` |

总结起来：

- `wan / hunyuan` 的行为和理论预期基本一致
  - workload 增大
  - 更多 prefetch 被 compute hide 掉
  - `new` 逐渐逼近 `no`
  - 并始终优于 `old`
- `flux` 不符合这个趋势
  - batch 增大后，`old -> new` 的收益显著缩小
  - `new -> no` 的 gap 仍然比较大

## 3. 新增发现：`new` 里确实仍然存在大量 `H2D / NCCL overlap`

之前一个自然的猜测是：

- 如果 `SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB=16` 已经把 copy 切得足够小
- 那么就算 comm 前没有完全 drain 干净，最坏情况也可能只是“拖着一个 chunk 的尾巴”

但 trace 里单个 chunk 的实际时长并不支持这个解释。

以 `Flux new` 为例，`copyKind=Host-to-Device` 的 memcpy 分布是：

- 最常见大小就是 `16 MB`
- 单次 H2D 时长：
  - `p50 ≈ 0.55 ms`
  - `p90 ≈ 1.10 ms`
  - `p95 ≈ 1.12 ms`
  - `p99 ≈ 7.1 ms`
  - `max ≈ 25 ms`

而 `Flux new - no` 的 NCCL inflation 是：

- `bs=8`: `~276-282 ms / rank / step`
- `bs=16`: `~246-262 ms / rank / step`

这说明：

- “最多只是一个 `16 MB` chunk 没 drain 干净”这个更窄的解释不成立
- 真正发生的事情更像是：`new` 里有大量 chunk 化 H2D 在 step 内持续进入 NCCL 窗口

为了验证这一点，我们直接在 steady-state denoising step (`SGL_DENOISING_STEP_1..9`) 内，计算了：

- `ncclDevKernel_SendRecv` 的实际运行时间窗口
- `Host-to-Device memcpy` 的实际运行时间窗口
- 二者的真实时间交集

结果如下：

| 模型 / 设置 | `no` overlap | `old` overlap | `new` overlap | 备注 |
|---|---:|---:|---:|---|
| Flux `bs=8` | `0 ms` | `0 ms` | `~561-567 ms / rank / step` | 约 `62%` 的 NCCL 时间与 H2D 重叠 |
| Flux `bs=16` | `0 ms` | `0 ms` | `~511-521 ms / rank / step` | 约 `33-34%` 的 NCCL 时间与 H2D 重叠 |
| Wan5B `frame_81` | `0 ms` | `0 ms` | `~350-353 ms / rank / step` | 约 `58%` 的 NCCL 时间与 H2D 重叠 |
| HunyuanVideo `frame_21` | `0 ms` | `0 ms` | `rank1 ~359 ms / step` | 活跃 rank 上约 `63%` 的 NCCL 时间与 H2D 重叠；另一张卡几乎没有 H2D |

这个结果有两个重要含义：

1. `new` 路径里的 `H2D / NCCL overlap` 是**真实存在**的，不是只靠 summary 间接推测。
2. 这不是 `flux` 独有的问题。
   - `wan` 和 `hunyuan` 的 `new` 里也有 overlap
   - 但它们的 net result 更好，因为 copy-hiding 带来的收益仍然大于这部分 NCCL inflation

换句话说：

- `old / no` 的 NCCL 窗口基本是干净的
- `new` 的 NCCL 窗口在当前实现下并不干净

因此现在更准确的判断是：

- 当前 `new` 的收益，来自“把很多 H2D 从 exposed critical path 上拿掉”
- 当前 `new` 的代价，则是“允许 chunk 化 H2D 实际进入 NCCL 窗口，导致 collective kernel 被拉长”

进一步看 overlap 在每个 NCCL kernel 内的位置，结果更偏向：

- 不是只在 kernel 的开头沾到一点
- 也不是只在 kernel 的尾部恢复一点
- 而是很多 overlapping kernel 的大部分生命周期都和 H2D 重叠

把 overlapping NCCL kernel 按位置粗分为：

- `begin_only`: overlap 主要出现在开头
- `end_only`: overlap 主要出现在结尾
- `both_edges`: 开头和结尾都有 overlap
- `interior_only`: 只在中间某段 overlap
- `full`: 几乎整个 kernel 都在 overlap

得到的结果是：

| 模型 / 设置 | `begin_only` | `end_only` | `both_edges` | `interior_only` | `full` |
|---|---:|---:|---:|---:|---:|
| Flux `bs=8 new` | `1.5-1.6%` | `3.6-4.2%` | `8.7-9.1%` | `0.2-0.3%` | `84.8-85.9%` |
| Flux `bs=16 new` | `29.1-30.1%` | `1.0%` | `5.4-7.8%` | `0.4-0.8%` | `60.7-63.7%` |
| Wan5B `frame_81 new` | `9.1-9.8%` | `10.8%` | `15.6-15.9%` | `11.0-12.2%` | `51.3-53.6%` |
| Hunyuan `frame_21 new` | `5.4%` | `1.9%` | `14.0%` | `0.3%` | `78.4%` |

对应的 kernel 内位置统计也说明同一件事：

- `Flux bs=8 new`
  - overlapping kernel 的 `median first_overlap_pos = 0.0`
  - `median last_overlap_pos = 1.0`
  - `median overlap_duty ≈ 0.91`
- `Flux bs=16 new`
  - `median first_overlap_pos = 0.0`
  - `median last_overlap_pos = 1.0`
  - `median overlap_duty ≈ 0.86-0.88`
  - 但 `mean last_overlap_pos ≈ 0.78`，说明有一批 kernel 是“从开头开始 overlap，但在结束前停掉”
- `Wan frame_81 new`
  - `median first_overlap_pos = 0.0`
  - `median last_overlap_pos = 1.0`
  - `median overlap_duty ≈ 0.72-0.73`
- `Hunyuan frame_21 new`
  - 活跃 rank 上 `median first_overlap_pos = 0.0`
  - `median last_overlap_pos = 1.0`
  - `median overlap_duty ≈ 0.87`

这说明：

- overlap 并不主要长在 NCCL 的尾部
- 更像是 overlap 从 NCCL kernel 一开始就存在，并且常常覆盖其大部分生命周期

因此仅仅说“comm 还没开始前，有一个 H2D chunk 没 drain 干净”已经不够了。更准确的说法是：

- 当前 `new` 路径里，communication 开始时 H2D 往往已经在场
- 或者 communication 一开始后，H2D 很快又恢复
- 最终形成了持续性的 `H2D / NCCL overlap`

## 4. 统一的 trace 解释框架

从 trace 看，所有模型都可以用同一个框架来解释：

- `old` 的问题主要是 **uncovered H2D 很大**
- `new` 的好处是把大部分 H2D hide 掉
- 但 `new` 同时会引入 **NCCL inflation**

最后 step time 的净结果，就是这两项的平衡。

### 4.1 Wan5B `frame_81`

analysis summary：

- `no = 1.302 s`
- `old = 1.886 s`
- `new = 1.483 s`

对应 trace 量化：

- `old` 的 uncovered H2D 约 `630 ms / rank / step`
- `new` 的 uncovered H2D 约 `10 ms / active rank / step`
- `new` 的 NCCL 总时长约 `608-611 ms / rank / step`
- `no` 的 NCCL 总时长约 `430-438 ms / rank / step`
- `new` 的 `H2D / NCCL overlap` 约 `350-353 ms / rank / step`

解释：

- `new` 几乎把 exposed copy 全部消掉了
- `new` 里确实也有 NCCL inflation
- `new` 里也确实存在显著 `H2D / NCCL overlap`
- 但这部分 inflation 小于 H2D hiding 带来的收益

所以：

- `new` 明显快于 `old`
- 同时 workload 变大后，`new` 也明显更接近 `no`

### 4.2 HunyuanVideo `frame_21`

analysis summary：

- `no = 1.340 s`
- `old = 1.979 s`
- `new = 1.601 s`

对应 trace 量化：

- `old` 的 uncovered H2D 约 `682 ms / rank / step`
- `new` 的 uncovered H2D 约 `73 ms / active rank / step`
- `new` 的 NCCL 总时长约 `517-568 ms / rank / step`
- `no` 的 NCCL 总时长约 `352-356 ms / rank / step`
- `new` 的 `H2D / NCCL overlap` 在活跃 rank 上约 `359 ms / step`

解释：

- `new` 的 H2D hiding 收益很大
- 也存在 NCCL inflation
- `new` 里活跃 rank 上同样存在显著 `H2D / NCCL overlap`
- 但 inflation 不足以抵消前面的收益

因此：

- `new` 仍然明显优于 `old`
- 并且 reasonably 接近 `no`

### 4.3 Flux `bs=8`

analysis summary：

- `no = 1.624 s`
- `old = 2.079 s`
- `new = 1.895 s`

对应 trace 量化：

- `old` 的 uncovered H2D 约 `503 ms / rank / step`
- `new` 的 uncovered H2D 约 `21 ms / rank / step`
- `new` 的 NCCL 总时长约 `908-914 ms / rank / step`
- `no` 的 NCCL 总时长约 `632 ms / rank / step`
- `new` 的 `H2D / NCCL overlap` 约 `561-567 ms / rank / step`
- `old / no` 的 `H2D / NCCL overlap` 基本都是 `0`

解释：

- `new` 在 `bs=8` 时其实已经把大部分 H2D hide 掉了
- 但 `new - no` 剩下来的主要 gap，已经不再是 exposed H2D
- 而是 **NCCL kernel 自身被拉长**
- 并且这个拉长与大量真实 `H2D / NCCL overlap` 对应得上

也就是说，这时 `new` 慢于 `no` 的主因，已经从“copy 没藏住”变成了“comm inflation 太大”。

### 4.4 Flux `bs=16`

analysis summary：

- `no = 3.131 s`
- `old = 3.437 s`
- `new = 3.385 s`

对应 trace 量化：

- `old` 的 uncovered H2D 约 `313 ms / rank / step`
- `new` 的 uncovered H2D 约 `6.8 ms / rank / step`
- `new` 的 NCCL 总时长约 `1523-1529 ms / rank / step`
- `no` 的 NCCL 总时长约 `1267-1277 ms / rank / step`
- `new` 的 `H2D / NCCL overlap` 约 `511-521 ms / rank / step`
- `old / no` 的 `H2D / NCCL overlap` 基本都是 `0`

解释：

- batch 变大后，`old` 的 exposed H2D 也开始被更多 compute hide 掉
- 所以 `old` 会向 `new` 靠近
- 但 `new` 在 `bs=8` 时就已经几乎把 H2D hide 完了
- 因此继续增大 batch，对 `new` 带来的额外收益很小
- 此时 `new - no` 的主要 gap 仍然是 NCCL inflation

这就是为什么：

- `old` 在 `bs=16` 明显追近 `new`
- `new` 并没有显著继续逼近 `no`

## 5. 为什么 design 明明要求 yield，trace 里仍然会有 overlap

从实现上看，当前设计确实已经显式尝试避免 communication 窗口内的 copy：

- background worker 每发一个 chunk 之前，都会先看 `comm_tracker.is_active()`
- communication wrapper 开始前，会调用 `quiesce_prefetch_for_comm()`
- `quiesce_prefetch_for_comm()` 最终会走到 `quiesce_copy_stream_for_comm()`

对应代码路径是：

- worker 线程在 [layerwise_offload_chunkwise.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py) 的 `_prefetch_worker_loop()` 中，每发一个 chunk 前都会调用 `_wait_for_comm_inactive()`
- comm-aware quiesce 在 [usp.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py) 和 [layer.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/layers/attention/layer.py) 里，都会在 collective 前调用 `_quiesce_prefetch_for_comm()`
- 但真正的 copy-stream quiesce 目前只是 [layerwise_offload_chunkwise.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py) 里的：
  - `torch.cuda.current_stream().wait_stream(self.copy_stream)`

这套实现之所以仍然允许 overlap，我目前的判断是：

1. `quiesce_copy_stream_for_comm()` 并不是真正的 “drain copy stream”
   - 它只是让 **当前 stream** 等 copy stream
   - 但 trace 里真正执行 NCCL kernel 的往往是 NCCL 自己的通信 stream
   - 因此它更像是在 collective launch 点插了一个 stream dependency，而不是把整个 comm 生命周期前的 copy 全部排空

2. `comm_tracker` 的 active 生命周期更像是“collective API 调用还在进行”
   - 而不是“对应的 NCCL kernel 已经在 GPU 上执行完成”
   - wrapper 在 Python 函数返回后就会 `mark_end`
   - 但底层 `dist.all_to_all_single` / `all_gather_into_tensor` / `ft_c.all_to_all_single` 对应的 NCCL kernel 仍可能继续在 NCCL stream 上运行
   - 这意味着 worker 是按 “Python comm scope” 来停/开 H2D，而不是按 “GPU NCCL kernel 生命周期” 来停/开 H2D

3. 等待 comm inactive 的粒度也是“每个 chunk 一次”
   - background worker 和 blocking catch-up 都是在发每个 chunk 前调用 `_wait_for_comm_inactive()`
   - 但只要 tracker 一旦变成 inactive，下一次 chunk launch 就可以继续
   - 因此它不是一个“从 comm 开始到 comm 完整结束都禁止 copy”的全局冻结

4. `worker` 还有一个入口 race
   - `_prefetch_worker_loop()` 是先 `_wait_for_comm_inactive()`，再进入 `_copy_one_chunk()`
   - `_copy_one_chunk()` 里会拿 `copy_lock`，但拿到锁之后不会再二次检查 `comm_tracker`
   - 因此如果 worker 恰好在 `mark_start()` 之前通过了 inactive 检查，它仍然可能在 quiesce 释放 `copy_lock` 后，把 **一个** chunk 发进 collective 窗口

5. 于是会出现下面这个时序：
   - communication wrapper 退出，`comm_tracker` 变成 inactive
   - background worker 看到 inactive，于是继续发下一个 H2D chunk
   - 但前一批 NCCL kernel 其实还没跑完
   - 于是新 H2D 就进入了 NCCL 窗口

这也解释了为什么：

- `old / no` 基本没有 `H2D / NCCL overlap`
- `new` 却在三个模型里都出现了 overlap

也就是说，当前问题不是“design 完全没有做 yield”，而是：

- **yield 的控制点是对的**
- 但 **当前 active / quiesce 的语义还不足以把 GPU 上真正正在执行的 NCCL kernel 排干净**

而前面的 overlap 位置统计也和这个实现判断是一致的：

- 大多数 overlapping NCCL kernel 的 `first_overlap_pos` 非常接近 `0`
- `full` 或 `both_edges` 的占比很高
- `end_only` 的比例反而很低

所以目前更像是：

- comm 开始时 H2D 往往已经在场，或者在 NCCL kernel 启动后极快恢复
- 而不是“绝大多数 overlap 只是出现在 comm 快结束的尾部”

## 6. 进一步收细到“单次 collective”粒度

前面的 overlap 统计是以单个 NCCL kernel 为粒度。为了进一步区分：

- 是不是只有 “最后一个没 drain 干净的 chunk” 跟着进了 comm
- 还是 comm 的 active scope 太短，导致 comm 过程中又恢复了新的 H2D

我们又把粒度收细到 **一次完整 collective wrapper**，也就是代码里嵌入的一次 `all_to_all`。

对 `USP` 路径，可以直接使用 trace 里的 `SGL_REAL_COMM_USP_DEV{0,1}` NVTX range 作为单次 collective 窗口；再去统计这个窗口内和它相交的 H2D memcpy / `SGL_PREFETCH_H2D` 事件。

### 6.1 `Flux new`：overlap 的 collective 基本都只带着一个 chunk，但这个 chunk 是在 comm 期间新发出的

steady-state (`SGL_DENOISING_STEP_1..9`) 下，`Flux new` 的结果很一致：

- `bs=8`
  - 每张卡约 `2052` 次 collective
  - 其中约 `1332` 次（`64.9%`）与 H2D 有真实 overlap
  - 这些 overlapping collective 中，**GPU 上重叠的 H2D chunk 数量几乎总是 `1`**
    - `median = 1`
    - `p90 = 1`
    - `max = 1`（另一张卡偶尔有一次 `2`）
- `bs=16`
  - 结论几乎一样
  - 同样约 `64.9%` 的 collective 与 H2D overlap
  - overlapping collective 的 GPU-overlap chunk 数量也几乎总是 `1`

如果只看这一层，会让人觉得：

- 它好像确实符合“最多只是一个尾部 chunk 没 drain 干净”

但再进一步把同一进程内的 `SGL_PREFETCH_H2D` host-side NVTX 也对齐进去，结论就不一样了。

对 `Flux new`：

- `bs=8`
  - `device0`: `1332` 个 overlapping collective 里，`1332` 个都能看到 **同进程的 `SGL_PREFETCH_H2D` 在 collective 窗口内启动**
  - `device1`: `1332` 个里有 `1331` 个满足同样条件，只有 `1` 个更像“纯尾巴”
- `bs=16`
  - 两张卡都是 `1332 / 1332`
  - 也就是所有 overlapping collective 都能看到 **同进程 H2D 在 comm 窗口内启动**

时间位置上也很集中：

- `SGL_PREFETCH_H2D` 往往在 collective 开始后约 `+0.04 ~ +0.05 ms` 就启动
- 对应的 GPU memcpy 往往在 collective 开始后约 `+0.07 ~ +0.13 ms` 出现

这比“comm 前有一个 chunk 还没完全 drain 干净”更像：

- collective 的 active / quiesce 覆盖范围偏短
- comm 刚开始后不久，worker 就重新恢复发 chunk
- 或者 worker 在 `mark_start()` 之前已经通过了 inactive 检查，并在 quiesce 之后漏进 **一个** chunk

所以对 `Flux` 来说，当前证据更支持：

- **主因不是“只拖着一个尾部 chunk”**
- 而是 **comm scope 过短，导致 H2D 在单次 collective 期间又恢复了**

### 6.2 `Wan new`：kernel 级 overlap 存在，但 generic collective range 内几乎看不到 GPU memcpy overlap

`Wan5B frame_81 new` 在 kernel 粒度上，已经确认存在显著 `H2D / NCCL overlap`：

- 约 `350-353 ms / rank / step`

但如果把粒度收细到 generic 的 `SGL_REAL_COMM_USP_DEV{0,1}` range，会得到另一种现象：

- 在这些 collective range 内，几乎看不到 GPU memcpy overlap
- 同时同一进程的 `SGL_PREFETCH_H2D` 也几乎都发生在 collective 开始之前，而不是 collective 期间
  - `wan new`：两张卡上约 `4130 / 4320` 个 collective 能看到“comm 前不久有 H2D”
  - 但 `host_during = 0`

这说明：

- `Wan` 里 generic USP wrapper 的 NVTX scope，本身并没有覆盖完整的 NCCL GPU kernel 生命周期
- kernel 级 overlap 发生在 “wrapper 看起来已经结束” 之后

换句话说，`Wan` 进一步支持了同一个方向：

- 问题不只是 “copy stream 没有在 comm 入口前完全 drain”
- 更像是 **当前 wrapper / tracker 看到的 communication 窗口，短于 GPU 上实际的 NCCL 执行窗口**

### 6.3 `Hunyuan new`

`HunyuanVideo` 当前 trace 里没有和 `USP` 一样的 `SGL_REAL_COMM_*` NVTX range，因此暂时还不能用完全同一口径做“单次 collective”分析。

但它在 kernel 粒度上的现象仍然一致：

- `old / no` 的 `H2D / NCCL overlap` 基本为 `0`
- 显著 overlap 只出现在 `new`

所以目前可以先保守地说：

- “`new` 会把 H2D 带进 communication 窗口” 并不是 `Flux` 独有
- 但能否在“单次 collective”粒度上复现同样的 host-side 恢复模式，还需要额外 instrumentation

### 6.4 这一层分析的结论

如果把 overlap 再收细到单次 collective 粒度，当前最关键的结论是：

1. 对 `Flux`，overlap 的 collective 几乎总是只 overlap **一个** H2D chunk
2. 但这个 chunk 大多数不是“comm 前已经在飞的尾巴”
3. 更常见的是：**comm 开始后不久，这个 chunk 才在同一进程里被重新发出**
4. 对 `Wan`，generic collective range 里则看不到 “comm 期间重新发 H2D”，更像是 wrapper scope 本身短于真实 NCCL kernel 生命周期

因此：

- “单个 chunk 很小，所以最坏情况不该有这么大的 inflation” 这个判断仍然成立
- 但根因并不是 chunk size 本身
- 更像是 **comm-aware 的 active / quiesce 作用域没有覆盖住整个 collective 的真实 GPU 生命周期**
- 从模型共性看，**原因 2（wrapper / tracker scope 过短）更像主因**
- 而 `Flux` 上额外看到的 “wrapper 内漏进 exactly one chunk”，更像是入口 race 或单-chunk 泄漏的附加因素

### 6.5 进一步定位：`mark_end` 到真实 NCCL kernel 结束到底差了多少

为了把 “scope 过短” 定成具体数字，我们又沿着：

`wrapper NVTX -> cuLaunchKernelEx(runtime) -> correlationId -> ncclDevKernel_SendRecv(kernel)`

做了精确配对。

steady-state 下，`USP` 路径上的结果非常一致：

- 每个 generic `SGL_REAL_COMM_USP_DEV*` wrapper，基本都只包含
  - `1` 个 `cuLaunchKernelEx`
  - 并最终对应 `1` 个 `ncclDevKernel_SendRecv` kernel
- 但 wrapper 结束和 kernel 真正开始/结束之间，存在非常大的时间差

量化结果如下：

| 模型 / 设置 | `wrapper_end - launch_end` | `kernel_start - wrapper_end` | `kernel_end - wrapper_end` |
|---|---:|---:|---:|
| Flux `bs=8 new` | `~0.18 ms (p50)` | `~225 ms (p50)` | `~230 ms (p50)` |
| Flux `bs=16 new` | `~0.18 ms (p50)` | `~1014-1031 ms (p50)` | `~1020-1037 ms (p50)` |
| Wan5B `frame_81 new` | `~0.17 ms (p50)` | `~203-208 ms (p50)` | `~204-210 ms (p50)` |

这说明：

- wrapper 结束几乎就跟在 `cuLaunchKernelEx` 后面
- 也就是说，当前 `mark_end` 覆盖的本质上只是 **“launch 已经发出”**
- 而不是 **“GPU 上这次 communication 已经执行完成”**

更关键的是：

- 对 `Flux` 和 `Wan`，这种 “wrapper 结束远早于 kernel 真正结束” 的现象都是共性的
- 因此从跨模型共性看，**原因 2 已经可以认为是主导原因**

换句话说，当前实现里：

- `comm_tracker` 看到的 active window，大致只覆盖了 host-side collective launch
- 但真实的 NCCL GPU kernel 往往在几百毫秒之后才开始，甚至更晚才结束

所以只要 worker 依据 `mark_end` 恢复 H2D，它几乎必然会在真实 NCCL kernel 生命周期内重新发起 copy。

### 6.6 `kernel-window` 版本的进一步诊断：方向对了，但 `start/end` 仍然都没有对准

在 `Flux bs=8` 上，我们又对比了：

- 旧版 `launch-window new`
- 新版 `kernel-window new`

新的 `kernel-window` 版本，step time 反而更差：

- `launch-window new`: `1.895 s/step`
- `kernel-window new`: `2.011 s/step`

但 trace 分解说明，这不是因为修复方向错了，而是因为：

1. 它确实减少了 `NCCL inflation`
2. 但同时把太多 H2D 重新暴露回了 critical path

量化结果如下（steady-state，按 `rank / step`）：

| 指标 | `launch-window new` | `kernel-window new` |
|---|---:|---:|
| NCCL 总时长 | `0.911 s` | `0.761 s` |
| `NCCL/H2D overlap` | `0.564 s` | `0.253 s` |
| uncovered H2D | `0.021 s` | `0.237 s` |
| `SGL_PREFETCH_WAIT_COMM_BG` | `~40.7 ms` | `~330.1 ms` |

所以这版的净效果是：

- overlap 和 `NCCL inflation` 确实下降了
- 但 comm pause 过于保守，导致 background prefetch 被卡住太久
- 最终 uncovered H2D 大幅上升，`new` 反而更慢

再往前看一层，`kernel-window` 到底修掉了什么、又没修掉什么：

- 在单次 `SGL_REAL_COMM_USP_DEV*` wrapper 粒度上：
  - 旧版 `launch-window new` 中，约 `64.9%` 的 collective wrapper 内还能看到同进程 `SGL_PREFETCH_H2D` 启动
  - 新版 `kernel-window new` 中，这个比例降到了约 `1.2% - 1.8%`
- 这说明：**wrapper 窗口内重新恢复 H2D** 这件事，基本已经被修掉了

但与此同时：

- 真实 GPU 上与 NCCL kernel 相交的 H2D collective 占比，仍然约是 `64.9%`
- 也就是说：**wrapper 内不再恢复 H2D**，并不等于 **真实 NCCL kernel 生命周期内不再 overlap**

这就把问题进一步拆成了两半：

1. `start` 仍然偏早
   - 当时的实现是在 wrapper 入口就开始 block 新的 H2D launch
   - 这会把 `quiesce` / host-side launch 前的这段时间也一起算进 active window
   - 结果是：H2D 可 overlap 的预算被白白缩短，`wait_comm` 增大

2. `end` 仍然偏早
   - 当时的实现虽然不再用纯 host-side `mark_end`
   - 但 `end_event` 仍然记录在 `current_stream` 上，而不是 NCCL 真正执行的 comm stream 上
   - 所以 H2D 虽然不再在 wrapper 里面恢复，但往往仍会在 wrapper 结束后约 `~0.9 ms` 就恢复
   - 而真实 `ncclDevKernel_SendRecv` 的结束，中位数仍在约 `~238 ms` 之后

一句话总结就是：

- `kernel-window` 版本已经部分修掉了“wrapper scope 太短”的问题
- 但它仍然没有真正把 active window 对准 **真实 NCCL kernel 的开始和结束**
- 因此：
  - `start` 侧过早，带来过多 pause
  - `end` 侧过早，kernel 级 overlap 仍然存在

这也正是后续修正的方向：

- `start` 需要后移到 “copy stream 已经 drain 干净、且 comm 即将真正 launch” 的位置
- `end` 需要绑定到更接近真实 collective completion 的信号，而不是仅仅绑定到 `current_stream` 上的 event

## 7. `wan` / `flux` / `hunyuan` 的关键差别

一个重要背景是：

- `Flux` 用的是 `USPAttention`
- `Wan` 也用的是 `USPAttention`
- `HunyuanVideo` 用的是 `UlyssesAttention`

对应代码位置：

- `USPAttention` 定义在 [layer.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/layers/attention/layer.py)
- `Flux` 接入在 [flux.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/models/dits/flux.py)
- `Wan` 接入在 [wanvideo.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py)
- `HunyuanVideo` 接入在 [hunyuanvideo.py](/home/mh/sglang/python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py)

这件事说明两点：

1. `hunyuan` 不能直接反证 `flux` 的问题  
因为它不是同一条通信实现路径。

2. `wan` 的结果说明“USP 路径本身并不必然失败”  
`wan` 和 `flux` 都是 USP，但 `wan` 仍然呈现出更符合预期的趋势。

同时还要再补一层：

- `wan` 和 `flux` 都是 USP，并且 `new` 里都存在显著 overlap
- 这说明 “当前 `new` 会把 H2D 带进 NCCL 窗口” 本身是一个更普遍的实现现象
- 但 `flux` 的 overlap 更大、H2D 量也更大，因此最终 inflation 对 step time 的伤害更重

所以目前更合理的判断是：

- `flux` 的异常并不只是“因为它是 USP”
- 更像是 `flux + 当前 USP comm-aware 接入方式` 的组合，导致 NCCL inflation 项异常偏大

## 8. 当前对 `Flux` 的最佳解释

当前最合理的结论是：

- 理论本身并没有错
- 但在 `flux` 上，当前 `USP` 路径引入了一个很大的 **NCCL inflation** 项
- 并且这个 inflation 与 trace 中大规模的 `H2D / NCCL overlap` 是对应的
- 这个项已经足以抵消大部分 copy-hiding 收益，尤其是在大 batch 时

因此，`flux` 当前不再是一个干净的：

`slowdown ~= max(T_pref - T_comp, 0)`

验证对象。

对 `flux new` 来说，更合适的分解应当是：

`new - no ~= residual exposed H2D + NCCL inflation`

而且当前第二项是主导项。

## 9. 实验含义

基于当前 trace，我觉得可以这样理解：

- `wan / hunyuan` 仍然是支持理论的正例
  - 尽管 `new` 里也有真实 overlap
  - 但 copy-hiding 的收益仍然大于 comm inflation
- `flux` 暂时更像是一个实现/接入问题暴露器，而不是理论验证样本
  - 因为在 `flux` 上，这部分 overlap 带来的 NCCL inflation 已经足够主导 `new - no`

如果还想继续抢救 `flux`，最值得优先验证的是：

1. 只针对当前 `new` 路径，把 `comm_tracker` / `quiesce` 的语义从“API 级 inactive”提升到“GPU kernel 级真正 drain”
2. 对比 `USP` 和 `Ulysses` 路径，确认哪个 collective wrapper 最早释放了 `comm_active`
3. 再次把 `SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB` 缩小，只作为辅助验证，而不要再把它当成主要根因

如果这些都不能显著压低 `flux new` 的 NCCL inflation，那么换模型会是合理选择。因为那时问题就不再是“compute 不够大，prefetch 没藏住”，而是 `flux` 这条具体路径本身不适合作为当前机制的验证对象。
