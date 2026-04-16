# Text-Conditioned DiT 的 Prefetch / Comm 建模

本文把当前代码库中最相关的三类 text-conditioned DiT backbone 放到同一套分析框架下：

- `wanvideo`：text-to-video
- `hunyuanvideo`：text-to-video
- `flux`：text-to-image

目标是回答四个问题：

1. 每个模型在 forward 主路径上，哪些位置会发生跨卡通信，因此需要插入 `quiesce` 做 comm-aware offload。
2. 每个 offload unit 的计算量如何拆解成更细的子项。
3. 每个 offload unit 的参数 prefetch 量如何估算。
4. 何时当前 block 的计算时间足以盖住下一 block 的 prefetch 时间。

## 1. 统一记号

记：

- `U`：Ulysses degree
- `S_v`：video token 的全局长度
- `S_x`：image token 的全局长度
- `\bar{S}_v = S_v / U`：每张 GPU 上的本地 video token 长度
- `\bar{S}_x = S_x / U`：每张 GPU 上的本地 image token 长度
- `S_t`：text token 长度
- `P_{\mathrm{peak}}`：设备理论峰值算力
- `BW_{\mathrm{peak}}`：host-to-device 理论峰值带宽
- `\eta_{\mathrm{comp}}`：计算效率系数
- `\eta_{\mathrm{h2d}}`：H2D 带宽效率系数

统一定义：

```math
T_{\mathrm{comp}} = \frac{F_{\mathrm{layer}}}{P_{\mathrm{eff}}}
= \frac{F_{\mathrm{layer}}}{\eta_{\mathrm{comp}} P_{\mathrm{peak}}}
```

```math
T_{\mathrm{pref}} = \frac{B_{\mathrm{layer}}}{BW_{\mathrm{h2d,eff}}}
= \frac{B_{\mathrm{layer}}}{\eta_{\mathrm{h2d}} BW_{\mathrm{peak}}}
```

其中：

- `F_{\mathrm{layer}}`：当前 offload unit 的总 FLOPs
- `B_{\mathrm{layer}}`：当前 offload unit 需要从 host 预取到 GPU 的参数字节数

对 comm-aware layerwise offload，最关键的判据是：

```math
T_{\mathrm{comp}}(S^\star) = T_{\mathrm{pref}}
```

其中 `S^\star` 表示临界序列规模。其含义是：

- 当 `S < S^\star` 时，compute window 不足以完全盖住下一层 prefetch，下一层入口更容易出现 exposed wait。
- 当 `S > S^\star` 时，prefetch 更容易被当前层计算完全隐藏，step latency 更接近 `no offload`。

更接近实现的近似写法为：

```math
T_{\mathrm{exposed}} \approx \max\!\left(T_{\mathrm{pref}} - T_{\mathrm{comp}}, 0\right)
```

参数量的精确表达总是：

```math
B_{\mathrm{layer}} = \sum_{p \in \mathcal{P}_{\mathrm{layer}}} \mathrm{numel}(p)\,\beta(p)
```

其中 `\beta(p)` 是参数 `p` 的字节宽度。下面为了得到可用的闭式模型，通常只保留 projection / FFN / modulation 这类主导项，把 norm、bias 等低阶项并入 `O(d)`。

## 2. Comm-Aware 应该插在哪

在当前 profiling setting 下只指定了 `ulysses_degree`，没有显式开启 tensor parallel，因此可简化为：

- `tp_size = 1`
- linear / FFN 本身不触发 TP collective
- 真正的通信集中在 attention 内部的 sequence-parallel collective

因此 comm-aware 的原则不是“任何算子前都停 prefetch”，而是：

- 保留 `prefetch ↔ compute` 的重叠
- 只在 collective 前调用 `quiesce`
- 避免 `prefetch ↔ comm` 在关键路径资源上相互争用

三类模型在当前实现里的通信位置如下：

| Model | Offload Unit | Attention Impl | Collective Count | 需要 `quiesce` 的位置 |
|---|---|---|---:|---|
| `wanvideo` | `blocks[i]` | `USPAttention` | 8 | `attn1` 与 `attn2` 内部的 all-to-all |
| `hunyuanvideo` double block | `double_blocks[i]` | `UlyssesAttention` | 3 | joint attention 内部的 `all_to_all / all_gather / all_to_all` |
| `hunyuanvideo` single block | `single_blocks[i]` | `UlyssesAttention` | 3 | joint attention 内部的 `all_to_all / all_gather / all_to_all` |
| `flux` double block | `transformer_blocks[i]` | `USPAttention` | 4 | joint attention 内部的 all-to-all |
| `flux` single block | `single_transformer_blocks[i]` | `USPAttention` | 4 | concatenated attention 内部的 all-to-all |

## 3. `wanvideo`

### 3.1 Block 执行顺序

对 text-only 路径，`WanTransformerBlock` 的主路径可按执行顺序写成：

1. `norm1 + modulation`
2. self-attn 的 `to_q / to_k / to_v`
3. `norm_q / norm_k + RoPE`
4. `attn1 = USPAttention`
5. self-attn 的 `to_out + residual norm`
6. text cross-attn 的 `to_q / to_k / to_v`
7. `attn2 = USPAttention`
8. cross-attn 的 `to_out + residual norm`
9. `ffn`

其中真正需要 comm-aware `quiesce` 的只有两次 `USPAttention`：

- `attn1`：4 个 collective
- `attn2`：4 个 collective

### 3.2 计算量

记：

- `d_w`：hidden size
- `f_w`：FFN hidden size

self-attention 分项 FLOPs：

```math
F_{\mathrm{sa,qkv}} = 6 \bar{S}_v d_w^2
```

```math
F_{\mathrm{sa,core}} = \frac{4 S_v^2 d_w}{U}
```

```math
F_{\mathrm{sa,out}} = 2 \bar{S}_v d_w^2
```

cross-attention 分项 FLOPs：

```math
F_{\mathrm{ca,q}} = 2 \bar{S}_v d_w^2
```

```math
F_{\mathrm{ca,kv}} = 4 S_t d_w^2
```

```math
F_{\mathrm{ca,core}} = 4 \bar{S}_v S_t d_w
```

```math
F_{\mathrm{ca,out}} = 2 \bar{S}_v d_w^2
```

FFN：

```math
F_{\mathrm{ffn}} = 4 \bar{S}_v d_w f_w
```

因此一个 Wan block 的总 FLOPs 为：

```math
F_{\mathrm{wan}} =
F_{\mathrm{sa,qkv}} +
F_{\mathrm{sa,core}} +
F_{\mathrm{sa,out}} +
F_{\mathrm{ca,q}} +
F_{\mathrm{ca,kv}} +
F_{\mathrm{ca,core}} +
F_{\mathrm{ca,out}} +
F_{\mathrm{ffn}}
```

```math
T_{\mathrm{comp,wan}} = \frac{F_{\mathrm{wan}}}{P_{\mathrm{eff}}}
```

### 3.3 Prefetch 参数量

主导参数来自：

- self-attn：`to_q / to_k / to_v / to_out`
- cross-attn：`to_q / to_k / to_v / to_out`
- FFN：`fc_in / fc_out`

因此主导 prefetch 量可近似为：

```math
B_{\mathrm{wan}}
\approx
\beta_w \left(8 d_w^2 + 2 d_w f_w\right) + O(d_w)
```

```math
T_{\mathrm{pref,wan}} = \frac{B_{\mathrm{wan}}}{BW_{\mathrm{h2d,eff}}}
```

## 4. `hunyuanvideo`

`HunyuanVideoTransformer3DModel` 的主 denoising trunk 由两类 block 组成：

- `double_blocks[i]`
- `single_blocks[i]`

因此计算和 prefetch 应分别建模。

### 4.1 Double Block

#### 执行顺序

`MMDoubleStreamBlock` 的主路径为：

1. video modulation
2. video `qkv` projection
3. text modulation
4. text `qkv` projection
5. 一次 joint `UlyssesAttention(img_qkv, txt_qkv)`
6. video output projection
7. text output projection
8. video MLP
9. text MLP

`UlyssesAttention` 在这里会触发三类 collective：

1. `ulysses_qkv_all_to_all`
2. `ulysses_replicated_all_gather`
3. `ulysses_output_all_to_all`

#### 计算量

记：

- `d_h`：hidden size
- `f_h`：MLP hidden size

```math
F_{\mathrm{dbl,img\_qkv}} = 6 \bar{S}_v d_h^2
```

```math
F_{\mathrm{dbl,txt\_qkv}} = 6 S_t d_h^2
```

```math
F_{\mathrm{dbl,core}} = \frac{4 (S_v + S_t)^2 d_h}{U}
```

```math
F_{\mathrm{dbl,out}} = 2 (\bar{S}_v + S_t) d_h^2
```

```math
F_{\mathrm{dbl,mlp}} = 4 (\bar{S}_v + S_t) d_h f_h
```

```math
F_{\mathrm{dbl}} =
F_{\mathrm{dbl,img\_qkv}} +
F_{\mathrm{dbl,txt\_qkv}} +
F_{\mathrm{dbl,core}} +
F_{\mathrm{dbl,out}} +
F_{\mathrm{dbl,mlp}}
```

```math
T_{\mathrm{comp,dbl}} = \frac{F_{\mathrm{dbl}}}{P_{\mathrm{eff}}}
```

#### Prefetch 参数量

主导参数来自：

- `img_mod`, `txt_mod`
- `img_attn_qkv`, `txt_attn_qkv`
- `img_attn_proj`, `txt_attn_proj`
- `img_mlp`, `txt_mlp`

因此：

```math
B_{\mathrm{dbl}}
\approx
\beta_h \left(20 d_h^2 + 4 d_h f_h\right) + O(d_h)
```

```math
T_{\mathrm{pref,dbl}} = \frac{B_{\mathrm{dbl}}}{BW_{\mathrm{h2d,eff}}}
```

### 4.2 Single Block

#### 执行顺序

在 `double_blocks` 之后，模型把 video / text token 拼接成一个序列，再进入 `MMSingleStreamBlock`：

1. modulation
2. `linear1`，同时生成 `qkv` 与 MLP branch
3. 一次 joint `UlyssesAttention`
4. `GELU`
5. `linear2`

同样只有这次 `UlyssesAttention` 需要 comm-aware `quiesce`。

#### 计算量

```math
F_{\mathrm{sng,lin1}} = 2 (\bar{S}_v + S_t) d_h (3 d_h + f_h)
```

```math
F_{\mathrm{sng,core}} = \frac{4 (S_v + S_t)^2 d_h}{U}
```

```math
F_{\mathrm{sng,lin2}} = 2 (\bar{S}_v + S_t) (d_h + f_h) d_h
```

```math
F_{\mathrm{sng}} =
F_{\mathrm{sng,lin1}} +
F_{\mathrm{sng,core}} +
F_{\mathrm{sng,lin2}}
```

```math
T_{\mathrm{comp,sng}} = \frac{F_{\mathrm{sng}}}{P_{\mathrm{eff}}}
```

#### Prefetch 参数量

主导参数来自：

- `modulation`
- `linear1`
- `linear2`

```math
B_{\mathrm{sng}}
\approx
\beta_h \left(7 d_h^2 + 2 d_h f_h\right) + O(d_h)
```

```math
T_{\mathrm{pref,sng}} = \frac{B_{\mathrm{sng}}}{BW_{\mathrm{h2d,eff}}}
```

### 4.3 平均层口径

若需要把 Hunyuan 收成单一一条曲线，设：

- `N_d`：double block 个数
- `N_s`：single block 个数

则可定义平均 FLOPs 与平均参数量：

```math
\bar{F}_{\mathrm{hunyuan}} =
\frac{N_d F_{\mathrm{dbl}} + N_s F_{\mathrm{sng}}}{N_d + N_s}
```

```math
\bar{B}_{\mathrm{hunyuan}} =
\frac{N_d B_{\mathrm{dbl}} + N_s B_{\mathrm{sng}}}{N_d + N_s}
```

```math
T_{\mathrm{comp,hunyuan}} = \frac{\bar{F}_{\mathrm{hunyuan}}}{P_{\mathrm{eff}}}, \qquad
T_{\mathrm{pref,hunyuan}} = \frac{\bar{B}_{\mathrm{hunyuan}}}{BW_{\mathrm{h2d,eff}}}
```

## 5. `flux`

`FluxTransformer2DModel` 虽然是 text-to-image，而不是 text-to-video，但它和前两类模型共享同样的问题结构：

- block 内存在 sequence-parallel attention collective
- linears 可以与 prefetch 重叠
- 关键问题仍然是 `T_{\mathrm{comp}}` 与 `T_{\mathrm{pref}}` 的相对大小

`flux` 的 denoising trunk 由两类 block 组成：

- `transformer_blocks[i]`
- `single_transformer_blocks[i]`

它的外层框架与 `hunyuanvideo` 基本一致：

- double block 与 single block 都是 offload unit
- communication 都发生在 block 内部 `USPAttention` 的 collective
- 平均层口径仍然是

```math
T_{\mathrm{comp}} = \frac{\bar{F}}{P_{\mathrm{eff}}}, \qquad
T_{\mathrm{pref}} = \frac{\bar{B}}{BW_{\mathrm{h2d,eff}}}
```

### 5.1 Double Block

#### 执行顺序

`FluxTransformerBlock` 的主路径为：

1. image stream `AdaLayerNormZero`
2. text stream `AdaLayerNormZero`
3. image `qkv` projection
4. text `qkv` projection
5. 一次 joint `FluxAttention`
6. image output projection
7. text output projection
8. image FFN
9. text FFN

`FluxAttention` 内部最终调用的是一次 `USPAttention`，因此在 `U > 1` 时会触发：

- `q` input all-to-all
- `k` input all-to-all
- `v` input all-to-all
- output all-to-all

#### 计算量

记：

- `d_f`：hidden size
- `f_f`：FFN hidden size

由于 `FluxAttention` 会把 text token 与本地 image token 先拼接，再送入 `USPAttention`，这里定义一个有效本地序列长度：

```math
\tilde{S}_x = \bar{S}_x + S_t
```

于是：

```math
F_{\mathrm{flux,dbl,img\_qkv}} = 6 \bar{S}_x d_f^2
```

```math
F_{\mathrm{flux,dbl,txt\_qkv}} = 6 S_t d_f^2
```

```math
F_{\mathrm{flux,dbl,core}} \approx 4 \tilde{S}_x^2 d_f
```

```math
F_{\mathrm{flux,dbl,out}} = 2 (\bar{S}_x + S_t) d_f^2
```

```math
F_{\mathrm{flux,dbl,ffn}} = 4 (\bar{S}_x + S_t) d_f f_f
```

```math
F_{\mathrm{flux,dbl}} =
F_{\mathrm{flux,dbl,img\_qkv}} +
F_{\mathrm{flux,dbl,txt\_qkv}} +
F_{\mathrm{flux,dbl,core}} +
F_{\mathrm{flux,dbl,out}} +
F_{\mathrm{flux,dbl,ffn}}
```

```math
T_{\mathrm{comp,flux\_dbl}} = \frac{F_{\mathrm{flux,dbl}}}{P_{\mathrm{eff}}}
```

#### Prefetch 参数量

主导参数来自：

- image / text 两路 `AdaLayerNormZero`
- image / text 两路 attention projections
- image / text 两路 FFN

因此可近似为：

```math
B_{\mathrm{flux,dbl}}
\approx
\beta_f \left(20 d_f^2 + 4 d_f f_f\right) + O(d_f)
```

```math
T_{\mathrm{pref,flux\_dbl}} = \frac{B_{\mathrm{flux,dbl}}}{BW_{\mathrm{h2d,eff}}}
```

### 5.2 Single Block

#### 执行顺序

`FluxSingleTransformerBlock` 的主路径为：

1. 将 text / image token 先拼接成一个序列
2. `AdaLayerNormZeroSingle`
3. `proj_mlp`
4. 一次 attention
5. `GELU`
6. `proj_out`

这一类 block 里同样只有 attention 内部的 `USPAttention` 需要 comm-aware `quiesce`。

#### 计算量

单块使用的序列长度同样记为：

```math
\tilde{S}_x = \bar{S}_x + S_t
```

于是：

```math
F_{\mathrm{flux,sng,proj\_mlp}} = 2 \tilde{S}_x d_f f_f
```

```math
F_{\mathrm{flux,sng,attn\_qkv}} = 6 \tilde{S}_x d_f^2
```

```math
F_{\mathrm{flux,sng,core}} \approx 4 \tilde{S}_x^2 d_f
```

```math
F_{\mathrm{flux,sng,proj\_out}} = 2 \tilde{S}_x (d_f + f_f) d_f
```

```math
F_{\mathrm{flux,sng}} =
F_{\mathrm{flux,sng,proj\_mlp}} +
F_{\mathrm{flux,sng,attn\_qkv}} +
F_{\mathrm{flux,sng,core}} +
F_{\mathrm{flux,sng,proj\_out}}
```

```math
T_{\mathrm{comp,flux\_sng}} = \frac{F_{\mathrm{flux,sng}}}{P_{\mathrm{eff}}}
```

#### Prefetch 参数量

主导参数来自：

- `AdaLayerNormZeroSingle`
- `proj_mlp`
- attention projections
- `proj_out`

因此可近似为：

```math
B_{\mathrm{flux,sng}}
\approx
\beta_f \left(7 d_f^2 + 2 d_f f_f\right) + O(d_f)
```

```math
T_{\mathrm{pref,flux\_sng}} = \frac{B_{\mathrm{flux,sng}}}{BW_{\mathrm{h2d,eff}}}
```

### 5.3 平均层口径

设：

- `N_d`：`transformer_blocks` 个数
- `N_s`：`single_transformer_blocks` 个数

则平均 FLOPs 与平均参数量可写成：

```math
\bar{F}_{\mathrm{flux}} =
\frac{N_d F_{\mathrm{flux,dbl}} + N_s F_{\mathrm{flux,sng}}}{N_d + N_s}
```

```math
\bar{B}_{\mathrm{flux}} =
\frac{N_d B_{\mathrm{flux,dbl}} + N_s B_{\mathrm{flux,sng}}}{N_d + N_s}
```

```math
T_{\mathrm{comp,flux}} = \frac{\bar{F}_{\mathrm{flux}}}{P_{\mathrm{eff}}}, \qquad
T_{\mathrm{pref,flux}} = \frac{\bar{B}_{\mathrm{flux}}}{BW_{\mathrm{h2d,eff}}}
```

## 6. 默认 Setting 下的 Critical Parameter（2xH100）

这一节继续使用 plain-text 公式，避免 Markdown 预览器不支持 LaTeX 时显示成原始源码。

这里把 `critical size` 直接改写成 profiling 真正会 sweep 的参数：

- `wanvideo` / `hunyuanvideo`：对应 `num_frames`
- `flux`：对应 `batch size`，也就是 `run_profile.sh` 里的 `NUM_OUTPUTS_PER_PROMPT`

同时不再引入 `kappa` 这类中间缩写，最终公式统一直接写成

`常数 * (eta_comp / eta_h2d)`。

统一采用：

- `run_profile.sh` 默认输入：
  - `wanvideo` / `hunyuanvideo`：`height=720`, `width=1280`
  - `flux`：`height=1024`, `width=1024`
- `NUM_GPUS=2`, `ULYSSES_DEGREE=2`
- `layerwise_offload_chunkwise.py` 默认口径：不保留 resident phase，按整层参数量估算 `B_layer`
- 参数 dtype 取 `bf16`，因此 `beta = 2` bytes
- 2xH100 峰值口径：`P_peak = 756 TFLOP/s`, `BW_peak = 63 GB/s`

因此

```text
P_peak / BW_peak = 12000 FLOP/byte
```

临界条件统一写成

```text
F_layer(critical) = 12000 * B_layer * (eta_comp / eta_h2d)
```

### 6.1 `wanvideo`：critical `num_frames`

这里把 `wanvideo` 校准到你实际使用的

`Wan-AI/Wan2.2-TI2V-5B-Diffusers`

而不是旧的 A14B / 14B 口径。

说明：

- `offload_profiling/model_profile_common.sh` 里的 `wanvideo` profile alias 当前默认路径仍然指向 `Wan2.2-T2V-A14B-Diffusers`
- 这一节的系数与临界公式已经按你指定的 `Wan2.2-TI2V-5B-Diffusers` 重新校准

- HF `transformer/config.json`：`num_attention_heads=24`, `attention_head_dim=128`, `ffn_dim=14336`, `num_layers=30`
- HF `vae/config.json`：`scale_factor_spatial=16`, `scale_factor_temporal=4`
- 这里仍按 `run_profile.sh` 默认输入口径取 `height=720`, `width=1280`
- `S_t = 512`

因此当 `num_frames = n` 且 `n mod 4 = 1` 时，本地 runtime 对应的 video token 长度是：

```text
S_v(n) = ((n - 1) / 4 + 1) * (720 / 16) * (1280 / 16) / (2 * 2)
       = 225 * (n + 3)
```

模型维度为：

```text
d_w = 24 * 128 = 3072
f_w = 14336
```

主导 prefetch 量为：

```text
B_wan = 327,155,712 bytes
```

把 `S_v(n)` 代入 3.2 节 FLOPs 公式，可得：

```text
F_wan(n) = 24576 * [225 * (n + 3)]^2
         + 239075328 * [225 * (n + 3)]
         + 19327352832
```

因此临界 `num_frames` `n*_wan` 满足：

```text
F_wan(n*_wan) = 3925868544000 * (eta_comp / eta_h2d)
```

等价展开后：

```text
1244160000 * (n*_wan)^2
+ 61256908800 * n*_wan
+ 191900639232
= 3925868544000 * (eta_comp / eta_h2d)
```

这里看起来数字很大，是因为上式两边的单位都是 FLOPs，不是帧数。

真正的 `critical num_frames` 需要解这个关于 `n` 的二次方程。

例如如果先取 `eta_comp / eta_h2d = 1`，则：

```text
n*_wan ~= 35.44
```

考虑到合法帧数需要满足 `n mod 4 = 1`，可近似记成：

```text
n*_wan ~= 37 frames
```

说明：

- 官方 Wan2.2 TI2V README 的示例输入是 `1280x704`，不是 `1280x720`
- 这里仍保留 `run_profile.sh` 默认 `720x1280` 口径；如果改成 README 示例分辨率，只需把 `225 * (n + 3)` 换成 `220 * (n + 3)`

### 6.2 `hunyuanvideo`：critical `num_frames`

这里对应你实际使用的

`tencent/HunyuanVideo`

在默认 `720p` 下：

```text
S_v(n) = ((n - 1) / 4 + 1) * (720 / 8) * (1280 / 8) / (2 * 2)
       = 900 * (n + 3)
```

对 `hunyuanvideo` 默认模型配置：

- `d_h = 24 * 128 = 3072`
- `f_h = 4 * d_h = 12288`
- `S_t = 161`
  - 这里取的是 Hunyuan pipeline 实际送入 DiT 的 text 长度：`256 - crop_start(95)`
- `N_d = 20`
- `N_s = 40`

按 average-layer 口径：

```text
B_hunyuan_avg = 415,236,096 bytes
```

把 4.3 节平均层公式在默认 setting 下整理后，可写成：

```text
F_hunyuan_avg(n) = 6144 * [900 * (n + 3)]^2
                 + 115224576 * [900 * (n + 3)]
                 + 36624537600
```

因此临界 `num_frames` `n*_hunyuan` 满足：

```text
F_hunyuan_avg(n*_hunyuan) = 4982833152000 * (eta_comp / eta_h2d)
```

等价展开后：

```text
4976640000 * (n*_hunyuan)^2
+ 133561958400 * n*_hunyuan
+ 392520652800
= 4982833152000 * (eta_comp / eta_h2d)
```

同样地，这里右边的大数也是 FLOPs 量级，不是帧数本身。

如果先取 `eta_comp / eta_h2d = 1`，则：

```text
n*_hunyuan ~= 19.78
```

按合法帧数 `n mod 4 = 1` 取整后，可近似记成：

```text
n*_hunyuan ~= 21 frames
```

### 6.3 `flux`：critical `batch size`

这里对应你实际使用的

`black-forest-labs/FLUX.1-dev`

对 `flux`，这里固定默认分辨率 `1024 x 1024`，只把 `batch size` 当成 sweep 变量。

在这个 setting 下：

- VAE scale factor = `8`
- 送入 DiT 前还会做一次 `2x2` latent packing
- 因此 image token 长度固定为：

```text
S_x = (1024 / 16)^2 = 4096
```

默认模型配置：

- `d_f = 24 * 128 = 3072`
- `f_f = 12288`
- `S_t = 512`
- `N_d = 19`
- `N_s = 38`

average-layer 口径下：

```text
B_flux_avg = 415,236,096 bytes
F_flux_avg_per_sample = 660,351,221,760
```

因此当 batch size 为 `b` 时：

```text
T_comp_flux(b) = b * F_flux_avg_per_sample / (eta_comp * P_peak)
```

临界 batch size `b*_flux` 满足：

```text
b*_flux = (12000 * B_flux_avg / F_flux_avg_per_sample) * (eta_comp / eta_h2d)
        = (12000 * 415236096 / 660351221760) * (eta_comp / eta_h2d)
```

整理后：

```text
b*_flux ~= 7.546 * (eta_comp / eta_h2d)
```

也就是说，在本文默认 `1024 x 1024` setting 下，`flux` 的 critical parameter 更适合记成 `critical batch size`，而不是 `critical image token length`。

## 7. 建模上的直接含义

把三类模型放在一起看，critical size 的解释仍然是统一的，只是映射到 profiling 参数时要区分：

- 对 `wanvideo` 与 `hunyuanvideo`，在本文默认分辨率固定时，critical size 直接对应 `num_frames`。
- 对 `flux`，在本文默认分辨率固定时，critical size 直接对应 `batch size`；如果以后固定 batch size 去 sweep 分辨率，那么临界变量应再切回 `S_x`。
- 对所有模型，只要 `tp_size = 1` 仍成立，comm-aware 的重点都应放在 attention 内部 collective，而不是 linear / FFN。

因此，论文里的统一分析框架可以概括为：

1. 先按模型 block 结构拆分 `F_{\mathrm{layer}}`。
2. 再按 offload unit 的参数集合估算 `B_{\mathrm{layer}}`。
3. 最后先用 `T_comp(S*) = T_pref`

求出临界序列规模 `S*`，再把它映射回当前 profiling 真正 sweep 的参数（这里是 `num_frames` 或 `batch size`），并用

`T_exposed ~= max(T_pref - T_comp, 0)`

解释何时 offload 会开始直接暴露在关键路径上。
