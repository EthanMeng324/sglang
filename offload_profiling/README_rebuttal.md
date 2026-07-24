# run_rebuttal.sh — 单机 rebuttal 基准脚本使用说明

替代 slurm 时代的 `run_access*.sh` 链条：一条命令 = 在一个 (SP degree, 帧数) 配置点上跑完论文
Figure-3 的模式矩阵（No-Offload / Layerwise / ChunkFlow），每个 mode 由 nsys 包裹，跑完自动分析。
产物命名与既有 `analyze_nsys.sh` / `summarize_profile_matrix.py` 管线完全兼容。目前仅支持 WanVideo
(Wan2.2-TI2V-5B-Diffusers)，论文设置全部内置：704×1280、10 步、guidance 4.0/3.0、batch 1、
chunk 16 MB、comm window = launch、prompt "A cat walks on the grass, realistic"。

## 前置条件

1. **Python 环境**：仓库根目录下的 `.venv`（脚本会自动 `source .venv/bin/activate`）
   ```bash
   uv venv .venv --python 3.12 --seed
   uv pip install -e "python[diffusion]"        # torch 2.9.1 cu128；Blackwell sm_120 原生支持
   ```
2. **模型**（HF 仓库共 34 GB；磁盘小的机器放 /dev/shm）：
   ```bash
   HF_HOME=/dev/shm/hf hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --exclude "assets/*" --exclude "examples/*"
   ```
3. `nsys` 在 PATH 上（分析步骤依赖 nsys export）。

## 用法

```bash
bash offload_profiling/run_rebuttal.sh <SP> <FRAMES> [MODES]
```

- `SP`：GPU 数 = Ulysses SP degree（2 或 4；SP=2 默认用 GPU 0,1）
- `FRAMES`：帧数，Wan 要求 4k+1（41 / 81 / 121 / 161 / 201 …）
- `MODES`：逗号分隔，默认 `warmup,no,old,new,analyze`

例子：

```bash
bash offload_profiling/run_rebuttal.sh 2 41                    # SP=2, 41 帧, 全矩阵
bash offload_profiling/run_rebuttal.sh 4 161                   # SP=4, 161 帧
bash offload_profiling/run_rebuttal.sh 2 81 no,analyze         # 只补跑 no-offload + 重新分析
bash offload_profiling/run_rebuttal.sh 2 241 warmup,new,no,analyze   # OOM 探针：warmup 用 new 配置编译，
                                                               # no 若 OOM 不影响其余 mode，日志留证据
TAG=r2 bash offload_profiling/run_rebuttal.sh 2 41             # 重复实验，结果存到独立目录 sp2_f41_r2
```

## 模式说明

| mode | 含义 |
|---|---|
| `warmup` | 用 MODES 里第一个 benchmark 模式的配置跑一遍（无 nsys），暖 torch.compile 缓存 |
| `no` | No Offload：`--dit-layerwise-offload false`，权重全驻留 |
| `old` | Layerwise（SGLang 整层 prefetch 基线）：offload 开、`SGLANG_DIT_COMM_AWARE_OFFLOAD=0` |
| `new` | ChunkFlow：`SGLANG_DIT_COMM_AWARE_OFFLOAD=1` + launch window + 16 MB chunk |
| `ratio` | ChunkFlow + partial residency（`RESIDENT_RATIO`，默认 0.4；不在默认 MODES 里） |
| `analyze` | nsys 导出 + 汇总 step time / peak memory 成 markdown + csv |

某个 mode 失败（例如刻意的 no-offload OOM 探针）**不会中断后续 mode**：该 mode 的半截产物会被清
掉以免污染分析，OOM traceback 留在 `logs/<mode>.log` 里作为证据，结尾按 mode 汇报 exit code。
ChunkFlow 的开关是 **env-only**（`SGLANG_DIT_COMM_AWARE_OFFLOAD` 在 import 时读取，无对应 CLI
flag），脚本已内置，手工复现单条命令时务必带上。

## 可覆盖的环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `TAG` | 空 | 结果目录后缀（重复实验用） |
| `RESIDENT_RATIO` | 0.4 | ratio 模式的驻留比例 |
| `NSYS` | 1 | 置 0 跳过 nsys 包裹（更快，但无法 analyze） |
| `ATTENTION_BACKEND` | `fa` | 论文跑法未用 sageattention；sm_120 上 `fa` 自动落到 Torch SDPA |
| `CUDA_VISIBLE_DEVICES` | SP=2 时 `0,1` | GPU 选择 |
| `SGLANG_WAN_VAE_TILING` | `0` | **0 = 论文原始配置（untiled VAE decode）**。置 1 启用 tiled decode，仅作诊断用途（会整体压低 peak memory 的 decode 段贡献，latency 不受影响） |

## 输出

```
/dev/shm/chunkflow_results/sp<SP>_f<FRAMES>[_TAG]/      # nsys 大 trace + 日志（tmpfs，重启即失）
offload_profiling/results_5090/sp<SP>_f<FRAMES>[_TAG]/  # json/md/csv/log 自动拷回（持久化）
```

关键产物（在 `profiles/wanvideo/` 下）：

- `analysis/analysis_summary.md` — step time 表 + peak memory 表 + per-step comm/prefetch 分解
- `analysis/profile_matrix.csv` — 机器可读
- `perf_<mode>_profiled.json` — 原始 perf dump（含 per-stage `memory_checkpoints`）

口径说明：
- **Step time** 取 NVTX `SGL_DENOISING_STEP_i` 的 steady 平均（去 step 0，若检测到 switch-step
  outlier 也去掉）；
- **Peak memory** 取 perf JSON 的 `mem_analysis` checkpoint（torch allocator 峰值，**累计值、含
  VAE decode 段**；per-stage 数值在 `memory_checkpoints` 里可分段查看）。

## 本机（Vast.ai 4×RTX 5090 32GB）已踩过的坑

脚本已内置处理，移植到别的机器时留意：

- 基础镜像 ambient env 有 `HF_HOME=/workspace/.hf_home`，而磁盘只有 32 GB —— 脚本无条件覆写
  `HF_HOME=/dev/shm/hf`，否则会重复下载 34 GB 模型把盘塞爆；
- nsys 中间文件很大 → `TMPDIR=/dev/shm/nsys-tmp`；但 `/dev/shm` 是 **noexec**，inductor/triton
  编译出的 .so 必须留在磁盘（`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` 指到仓库 `.cache/`）；
- 32 GB 卡上 untiled no-offload 在大帧数会 OOM —— 这本身是 rebuttal 要的结果，不是故障。

## 本次配套的代码改动（相对 1e58a0348）

- `offload_profiling/run_rebuttal.sh`：新增（本脚本）；
- `offload_profiling/analyze_nsys.sh`：`module load` 行加 guard（非 slurm 机器直接跳过）；
  summarize 传 per-config `--export-dir`，修复 sqlite 导出缓存跨配置同名污染（补跑旧配置时会
  错用新配置的缓存）；
- `offload_profiling/run_access*.sh`：`module load` 行加 guard；
- `runtime/managers/gpu_worker.py`：OutputBatch 回传前把输出张量搬到 CPU —— 反序列化 CUDA 张量
  会在接收进程按原 device 分配显存，32 GB 卡上会在**生成成功之后**的结果传输阶段 OOM，导致
  perf JSON 丢失（测量区间之外的 harness 修复，不影响任何指标）；
- `runtime/entrypoints/utils.py`：`post_process_sample` 改在 CPU 上做像素后处理（同上原因），并
  兼容 tiled VAE 返回的 5D 张量；
- `configs/models/vaes/wanvae.py`：新增 `SGLANG_WAN_VAE_TILING` 开关（默认 0，行为与原始完全一致）。

实验数据（`results_5090/`、`results/`）不进 git。Claude 会话完整记录见
`offload_profiling/rebuttal_notes/claude_history/`（主转录 jsonl + workflow 子代理转录 + memory 笔记）。
