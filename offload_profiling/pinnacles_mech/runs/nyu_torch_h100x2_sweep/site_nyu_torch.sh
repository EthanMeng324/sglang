# NYU Torch local equivalents for the runbook's "adjust for this cluster" items (recorded in REPORT).
export WS=/scratch/yx2432/rebuttal_mn
export CUDA_HOME=$WS/cuda-12.9                       # copied from /share/apps/images/cuda-12.9.1-cudnn-9.10.2-ubuntu-24.04.2.sif
export PATH=$CUDA_HOME/bin:/scratch/yx2432/chunkflow/nsight/Nsight_Systems_2025.3/bin:$PATH
export CC=gcc CXX=g++
export HF_HOME=$WS/hf HF_HUB_DISABLE_TELEMETRY=1      # models are local dirs: $WS/models/<name> -> HF snapshot
export TORCHINDUCTOR_CACHE_DIR=$WS/.cache/torchinductor TRITON_CACHE_DIR=$WS/.cache/triton   # shared (VAST NFS)
export TMPDIR=/dev/shm/$USER-tmp; mkdir -p $TMPDIR
