bash offload_profiling/run_access.sh
bash offload_profiling/run_access_old.sh
bash offload_profiling/run_access_no.sh
module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
module load GCC/12.3.0
bash offload_profiling/analyze_nsys.sh real
