#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/vllm_server-%j.out


model_name=$1
port=$2

eval "$(conda shell.bash hook)"
conda activate vllm

vllm serve $model_name \
    --host $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --port $port \
    --disable-frontend-multiprocessing \
    --gpu-memory-utilization 0.95 \
    --served-model-name llm