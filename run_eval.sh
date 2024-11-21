#!/bin/bash
#SBATCH --job-name=math_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/math_eval-%j.out


eval "$(conda shell.bash hook)"
conda activate manyshot-math 

cd py_scripts

python eval_math.py \
    --model_name $4 \
    --hostname $5 \
    --port $6 