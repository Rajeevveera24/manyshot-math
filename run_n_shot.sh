#!/bin/bash
#SBATCH --job-name=inf_math_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/math_eval-%j.out


eval "$(conda shell.bash hook)"
conda activate manyshot-math 

cd src

# Add cli arguments - example usage:
# --n_samples 3 --generate --generate_cot : to generate CoT synthetic data with 3 samples
# --n_samples 3 --predict --pred_use_cot : to predict using CoT synthetic data with 3 shot
# --n_samples 3 --predict --pred_use_cot --supervised : to predict using supervised CoT synthetic data with 3 shot
srun python generate_and_n_shot.py "$@"
