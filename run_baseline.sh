#!/bin/bash
#SBATCH --job-name=math_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/math_eval-%j.out


eval "$(conda shell.bash hook)"
conda activate manyshot-math 

cd src

python baseline.py
