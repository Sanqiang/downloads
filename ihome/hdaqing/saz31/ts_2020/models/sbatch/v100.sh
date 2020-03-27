#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --job-name=pretrain
#SBATCH --output=pretrain.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --mem=32g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"
wandb login 4bc424c09cbfe38419de3532e74935ed7f257124

# Run the job
srun python ../utils/pretrain.py