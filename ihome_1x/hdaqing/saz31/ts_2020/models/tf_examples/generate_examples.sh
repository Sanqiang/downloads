#!/usr/bin/env bash

#SBATCH --cluster=smp
#SBATCH --job-name=generate_examples
#SBATCH --output=log/generate_examples_%A_%a.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore
export PYTHONPATH="${PYTHONPATH}:/ihome/hdaqing/saz31/ts_2020"
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
# Run the job
export PYTHONHASHSEED=0
srun python generate_examples.py --cur_thread $SLURM_ARRAY_TASK_ID --num_thread 512


