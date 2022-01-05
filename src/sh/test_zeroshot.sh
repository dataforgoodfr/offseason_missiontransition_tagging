#!/bin/bash
#SBATCH --job-name=test-zeroshot
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/test-zeroshot-%j.out
#SBATCH --error=slurm_out/test-zeroshot-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}


srun python -u src/test_zeroshot.py -thr 0.7
srun python -u src/test_zeroshot.py -thr 0.8
srun python -u src/test_zeroshot.py -thr 0.9