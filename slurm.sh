#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -o ./slurm/%x/job_%A.out # STDOUT
# SBATCH -p HGPU
#SBATCH --gres=gpu:H100:1    # Request N GPUs per machine
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/efficient-reasoning

# script=examples/ppo_trainer/verify_qwen2.5-math-1.5b_gsm8k.sh
# script=examples/ppo_trainer/verify_deepseek-1.5b_gsm8k.sh
script=scripts/generator/math.sh


local_dir=.local_daman
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif /bin/bash $script
