#!/bin/bash
#SBATCH --job-name=gsm8k_train_deepseek
#SBATCH -o ./slurm/%x/job_%A.out # STDOUT
#SBATCH -p HGPU
#SBATCH --gres=gpu:H200:1    # Request N GPUs per machine
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 8
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/efficient-reasoning

local_dir=.local_daman
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif \
python get_responses.py \
    --model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset=openai/gsm8k \
    --scale=1.5B \
    --split=train