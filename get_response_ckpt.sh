#!/bin/bash
#SBATCH --job-name=math_1.5B_ckpt
#SBATCH -o ./slurm/%x/job_%A.out # STDOUT
# SBATCH -p HGPU
#SBATCH --gres=gpu:H100:1    # Request N GPUs per machine
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 2
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/efficient-reasoning
#SBATCH --time 5:00:00

# model=Qwen/Qwen2.5-0.5B-Instruct
# model=./checkpoints/verl_qwen_1.5B_math_rloo
model=Qwen/Qwen2.5-MATH-1.5B
# model=zzy1123/qwen_0.5B_rloo
main=get_responses_ckpt.py
# main=get_responses.py
# data=openai/gsm8k
data=MATH
local_dir=.local_daman
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif \
python -u $main \
    --model_path=$model \
    --base_model_path=$model \
    --dataset=$data \
    --split=test \
    --test_n=32 \
    --save_freq=512