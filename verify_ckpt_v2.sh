#!/bin/bash
#SBATCH --job-name=verify_ckpt
#SBATCH -o ./slurm/%x/job_%A.out # STDOUT
#SBATCH -p HGPU
#SBATCH --gres=gpu:H200:1    # Request N GPUs per machine
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --chdir /home/zhaoyiz/projects/reasoning/efficient-reasoning


# path=outputs/openai_gsm8k_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json
path=outputs/test.json

local_dir=.local_daman
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif \
python evaluate_response_ckpt_v2.py \
    --model_path="/home/zhaoyiz/projects/reasoning/reasoning_verifier/checkpoint/math_qwen2.5_math_1.5b/checkpoint-112" \
    --response_path=$path \
    --task verify \
    --scale=1.5B \
    --result_name "math500-verify-rm-ckpt"
# local_dir=.local_daman
# apptainer exec --nv -B $HOME/$local_dir:$HOME/.local --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_23.11-py3.sif \
# python evaluate_response_ckpt.py \
#     --model_path="/home/zhaoyiz/projects/reasoning/verl/checkpoints/verl/deepseek-1.5b_gsm8k/global_step_105/actor/huggingface" \
#     --response_path=$path \
#     --task verify \
#     --scale=1.5B \
#     --result_name "gsm8k_verify_deepseek-1.5b_deepseek-1.5b"