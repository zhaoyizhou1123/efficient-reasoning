path=outputs/openai_gsm8k_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json
# path=outputs/test.json

python evaluate_response.py \
    --model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --response_path=$path \
    --task verify \
    --scale=1.5B