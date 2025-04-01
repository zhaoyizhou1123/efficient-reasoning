split=test
for t in 0.6 0.4 0.8 0.2; do
    for p in generator4; do
        python get_responses.py --model_path="Qwen/Qwen2.5-Math-1.5B" --dataset="MATH" --tok_limit=4096 --test_n=10 --save_freq 50 --temperature $t --split ${split} \
        --prompt_type $p
    done
done

# split=train
# python get_responses.py --model_path="Qwen/Qwen2.5-Math-1.5B" --dataset="MATH" --tok_limit=4096 --test_n=10 --save_freq 50 --temp 0.6 --split ${split}