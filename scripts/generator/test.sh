split=test

python debug/test_tokenizer.py --model_path="Qwen/Qwen2.5-Math-1.5B" --dataset="MATH" --tok_limit=4096 --test_n=1 --save_freq 250 --split ${split}