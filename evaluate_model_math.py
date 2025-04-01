# Credit: Guanning https://github.com/guanning03/reasoning-verifier/blob/master/evaluate_generator.py

import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

os.makedirs('evaluations', exist_ok=True)

# Examples 
'''
CUDA_VISIBLE_DEVICES=1 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/gsm8k" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_gsm8k_8shot.txt" --post_truncate
'''
'''
CUDA_VISIBLE_DEVICES=2 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/competition_math" --tok_limit=4096 --split=train --test_n=1 --template="templates/Qwen_MATH_4shot.txt" --post_truncate
'''
'''
CUDA_VISIBLE_DEVICES=3 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/MATH-500" --tok_limit=4096 --split=test --test_n=1 --template="templates/Qwen_MATH_4shot.txt" --post_truncate
'''
'''
CUDA_VISIBLE_DEVICES=7 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/MATH-500" --tok_limit=4096 --split=test --test_n=256 --template="templates/Qwen_MATH_0shot.txt"
CUDA_VISIBLE_DEVICES=6 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/MATH-500" --tok_limit=4095 --split=test --test_n=256 --template="templates/Qwen_MATH_0shot.txt"
CUDA_VISIBLE_DEVICES=5 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/MATH-500" --tok_limit=4094 --split=test --test_n=256 --template="templates/Qwen_MATH_0shot.txt"
CUDA_VISIBLE_DEVICES=4 python evaluate_generator.py --model_path="./models/Qwen2.5-Math-1.5B" --dataset="./benchmarks/MATH-500" --tok_limit=4093 --split=test --test_n=256 --template="templates/Qwen_MATH_0shot.txt"
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')
parser.add_argument('--dataset', type=str, default='openai/gsm8k')
parser.add_argument('--tok_limit', type=int, default=4096)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--temperature', type=float, default=None)
parser.add_argument('--test_n', type=int, default=None)
parser.add_argument('--template', type=str, default='templates/Qwen_gsm8k_8shot.txt')
parser.add_argument('--post_truncate', action='store_true', default=False)
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

dataset_name = args.dataset
model_path = args.model_path
tok_limit = args.tok_limit
split = args.split
dataset_name = args.dataset
dataset_short_name = dataset_name.split('/')[-1]
template = args.template
template_short_name = template.split('/')[-1].split('.')[0]
results = {}

# with open(template, 'r', encoding='utf-8') as f:
#     template = f.read()

with open("prompt.json", 'r') as f:
    prompt_list = json.load(f)
for p in prompt_list:
    if p['name'] == "generator":
        template = p['prompt_template']
        break

print("Dataset:", dataset_short_name, "\nModel:", model_path)

QUESTION_KEY = DATASET_KEYS[dataset_short_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_short_name]["answer"]
eq = RESPONSE_COMPARATOR[dataset_short_name]

dataset = load_dataset(dataset_name, split=split)
TEST_N = 3
MAX_TOKENS = tok_limit
TEST_TEMPERATURE = 0.6
MAX_TEST_SAMPLES = 500


# if dataset_short_name == 'converted_aime_dataset':
#     dataset = load_from_disk(dataset_name)
#     TEST_N = 10
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 100
# elif dataset_short_name in ['MATH500', 'MATH-500']:
#     dataset = load_from_disk(dataset_name)
#     TEST_N = 3
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 500
# elif dataset_short_name == 'gsm8k':
#     dataset = load_from_disk(dataset_name, 'main')
#     TEST_N = 1
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.0
#     MAX_TEST_SAMPLES = 7500
# elif dataset_short_name == 'AIME_2024':
#     dataset = load_from_disk(dataset_name)
#     print("\nDataset columns:", dataset['train'].column_names) 
#     TEST_N = 5
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 30
# elif dataset_short_name == 'AIME2025':
#     dataset = load_from_disk(dataset_name)
#     print("\nDataset columns:", dataset['train'].column_names)  
#     TEST_N = 5
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 15
# elif dataset_short_name == 'competition_math':
#     dataset = load_from_disk(dataset_name) 
#     TEST_N = 1
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.4
#     MAX_TEST_SAMPLES = 100
    
print("Available splits in dataset:", dataset.keys()) 
print("Available keys in dataset:", dataset[split].column_names)

# override the default values
if args.temperature is not None:
    TEST_TEMPERATURE = float(args.temperature)
if args.test_n is not None:
    TEST_N = int(args.test_n)
    
def post_truncate(response):
    response = response.split("<|endoftext|>")[0]
    # response = response.split("\n\n\n")[0]
    # response = response.split("\n\n")[0]
    response = response.split("Question:")[0]
    response = response.split("Problem:")[0]
    return response

def get_scores(ds, outputs, tokenizer_encode, save_file_name=None):
    predictions, golds = [], []
    results = []
    for input, output in tqdm(zip(ds, outputs), total=len(ds), desc='Analysed responses'):
        gold = RESPONSE_EXTRACTOR[dataset_short_name](input[ANSWER_KEY])
        if args.post_truncate:
            truncated_responses = [post_truncate(resp.text) for resp in output.outputs]
        else:
            truncated_responses = [resp.text for resp in output.outputs]
        prediction = [
            RESPONSE_EXTRACTOR[dataset_short_name](truncated_resp)
            for truncated_resp in truncated_responses
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                QUESTION_KEY: input[QUESTION_KEY],
                ANSWER_KEY: input[ANSWER_KEY],
                "responses": truncated_responses,  
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(tokenizer_encode(truncated_resp)) for truncated_resp in truncated_responses]) / len(truncated_responses),
                "accuracy": [eq(gold, pred) for pred in prediction],
            }
        )
    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)
    
    return {} # TODO: add various metrics when required

def evaluate_model():
    test_prompts = []
    model = LLM(model_path, tokenizer=model_path, gpu_memory_utilization=0.9, 
                tensor_parallel_size=1, max_model_len = MAX_TOKENS, swap_space=80)    
    
    test_ds = dataset[split].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset[split]))))
    
    for x in test_ds:
        prompt = template.replace('<question>', x[QUESTION_KEY])
        prompt_tokens = model.llm_engine.tokenizer.tokenizer.encode(prompt)
        test_prompts.append(prompt_tokens)
    
    sampling_params = SamplingParams(
        temperature=TEST_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=TEST_N
    )

    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    start_time = time.time()
    test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
    test_scores = get_scores(test_ds, 
                             test_outputs, 
                             model.llm_engine.tokenizer.tokenizer.encode,
                             f"evaluations/outputs_{dataset_short_name}_{model_path.split('/')[-1]}_{template_short_name}_{TEST_TEMPERATURE}_{tok_limit}.json")
    print("Test:", test_scores)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}

print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model()
results[model_path] = scores

with open(f'evaluations/results_{dataset_short_name}_{model_path.split("/")[-1]}_{template_short_name}_{TEST_TEMPERATURE}_{tok_limit}.json', 'w') as f:
    json.dump(results, f, indent=4)