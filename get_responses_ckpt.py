import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
from pathlib import Path


# This script evaluates a model on a dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--base_model_path', type=str, default='')
parser.add_argument('--dataset', type=str)
# parser.add_argument('--scale', type=str, default='1.5B')
parser.add_argument('--tok_limit', type=int, default=32768)
parser.add_argument('--split', type=str, default='test', help="train or test")
parser.add_argument('--temperature', type=float, default=None)
parser.add_argument('--test_n', type=int, default=None)
parser.add_argument('--post_truncate', action='store_true')
parser.add_argument('--save_freq', type=int, default=-1)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--prompt_type', type=str)
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

dataset_name = args.dataset
model_path = args.model_path
# scale = args.scale
tok_limit = args.tok_limit
dataset_name = args.dataset
results = {}

# print("Dataset:", dataset_name, "\nScale:", scale)

QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
eq = RESPONSE_COMPARATOR[dataset_name]

if dataset_name == 'datasets/converted_aime_dataset':
    dataset = load_from_disk(dataset_name)
    TEST_N = 10
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 100
elif dataset_name == 'di-zhang-fdu/MATH500':
    dataset = load_dataset(dataset_name)
    TEST_N = 3
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 500
elif dataset_name == 'openai/gsm8k':
    dataset = load_dataset(dataset_name, 'main')
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 1319
elif dataset_name == 'MATH':
    dataset = load_dataset("zzy1123/MATH_train_test_split")
    # dataset = load_from_disk("/home/zhaoyiz/projects/reasoning/guanning/reasoning-verifier/benchmarks/MATH-500")
    TEST_N = 2
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 500
else:
    raise NotImplementedError

with open("prompt.json", 'r') as f:
    prompt_list = json.load(f)
for p in prompt_list:
    if p['name'] == args.prompt_type:
        prompt_template = p['prompt_template']
        break

if args.temperature is not None:
    TEST_TEMPERATURE = float(args.temperature)
if args.test_n is not None:
    TEST_N = int(args.test_n)

print(TEST_N, MAX_TOKENS, TEST_TEMPERATURE)

def post_truncate(response):
    response = response.split("<|endoftext|>")[0]
    # response = response.split("\n\n\n")[0]
    # response = response.split("\n\n")[0]
    response = response.split("Question:")[0]
    response = response.split("Problem:")[0]
    return response

def process_outputs(ds, outputs, save_file_name):
    assert save_file_name is not None

    predictions, golds = [], []
    results = []
    for model_input, output in zip(ds, outputs):
        gold = RESPONSE_EXTRACTOR[dataset_name](model_input[ANSWER_KEY])
        if args.post_truncate:
            truncated_responses = [post_truncate(resp.text) for resp in output.outputs]
        else:
            truncated_responses = [resp.text for resp in output.outputs]
        prediction = [
            RESPONSE_EXTRACTOR[dataset_name](truncated_resp)
            for truncated_resp in truncated_responses
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                QUESTION_KEY: model_input[QUESTION_KEY],
                ANSWER_KEY: model_input[ANSWER_KEY],
                "responses": [resp.text for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
                "accuracy": [eq(gold, pred) for pred in prediction],
            }
        ) 
    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    return results

def get_scores(results):
    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))
    # for i in range(k):
    #     pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    #     acc_at_i = sum([eq(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
    #     acc_at_k_list.append(acc_at_i)
    #     pass_at_k_list.append(pass_at_i)
    #     print(
    #         f"Pass @ {i+1}: {pass_at_i}"
    #     )

    def get_most_common(solns):
        soln_counts = {}
        for soln in solns:
            if soln is None:
                continue
            added = False
            for other_solns in solns:
                if eq(soln, other_solns):
                    added = True
                    soln_counts[soln] = soln_counts.get(soln, 0) + 1
            if not added:
                soln_counts[soln] = 1
        if len(soln_counts) == 0:
            return None
        return max(soln_counts, key=soln_counts.get)
    
    predictions_maj = [get_most_common(p) for p in predictions]
    all_preds = sum([[eq(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
    avg_pass_rate = sum(all_preds) / len(all_preds)
    pass_at_n = sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
    print(
        f"Pass @ 1(with majority): {pass_at_n}"
    )
    
    return {
        'pass@1': pass_at_1,
        'pass@1(majority)': sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
        'average_pass_rate': avg_pass_rate,
        # 'std_pass_rate': np.std(acc_at_k_list),
        # 'acc@k': acc_at_k_list,
        # 'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }


def evaluate_model(model_name, base_model_name, split):
    test_prompts = []
    model = LLM(model_name, tokenizer=base_model_name, gpu_memory_utilization=0.9, tensor_parallel_size=1)   
    test_ds = dataset[split]
    # test_ds = test_ds.select(range(50))
    # test_ds = dataset[split].shuffle(seed=0)
    
    for cnt, x in enumerate(test_ds):
        if args.apply_chat_template:
            prompt = [{
                "role": "user",
                "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {x[QUESTION_KEY]}",
            }]
            prompt_tokens = model.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        else:
            prompt=f"{x[QUESTION_KEY]} Let's think step by step and output the final answer after \'####\'."
            # prompt=f"Question: {x[QUESTION_KEY]}\nLet's think step by step"
            # prompt=x[QUESTION_KEY]
            # prompt=prompt_template.format(question=x[QUESTION_KEY])
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
    if args.save_freq > 0:
        n_groups = (len(test_prompts)-1) // args.save_freq + 1
        full_result_list = []
        for i in range(n_groups):
            test_outputs = model.generate(prompt_token_ids=test_prompts[i*args.save_freq: (i+1)*args.save_freq], sampling_params=sampling_params, use_tqdm=True)
            result_list = process_outputs(test_ds, test_outputs, f"outputs/checkpoints/{dataset_name.replace('/', '_')}_{args.split}_results_{model_path.replace('/', '_')}_{args.prompt_type}_t{args.temperature}_{tok_limit}_ckpt{i*args.save_freq}.json")
            full_result_list += result_list
        with open(f"outputs/{dataset_name.replace('/', '_')}_{args.split}_results_{model_path.replace('/', '_')}_{args.prompt_type}_t{args.temperature}_{tok_limit}.json", 'w') as f:
            json.dump(full_result_list, f, indent=4)
    else:
        test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
        full_result_list = process_outputs(test_ds, test_outputs, f"outputs/{dataset_name.replace('/', '_')}_{args.split}_results_{model_path.replace('/', '_')}_{tok_limit}.json")
        
    end_time = time.time()
    test_scores = get_scores(full_result_list)
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}

os.makedirs("results", exist_ok=True)
if args.save_freq > 0:
    os.makedirs("outputs/checkpoints", exist_ok=True)
else:
    os.makedirs("outputs", exist_ok=True)

print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model(model_path, args.base_model_path, args.split)
results[model_path] = scores

with open(f"results/{dataset_name.replace('/', '_')}_{args.split}_results_{model_path.replace('/', '_')}_{args.prompt_type}_t{args.temperature}_{tok_limit}.json", 'w') as f:
    json.dump(results, f, indent=4)
