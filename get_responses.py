import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np


# This script evaluates a model on a dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dataset', type=str)
parser.add_argument('--scale', type=str, default='1.5B')
parser.add_argument('--tok_limit', type=int, default=32768)
parser.add_argument('--split', type=str, default='test', help="train or test")
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

dataset_name = args.dataset
model_path = args.model_path
scale = args.scale
tok_limit = args.tok_limit
dataset_name = args.dataset
results = {}

print("Dataset:", dataset_name, "\nScale:", scale)

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


def get_scores(ds, outputs, save_file_name=None):
    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs):
        gold = RESPONSE_EXTRACTOR[dataset_name](input[ANSWER_KEY])
        prediction = [
            RESPONSE_EXTRACTOR[dataset_name](resp.text)
            for resp in output.outputs
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                QUESTION_KEY: input[QUESTION_KEY],
                ANSWER_KEY: input[ANSWER_KEY],
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

    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))
    for i in range(k):
        pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_i = sum([eq(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_k_list.append(acc_at_i)
        pass_at_k_list.append(pass_at_i)
        print(
            f"Pass @ {i+1}: {pass_at_i}"
        )

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
        'std_pass_rate': np.std(acc_at_k_list),
        'acc@k': acc_at_k_list,
        'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }


def evaluate_model(model_name, split):
    test_prompts = []
    model = LLM(model_name, tokenizer=f'deepseek-ai/DeepSeek-R1-Distill-Qwen-{scale}', gpu_memory_utilization=0.9, tensor_parallel_size=1)    
    test_ds = dataset[split]
    
    for x in test_ds:
        prompt = [{
            "role": "user",
            "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {x[QUESTION_KEY]}",
        }]
        prompt_tokens = model.llm_engine.tokenizer.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
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
    end_time = time.time()
    test_scores = get_scores(test_ds, test_outputs, f"outputs/{dataset_name.replace('/', '_')}_{args.split}_results_{model_path.replace('/', '_')}_{tok_limit}.json")
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}

os.makedirs("results", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model(model_path, args.split)
results[model_path] = scores

with open(f'results/{dataset_name.replace("/", "_")}_{args.split}_results_{model_path.replace("/", "_")}_{tok_limit}.json', 'w') as f:
    json.dump(results, f, indent=4)
