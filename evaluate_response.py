import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
import json

# This script evaluates a model on a dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--task', type=str)
parser.add_argument('--response_path', type=str, help="json file path of the response")
parser.add_argument('--scale', type=str, default='1.5B')
parser.add_argument('--tok_limit', type=int, default=32768)
parser.add_argument('--test_n', type=int, default=1)
parser.add_argument('--temp', type=float, default=0.9)
parser.add_argument('--n_sample', type=int, default=100)
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# dataset_name = args.dataset
model_path = args.model_path
scale = args.scale
tok_limit = args.tok_limit
dataset_name = args.task
results = {}

# print("Dataset:", dataset_name, "\nScale:", scale)

QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
RESPONSE_KEY = DATASET_KEYS[dataset_name]["responses"]
CORRECTNESS_KEY = DATASET_KEYS[dataset_name]["correctness"]
eq = RESPONSE_COMPARATOR[dataset_name]

# if dataset_name == 'datasets/converted_aime_dataset':
#     dataset = load_from_disk(dataset_name)
#     TEST_N = 10
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 100
# elif dataset_name == 'di-zhang-fdu/MATH500':
#     dataset = load_dataset(dataset_name)
#     TEST_N = 3
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 500
# elif dataset_name == 'openai/gsm8k':
#     dataset = load_dataset(dataset_name, 'main')
#     TEST_N = 1
#     MAX_TOKENS = tok_limit
#     TEST_TEMPERATURE = 0.6
#     MAX_TEST_SAMPLES = 1319
TEST_N = args.test_n
MAX_TOKENS = tok_limit
TEST_TEMPERATURE = args.temp
MAX_TEST_SAMPLES = args.n_sample

with open(args.response_path, 'r') as f:
    data = json.load(f)
data = data[:4]

with open("prompt.json", 'r') as f:
    prompt_list = json.load(f)
for p in prompt_list:
    if p['name'] == "verifier2":
        prompt_template = p['prompt_template']
        break

def get_scores(ds, outputs, save_file_name=None):
    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs):
        # gold = RESPONSE_EXTRACTOR[dataset_name](input[ANSWER_KEY])
        gold = str(input["accuracy"][0])
        # print([
        #     resp.text
        #     for resp in output.outputs
        # ])
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
                RESPONSE_KEY: input[RESPONSE_KEY],
                "prediction": input["prediction"][0],
                "gold": input["gold"],
                "accuracy": input["accuracy"][0],
                'verifier_responses': [resp.text for resp in output.outputs],
                'verifier_prediction': prediction,
                'verifier_tokens': [len(resp.token_ids) for resp in output.outputs],
                'verifier_accuracy': [eq(pred, gold) for pred in prediction]
            }
        )
    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    predictions, golds = results["verifier_prediction"], results["accuracy"]
    predictions = [str(p) for p in predictions]
    golds = [str(g) for g in golds]
    # print(predictions, golds)
    pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    # print("Average tokens:", sum(tokens) / len(tokens))
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
    print("Finished")
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
        # 'avg_tokens': sum(tokens) / len(tokens)
    }


def evaluate_model(model_name):
    test_prompts = []
    model = LLM(model_name, tokenizer=f'deepseek-ai/DeepSeek-R1-Distill-Qwen-{scale}', gpu_memory_utilization=0.9, tensor_parallel_size=1)    
    # test_ds = dataset['test'].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset['test']))))
    
    for x in data:
        content = prompt_template.format(
                question = x['question'],
                response = x['responses']
            )
        print(content)
        prompt = [{
            "role": "user",
            "content": content,
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
    test_scores = get_scores(data, test_outputs, f"outputs/verify/{result_name}.json")
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'test': test_scores, 'time_taken': time_taken}

os.makedirs("results/verify", exist_ok=True)
os.makedirs("outputs/verify", exist_ok=True)
result_name = args.response_path.split("/")[-1][:-5] # to save files
result_name += "_verifier2"


print("Found model_path:", model_path)
print("This is not a checkpoint, will evaluate directly...")
scores = evaluate_model(model_path)
results[model_path] = scores

with open(f'results/verify/{result_name}.json', 'w') as f:
    json.dump(results, f, indent=4)
