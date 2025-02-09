'''
This file is the server for the reward function. It listens for incoming connections from the client and sends the reward to the OpenRLHF environment.
'''

import json
from flask import Flask, request, jsonify
from datasets import load_from_disk
import argparse
import numpy as np
from utils import DATASET_KEYS, RESPONSE_COMPARATOR, RESPONSE_EXTRACTOR
from transformers import AutoTokenizer

app = Flask(__name__)


def load_dataset_dicts(datasets):
    # Load datasets into memory
    dataset_dict = {}
    print(f"Loading {datasets}...")
    for dataset_name in datasets:
        dataset = load_from_disk(dataset_name)
        if 'train' in dataset:
            dataset = dataset['train']
        print(f"Picking {dataset_name} consisting of {len(dataset)} examples.")
        question_key = DATASET_KEYS[dataset_name]['question']
        answer_key = DATASET_KEYS[dataset_name]['answer']
        for entry in dataset:
            dataset_dict[entry[question_key]] = entry[answer_key]
        
    return dataset_dict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.route('/query', methods=['POST'])
def query():
    # Main entry point for the server
    try:
        metrics = {'rewards': []}
        for dataset_name in app.config['dataset_names']:
            metrics[f"{dataset_name}_accuracy"] = []
            metrics[f"{dataset_name}_response_length"] = []
            metrics[f"is_{dataset_name}"] = []

        tokenizer = app.config['tokenizer']
        query_dict = request.get_json()

        # Compute length of only the correct responses grouped by 'question'
        avg_length = {}
        avg_length_of_batch = []
        for query in query_dict.get('query', []):
            aux_info = query.get('aux_info', None)
            curr_dataset_name = aux_info.get('dataset_name', None)
            question = aux_info[DATASET_KEYS[curr_dataset_name]['question']]
            if question in avg_length:
                continue
            all_responses = aux_info['all_responses']
            answer = app.config['dataset_dict'].get(aux_info[DATASET_KEYS[curr_dataset_name]['question']], None)
            for response in all_responses:
                response_len = len(tokenizer.encode(tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True)))

                extracted_response = RESPONSE_EXTRACTOR[curr_dataset_name](response)
                extracted_answer = RESPONSE_EXTRACTOR[curr_dataset_name](answer)
                contains_eos = True
                if tokenizer.eos_token_id not in tokenizer.encode(response, add_special_tokens=False):
                    contains_eos = False
                if not contains_eos and app.config['check_eos']:
                    accuracy = 0
                else:
                    accuracy = float(RESPONSE_COMPARATOR[curr_dataset_name](extracted_response, extracted_answer))
                
                if accuracy > 0:
                    # Collect only if acc > 0
                    if question not in avg_length:
                        avg_length[question] = []
                    avg_length[question].append(response_len)
                    avg_length_of_batch.append(response_len)

        for query in query_dict.get('query', []):
            aux_info = query.get('aux_info', None)
            curr_dataset_name = aux_info.get('dataset_name', None)
            response = query.get('response', None)
            contains_eos = True
            if tokenizer.eos_token_id not in tokenizer.encode(response, add_special_tokens=False):
                contains_eos = False
            if not contains_eos and app.config['check_eos']:
                accuracy = 0
                reward = 0
            else:
                response_len = len(tokenizer.encode(tokenizer.decode(tokenizer.encode(response, add_special_tokens=False), skip_special_tokens=True))) # needed because there are special tokens in the response
                answer = app.config['dataset_dict'].get(aux_info[DATASET_KEYS[curr_dataset_name]['question']], None)
                extracted_response = RESPONSE_EXTRACTOR[curr_dataset_name](response)
                extracted_answer = RESPONSE_EXTRACTOR[curr_dataset_name](answer)
                accuracy = float(RESPONSE_COMPARATOR[curr_dataset_name](extracted_response, extracted_answer))

            if app.config['reward_type'] == 'sigmoid':
                if accuracy > 0:
                    lens = avg_length[aux_info[DATASET_KEYS[curr_dataset_name]['question']]]
                    relative_length = (response_len - np.mean(lens)) / (np.std(lens) + 1e-7) # Reward only when answer is correct.
                    reward = accuracy * (1 - app.config['alpha'] * (sigmoid(relative_length)))
                else:
                    reward = 0.0

            metrics['rewards'].append(reward) # score can be something else as well, not just correctness
                                                
            for dataset_name in app.config['dataset_names']:
                if dataset_name == curr_dataset_name:
                    metrics[f"is_{dataset_name}"].append(1.0)
                    metrics[f"{dataset_name}_accuracy"].append(accuracy)
                    metrics[f"{dataset_name}_response_length"].append(response_len)
                else:
                    metrics[f"is_{dataset_name}"].append(float('nan'))
                    metrics[f"{dataset_name}_accuracy"].append(float('nan'))
                    metrics[f"{dataset_name}_response_length"].append(float('nan'))
    
        return jsonify(metrics), 200
    
    except Exception as e:
        # Save the dict for debugging purposes.
        print("Query:", json.dump(query_dict, open('error.json', 'w'), indent=4))
        print(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='0.0.0.0:100')
    parser.add_argument('--dataset_names', type=str, default='openai/gsm8k')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--reward_type', type=str, default='linear') # can be linear or sigmoid
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--check_eos', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    dataset_names = args.dataset_names.split(',')
    app.config['dataset_names'] = dataset_names
    
    dataset_dict = load_dataset_dicts(dataset_names)
    print(f"Server will start running on port: {args.address}. Use the URI 'http://{args.address}/query' to send queries.")
    app.config['dataset_dict'] = dataset_dict
    app.config['tokenizer'] = AutoTokenizer.from_pretrained(args.tokenizer)
    app.config['reward_type'] = args.reward_type
    app.config['alpha'] = args.alpha
    app.config['check_eos'] = args.check_eos
    
    app.run(host=args.address.split(":")[0], port=int(args.address.split(":")[1]))