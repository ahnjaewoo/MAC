import os
import json
import argparse

from evaluate_scores import (
    COCOCandidateDataset,
    MSRVTTCandidateDataset,
    AudioCapsCandidateDataset,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default='coco', choices=['coco', 'msrvtt', 'audiocaps'])
parser.add_argument("--data_split", default='train', choices=['train', 'val', 'test'])
parser.add_argument('--gen_mode', default='deceptive-general', type=str, choices=['deceptive-general', 'deceptive-specific', 'seetrue-general', 'videocon-specific'])
parser.add_argument("--generation_model", default='llama3.1:8b', choices=['llama3.1:8b', 'gemma2:9b', 'qwen2.5:7b', 'gpt-4o'])
parser.add_argument('--temperature', default=0.7, type=float)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--num_iterations', default=0, type=int)
parser.add_argument('--num_return_sequences', default=1, type=int)
parser.add_argument('--num_return_sequences_at_test_time', default=-1, type=int)
parser.add_argument("--model_checkpoint_root_dir", default='../checkpoints', type=str)
parser.add_argument("--model_checkpoint_fname", default=None, type=str)
parser.add_argument("--model_checkpoint_steps", default=None, type=str)
parser.add_argument("--crossmodal_model", default='clip', choices=['clip', 'siglip', 'llava', 'languagebind_video', 'languagebind_audio', 'clap'])
args = parser.parse_args()

if args.num_return_sequences_at_test_time in [-1, args.num_return_sequences]:
    args.num_return_sequences_at_test_time = args.num_return_sequences
else:
    assert args.data_split == 'train', f"num_return_sequences_at_test_time should be set only for train split!"

if args.dataset_name == 'coco':
    dataset = COCOCandidateDataset(args)
elif args.dataset_name == 'msrvtt':
    dataset = MSRVTTCandidateDataset(args)
elif args.dataset_name == 'audiocaps':
    dataset = AudioCapsCandidateDataset(args)
else:
    raise NotImplementedError

eval_outputs_fname = os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"evaluated_{args.crossmodal_model}.json")
with open(eval_outputs_fname, 'r') as f:
    data = json.load(f)

# Desired order of keys
desired_order = ['cross', 'uni', 'distance', 'lexical', 'total']

# Reordering result based on the desired order
global_deception_result = [
    data['global_results_overall'][key].split('=')[1].split('%')[0].strip()
    if '%' in data['global_results_overall'][key]
    else data['global_results_overall'][key].split('=')[1].strip()
    for key in desired_order
    if key in data['global_results_overall']
]

# Reordering result based on the desired order
local_deception_result = [
    data['local_results_overall'][key].split('=')[1].split('%')[0].strip()
    if '%' in data['local_results_overall'][key]
    else data['local_results_overall'][key].split('=')[1].strip()
    for key in desired_order
    if key in data['local_results_overall']
]

# Extracting required values from 'global_diversity_overall'
global_diversity_data = data['global_diversity_overall']

# Calculating dist-1
unique_keys = float(global_diversity_data['unique_keys'])
total_operations = float(global_diversity_data['total_operations'])
dist_1 = unique_keys / total_operations if total_operations > 0 else 0

# Formatting the result
global_diversity_result = [
    global_diversity_data['entropy'],
    global_diversity_data['normalized_entropy'],
    global_diversity_data['unique_keys'],
    global_diversity_data['total_operations'],
    f"{dist_1:.4f}"
]

# Calculating dist-1
unique_keys = float(global_diversity_data['unique_keys'])
total_operations = float(global_diversity_data['total_operations'])
dist_1 = unique_keys / total_operations if total_operations > 0 else 0

print("\n========== Deception Evaluation ==========")
print("[Global Deception]")
for k, v in zip(desired_order, global_deception_result):
    print(f"  {k.ljust(10)}: {v}%")

print("\n[Local Deception]")
for k, v in zip(desired_order, local_deception_result):
    print(f"  {k.ljust(10)}: {v}%")

print("\n========== Diversity Evaluation ==========")
print("[Global Diversity]")
print(f"  Entropy             : {global_diversity_result[0]}")
print(f"  Normalized Entropy  : {global_diversity_result[1]}")
print(f"  Unique Keys         : {global_diversity_result[2]}")
print(f"  Total Operations    : {global_diversity_result[3]}")
print(f"  Dist-1              : {global_diversity_result[4]}")

print('Good Job Computer!')