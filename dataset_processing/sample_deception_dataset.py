import os
import json
import copy
import random
import argparse
import numpy as np
from math import log
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer

from evaluate_diversity import levenshtein_parser

base_dir = os.path.dirname(os.path.abspath(__file__))

random.seed(1234)

def compute_metrics(dist_dict):
    total_cnt = sum(dist_dict.values())
    prob_dict = {k: v / total_cnt for k, v in dist_dict.items()}
    entropy = -sum([p * log(p) for p in prob_dict.values()])
    distinct_1 = len(dist_dict) / total_cnt if total_cnt > 0 else 0
    norm_entropy = entropy/log(len(dist_dict)) if len(dist_dict) > 0 else 0
    return entropy, norm_entropy, distinct_1

def print_metrics(dist_dict):
    ent, norm_ent, dist = compute_metrics(dist_dict)
    print(f"ENT: {ent:.4f} | ENT (norm): {norm_ent:.4f} | D-1: {dist:.4f}")

def get_diverse_sft_dataset(eval_outputs, prompt_templates, modality_type, max_levenshtein_distance, sft_fpath, args):
    sft_dataset = []
    out = list()

    for item in tqdm(eval_outputs.values(), ncols=80):
        item = item[0]
        for gt_idx in range(len(item["gt"])):
            gt = item["gt"][gt_idx]
            gen_type = item["type"][gt_idx]
            gen_list = item["gen"][gt_idx]
            dist_list = item["distance"][gt_idx]
            success_list = [
                item["is_deceptive"]["cross"][gt_idx][i] and
                item["is_deceptive"]["uni"][gt_idx][i] and
                item["is_deceptive"]["distance"][gt_idx][i] and
                item["is_deceptive"]["lexical"][gt_idx][i]
                for i in range(len(gen_list))
            ]

            if any(success_list):
                out.append([
                    [levenshtein_parser(gt, gen_list[i]), dist_list[i], success_list[i], gt, gen_list[i], gen_type]
                    for i in range(len(gen_list)) if success_list[i]
                ])

    dist_dict = dict() # This instance will be used in multiple rounds

    # Prepare init distribution via random selection
    old_list = list()  # list of selected samples from previous iter
    for item in out:
        # Select random sample while prioritizing successful ones if any
        valid_list = [x for x in item if x[2]] # success
        sample = random.choice(valid_list if len(valid_list) > 0 else item)
        old_list.append(sample)
        if sample[1] <= max_levenshtein_distance:
            for t in sample[0][0]:
                dist_dict[t] = 1 if t not in dist_dict else dist_dict[t] + 1
        else:
            # Does not satisfy distance criteria; ignoring
            old_list[-1][0] = [[], []]

    print_metrics(dist_dict)
    ent, norm_ent, d1 = compute_metrics(dist_dict)

    # Apply Gibbs-ish greedy sampling
    idx_list = list(range(len(out)))

    for round in range(4):
        new_list = list() # list of selected samples from current iter
        for item_idx in tqdm(idx_list, ncols=80):
            item = out[item_idx]
            # Select target list while prioritizing successful ones if any
            valid_list = [x for x in item if x[2]]
            target_list = valid_list if len(valid_list) > 0 else item
            max_gain, max_idx = 0, -1
            for sample_idx, new_sample in enumerate(target_list):
                old_sample = old_list[item_idx]
                if new_sample[1] > max_levenshtein_distance:
                    continue
                # Modify dict
                for t in old_sample[0][0]:
                    dist_dict[t] = dist_dict[t] - 1
                    if dist_dict[t] == 0:
                        del dist_dict[t]
                for t in new_sample[0][0]:
                    dist_dict[t] = 1 if t not in dist_dict else dist_dict[t] + 1
                # Greedy improvement of objective function
                step_ent, step_norm_ent, step_d1 = compute_metrics(dist_dict)
                if step_ent > ent and step_ent - ent > max_gain:
                    max_gain = step_ent - ent
                    max_idx = sample_idx
                # Rollback dict
                for t in old_sample[0][0]:
                    dist_dict[t] = 1 if t not in dist_dict else dist_dict[t] + 1
                for t in new_sample[0][0]:
                    dist_dict[t] = dist_dict[t] - 1
                    if dist_dict[t] == 0:
                        del dist_dict[t]

            if max_idx > -1:
                # Update with a sample with the most performance boost if any
                new_list.append(target_list[max_idx])
                old_sample = old_list[item_idx]
                # Modify dict
                for t in old_sample[0][0]:
                    dist_dict[t] = dist_dict[t] - 1
                    if dist_dict[t] == 0:
                        del dist_dict[t]
                for t in target_list[max_idx][0][0]:
                    dist_dict[t] = 1 if t not in dist_dict else dist_dict[t] + 1
            else:
                new_list.append(old_list[item_idx])

        # Compute metrics with newly updated lists
        dist_dict = dict()
        for sample in new_list:
            for t in sample[0][0]:
                dist_dict[t] = 1 if t not in dist_dict else dist_dict[t] + 1
        old_list = [x for x in new_list]
        ent, norm_ent, d1 = compute_metrics(dist_dict)
        print_metrics(dist_dict)

        # save intermediate results if necessary
        sft_dataset_temp = []
        for item in new_list:
            gt_cap, gen_cap, gen_type = item[3], item[4], item[5]
            prompt_temp = prompt_templates[gen_type].format(caption=gt_cap,
                                                    contents_modality=modality_type,
                                                    max_word_distance_plus_one=max_levenshtein_distance+1)

            sft_example_temp = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_temp,
                    },
                    {
                        "role": "assistant",
                        "content": f"Generated Caption: {gen_cap}",
                    }
                ]
            }
            sft_dataset_temp.append(sft_example_temp)

        os.makedirs(base_dir / Path("./data") / "sft" / sft_fpath, exist_ok=True)
        with open(base_dir / Path("./data") / "sft" / sft_fpath / "instruction_dataset_temp.json", 'w') as fp:
            json.dump(sft_dataset_temp, fp, indent=4)

    # use new_list to generate sft_dataset
    for item in new_list:
        gt_cap, gen_cap, gen_type = item[3], item[4], item[5]
        prompt = prompt_templates[gen_type].format(caption=gt_cap,
                                                   contents_modality=modality_type,
                                                   max_word_distance_plus_one=max_levenshtein_distance+1)

        sft_example = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": f"Generated Caption: {gen_cap}",
                }
            ]
        }
        sft_dataset.append(sft_example)
    return sft_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='coco', choices=['coco', 'msrvtt', 'audiocaps'])
    parser.add_argument("--data_split", default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--gen_mode', default='deceptive-general', type=str, choices=['deceptive-general', 'deceptive-specific'])
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
    parser.add_argument("--sample_sft_method", default='random', choices=['random', 'maxent'])
    parser.add_argument('--partition', default=0, type=int, help='partition index of the dataset')
    parser.add_argument('--num_partitions', default=1, type=int, help='number of partitions to split the dataset')
    args = parser.parse_args()

    if args.num_return_sequences_at_test_time in [-1, args.num_return_sequences]:
        args.num_return_sequences_at_test_time = args.num_return_sequences
    else:
        assert args.data_split == 'train', f"num_return_sequences_at_test_time should be set only for train split!"

    model_checkpoint_subdir = f"{args.crossmodal_model}"

    if args.num_iterations == 0:
        assert args.model_checkpoint_fname is None, f"model_checkpoint_fname should not be set!"
        assert args.model_checkpoint_steps is None, f"model_checkpoint_steps should not be set!"
        generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter{args.num_iterations}'
        sft_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}_{args.sample_sft_method}/iter{args.num_iterations}/{model_checkpoint_subdir}'
    else:
        assert args.model_checkpoint_fname is not None, f"model_checkpoint_fname should be set!"
        assert args.model_checkpoint_steps is not None, f"model_checkpoint_steps should be set!"
        generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter{args.num_iterations}/{model_checkpoint_subdir}/{args.model_checkpoint_fname}_ckpt{args.model_checkpoint_steps}'
        sft_fpath = copy.deepcopy(generation_fpath)

    eval_outputs_fname = f"evaluated_{args.crossmodal_model}"
    with open(base_dir / Path("./data") / "candidates" / generation_fpath / f"{eval_outputs_fname}.json", 'r') as fp:
        eval_outputs = json.load(fp)

    with open(base_dir / Path("./data") / "candidates" / generation_fpath / f"{eval_outputs_fname}_full.json", 'r') as fp:
        full_eval_outputs = json.load(fp)

    prompt_templates = {}
    if args.gen_mode == 'deceptive-general':
        with open(os.path.join(base_dir, f'instructions/{args.gen_mode}_prompt_text.txt'), 'r') as fp:
            prompt_templates['general'] = fp.read()
    elif args.gen_mode == 'deceptive-specific':
        for type in ['object-add', 'object-replace', 'object-swap', 'attribute-add', 'attribute-replace', 'attribute-swap', 'relation-replace', 'count-replace']:
            with open(os.path.join(base_dir, f'instructions/deceptive-{type}_prompt_text.txt'), 'r') as fp:
                prompt_templates[type] = fp.read()
    else:
        raise NotImplementedError

    # Load modality type
    if args.dataset_name == 'coco':
        modality_type = 'image'
        max_levenshtein_distance = 5
    elif args.dataset_name == 'msrvtt':
        modality_type = 'video'
        max_levenshtein_distance = 4
    elif args.dataset_name == 'audiocaps':
        modality_type = 'audio'
        max_levenshtein_distance = 5
    else:
        raise NotImplementedError

    if args.sample_sft_method == 'maxent':
        # code for 'greedy sampling (max entropy)' for lexical diversity enhancement
        sft_dataset = get_diverse_sft_dataset(full_eval_outputs, prompt_templates, modality_type, max_levenshtein_distance, sft_fpath, args)
    else:
        sft_dataset = []
        for example in tqdm(eval_outputs['examples'].values(), ncols=80):
            is_dec = example[0]['is_deceptive']
            for idx in range(len(example[0]['gt'])):
                if is_dec['cross'][idx] and is_dec['uni'][idx] and is_dec['distance'][idx] and is_dec['lexical'][idx]:
                    gt_cap = example[0]['gt'][idx]
                    gen_cap = example[0]['gen'][idx]
                    gen_type = example[0]['type'][idx]
                    prompt = prompt_templates[gen_type].format(caption=gt_cap,
                                                            contents_modality=modality_type,
                                                            max_word_distance_plus_one=max_levenshtein_distance+1)

                    sft_example = {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                            {
                                "role": "assistant",
                                "content": f"Generated Caption: {gen_cap}",
                            }
                        ]
                    }
                    sft_dataset.append(sft_example)
                else:
                    pass

    print(f"# of sft dataset: {len(sft_dataset)}")
    os.makedirs(base_dir / Path("./data") / "sft" / sft_fpath, exist_ok=True)
    with open(base_dir / Path("./data") / "sft" / sft_fpath / "instruction_dataset.json", 'w') as fp:
        json.dump(sft_dataset, fp, indent=4)

    print(f'Good Job Computer!')

if __name__ == "__main__":
    main()