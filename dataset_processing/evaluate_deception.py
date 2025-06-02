import os
import json
import random
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from evaluate_scores import (
    COCOCandidateDataset,
    MSRVTTCandidateDataset,
    AudioCapsCandidateDataset,
    get_cross_score,
    get_uni_score
)

random.seed(1234)

base_dir = os.path.dirname(os.path.abspath(__file__))

def sample_gen(labeled_dataset, cross_score):
    # Initialize the new data structure
    sampled_dataset = {}
    sampled_cross_score_gen = {}  # To store the sampled cross scores per example

    # Iterate over each entry in 'labeled_dataset'
    for key, value in labeled_dataset.items():
        data_entry = value[0]  # The main data entry
        image_metadata = value[1]  # Image metadata

        # Extract ground truth captions and generated captions
        gt_list = data_entry['gt']  # List of ground truth captions
        gen_list = data_entry['gen']  # List of lists, each containing N generated captions
        is_deceptive_dict = data_entry['is_deceptive']
        distance_list = data_entry['distance']  # List of lists of distances
        type_list = data_entry['type']  # List of types

        cross_score_gen = cross_score['gen'][key]

        # Initialize new data structures
        selected_gen_captions = []
        selected_is_deceptive = {'cross': [], 'uni': [], 'distance': [], 'lexical': []}
        selected_distances = []
        sampled_cross_score_gen_per_example = []  # To store the cross scores 'gen' for selected captions

        # Iterate over each ground truth caption
        for i in range(len(gt_list)):
            # Get the generated captions and 'is_deceptive' lists for the current ground truth
            gen_captions = gen_list[i]  # List of N generated captions for ground truth i

            # Extract 'is_deceptive' lists for different criteria
            is_deceptive_cross = is_deceptive_dict['cross'][i]
            is_deceptive_uni = is_deceptive_dict['uni'][i]
            is_deceptive_distance = is_deceptive_dict['distance'][i]
            is_deceptive_lexical = is_deceptive_dict['lexical'][i]

            # Combine 'is_deceptive' flags using logical AND
            is_deceptive_combined = [
                cross and uni and distance and lexical
                for cross, uni, distance, lexical in zip(
                    is_deceptive_cross,
                    is_deceptive_uni,
                    is_deceptive_distance,
                    is_deceptive_lexical
                )
            ]

            # Find indices where 'is_deceptive' is True
            deceptive_indices = [idx for idx, val in enumerate(is_deceptive_combined) if val]
 
            # deception_priority_sample
            if deceptive_indices:
                # If any deceptive captions exist, sample one from them
                selected_idx = random.choice(deceptive_indices)
            else:
                # Else, sample one from all generated captions
                selected_idx = random.randint(0, len(gen_captions) - 1)

            # Append the selected generated caption to the list
            selected_gen_captions.append(gen_captions[selected_idx])

            # Append the 'is_deceptive' values for the selected caption
            selected_is_deceptive['cross'].append(is_deceptive_cross[selected_idx])
            selected_is_deceptive['uni'].append(is_deceptive_uni[selected_idx])
            selected_is_deceptive['distance'].append(is_deceptive_distance[selected_idx])
            selected_is_deceptive['lexical'].append(is_deceptive_lexical[selected_idx])

            # Append the 'distance' value for the selected caption
            selected_distances.append(distance_list[i][selected_idx])

            # Append the cross_score value for the selected caption to the separate list
            sampled_cross_score_gen_per_example.append(cross_score_gen[i][selected_idx])

        # Create the new data entry
        new_data_entry = {
            'gt': gt_list,
            'gen': selected_gen_captions,
            'type': type_list,
            'distance': selected_distances,
            'is_deceptive': selected_is_deceptive
        }

        # Add the new entry to 'sampled_dataset'
        sampled_dataset[key] = [new_data_entry, image_metadata]

        # Store the sampled cross scores in the separate dictionary
        sampled_cross_score_gen[key] = sampled_cross_score_gen_per_example
    return sampled_dataset, sampled_cross_score_gen

def evaluate_and_print_results(outputs):
    results = {}
    for key in outputs.keys():
        if key in ['cross-gap']:
            cross_score_gap_list = [x for x,y in zip(outputs['cross-gap'], outputs['total']) if y]
            cross_score_gap = sum(cross_score_gap_list) / len(cross_score_gap_list) if len(cross_score_gap_list) > 0 else 0
            results[key] = f'cross-modal score gap (among {len(cross_score_gap_list)} examples) = {round(cross_score_gap, 4)}'
            print(results[key])
        elif key in ['cross', 'uni', 'distance', 'lexical', 'total']:
            deception_rate = sum(outputs[key]) / len(outputs[key])
            results[key] = f'deception rate ({key}) = {round(deception_rate*100, 2)}% ({sum(outputs[key])}/{len(outputs[key])})'
            print(results[key])
        elif key in ['type']:
            pass
        else:
            raise ValueError

    return results

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
    args = parser.parse_args()

    if args.num_return_sequences_at_test_time in [-1, args.num_return_sequences]:
        args.num_return_sequences_at_test_time = args.num_return_sequences
    else:
        assert args.data_split == 'train', f"num_return_sequences_at_test_time should be set only for train split!"

    if args.dataset_name == 'coco':
        dataset = COCOCandidateDataset(args)
        max_levenshtein_distance = 5
    elif args.dataset_name == 'msrvtt':
        dataset = MSRVTTCandidateDataset(args)
        max_levenshtein_distance = 4
    elif args.dataset_name == 'audiocaps':
        dataset = AudioCapsCandidateDataset(args)
        max_levenshtein_distance = 5
    else:
        raise NotImplementedError

    cross_score = get_cross_score(args, dataset)
    # label each sample as deceptive or not
    labeled_dataset = json.load(
            open(os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, "generated.json"))
        )

    uni_score_complete = defaultdict(lambda: defaultdict())
    unimodal_models = ['roberta_large_mnli', 'deberta_xlarge_mnli', 'bart_large_mnli']
    for unimodal_model in unimodal_models:
        args.unimodal_model = unimodal_model
        uni_score = get_uni_score(args, dataset)
        for contents_id, sample_uni in uni_score['gen'].items():
            uni_score_complete[unimodal_model][contents_id] = sample_uni

    # check cross modal score, uni modal score, and distance
    for contents_id, example in tqdm(labeled_dataset.items(), ncols=80):
        cross_score_gen = cross_score['gen'][contents_id]
        cross_score_gt = cross_score['gt'][contents_id] # [5, num_return_sequences]
        uni_score_gen = [uni_score_complete[unimodal_model][contents_id] for unimodal_model in unimodal_models] # [3, 5, num_return_sequences, 3]

        # threshold-based filtering
        uni_score_gen = np.array(uni_score_gen)[:,:,:,0] * 1 + np.array(uni_score_gen)[:,:,:,1] * 0.5
        output_uni = np.all(uni_score_gen < 0.5, axis=0).tolist() # [5, num_return_sequences]

        output_lexical = []
        for gt_idx, gt_c in enumerate(example[0]['gt']):
            gt_w = gt_c.lower().rstrip('.').split()
            output_lexical_inner = []
            for gen_idx, gen_c in enumerate(example[0]['gen'][gt_idx]):
                gen_w = gen_c.lower().rstrip('.').split()
                gen_type = example[0]['type'][gt_idx]
                dist = example[0]['distance'][gt_idx][gen_idx]
                is_safe_lexical = True

                # 1. negation check
                for neg in ['no', 'not', 'empty', 'without']:
                    if neg in gen_w and neg not in gt_w:
                        is_safe_lexical = False
                        break
                # 2. type check
                if gen_type in ['object-swap', 'attribute-swap']:
                    # check if the word set is the same, while the order is different
                    if set(gt_w) != set(gen_w) or gt_w == gen_w:
                        is_safe_lexical = False
                elif gen_type in ['general', 'attribute-add', 'attribute-replace', 'object-add', 'object-replace', 'relation-replace']:
                    pass
                else:
                    raise ValueError
                output_lexical_inner.append(is_safe_lexical)
            output_lexical.append(output_lexical_inner)

        example[0]['is_deceptive'] = {
            'cross': (np.array(cross_score_gen) >= np.array(cross_score_gt)[:, np.newaxis]).tolist(),
            'uni': output_uni,
            'distance': ((np.array(example[0]['distance']) > 0) & (np.array(example[0]['distance']) <= max_levenshtein_distance)).tolist(),
            'lexical': output_lexical,
        }

    sampled_dataset, sampled_cross_score_gen = sample_gen(labeled_dataset, cross_score)

    # global deception
    print(f'\n### Global Deception Rate of {dataset.generation_fpath} {args.crossmodal_model} ###\n')

    outputs = {'cross': [], 'uni': [], 'distance': [], 'lexical': [], 'total': [], 'cross-gap': [], 'type': []}
    eval_outputs = {'examples': sampled_dataset, 'global_results_overall': {}, 'local_results_overall': {}}
    for contents_id, example in sampled_dataset.items():
        for key, value in example[0]['is_deceptive'].items():
            outputs[key].extend(value)
        outputs['type'].extend(example[0]['type'])
        outputs['cross-gap'].extend([x - y for x,y in zip(sampled_cross_score_gen[contents_id], cross_score['gt'][contents_id])])
    outputs['total'] = [x and y and z and w for x,y,z,w in zip(outputs['cross'], outputs['uni'], outputs['distance'], outputs['lexical'])]

    # overall results (global)
    eval_outputs['global_results_overall'] = evaluate_and_print_results(outputs)

    # results for each type (global)
    type_list = sorted(list(set(outputs['type'])))
    for type in type_list:
        indices = [i for i, x in enumerate(outputs['type']) if x == type]
        type_outputs = {key: [outputs[key][i] for i in indices] for key in outputs.keys()}
        print(f"\n### Global Deception Rate of {dataset.generation_fpath} {args.crossmodal_model} ({type}) ###\n")
        eval_outputs[f'global_results_{type}'] = evaluate_and_print_results(type_outputs)

    # show word statistics
    import itertools
    from collections import Counter
    gt_captions = list(itertools.chain(*[x[0]['gt'] for x in sampled_dataset.values()]))
    gt_words = Counter(list(itertools.chain(*[x.lower().rstrip('.').split() for x in gt_captions])))
    gen_captions = list(itertools.chain(*[x[0]['gen'] for x in sampled_dataset.values()]))
    gen_words = Counter(list(itertools.chain(*[x.lower().rstrip('.').split() for x in gen_captions])))

    gen_deceptive_captions = list(itertools.chain(*[[y for i,y in enumerate(x[0]['gen']) if x[0]['is_deceptive']['cross'][i] and x[0]['is_deceptive']['uni'][i] and x[0]['is_deceptive']['distance'][i] and x[0]['is_deceptive']['lexical'][i]] for x in sampled_dataset.values()]))
    gen_deceptive_words = Counter(list(itertools.chain(*[x.lower().rstrip('.').split() for x in gen_deceptive_captions])))

    print(f"\n### Word Statistics (global) ###\n")
    print(f"GT: {gt_words.most_common(20)}")
    print(f"\nGEN: {gen_words.most_common(20)}")
    print(f"\nGEN (emergent words): {(gen_words - gt_words).most_common(20)}")
    print(f"\nGEN (deceptive): {(gen_deceptive_words).most_common(20)}")
    
    # local deception
    print(f'\n### Local Deception Rate of {dataset.generation_fpath} {args.crossmodal_model} ###\n')
    from itertools import chain

    outputs = {'cross': [], 'uni': [], 'distance': [], 'lexical': [], 'total': [], 'cross-gap': [], 'type': []}
    for contents_id, example in labeled_dataset.items():
        for key, value in example[0]['is_deceptive'].items():
            outputs[key].extend(list(chain(*value)))
        type_list = example[0]['type']
        expanded_type = [t for t in type_list for _ in range(args.num_return_sequences)]
        outputs['type'].extend(expanded_type)
        cross_gap = [
            gen_value - gt_value
            for gt_value, sublist_gen in zip(cross_score['gt'][contents_id], cross_score['gen'][contents_id])
            for gen_value in sublist_gen
        ]
        outputs['cross-gap'].extend(cross_gap)
    outputs['total'] = [x and y and z and w for x,y,z,w in zip(outputs['cross'], outputs['uni'], outputs['distance'], outputs['lexical'])]

    # overall results (local)
    eval_outputs['local_results_overall'] = evaluate_and_print_results(outputs)

    # results for each type (local)
    type_list = sorted(list(set(outputs['type'])))
    for type in type_list:
        indices = [i for i, x in enumerate(outputs['type']) if x == type]
        type_outputs = {key: [outputs[key][i] for i in indices] for key in outputs.keys()}
        print(f"\n### Local Deception Rate of {dataset.generation_fpath} {args.crossmodal_model} ({type}) ###\n")
        eval_outputs[f'local_results_{type}'] = evaluate_and_print_results(type_outputs)

    # save results
    eval_outputs_fname = f"evaluated_{args.crossmodal_model}"
    with open(os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"{eval_outputs_fname}.json"), 'w') as fp:
        json.dump(eval_outputs, fp, indent=4)

    # save results (full)
    with open(os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"{eval_outputs_fname}_full.json"), 'w') as fp:
        json.dump(labeled_dataset, fp, indent=4)

    print(f"Good Job Computer!")

if __name__ == "__main__":
    main()
