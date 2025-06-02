import os
import json
import argparse
import multiprocessing as mp
from math import log
from tqdm import tqdm
from pathlib import Path
from functools import partial

import spacy
import numpy as np

from evaluate_scores import (
    COCOCandidateDataset,
    MSRVTTCandidateDataset,
    AudioCapsCandidateDataset,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

UPOS_LIST = [
    "ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB", # Open-class
    "ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ", # Closed-class
    "PUNCT", "SYM", "X" # ETC
]
TAGGER = spacy.load("en_core_web_sm")
NUM_WORKERS = 24

def levenshtein_parser(sentence1, sentence2, config=dict()):
    use_upos = config.get("use_upos", True)
    lemmatize = config.get("lemmatize", True)

    sentence1 = TAGGER(sentence1.lower().strip().rstrip('.'))
    sentence2 = TAGGER(sentence2.lower().strip().rstrip('.'))

    if lemmatize:
        words1 = [token.lemma_ for token in sentence1]
        words2 = [token.lemma_ for token in sentence2]
    else:
        words1 = [token.text for token in sentence1]
        words2 = [token.text for token in sentence2]
    if use_upos:
        pos1 = [token.pos_ for token in sentence1]
        pos2 = [token.pos_ for token in sentence2]
    else:
        pos1 = [token.tag_ for token in sentence1]
        pos2 = [token.tag_ for token in sentence2]

    len1, len2 = len(words1) + 1, len(words2) + 1
    distance_matrix = np.zeros((len1, len2))
    distance_matrix[:,0] = np.arange(len1)
    distance_matrix[0,:] = np.arange(len2)
    pointer_matrix = np.zeros_like(distance_matrix)

    # Compute levenshtein distance
    for i in range(1, len1):
        for j in range(1, len2):
            cost = 0 if words1[i-1] == words2[j-1] else 1
            candidate = np.array([
                distance_matrix[i-1][j] + 1,    # Deletion
                distance_matrix[i][j-1] + 1,    # Insertion
                distance_matrix[i-1][j-1] + cost  # Substitution
            ])
            distance_matrix[i][j] = np.min(candidate)
            pointer_matrix[i][j] = 1 + np.argmin(candidate)

    # Backtrack
    parsed_list, parsed_list_with_sub = list(), list()
    it_i, it_j = len1 - 1, len2 - 1
    while pointer_matrix[it_i, it_j] != 0:
        if pointer_matrix[it_i, it_j] == 1:
            parsed_list.append(f"D_{pos1[it_i-1]}_{words1[it_i-1]}")
            parsed_list_with_sub.append(f"D_{pos1[it_i-1]}_{words1[it_i-1]}")
            it_i -= 1
        elif pointer_matrix[it_i, it_j] == 2:
            parsed_list.append(f"I_{pos2[it_j-1]}_{words2[it_j-1]}")
            parsed_list_with_sub.append(f"I_{pos2[it_j-1]}_{words2[it_j-1]}")
            it_j -= 1
        elif pointer_matrix[it_i, it_j] == 3:
            if distance_matrix[it_i, it_j] != distance_matrix[it_i-1, it_j-1]:
                parsed_list.append(f"D_{pos1[it_i-1]}_{words1[it_i-1]}")
                parsed_list.append(f"I_{pos2[it_j-1]}_{words2[it_j-1]}")
                parsed_list_with_sub.append(f"S_{pos2[it_j-1]}_{words2[it_j-1]}")
            it_j -= 1
            it_i -= 1

    return parsed_list, parsed_list_with_sub

def check_validity(item, gt_idx, criteria):
    return {
        "crossmodal": item["cross"][gt_idx],
        "unimodal": item["uni"][gt_idx],
        "distance": item["distance"][gt_idx],
        "lexical": item["lexical"][gt_idx]
    }

def get_corpus(eval_outputs_fname, config):
    # Compute with multiprocessing
    print(f'Loading eval outputs from {eval_outputs_fname}')
    data = json.load(open(eval_outputs_fname))['examples'].values()
    corpus = dict()
    valid_cnt = 0
    do_parse = partial(get_parsed_corpus, config=config)

    p = mp.Pool(NUM_WORKERS)
    for out in tqdm(p.imap_unordered(do_parse, data), total=len(data), ncols=80):
        corpus[out[0]] = out[1]
    p.close()
    p.join()

    dist_dict, dist_dict_with_sub = dict(), dict()
    sent_total = 0
    for key, value in corpus.items():
        for item in value:
            sentence, sentece_with_sub = item[0][0], item[0][1]
            validity = item[1]
            is_valid = True
            for crit in config["criteria"]:
                is_valid = is_valid and validity[crit]
            if is_valid:
                sent_total += 1
                for v in sentence:
                    dist_dict[v] = 1 if v not in dist_dict else dist_dict[v] + 1
                for v in sentece_with_sub:
                    dist_dict_with_sub[v] = 1 if v not in dist_dict_with_sub else dist_dict_with_sub[v] + 1

    if config["use_upos"]:
        dist_dict = {
            k: v for k, v in dist_dict.items()
            if k.split("_")[1] in config["pos_list"]
        }
        dist_dict_with_sub = {
            k: v for k, v in dist_dict_with_sub.items()
            if k.split("_")[1] in config["pos_list"]
        }

    return dist_dict, dist_dict_with_sub, sent_total

def report_stat(dist_dict, sent_total, eval_outputs, with_sub=False):
    # Operation-wise statistics
    id_dict = dict()
    ops_dict = dict()
    for k, v in dist_dict.items():
        id_key = k.split("_")[0]
        ops_key = "_".join(k.split("_")[:-1])
        id_dict[id_key] = v if id_key not in id_dict else id_dict[id_key] + v
        ops_dict[ops_key] = v if ops_key not in ops_dict else ops_dict[ops_key] + v
    ops_dict.update(id_dict)
    ops_dict = dict(sorted(ops_dict.items(), key=lambda x: x[1], reverse=True))

    # Probability-wise statistics
    len_key = len(dist_dict)
    total_cnt = sum(dist_dict.values())
    distinct_1 = len_key / total_cnt if total_cnt != 0 else 0
    prob_dict = {k: v/total_cnt for k, v in dist_dict.items()}
    entropy = -sum([p*log(p) for p in prob_dict.values()])

    # Formatted output
    print(f"### Metrics Summary (with_sub={with_sub})")
    print(f"**Metrics**           | **Value** ")
    print(f"----------------------|-----------")
    print(f"Total sentences       | {sent_total} ")
    print(f"Unique keys           | {len_key} ")
    print(f"Total operations      | {total_cnt} ")
    print(f"Distinct-1            | {distinct_1:.04f} ")
    print(f"Entropy               | {entropy:.04f} ")
    print(f"Normalized entropy    | {entropy/log(len_key):.04f} ")
    print(f"Perplexity            | {2**entropy:.04f} ")
    print(f"Normalized perplexity | {2**(entropy/log(len_key)):.04f} ")

    print("\n- Operation statistics")
    print(f"```json\n{json.dumps(ops_dict, indent=4)}\n```")

    output_key = "global_diversity_overall_with_sub" if with_sub else "global_diversity_overall"
    eval_outputs[output_key] = {
        "total_sentences": f"{sent_total}",
        "unique_keys": f"{len_key}",
        "total_operations": f"{sum(dist_dict.values())}",
        "entropy": f"{entropy:.04f}",
        "normalized_entropy": f"{entropy/log(len_key):.04f}",
        "perplexity": f"{2**entropy:.04f}",
        "normalized_perplexity": f"{2**(entropy/log(len_key)):.04f}",
    }

    return eval_outputs

def get_parsed_corpus(item, config):
    key = item[1]["contents_file_name"]
    corpus_list = list()
    for gt_idx in range(len(item[0]["gt"])):
        gt = item[0]["gt"][gt_idx]

        dist = item[0]["distance"][gt_idx]
        if 0 < dist <= config["max_levenshtein_distance"]:
            gen = item[0]["gen"][gt_idx]
        else:
            gen = item[0]["gt"][gt_idx]

        corpus_list.append([
            levenshtein_parser(gt, gen, config),
            check_validity(
                item[0]["is_deceptive"], gt_idx, config["criteria"]
            )
        ])

    return key, corpus_list

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for Corpus Processing")
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

    parser.add_argument('--pos_type', type=str,
        choices=['universal', 'detailed'],
        default='universal',
        help="POS type: 'universal' or 'detailed'"
    )

    parser.add_argument(
        '--tok_type',
        type=str,
        choices=['lemma', 'original'],
        default='lemma',
        help="Token type: 'lemma' or 'original'"
    )

    parser.add_argument(
        '--criteria',
        type=str,
        nargs='+',
        choices=["crossmodal", "unimodal", "distance", "lexical"],
        default=[],
        help="Deception criteria: choose one or more from ['crossmodal', 'unimodal', 'distance', 'lexical']"
    )

    parser.add_argument(
        '--pos_list',
        type=str,
        nargs='+',
        choices=UPOS_LIST,
        default=["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"],
        help="Target UPOS tags (effective only if POS type is 'universal')"
    )

    args = parser.parse_args()

    if args.num_return_sequences_at_test_time in [-1, args.num_return_sequences]:
        args.num_return_sequences_at_test_time = args.num_return_sequences
    else:
        assert args.data_split == 'train', f"num_return_sequences_at_test_time should be set only for train split!"

    return args

def main():
    args = get_args()

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
    
    config = {
        "use_upos": args.pos_type == "universal",
        "lemmatize": args.tok_type == "lemma",
        "criteria": args.criteria,
        "pos_list": args.pos_list,
        "max_levenshtein_distance": max_levenshtein_distance,
    }

    eval_outputs_fname = os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"evaluated_{args.crossmodal_model}.json")
    with open(eval_outputs_fname, 'r') as f:
        eval_outputs = json.load(f)

    corpus, corpus_with_sub, sent_total = get_corpus(eval_outputs_fname, config)
    
    corpus = sorted(corpus.items(), key=lambda x: x[1], reverse=True)
    report_stat(dict(corpus), sent_total, eval_outputs, with_sub=False)

    corpus_with_sub = sorted(corpus_with_sub.items(), key=lambda x: x[1], reverse=True)
    report_stat(dict(corpus_with_sub), sent_total, eval_outputs, with_sub=True)

    print(f'Saving eval outputs to {eval_outputs_fname}')
    with open(eval_outputs_fname, 'w') as f:
        json.dump(eval_outputs, f, indent=4)

    print(f'Good Job Computer!')

if __name__ == '__main__':
    main()
