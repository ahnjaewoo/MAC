import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

random.seed(1234)

base_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess_prompt(caption_list, pos_stat_list, prompt_templates, modality_type, max_word_distance, gen_mode):
    selected_types = []
    if gen_mode == 'deceptive-general':
        selected_types += ['general' for _ in range(len(caption_list))]
        prompts = [prompt_templates['general'].format(caption=x,
                                                      contents_modality=modality_type,
                                                      max_word_distance_plus_one=max_word_distance+1) for x in caption_list]
    elif gen_mode == 'deceptive-specific':
        for pos_stat in pos_stat_list:
            possible_types = ['object-add', 'attribute-add']
            if 'NOUN' in pos_stat.keys() and pos_stat['NOUN'] >= 1:
                possible_types += ['object-replace']
            if 'NOUN' in pos_stat.keys() and pos_stat['NOUN'] >= 2:
                possible_types += ['object-swap']
            if 'ADJ' in pos_stat.keys() and pos_stat['ADJ'] >= 1:
                possible_types += ['attribute-replace']
            if 'ADJ' in pos_stat.keys() and pos_stat['ADJ'] >= 2:
                possible_types += ['attribute-swap']
            if 'VERB' in pos_stat.keys() and pos_stat['VERB'] >= 1:
                possible_types += ['relation-replace']
            if 'NUM' in pos_stat.keys() and pos_stat['NUM'] >= 1:
                possible_types += ['count-replace']
            selected_types.append(random.choice(possible_types))
        prompts = [prompt_templates[type].format(caption=x,
                                                 contents_modality=modality_type,
                                                 max_word_distance_plus_one=max_word_distance+1) for x, type in zip(caption_list, selected_types)]
    else:
        raise NotImplementedError
    return prompts, selected_types

# Function to calculate Levenshtein distance between two sentences based on words
def levenshtein_word_distance(sentence1, sentence2):
    # Strip leading/trailing spaces and remove periods at the end of sentences
    sentence1 = sentence1.lower().strip().rstrip('.')
    sentence2 = sentence2.lower().strip().rstrip('.')

    # Split sentences into words
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Initialize the distance matrix
    len1 = len(words1) + 1
    len2 = len(words2) + 1
    distance_matrix = np.zeros((len1, len2))

    # Fill the distance matrix
    for i in range(len1):
        distance_matrix[i][0] = i
    for j in range(len2):
        distance_matrix[0][j] = j

    # Compute distances
    for i in range(1, len1):
        for j in range(1, len2):
            if words1[i-1] == words2[j-1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + 1,    # Deletion
                distance_matrix[i][j-1] + 1,    # Insertion
                distance_matrix[i-1][j-1] + cost  # Substitution
            )

    # The Levenshtein distance is the bottom-right value in the matrix
    return int(distance_matrix[len1-1][len2-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='coco', choices=['coco', 'msrvtt', 'audiocaps'])
    parser.add_argument("--data_split", default='train', choices=['train', 'val', 'test'])
    parser.add_argument("--model_name", default='llama3.1:8b', choices=['llama3.1:8b', 'gemma2:9b', 'qwen2.5:7b', 'gpt-4o'])
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--gen_mode', default='deceptive-general', type=str, choices=['deceptive-general', 'deceptive-specific'])
    parser.add_argument("--do_batch_decoding", action='store_true', help="Whether to run batch decoding.")
    parser.add_argument('--num_iterations', default=0, type=int)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--num_return_sequences_at_test_time', default=-1, type=int)
    parser.add_argument("--model_checkpoint_root_dir", default='./checkpoints', type=str)
    parser.add_argument("--model_checkpoint_fname", default=None, type=str)
    parser.add_argument("--model_checkpoint_steps", default=None, type=str)
    parser.add_argument("--crossmodal_model", default='clip', choices=['clip', 'siglip', 'llava', 'languagebind_video', 'languagebind_audio', 'clap'])
    parser.add_argument('--partition', default=0, type=int, help='partition index of the dataset')
    parser.add_argument('--num_partitions', default=1, type=int, help='number of partitions to split the dataset')    
    args = parser.parse_args()

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.num_return_sequences_at_test_time in [-1, args.num_return_sequences]:
        args.num_return_sequences_at_test_time = args.num_return_sequences
    else:
        assert args.data_split == 'train', f"num_return_sequences_at_test_time should be set only for train split!"

    # model selection
    if args.model_name in ['llama3.1:8b', 'gemma2:9b', 'qwen2.5:7b']:
        if args.model_name == 'llama3.1:8b':
            model_full_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            attn_implementation = "flash_attention_2"
        elif args.model_name == 'gemma2:9b':
            model_full_name = "google/gemma-2-9b-it"
            attn_implementation = "eager"
        elif args.model_name == 'qwen2.5:7b':
            model_full_name = "Qwen/Qwen2.5-7B-Instruct"
            attn_implementation = "flash_attention_2"
        from transformers import AutoTokenizer
        if args.do_batch_decoding:
            # Required for batch decoding
            tokenizer = AutoTokenizer.from_pretrained(model_full_name,
                                                    padding_side='left',)
            # Ensure the pad token is set (if not, set it to the EOS token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_full_name)
        if args.num_iterations == 0:
            assert args.model_checkpoint_fname is None, f"model_checkpoint_fname should not be set!"
            assert args.model_checkpoint_steps is None, f"model_checkpoint_steps should not be set!"
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_full_name,
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation=attn_implementation,
                                                        device_map=device)
        else:
            assert args.model_checkpoint_fname is not None, f"model_checkpoint_fname should be set!"
            assert args.model_checkpoint_steps is not None, f"model_checkpoint_steps should be set!"
            model_checkpoint_subdir = f"{args.crossmodal_model}"
            model_checkpoint_dir = os.path.join(args.model_checkpoint_root_dir, f"{args.dataset_name}/train/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.model_name}_{args.temperature}_{args.top_p}", f"iter{args.num_iterations}", model_checkpoint_subdir, args.model_checkpoint_fname, f"checkpoint-{args.model_checkpoint_steps}")
            assert os.path.exists(model_checkpoint_dir), f"model checkpoint dir does not exist!"
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(model_checkpoint_dir,
                                                             torch_dtype=torch.bfloat16,
                                                             attn_implementation=attn_implementation,
                                                             device_map=device)
    elif args.model_name in ['gpt-4o']:
        assert args.num_return_sequences == 1, f"num_return_sequences should be 1!"
        import openai
        from openai import OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"
        client = OpenAI()
    else:
        raise NotImplementedError

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

    if args.dataset_name == 'coco':
        if args.data_split == 'train':
            with open('dataset_processing/data/COCO/annotations/captions_train2014.json', 'r') as fp:
                data = json.load(fp)

            contents_dict = {}
            for example_image in data['images']:
                contents_dict[example_image['id']] = example_image
                contents_dict[example_image['id']]['url'] = example_image['coco_url']

            annotation_dict = defaultdict(list) # image_id: [caption1, caption2, ...]
            for example_annotation in data['annotations']:
                annotation_dict[example_annotation['image_id']].append(example_annotation['caption'])
        # Load COCO "Karpathy" val, test set
        elif args.data_split in ['val', 'test']:
            with open(f'dataset_processing/data/COCO/annotations/karpathy_{args.data_split}_coco_ids.txt', 'r') as fp:
                cocoids = set([int(x.strip()) for x in fp.readlines()])

            with open('dataset_processing/data/COCO/annotations/captions_val2014.json', 'r') as fp:
                data = json.load(fp)

            contents_dict = {} # image_id: {'file_name': file_name, }
            for example_image in data['images']:
                contents_dict[example_image['id']] = example_image
                contents_dict[example_image['id']]['url'] = example_image['coco_url']

            annotation_dict = defaultdict(list)
            for example_annotation in data['annotations']:
                if example_annotation['image_id'] not in cocoids:
                    continue
                annotation_dict[example_annotation['image_id']].append(example_annotation['caption'])
        else:
            raise ValueError
        with open(f'dataset_processing/data/COCO/annotations/pos_stats_{args.data_split}.json', 'r') as fp:
            pos_stat_dict = json.load(fp) # image_id: {'NOUN': 1, 'ADJ': 0, 'VERB': 0, 'NUM': 0}
        modality_type = 'image'
        # average num words in test captions = 10.43
        max_levenshtein_distance = 5
        max_new_tokens = 40
    elif args.dataset_name == 'msrvtt':
        if args.data_split == 'train':
            with open(f'dataset_processing/data/MSR-VTT/retrieval_task/train.json', 'r') as fp:
                data = json.load(fp)
        elif args.data_split == 'test':
            with open(f'dataset_processing/data/MSR-VTT/retrieval_task/test_jsfusion.json', 'r') as fp:
                data = json.load(fp)
        else:
            raise ValueError

        contents_dict = {}
        for example in data:
            contents_dict[example['video_id']] = {'file_name': f"{example['video_id']}.mp4", 'url': example['url']}

        annotation_dict = defaultdict(list)
        for example in data:
            annotation_dict[example['video_id']] = example['sentences']

        with open(f'dataset_processing/data/MSR-VTT/retrieval_task/pos_stats_{args.data_split}.json', 'r') as fp:
            pos_stat_dict = json.load(fp)
        modality_type = 'video'
        # average num words in test captions = 9.41
        max_levenshtein_distance = 4
        max_new_tokens = 40
    elif args.dataset_name == 'audiocaps':
        if args.data_split == 'train':
            with open('dataset_processing/data/AudioCaps/retrieval/retrieval_train.json', 'r') as fp:
                data = json.load(fp)
        elif args.data_split == 'test':
            with open('dataset_processing/data/AudioCaps/retrieval/retrieval_test.json', 'r') as fp:
                data = json.load(fp)
        else:
            raise ValueError

        contents_dict = {}
        for example_key in data.keys():
            contents_dict[example_key] = {'file_name': f"{example_key}.wav", 'url': f"https://www.youtube.com/watch?v={example_key}"}

        annotation_dict = defaultdict(list)
        for example_key, example in data.items():
            annotation_dict[example_key] = example['captions']

        with open(f'dataset_processing/data/AudioCaps/retrieval/pos_stats_{args.data_split}.json', 'r') as fp:
            pos_stat_dict = json.load(fp)
        modality_type = 'audio'
        # average num words in test captions = 10.26
        max_levenshtein_distance = 5
        max_new_tokens = 40
    else:
        raise NotImplementedError
    contents_id_list = sorted(annotation_dict.keys())

    # get start & end index
    if args.partition < args.num_partitions - 1:
        start_idx = (len(contents_id_list) // args.num_partitions) * args.partition
        end_idx = start_idx + (len(contents_id_list) // args.num_partitions)
    else:
        assert args.partition == args.num_partitions - 1
        start_idx = (len(contents_id_list) // args.num_partitions) * args.partition
        end_idx = len(contents_id_list)

    if args.num_iterations == 0:
        outputs_fpath = os.path.join(base_dir, f'data/candidates/{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.model_name}_{args.temperature}_{args.top_p}/iter0')
    else:
        outputs_fpath = os.path.join(base_dir, f'data/candidates/{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.model_name}_{args.temperature}_{args.top_p}/iter{args.num_iterations}/{model_checkpoint_subdir}/{args.model_checkpoint_fname}_ckpt{args.model_checkpoint_steps}')
    os.makedirs(outputs_fpath, exist_ok=True)
    outputs_fname = os.path.join(outputs_fpath, "generated")

    if args.partition == 0 and args.num_partitions == 1:
        outputs_fname += ".json"
    else:
        outputs_fname += f"_{args.partition}_{args.num_partitions}.json"

    outputs = dict()
    data_cnt = 0
    for contents_id in tqdm(contents_id_list[start_idx:end_idx], ncols=80):
        contents_fname = contents_dict[contents_id]['file_name']
        contents_url = contents_dict[contents_id]['url']

        gold_caption_list = annotation_dict[contents_id]
        pos_stat_list = pos_stat_dict[str(contents_id)]
        prompts, selected_types = preprocess_prompt(gold_caption_list, pos_stat_list, prompt_templates, modality_type, max_levenshtein_distance, args.gen_mode)

        if args.model_name in ['llama3.1:8b', 'gemma2:9b', 'qwen2.5:7b']:
            if args.do_batch_decoding:
                inputs = []
                for prompt in prompts:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                    input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs.append(input)

                # Tokenize the inputs with padding
                inputs_tokenized = tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512  # Adjust as needed
                ).to(model.device)

                with torch.no_grad():
                    # Generate
                    generated_ids = model.generate(**inputs_tokenized,
                                                pad_token_id=tokenizer.eos_token_id,
                                                do_sample=True,
                                                num_return_sequences=args.num_return_sequences,
                                                temperature=args.temperature,
                                                top_p=args.top_p,
                                                max_new_tokens=max_new_tokens)
                generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_captions = [x.split('assistant\n')[-1].split('Generated Caption:')[-1].split('\n\n')[0].strip() for x in generated_tokens]
                generated_caption_list = [[generated_captions[i + j * args.num_return_sequences] for i in range(args.num_return_sequences)] for j in range(len(gold_caption_list))]
            else:
                generated_caption_list = []
                for prompt in prompts:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
                    input_length = input_ids.shape[1]
                    attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=device)

                    with torch.no_grad():
                        generated_ids = model.generate(input_ids,
                                                    attention_mask=attention_mask,
                                                    pad_token_id=tokenizer.eos_token_id,
                                                    do_sample=True,
                                                    num_return_sequences=args.num_return_sequences,
                                                    temperature=args.temperature,
                                                    top_p=args.top_p,
                                                    max_new_tokens=max_new_tokens)
                    generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    generated_caption_list.append([x.split('assistant\n')[-1].split('Generated Caption:')[-1].split('\n\n')[0].strip() for x in generated_tokens])
        elif args.model_name in ['gpt-4o']:
            assert args.num_return_sequences == 1, f"num_return_sequences should be 1!"

            generated_caption_list = []
            for prompt in prompts:
                completion = client.chat.completions.create(
                    # gpt-4o-2024-08-06
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                response = completion.choices[0].message.content
                response = response.split('Generated Caption:')[-1].strip()
                if response[:2] == '- ' and prompt[:2] != '- ':
                    response = response[2:]
                generated_caption_list.append([response])
        else:
            raise NotImplementedError

        distance_list = []
        for gt, gens in zip(gold_caption_list, generated_caption_list):
            distance_list.append([levenshtein_word_distance(gt, gen) for gen in gens])

        output = [
            {
                'gt': gold_caption_list,
                'gen': generated_caption_list,
                'type': selected_types,
                'distance': distance_list,
            },
            {
                'contents_file_name': contents_fname,
                'contents_url': contents_url,
            }
        ]

        outputs[contents_id] = output
        data_cnt += 1

        if data_cnt % 10 == 0:
            with open(f'{outputs_fname}.temp', 'w') as fp:
                json.dump(outputs, fp, indent=4)

    with open(outputs_fname, 'w') as fp:
        json.dump(outputs, fp, indent=4)

    print(f'Good Job Computer!')

if __name__ == '__main__':
    main()
