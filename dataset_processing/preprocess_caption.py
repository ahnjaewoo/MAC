import os
import json
import spacy
import argparse
from tqdm import tqdm
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='coco', choices=['coco', 'msrvtt', 'audiocaps'])
    parser.add_argument("--data_split", default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    if args.dataset_name == 'coco':
        if args.data_split == 'train':
            with open('dataset_processing/data/COCO/annotations/captions_train2014.json', 'r') as fp:
                data = json.load(fp)

            image_dict = {}
            for example_image in data['images']:
                image_dict[example_image['id']] = example_image

            annotation_dict = defaultdict(list)
            for example_annotation in data['annotations']:
                annotation_dict[example_annotation['image_id']].append(example_annotation['caption'])
        # Load COCO "Karpathy" val, test set
        elif args.data_split in ['val', 'test']:
            with open(f'dataset_processing/data/COCO/annotations/karpathy_{args.data_split}_coco_ids.txt', 'r') as fp:
                cocoids = set([int(x.strip()) for x in fp.readlines()])

            with open('dataset_processing/data/COCO/annotations/captions_val2014.json', 'r') as fp:
                data = json.load(fp)

            image_dict = {}
            for example_image in data['images']:
                image_dict[example_image['id']] = example_image

            annotation_dict = defaultdict(list)
            for example_annotation in data['annotations']:
                if example_annotation['image_id'] not in cocoids:
                    continue
                annotation_dict[example_annotation['image_id']].append(example_annotation['caption'])
        else:
            raise ValueError
    elif args.dataset_name == 'msrvtt':
        if args.data_split == 'train':
            with open('dataset_processing/data/MSR-VTT/retrieval_task/train.json', 'r') as fp:
                data = json.load(fp)
        elif args.data_split == 'test':
            with open('dataset_processing/data/MSR-VTT/retrieval_task/test_jsfusion.json', 'r') as fp:
                data = json.load(fp)
        else:
            raise ValueError
        
        annotation_dict = defaultdict(list)
        for example in data:
            annotation_dict[example['video_id']] = example['sentences']

    elif args.dataset_name == 'audiocaps':
        if args.data_split == 'train':
            with open('dataset_processing/data/AudioCaps/retrieval/retrieval_train.json', 'r') as fp:
                data = json.load(fp)
        elif args.data_split == 'test':
            with open('dataset_processing/data/AudioCaps/retrieval/retrieval_test.json', 'r') as fp:
                data = json.load(fp)
        else:
            raise ValueError

        annotation_dict = defaultdict(list)
        for k,v in data.items():
            annotation_dict[str(k)] = v['captions']
    else:
        raise NotImplementedError
    image_id_list = sorted(annotation_dict.keys())

    nlp = spacy.load("en_core_web_sm")
    
    pos_stat_dict = {}
    for image_id in tqdm(image_id_list, ncols=80):
        gold_caption_list = annotation_dict[image_id]
        # pos tagging
        pos_stat_list = []
        for gold_caption in gold_caption_list:
            doc = nlp(gold_caption)
            pos = [token.pos_ for token in doc]
            pos_stat = {p: pos.count(p) for p in set(pos)}
            pos_stat_list.append(pos_stat)
        pos_stat_dict[image_id] = pos_stat_list

    if args.dataset_name == 'coco':
        save_dir = 'dataset_processing/data/COCO/annotations/'
    elif args._get_args.dataset_name == 'msrvtt':
        save_dir = 'dataset_processing/data/MSR-VTT/retrieval_task/'
    elif args.dataset_name == 'audiocaps':
        save_dir = 'dataset_processing/data/AudioCaps/retrieval/'
    else:
        raise NotImplementedError
    with open(os.path.join(save_dir, f'pos_stats_{args.data_split}.json'), 'w') as fp:
        json.dump(pos_stat_dict, fp, indent=2)

    print(f'Good Job Computer!')

if __name__ == '__main__':
    main()
