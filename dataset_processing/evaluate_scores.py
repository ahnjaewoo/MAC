import os
import json
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification as AutoModelForSeqCls,
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
)

base_dir = os.path.dirname(os.path.abspath(__file__))

class COCOCandidateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading COCO Candidates"""

    def __init__(self, args):
        super().__init__()
        self.data_path = Path('./data')

        model_checkpoint_subdir = f"{args.crossmodal_model}"
        if args.num_iterations == 0:
            assert args.model_checkpoint_fname is None, f"model_checkpoint_fname should not be set!"
            assert args.model_checkpoint_steps is None, f"model_checkpoint_steps should not be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter0'
        else:
            assert args.model_checkpoint_fname is not None, f"model_checkpoint_fname should be set!"
            assert args.model_checkpoint_steps is not None, f"model_checkpoint_steps should be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter{args.num_iterations}/{model_checkpoint_subdir}/{args.model_checkpoint_fname}_ckpt{args.model_checkpoint_steps}'
        self.meta = json.load(
            open(os.path.join(base_dir, self.data_path, "candidates", self.generation_fpath, "generated.json"))
        )
        self.meta = list(self.meta.values())

        self.data_split = args.data_split
        assert self.data_split in ['train', 'val', 'test'], \
            f"data_split should be one of ['train', 'val', 'test'] but got {self.data_split}"
        self.coco_split = "train2014" if self.data_split == "train" else "val2014"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        return {
            "id": str(int(meta[1]["contents_file_name"].split(".")[0].split("_")[-1])),
            "contents": Image.open(
                base_dir / self.data_path / "COCO" / self.coco_split /
                meta[1]["contents_file_name"]
            ).convert("RGB"),
            "text_gt": meta[0]["gt"],
            "text_gen": meta[0]["gen"]
        }

class MSRVTTCandidateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading MSRVTT Candidates"""

    def __init__(self, args):
        super().__init__()
        self.data_path = Path('./data')

        model_checkpoint_subdir = f"{args.crossmodal_model}"
        if args.num_iterations == 0:
            assert args.model_checkpoint_fname is None, f"model_checkpoint_fname should not be set!"
            assert args.model_checkpoint_steps is None, f"model_checkpoint_steps should not be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter0'
        else:
            assert args.model_checkpoint_fname is not None, f"model_checkpoint_fname should be set!"
            assert args.model_checkpoint_steps is not None, f"model_checkpoint_steps should be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter{args.num_iterations}/{model_checkpoint_subdir}/{args.model_checkpoint_fname}_ckpt{args.model_checkpoint_steps}'
        self.meta = json.load(
            open(os.path.join(base_dir, self.data_path, "candidates", self.generation_fpath, "generated.json"))
        )
        self.meta = list(self.meta.values())

        self.data_split = args.data_split
        assert self.data_split in ['train', 'val', 'test'], \
            f"data_split should be one of ['train', 'val', 'test'] but got {self.data_split}"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        return {
            "id": meta[1]["contents_file_name"].split(".")[0],
            "contents": os.path.join(base_dir, self.data_path, "MSR-VTT", "videos", meta[1]["contents_file_name"]),
            "text_gt": meta[0]["gt"],
            "text_gen": meta[0]["gen"]
        }

class AudioCapsCandidateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading AudioCaps Candidates"""

    def __init__(self, args):
        super().__init__()
        self.data_path = Path('./data')

        model_checkpoint_subdir = f"{args.crossmodal_model}"
        if args.num_iterations == 0:
            assert args.model_checkpoint_fname is None, f"model_checkpoint_fname should not be set!"
            assert args.model_checkpoint_steps is None, f"model_checkpoint_steps should not be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter0'
        else:
            assert args.model_checkpoint_fname is not None, f"model_checkpoint_fname should be set!"
            assert args.model_checkpoint_steps is not None, f"model_checkpoint_steps should be set!"
            self.generation_fpath = f'{args.dataset_name}/{args.data_split}/{args.gen_mode}-{args.num_return_sequences_at_test_time}_{args.generation_model}_{args.temperature}_{args.top_p}/iter{args.num_iterations}/{model_checkpoint_subdir}/{args.model_checkpoint_fname}_ckpt{args.model_checkpoint_steps}'
        self.meta = json.load(
            open(os.path.join(base_dir, self.data_path, "candidates", self.generation_fpath, "generated.json"))
        )
        self.meta = list(self.meta.values())

        self.data_split = args.data_split
        assert self.data_split in ['train', 'val', 'test'], \
            f"data_split should be one of ['train', 'val', 'test'] but got {self.data_split}"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        return {
            "id": meta[1]["contents_file_name"].split('.wav')[0],
            "contents": os.path.join(base_dir, self.data_path, "AudioCaps", "audio", meta[1]["contents_file_name"]),
            "text_gt": meta[0]["gt"],
            "text_gen": meta[0]["gen"]
        }

class CrossScore(nn.Module):
    """Model for cross-modal score computation"""

    def __init__(self, crossmodal_model, args, data_path):
        super().__init__()
        self.crossmodal_model = crossmodal_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if crossmodal_model == "clip":
            assert args.dataset_name in ['coco']
            self.model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            self.model = AutoModel.from_pretrained(self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        elif crossmodal_model == "siglip":
            assert args.dataset_name in ['coco']
            self.model_id = "google/siglip-so400m-patch14-384"
            self.model = AutoModel.from_pretrained(self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        elif crossmodal_model == 'languagebind_video':
            assert args.dataset_name in ['msrvtt'], "LanguageBind (video) model is only supported for MSRVTT dataset"
            from LanguageBind.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
            self.model_id = 'LanguageBind/LanguageBind_Video_FT'
            self.model = LanguageBindVideo.from_pretrained(self.model_id)
            tokenizer = LanguageBindVideoTokenizer.from_pretrained(self.model_id)
            self.processor = LanguageBindVideoProcessor(self.model.config, tokenizer)
        elif crossmodal_model == 'languagebind_audio':
            assert args.dataset_name in ['audiocaps'], "LanguageBind (audio) model is only supported for AudioCaps dataset"
            from LanguageBind.languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
            self.model_id = 'LanguageBind/LanguageBind_Audio_FT'
            self.model = LanguageBindAudio.from_pretrained(self.model_id)
            tokenizer = LanguageBindAudioTokenizer.from_pretrained(self.model_id)
            self.processor = LanguageBindAudioProcessor(self.model.config, tokenizer)
        elif crossmodal_model == 'clap':
            import librosa
            assert args.dataset_name in ['audiocaps'], "LanguageBind (audio) model is only supported for AudioCaps dataset"
            self.model_id = "laion/clap-htsat-unfused"
            self.model = AutoModel.from_pretrained(self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        else:
            raise NotImplementedError(f"{crossmodal_model} not supported")
        self.postprocess = lambda out: nn.CosineSimilarity()(
                out.image_embeds, out.text_embeds)
        pixel_cache_dir = os.path.join(base_dir, data_path, "candidates", f'{args.dataset_name}/{args.data_split}')
        os.makedirs(pixel_cache_dir, exist_ok=True)
        self.pixel_cache_file = os.path.join(base_dir, pixel_cache_dir, f"{args.crossmodal_model}_pixels.pt")
        self.pixel_value_dict = self._load_pixel_cache()  # Load all cached pixel values

    def _load_pixel_cache(self):
        """Load cached pixel values from a single file."""
        if os.path.exists(self.pixel_cache_file):
            logging.warning(f"Loading pixel cache from {self.pixel_cache_file}")
            return torch.load(self.pixel_cache_file)
        else:
            logging.warning(f"No pixel cache found. Initializing empty cache.")
            return {}

    def _save_pixel_cache(self):
        """Save all pixel values into a single cache file."""
        logging.warning(f"Saving pixel cache to {self.pixel_cache_file}")
        torch.save(self.pixel_value_dict, self.pixel_cache_file)

    def finalize(self):
        """Save the pixel cache to disk."""
        if not os.path.exists(self.pixel_cache_file):
            self._save_pixel_cache()

    def get_output(self, inputs, pixel_values):
        inputs['pixel_values'] = pixel_values.to(self.device)
        outputs = self.model(**inputs)
        outputs = self.postprocess(outputs)
        # outputs = self.postprocess(outputs).detach().cpu().tolist()
        return outputs

    def forward(self, data, mode='gen'):
        contents_id = data["id"]
        if contents_id in self.pixel_value_dict:
            pixel_values = self.pixel_value_dict[contents_id]
        else:
            pixel_values = self.processor(images=data['contents'], return_tensors='pt')['pixel_values']
            self.pixel_value_dict[contents_id] = pixel_values

        if mode == 'gt':
            if self.crossmodal_model in ['clip', 'siglip']:
                inputs = self.processor(
                    text=data[f"text_{mode}"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
            elif self.crossmodal_model in ['languagebind_video', 'languagebind_audio']:
                inputs = self.processor(
                    text=data[f"text_{mode}"],
                    return_tensors="pt"
                ).to(self.device)
            else:
                raise NotImplementedError
            outputs = self.get_output(inputs, pixel_values)
            outputs = outputs.detach().cpu().tolist()

        elif mode == 'gen':
            num_sequences = len(data['text_gen'][0])
            gen_list = [item for sublist in data['text_gen'] for item in sublist]
            if self.crossmodal_model in ['clip', 'siglip']:
                inputs = self.processor(
                    text=gen_list,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
            elif self.crossmodal_model in ['languagebind_video', 'languagebind_audio']:
                inputs = self.processor(
                    text=gen_list,
                    return_tensors="pt"
                ).to(self.device)
            else:
                raise NotImplementedError
            outputs = self.get_output(inputs, pixel_values)
            assert outputs.numel() == len(data['text_gen']) * num_sequences, "Output size mismatch"
            outputs = outputs.view(len(data['text_gen']), num_sequences).detach().cpu().tolist()

        else:
            raise ValueError
        return outputs

class CrossScoreCLAP(CrossScore):
    """Model for cross-modal score computation"""

    def __init__(self, crossmodal_model, args, data_path):
        super().__init__(crossmodal_model, args, data_path)
        self.postprocess = lambda out: nn.CosineSimilarity()(
                out.audio_embeds, out.text_embeds)

    def forward(self, data, mode='gen'):
        contents_id = data["id"]

        audio_path = data['contents']
        audio_array, _ = librosa.load(audio_path, sr=48000)

        if mode == 'gt':
            inputs = self.processor(
                text=data[f"text_{mode}"],
                padding=True,
                audios=audio_array,
                sampling_rate=48000,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            outputs = self.postprocess(outputs)
            outputs = outputs.detach().cpu().tolist()

        elif mode == 'gen':
            num_sequences = len(data['text_gen'][0])
            gen_list = [item for sublist in data['text_gen'] for item in sublist]
            inputs = self.processor(
                text=gen_list,
                padding=True,
                audios=audio_array,
                sampling_rate=48000,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            outputs = self.postprocess(outputs)
            assert outputs.numel() == len(data['text_gen']) * num_sequences, "Output size mismatch"
            outputs = outputs.view(len(data['text_gen']), num_sequences).detach().cpu().tolist()
        else:
            raise ValueError
        return outputs

class CrossScoreLLAVA(nn.Module):
    """Model for cross-modal score computation"""

    def __init__(self, crossmodal_model, args, data_path):
        assert args.dataset_name == 'coco'
        super().__init__()
        self.crossmodal_model = crossmodal_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id = "llava-hf/llava-1.5-7b-hf"
        from transformers import LlavaForConditionalGeneration
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id,
                                                                   torch_dtype=torch.bfloat16,
                                                                   attn_implementation='flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False)

    def finalize(self):
        pass

    def get_output(self, inputs):
        outputs = self.model(**inputs)

        batch_last_word_logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
        yes_idx = 3582
        no_idx = 1217

        batch_yes_logits = batch_last_word_logits[:, yes_idx].float()  # [batch_size]
        batch_no_logits = batch_last_word_logits[:, no_idx].float()  # [batch_size]

        batch_yes_exp_logit = torch.exp(batch_yes_logits)
        batch_no_exp_logit = torch.exp(batch_no_logits)

        outputs = batch_yes_exp_logit / (batch_yes_exp_logit + batch_no_exp_logit)  # [batch_size]

        return outputs

    def forward(self, data, mode='gen'):
        pixel_values = self.processor.image_processor(images=data['contents'], return_tensors='pt')['pixel_values']
        user_question_template = "USER: <image>\nDoes this image match the following caption: '{caption}'?\nAnswer Yes/No directly. ASSISTANT:"
        if mode == 'gt':
            gt_list = [user_question_template.format(caption=item) for item in data[f"text_{mode}"]]
            inputs = self.processor.tokenizer(
                text=gt_list,
                truncation=True,
                max_length=64,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            inputs['pixel_values'] = pixel_values.repeat(len(data['text_gt']), 1, 1, 1).to(self.device)
            outputs = self.get_output(inputs)
            outputs = outputs.detach().cpu().tolist()

        elif mode == 'gen':
            num_sequences = len(data['text_gen'][0])
            ########################################################################################
            # Code for 'small N' (fast, may OOM on large input)
            # Use this when data fits in memory. Switch to 'large N' below if OOM occurs.
            gen_list = [user_question_template.format(caption=item) for sublist in data['text_gen'] for item in sublist]

            inputs = self.processor.tokenizer(
                text=gen_list,
                truncation=True,
                max_length=64,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            inputs['pixel_values'] = pixel_values.repeat(len(gen_list), 1, 1, 1).to(self.device)
            outputs = self.get_output(inputs)
            assert outputs.numel() == len(data['text_gen']) * num_sequences, "Output size mismatch"
            outputs = outputs.view(len(data['text_gen']), num_sequences).detach().cpu().tolist()
            ########################################################################################
            # # Code for 'large N' (training purporse which can solve OOM)
            # all_outputs = []

            # max_batch_size = 32

            # for i in range(0, len(data['text_gen']), max_batch_size):
            #     batch_outputs = []
            #     batch_data = data['text_gen'][i:i+max_batch_size]
            #     batch_images = data['contents'][i:i+max_batch_size] if isinstance(data['contents'], list) else [data['contents']] * len(batch_data)

            #     for j, (image, captions) in enumerate(zip(batch_images, batch_data)):
            #         captions_output = []

            #         for k in range(0, len(captions), max_batch_size):
            #             mini_batch_captions = captions[k:k+max_batch_size]
            #             gen_list = [user_question_template.format(caption=item) for item in mini_batch_captions]

            #             pixel_values = self.processor.image_processor(
            #                 images=[image] * len(mini_batch_captions),
            #                 return_tensors='pt'
            #             )['pixel_values'].to(self.device)

            #             inputs = self.processor.tokenizer(
            #                 text=gen_list,
            #                 truncation=True,
            #                 max_length=64,
            #                 padding=True,
            #                 return_tensors="pt"
            #             ).to(self.device)

            #             inputs['pixel_values'] = pixel_values
            #             mini_outputs = self.get_output(inputs)
            #             captions_output.extend(mini_outputs.tolist())

            #             del inputs, mini_outputs, pixel_values
            #             torch.cuda.empty_cache()

            #         batch_outputs.append(captions_output)

            #     all_outputs.extend(batch_outputs)

            # outputs = []
            # for output_group in all_outputs:
            #     outputs.append(output_group[:num_sequences])
            ########################################################################################
        else:
            raise ValueError
        return outputs

class UniScore(nn.Module):
    """Model for unimodal entailment score computation"""

    def __init__(self, unimodal_model, args, data_path):
        super().__init__()
        if unimodal_model == "bart_large_mnli":
            self.model_id = "facebook/bart-large-mnli"
        elif unimodal_model == "roberta_large_mnli":
            self.model_id = "FacebookAI/roberta-large-mnli"
        elif unimodal_model == "deberta_xlarge_mnli":
            self.model_id = "microsoft/deberta-xlarge-mnli"
        else:
            raise NotImplementedError(f"{unimodal_model} not supported")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, clean_up_tokenization_spaces=True
        )
        self.model = AutoModelForSeqCls.from_pretrained(self.model_id)
        self.postprocess = lambda out: out.softmax(-1).flip(dims=[1])

    def forward(self, data, mode='gen'):
        num_sequences = len(data['text_gen'][0])
        gt_list = [gt for gt in data['text_gt'] for _ in range(num_sequences)]
        gen_list = [item for sublist in data['text_gen'] for item in sublist]

        inputs = self.tokenizer(
            gt_list, gen_list,
            return_tensors="pt",
            truncation=True,
            padding=True)
        # Move tensors to CUDA
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs).logits
        out_total = self.postprocess(outputs)
        out_total = out_total.view(len(data['text_gen']), num_sequences, 3).detach().cpu().numpy().tolist()
        
        return out_total

def get_cross_score(args, dataset):
    cross_data_dict = dict()
    score_cache_path_dict = {
        'gen': os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"{args.crossmodal_model}_norm_gen.pt"),
        'gt': os.path.join(base_dir, dataset.data_path, "candidates", f'{args.dataset_name}/{args.data_split}/{args.crossmodal_model}_norm_gt.pt'),
        'gt_dir': os.path.join(base_dir, dataset.data_path, "candidates", f'{args.dataset_name}/{args.data_split}'),
    }
    os.makedirs(score_cache_path_dict['gt_dir'], exist_ok=True)
    for mode in ['gen', 'gt']:
        score_cache_path = score_cache_path_dict[mode]
        if os.path.exists(score_cache_path):
            logging.warning(f"Loading precomputed crossmodal scores {args.crossmodal_model}_norm_{mode}")
            cross_data = torch.load(score_cache_path)
        else:
            if args.crossmodal_model == 'llava':
                cross_score = CrossScoreLLAVA(args.crossmodal_model, args, dataset.data_path)
            elif args.crossmodal_model in ["clap"]:
                cross_score = CrossScoreCLAP(args.crossmodal_model, args, dataset.data_path)
            else:
                cross_score = CrossScore(args.crossmodal_model, args, dataset.data_path)
            cross_score.cuda()
            cross_score.eval()
            cross_data = dict()
            with torch.no_grad():
                for data in tqdm(dataset, ncols=80):
                    c_score = cross_score(data, mode=mode)
                    cross_data[data['id']] = c_score
            ################################################################################################
            if args.partition > 0 or args.num_partitions > 1:
                if True:
                    score_cache_path = score_cache_path.replace(".pt", f"_{args.partition}_{args.num_partitions}.pt")
            ################################################################################################
            torch.save(cross_data, score_cache_path)
            cross_score.finalize()
        cross_data_dict[mode] = cross_data
    return cross_data_dict

def get_uni_score(args, dataset):
    uni_data_dict = dict()
    for mode in ['gen']:
        score_cache_path = os.path.join(base_dir, dataset.data_path, "candidates", dataset.generation_fpath, f"{args.unimodal_model}_softmax_{mode}.pt")
        if os.path.exists(score_cache_path):
            logging.warning(f"Loading precomputed unimodal scores {args.unimodal_model}_softmax_{mode}")
            uni_data = torch.load(score_cache_path)
        else:
            uni_score = UniScore(args.unimodal_model, args, dataset.data_path)
            uni_score.cuda()
            uni_score.eval()
            uni_data = dict()
            with torch.no_grad():
                for data in tqdm(dataset, ncols=80):
                    # TODO: should process special token (ex. EOS token)
                    u_score = uni_score(data, mode=mode)
                    uni_data[data['id']] = u_score
                ################################################################################################
                if args.partition > 0 or args.num_partitions > 1:
                    score_cache_path = score_cache_path.replace(".pt", f"_{args.partition}_{args.num_partitions}.pt")
                ################################################################################################
                torch.save(uni_data, score_cache_path)
        uni_data_dict[mode] = uni_data
    return uni_data_dict

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
    parser.add_argument("--unimodal_model", default='roberta_large_mnli', choices=['roberta_large_mnli', 'deberta_xlarge_mnli', 'bart_large_mnli', ])
    parser.add_argument("--compute_crossmodal", action='store_true', help="Whether to compute crossmodal score.")
    parser.add_argument("--compute_unimodal", action='store_true', help="Whether to compute unimodal score.")
    parser.add_argument('--partition', default=0, type=int, help='partition index of the dataset')
    parser.add_argument('--num_partitions', default=1, type=int, help='number of partitions to split the dataset')
    args = parser.parse_args()

    # add partition & num_partitions
    assert args.partition < args.num_partitions

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

    # dataset partitioning
    total_length = len(dataset)
    if args.num_partitions > 1:
        chunk_size = total_length // args.num_partitions
        start_idx = args.partition * chunk_size
        if args.partition == args.num_partitions - 1:
            end_idx = total_length
        else:
            end_idx = start_idx + chunk_size

        dataset.meta = dataset.meta[start_idx:end_idx]
        print(f"[Partitioning] total={total_length}, num_partitions={args.num_partitions}, "
              f"partition={args.partition}, start={start_idx}, end={end_idx}, subset_len={len(dataset)}")
    
    if args.compute_crossmodal:
        print(f'Computing cross-modal scores...')
        cross_score = get_cross_score(args, dataset)
    if args.compute_unimodal:
        print(f'Computing uni-modal scores...')
        uni_score = get_uni_score(args, dataset)

    print(f'Good Job Computer!')

if __name__ == '__main__':
    main()
