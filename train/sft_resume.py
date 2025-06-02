import os
from trl.commands.cli_utils import SFTScriptArguments, TrlParser
from datasets import load_dataset

import transformers
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint  # 최신 체크포인트 자동 검색 함수

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

import torch

transformers.logging.set_verbosity_info()

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    model_config.torch_dtype = torch.bfloat16

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("json", data_files=args.dataset_name)
    train_dataset = raw_datasets[args.dataset_train_split]

    ################
    # Resume checkpoint (if exists)
    ################
    last_ckpt = None

    if (
        os.path.isdir(training_args.output_dir)
        and os.listdir(training_args.output_dir)
    ):
        potential_ckpt = get_last_checkpoint(training_args.output_dir)
        if potential_ckpt is not None:
            print(f"Detected recent checkpoint: {potential_ckpt}")
            last_ckpt = potential_ckpt

    if training_args.resume_from_checkpoint is not None:
        print(f"Resume training from the checkpoint '{training_args.resume_from_checkpoint}'.")
        last_ckpt = training_args.resume_from_checkpoint

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # Add this line to move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)

    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(training_args.output_dir)

    print(f'Good Job Computer!')