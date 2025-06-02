# MAC (Multimodal Adversarial Compositionality)

Welcome! 👋
This is the official repository for our ACL 2025 main paper:

**Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**
by [Jaewoo Ahn*](https://ahnjaewoo.github.io/), [Heeseung Yun*](https://hs-yn.github.io/), [Dayoon Ko](https://dayoon-ko.github.io/), and [Gunhee Kim](https://vision.snu.ac.kr/gunhee/).

<p align="center">
    <img src="./img/main_figure.png" alt="main figure" width="100%" height="100%">
</p>

## Getting Started

### Environment Setup
We recommend using Anaconda. The following command will create a new conda environment mpchat with all the dependencies.
```bash
conda env create -f environment.yml
```

To activate the environment:
```bash
conda activate MAC
```

Additionally, install the English language model for spaCy:
```bash
python -m spacy download en_core_web_sm
```

### Dataset Preparation
Please refer to [DATASET.md](DATASET.md) for setup instructions for COCO, MSR-VTT, and AudioCaps

### Setup LanguageBind (Optional)
To use the `LanguageBind` model, please run the following commands:
```bash
git submodule update --init --recursive
cd dataset_processing/LanguageBind

# Create a separate environment for LanguageBind
conda create -n LanguageBind python=3.10.10
conda activate LanguageBind

# Install required dependencies
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install -r requirements.txt

# Return to the main environment
conda activate MAC
```

### How to Run
```bash
# Deceptive-General Prompt (Zero-Shot)
sh scripts/zero_shot.sh coco clip

# + Self-Train + Large-N Distilled + Diversity-Promoted (Ours)
sh scripts/self_train_with_large_N_ours.sh coco clip
```

### Load Pretrained Checkpoints from HuggingFace
LoRA checkpoint (ASR_total 42.1%) fine-tuned on LLaMA 3.1:8B to deceive CLIP on the COCO dataset.
#### Step 1: Modify `generate_candidates.py` 
Replace the model loading part with the following code:
```python
model = AutoPeftModelForCausalLM.from_pretrained(
    'ahnpersie/llama3.1-8b-lora-coco-deceptive-clip', # changed from local "model_checkpoint_dir" to HuggingFace repo
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    device_map=device
)
```
#### Step 2: Run evaluation
```bash
sh scripts/generate_evaluate_iter1_example.sh
```

> **Note**: All experiments were conducted on a single NVIDIA RTX A6000 GPU (48GB VRAM).

## Contact

If you have any questions, feel free to ask us: Jaewoo Ahn (jaewoo.ahn@vision.snu.ac.kr) or Heeseung Yun (heeseung.yun@vision.snu.ac.kr)