#!/bin/zsh

DATASET_NAME=${1:-"audiocaps"}
CROSSMODAL_MODEL=${2:-"languagebind_audio"}

# =====================
# 0th Iteration
# =====================
sh scripts/generate_evaluate_iter0.sh $DATASET_NAME test llama3.1:8b 0.7 0.9 deceptive-general 4 0 $CROSSMODAL_MODEL

# =====================
# 1st Iteration
# =====================
# The following two commands can take a very long time to run,
# especially for large datasets like COCO or MSR-VTT (over 1 week on a single GPU):
#   (1) generate_evaluate_iter0.sh with train split
#   (2) train_iter0.sh for fine-tuning
#
# We recommend using **dataset partitioning** to parallelize them across multiple GPUs.
# Alternatively, you may skip training by using our provided fine-tuned checkpoints on HuggingFace.
sh scripts/generate_evaluate_iter0.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 64 0 $CROSSMODAL_MODEL maxent
sh scripts/train_iter0.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 64 4 0 lr2e-4_lora_r16_lora_alpha32_from-64-maxent $CROSSMODAL_MODEL maxent

# Note: The number `1000` below refers to the checkpoint step to evaluate.
# It must be set to the actual checkpoint you want to evaluate after training.
# Replace `1000` with the appropriate step number as needed.
sh scripts/generate_evaluate_iter1.sh $DATASET_NAME test llama3.1:8b 0.7 0.9 deceptive-general 4 -1 1 lr2e-4_lora_r16_lora_alpha32_from-64-maxent 1000 $CROSSMODAL_MODEL

# =====================
# 2nd Iteration (optional)
# =====================
# sh scripts/generate_evaluate_iter1.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 64 4 1 lr2e-4_lora_r16_lora_alpha32_from-64-maxent 1000 $CROSSMODAL_MODEL maxent
# sh scripts/train_iter1.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 4 1 lr2e-4_lora_r16_lora_alpha32_from-64-maxent $CROSSMODAL_MODEL 1000
# sh scripts/generate_evaluate_iter1.sh $DATASET_NAME test llama3.1:8b 0.7 0.9 deceptive-general 4 -1 2 lr2e-4_lora_r16_lora_alpha32_from-64-maxent 2000 $CROSSMODAL_MODEL

# =====================
# 3rd Iteration (optional)
# =====================
# sh scripts/generate_evaluate_iter1.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 64 4 2 lr2e-4_lora_r16_lora_alpha32_from-64-maxent 2000 $CROSSMODAL_MODEL maxent
# sh scripts/train_iter1.sh $DATASET_NAME train llama3.1:8b 0.7 0.9 deceptive-general 4 2 lr2e-4_lora_r16_lora_alpha32_from-64-maxent $CROSSMODAL_MODEL 2000
# sh scripts/generate_evaluate_iter1.sh $DATASET_NAME test llama3.1:8b 0.7 0.9 deceptive-general 4 -1 3 lr2e-4_lora_r16_lora_alpha32_from-64-maxent 3000 $CROSSMODAL_MODEL