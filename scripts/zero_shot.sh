#!/bin/zsh

DATASET_NAME=${1:-"audiocaps"}
CROSSMODAL_MODEL=${2:-"languagebind_audio"}
MODEL_NAME=${3:-"llama3.1:8b"}

# =====================
# 0th Iteration
# =====================
zsh scripts/generate_evaluate_iter0.sh $DATASET_NAME test $MODEL_NAME 0.7 0.9 deceptive-general 4 0 $CROSSMODAL_MODEL