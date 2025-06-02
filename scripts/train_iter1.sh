#!/bin/zsh

# Input arguments with defaults
DATASET_NAME=${1:-"coco"}
DATA_SPLIT=${2:-"train"}
MODEL_NAME=${3:-"llama3.1:8b"}
TEMPERATURE=${4:-0.7}
TOP_P=${5:-0.9}
GEN_MODE=${6:-"deceptive-general"}
NUM_RETURN_SEQUENCES=${7:-1}
NUM_ITERATIONS=${8:-1}
MODEL_CHECKPOINT_FNAME=${9:-"lr2e-4_lora_r16_lora_alpha32"}
CROSSMODAL_MODEL=${10:-"clip"}
RESUME_STEP=${11:-3}

# Echo input arguments for verification
echo "=== Arguments ==="
echo "Dataset Name                  : $DATASET_NAME"
echo "Data Split                    : $DATA_SPLIT"
echo "Model Name                    : $MODEL_NAME"
echo "Temperature                   : $TEMPERATURE"
echo "Top-p                         : $TOP_P"
echo "Generation Mode               : $GEN_MODE"
echo "Number of Sequences           : $NUM_RETURN_SEQUENCES"
echo "Number of Iterations          : $NUM_ITERATIONS"
echo "Model Checkpoint Filename     : $MODEL_CHECKPOINT_FNAME"
echo "Crossmodal Model              : $CROSSMODAL_MODEL"
echo "Resume Checkpoint ID          : $RESUME_STEP"
echo "==================="

# Only support llama3.1:8b for now
if [[ "$MODEL_NAME" != "llama3.1:8b" ]]; then
  echo "Error: Model '${MODEL_NAME}' is not implemented for fine-tuning."
  exit 1
fi

# Construct path strings
DATASET_PATH="dataset_processing/data/sft/${DATASET_NAME}/${DATA_SPLIT}/${GEN_MODE}-${NUM_RETURN_SEQUENCES}_${MODEL_NAME}_${TEMPERATURE}_${TOP_P}/iter${NUM_ITERATIONS}/${CROSSMODAL_MODEL}/${MODEL_CHECKPOINT_FNAME}_ckpt${RESUME_STEP}/instruction_dataset.json"

RESUME_CKPT="checkpoints/${DATASET_NAME}/${DATA_SPLIT}/${GEN_MODE}-${NUM_RETURN_SEQUENCES}_${MODEL_NAME}_${TEMPERATURE}_${TOP_P}/iter${NUM_ITERATIONS}/${CROSSMODAL_MODEL}/${MODEL_CHECKPOINT_FNAME}/checkpoint-${RESUME_STEP}"

OUTPUT_DIR="checkpoints/${DATASET_NAME}/${DATA_SPLIT}/${GEN_MODE}-${NUM_RETURN_SEQUENCES}_${MODEL_NAME}_${TEMPERATURE}_${TOP_P}/iter$((NUM_ITERATIONS + 1))/${CROSSMODAL_MODEL}/${MODEL_CHECKPOINT_FNAME}"

python train/sft.py \
    --dataset_name "${DATASET_PATH}" \
    --resume_from_checkpoint "${RESUME_CKPT}" \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --report_to "none" \
    --bf16 \
    --max_seq_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --use_peft \
    --attn_implementation "flash_attention_2" \
    --logging_steps=10 \
    --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}"