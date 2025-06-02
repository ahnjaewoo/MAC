#!/bin/zsh

# Usage:
# sh generate_evaluate_iter1.sh [DATASET_NAME] [DATA_SPLIT] [MODEL_NAME] [TEMPERATURE] [TOP_P] [GEN_MODE] [NUM_RETURN_SEQUENCES] [NUM_RETURN_SEQUENCES_AT_TEST_TIME] [NUM_ITERATIONS] [MODEL_CHECKPOINT_FNAME] [MODEL_CHECKPOINT_STEPS] [CROSSMODAL_MODEL] [SAMPLE_SFT_METHOD]

# Input arguments with defaults
DATASET_NAME=${1:-"coco"}
DATA_SPLIT=${2:-"test"}
MODEL_NAME=${3:-"llama3.1:8b"}
TEMPERATURE=${4:-0.7}
TOP_P=${5:-0.9}
GEN_MODE=${6:-"deceptive-general"}
NUM_RETURN_SEQUENCES=${7:-1}
NUM_RETURN_SEQUENCES_AT_TEST_TIME=${8:-"-1"}
NUM_ITERATIONS=${9:-1}
MODEL_CHECKPOINT_FNAME=${10:-"lr2e-4_lora_r16_lora_alpha32"}
MODEL_CHECKPOINT_STEPS=${11:-1000}
CROSSMODAL_MODEL=${12:-"clip"}
SAMPLE_SFT_METHOD=${13:-"random"}

# Echo input arguments
echo "=== Arguments ==="
echo "Dataset Name                  : $DATASET_NAME"
echo "Data Split                    : $DATA_SPLIT"
echo "Model Name                    : $MODEL_NAME"
echo "Temperature                   : $TEMPERATURE"
echo "Top-p                         : $TOP_P"
echo "Generation Mode               : $GEN_MODE"
echo "Num Return Sequences          : $NUM_RETURN_SEQUENCES"
echo "Num Return Sequences (Test)   : $NUM_RETURN_SEQUENCES_AT_TEST_TIME"
echo "Number of Iterations          : $NUM_ITERATIONS"
echo "Model Checkpoint Filename     : $MODEL_CHECKPOINT_FNAME"
echo "Model Checkpoint Steps        : $MODEL_CHECKPOINT_STEPS"
echo "Crossmodal Model              : $CROSSMODAL_MODEL"
echo "Sample SFT Method             : $SAMPLE_SFT_METHOD"
echo "==================="

echo "Executing generate_candidates.py..."
python dataset_processing/generate_candidates.py \
  --dataset_name "$DATASET_NAME" \
  --data_split "$DATA_SPLIT" \
  --model_name "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --gen_mode "$GEN_MODE" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
  --num_iterations "$NUM_ITERATIONS" \
  --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
  --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
  --crossmodal_model "$CROSSMODAL_MODEL" \
  --do_batch_decoding
  # IMPORTANT: If you get an OOM (Out-Of-Memory) error, remove the --do_batch_decoding option

# Conditionally activate LanguageBind
if [[ "$CROSSMODAL_MODEL" == "languagebind_video" || "$CROSSMODAL_MODEL" == "languagebind_audio" ]]; then
  echo "Activating conda environment: LanguageBind"
  conda activate LanguageBind
else
  echo "Using current conda environment: MAC"
fi

echo "Executing evaluate_scores.py with crossmodal model..."
python dataset_processing/evaluate_scores.py \
  --dataset_name "$DATASET_NAME" \
  --data_split "$DATA_SPLIT" \
  --gen_mode "$GEN_MODE" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
  --generation_model "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --num_iterations "$NUM_ITERATIONS" \
  --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
  --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
  --crossmodal_model "$CROSSMODAL_MODEL" \
  --compute_crossmodal

# Switch back if necessary
if [[ "$CROSSMODAL_MODEL" == "languagebind_video" || "$CROSSMODAL_MODEL" == "languagebind_audio" ]]; then
  echo "Re-activating conda environment: MAC"
  conda activate MAC
fi

echo "Executing evaluate_scores.py with unimodal models..."
for UNIMODAL_MODEL in roberta_large_mnli deberta_xlarge_mnli bart_large_mnli; do
  python dataset_processing/evaluate_scores.py \
    --dataset_name "$DATASET_NAME" \
    --data_split "$DATA_SPLIT" \
    --gen_mode "$GEN_MODE" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
    --generation_model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --num_iterations "$NUM_ITERATIONS" \
    --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
    --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
    --crossmodal_model "$CROSSMODAL_MODEL" \
    --unimodal_model "$UNIMODAL_MODEL" \
    --compute_unimodal
done

echo "Executing evaluate_deception.py..."
python dataset_processing/evaluate_deception.py \
  --dataset_name "$DATASET_NAME" \
  --data_split "$DATA_SPLIT" \
  --gen_mode "$GEN_MODE" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
  --generation_model "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --num_iterations "$NUM_ITERATIONS" \
  --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
  --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
  --crossmodal_model "$CROSSMODAL_MODEL"

echo "Executing evaluate_diversity.py..."
python dataset_processing/evaluate_diversity.py \
  --dataset_name "$DATASET_NAME" \
  --data_split "$DATA_SPLIT" \
  --gen_mode "$GEN_MODE" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
  --generation_model "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --num_iterations "$NUM_ITERATIONS" \
  --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
  --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
  --crossmodal_model "$CROSSMODAL_MODEL"

echo "Executing print_overall_results.py..."
python dataset_processing/print_overall_results.py \
  --dataset_name "$DATASET_NAME" \
  --data_split "$DATA_SPLIT" \
  --gen_mode "$GEN_MODE" \
  --num_return_sequences "$NUM_RETURN_SEQUENCES" \
  --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
  --generation_model "$MODEL_NAME" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --num_iterations "$NUM_ITERATIONS" \
  --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
  --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
  --crossmodal_model "$CROSSMODAL_MODEL"

# Only run this in training split
if [[ "$DATA_SPLIT" == "train" ]]; then
  echo "Executing sample_deception_dataset.py..."
  python dataset_processing/sample_deception_dataset.py \
    --dataset_name "$DATASET_NAME" \
    --data_split "$DATA_SPLIT" \
    --gen_mode "$GEN_MODE" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --num_return_sequences_at_test_time "$NUM_RETURN_SEQUENCES_AT_TEST_TIME" \
    --generation_model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --num_iterations "$NUM_ITERATIONS" \
    --model_checkpoint_fname "$MODEL_CHECKPOINT_FNAME" \
    --model_checkpoint_steps "$MODEL_CHECKPOINT_STEPS" \
    --crossmodal_model "$CROSSMODAL_MODEL" \
    --sample_sft_method "$SAMPLE_SFT_METHOD"
else
  echo "Skipping sample_deception_dataset.py (data_split=$DATA_SPLIT)"
fi

echo "✅ All tasks executed successfully."