#!/bin/bash
set -e

# Load FluxFlow configuration and activate venv
if [ -f ".fluxflow_config" ]; then
    source .fluxflow_config
    if [ -d "$VENV_PATH" ]; then
        source $VENV_PATH/bin/activate
    fi
fi

if [ -f "config.local.sh" ]; then
    source config.local.sh
else
    echo "Error: config.local.sh not found!"
    echo "Please copy config.example.sh to config.local.sh and customize it."
    exit 1
fi

if [ "$USE_TT2M" = true ]; then
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "Error: HUGGINGFACE_TOKEN not set in config.local.sh"
        exit 1
    fi
    USE_TT2M_FLAG="--use_tt2m"
    TT2M_TOKEN_FLAG="--tt2m_token $HUGGINGFACE_TOKEN"
else
    if [ ! -d "$DATA_PATH" ]; then
        echo "Error: DATA_PATH directory not found: $DATA_PATH"
        exit 1
    fi
    if [ ! -f "$DATASET" ]; then
        echo "Error: DATASET file not found: $DATASET"
        exit 1
    fi
    USE_TT2M_FLAG=""
    TT2M_TOKEN_FLAG=""
fi

mkdir -p "$OUTPUT_FOLDER"

TRAIN_FLAGS=""
[ "$TRAIN_VAE" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --train_vae"
[ "$GAN_TRAINING" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --gan_training"
[ "$TRAIN_SPADE" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --train_spade"
[ "$TRAIN_DIFF_FULL" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --train_diff_full"
[ "$TRAIN_DIFF" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --train_diff"
[ "$PRESERVE_LR" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --preserve_lr"
[ "$REBUILD_CACHE" = true ] && TRAIN_FLAGS="$TRAIN_FLAGS --rebuild_cache"

KL_FREE_BITS=$(echo "${VAE_DIM} * ${KL_FREE_BITS_MULTIPLIER}" | bc)

CAPTION_ARGS=""
for caption in "${SAMPLE_CAPTIONS[@]}"; do
    CAPTION_ARGS="$CAPTION_ARGS \"$caption\""
done

SEED_FLAG=""
[ -n "$SEED" ] && SEED_FLAG="--seed $SEED"

OPTIM_SCHED_CONFIG_FLAG=""
[ -n "$OPTIM_SCHED_CONFIG" ] && [ -f "$OPTIM_SCHED_CONFIG" ] && OPTIM_SCHED_CONFIG_FLAG="--optim_sched_config $OPTIM_SCHED_CONFIG"

fluxflow-train \
    --data_path "$DATA_PATH" \
    --captions_file "$DATASET" \
    $TRAIN_FLAGS \
    $USE_TT2M_FLAG \
    $TT2M_TOKEN_FLAG \
    --kl_free_bits "$KL_FREE_BITS" \
    --kl_warmup_steps "$KL_WARMUP_STEPS" \
    --output_path "$OUTPUT_FOLDER" \
    --n_epochs "$EPOCHS" \
    --training_steps "$TRAINING_STEPS" \
    --batch_size "$BATCH" \
    --feature_maps_dim "$FEAT_DIM" \
    --feature_maps_dim_disc "$FEAT_DIM_DISC" \
    --vae_dim "$VAE_DIM" \
    --lr "$LR" \
    --lr_min "$LR_MIN" \
    --checkpoint_save_interval "$CHECKPOINT_SAVE_INTERVAL" \
    --samples_per_checkpoint "$SAMPLES_PER_CHECKPOINT" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --log_interval 1 \
    --workers "$WORKERS" \
    --pretrained_bert_model "$PRETRAINED_BERT" \
    --test_image_address "$TEST_IMAGE" \
    $SEED_FLAG \
    $OPTIM_SCHED_CONFIG_FLAG \
    --sample_captions $CAPTION_ARGS
