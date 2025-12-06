#!/bin/bash
#
# FluxFlow Training Configuration Template
# Copy this file to config.local.sh and customize for your environment
#

# Dataset Configuration
DATA_PATH="/path/to/your/images"
DATASET="/path/to/your/captions.txt"

# Alternative: TTI-2M Dataset Configuration
# Uncomment to use TTI-2M streaming dataset instead of local files
# USE_TT2M=true
# HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Model Configuration
# Recommended: 32 for limited resources, 128 for better quality, 256 for high-end GPUs
VAE_DIM="${VAE_DIM:-32}"
FEAT_DIM="${FEAT_DIM:-32}"
FEAT_DIM_DISC="${FEAT_DIM_DISC:-32}"

# Training Configuration
LR="1e-5"
LR_MIN="1e-2"
EPOCHS=100
BATCH=1
TRAINING_STEPS=1
WORKERS=8

# Output Configuration
OUTPUT_FOLDER="outputs/flux"
CHECKPOINT_SAVE_INTERVAL=20
SAMPLES_PER_CHECKPOINT=1

# Training Modes
TRAIN_VAE=true
GAN_TRAINING=true
TRAIN_SPADE=true
TRAIN_DIFF_FULL=false
TRAIN_DIFF=false

# KL Divergence Settings
KL_FREE_BITS_MULTIPLIER=2
KL_WARMUP_STEPS=0

# Model Checkpoint
MODEL_CHECKPOINT="${OUTPUT_FOLDER}/flxflow_final.safetensors"
PRETRAINED_BERT="distilbert-base-uncased"
PRESERVE_LR=true

# Test Image for VAE Validation
TEST_IMAGE="template.png"

# Dimension Cache and Resume Settings
# Cache directory for image dimension metadata (speeds up training startup)
DIMENSION_CACHE_DIR=".cache/dimensions"

# Force rebuild dimension cache (set to true to rescan all images)
REBUILD_CACHE=false

# Dimension grouping granularity - images rounded to this multiple for batching
# Higher values = fewer groups but more size variation within batches
# Lower values = more groups but tighter size matching
DIMENSION_MULTIPLE=32

# Random seed for reproducible training (leave empty for random)
SEED=""

# Optimizer and Scheduler Configuration (Optional)
# Path to JSON file with per-model optimizer/scheduler settings
# Leave empty to use default configuration
# OPTIM_SCHED_CONFIG="${OUTPUT_FOLDER}/optim_sched_config.json"
OPTIM_SCHED_CONFIG=""

# Sample Generation Prompts
SAMPLE_CAPTIONS=(
    "illustration of a boy playing guitar in the forest in autumn"
    "photo of a girl elf with blue Santa costume by a cottage in the snow"
    "on a yellow background, a blue triangle on the bottom"
    "A white circle is positioned in the top-left corner on a black background"
    "an orange banana"
    "a pink banana"
    "a green banana"
)
