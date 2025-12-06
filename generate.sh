#!/bin/bash
#
# FluxFlow Image Generation Script
# Generates images from text prompts using trained model
#

# Load FluxFlow configuration and activate venv
if [ -f ".fluxflow_config" ]; then
    source .fluxflow_config
    if [ -d "$VENV_PATH" ]; then
        source $VENV_PATH/bin/activate
    fi
fi

# Configuration
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-outputs/flux/flxflow_final.safetensors}"
TEXT_PROMPTS_PATH="${TEXT_PROMPTS_PATH:-prompts}"
OUTPUT_PATH="${OUTPUT_PATH:-generated}"
IMG_SIZE="${IMG_SIZE:-512}"
DDIM_STEPS="${DDIM_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-4}"
VAE_DIM="${VAE_DIM:-128}"
FEAT_DIM="${FEAT_DIM:-128}"
TEXT_EMB_DIM="${TEXT_EMB_DIM:-1024}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║              FluxFlow Image Generation                       ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check if model exists
if [[ ! -f "$MODEL_CHECKPOINT" ]]; then
    echo -e "${YELLOW}[WARNING]${NC} Model checkpoint not found: $MODEL_CHECKPOINT"
    echo "Please train a model first using train.sh"
    exit 1
fi

# Check if prompts directory exists
if [[ ! -d "$TEXT_PROMPTS_PATH" ]]; then
    echo -e "${YELLOW}[WARNING]${NC} Prompts directory not found: $TEXT_PROMPTS_PATH"
    echo "Creating directory and example prompt..."
    mkdir -p "$TEXT_PROMPTS_PATH"
    cat > "$TEXT_PROMPTS_PATH/example.txt" << 'EOF'
A serene landscape with mountains and a lake at sunset
EOF
    echo -e "${GREEN}[OK]${NC} Created example prompt in $TEXT_PROMPTS_PATH/example.txt"
fi

# Count prompts
PROMPT_COUNT=$(find "$TEXT_PROMPTS_PATH" -name "*.txt" | wc -l | tr -d ' ')
echo -e "${GREEN}[INFO]${NC} Found $PROMPT_COUNT prompt(s) in $TEXT_PROMPTS_PATH"
echo -e "${GREEN}[INFO]${NC} Model: $MODEL_CHECKPOINT"
echo -e "${GREEN}[INFO]${NC} Output: $OUTPUT_PATH"
echo -e "${GREEN}[INFO]${NC} Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo -e "${GREEN}[INFO]${NC} Diffusion steps: $DDIM_STEPS"
echo

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run generation
echo -e "${BLUE}[STARTING]${NC} Generating images..."
echo

fluxflow-generate \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --text_prompts_path "$TEXT_PROMPTS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --img_size "$IMG_SIZE" \
    --ddim_steps "$DDIM_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --vae_dim "$VAE_DIM" \
    --feature_maps_dim "$FEAT_DIM" \
    --text_embedding_dim "$TEXT_EMB_DIM"

EXIT_CODE=$?

echo
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}[SUCCESS]${NC} Generation complete!"
    echo -e "${GREEN}[INFO]${NC} Images saved to: $OUTPUT_PATH"
    
    # Show generated files
    GENERATED_COUNT=$(find "$OUTPUT_PATH" -name "*_gen.webp" | wc -l | tr -d ' ')
    if [[ $GENERATED_COUNT -gt 0 ]]; then
        echo -e "${GREEN}[INFO]${NC} Generated $GENERATED_COUNT image(s)"
        echo
        echo "Generated files:"
        find "$OUTPUT_PATH" -name "*_gen.webp" -exec basename {} \; | sort
    fi
else
    echo -e "${YELLOW}[ERROR]${NC} Generation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Generation Complete                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
