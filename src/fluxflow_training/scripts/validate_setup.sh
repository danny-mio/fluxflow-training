#!/bin/bash

echo "=== FluxFlow Setup Validation ==="
echo

errors=0

echo "1. Checking Python version..."
python_version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [ "$(echo "$python_version >= 3.10" | bc)" -eq 1 ]; then
    echo "   ✓ Python $python_version (OK)"
else
    echo "   ✗ Python $python_version (Requires 3.10+)"
    errors=$((errors + 1))
fi

echo
echo "2. Checking required packages..."
packages=("torch" "torchvision" "transformers" "diffusers" "safetensors" "einops" "webdataset")
for pkg in "${packages[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "   ✓ $pkg"
    else
        echo "   ✗ $pkg (missing)"
        errors=$((errors + 1))
    fi
done

echo
echo "3. Checking configuration..."
if [ -f "config.local.sh" ]; then
    echo "   ✓ config.local.sh exists"
    source config.local.sh

    if [ "$USE_TT2M" != true ]; then
        if [ ! -z "$DATA_PATH" ] && [ -d "$DATA_PATH" ]; then
            echo "   ✓ DATA_PATH exists: $DATA_PATH"
        else
            echo "   ✗ DATA_PATH not found: $DATA_PATH"
            errors=$((errors + 1))
        fi

        if [ ! -z "$DATASET" ] && [ -f "$DATASET" ]; then
            echo "   ✓ DATASET file exists: $DATASET"
        else
            echo "   ✗ DATASET file not found: $DATASET"
            errors=$((errors + 1))
        fi
    else
        if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
            echo "   ✓ HUGGINGFACE_TOKEN set"
        else
            echo "   ✗ HUGGINGFACE_TOKEN not set"
            errors=$((errors + 1))
        fi
    fi
else
    echo "   ✗ config.local.sh not found"
    echo "     Copy config.example.sh to config.local.sh and customize it"
    errors=$((errors + 1))
fi

echo
echo "4. Checking tokenizer cache..."
if python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./_cache', local_files_only=True)" 2>/dev/null; then
    echo "   ✓ Tokenizer cache ready"
else
    echo "   ⚠ Tokenizer cache not found"
    echo "     Run: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./_cache')\""
fi

echo
echo "5. Checking device availability..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "   ✓ CUDA available: $gpu_name"
elif python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo "   ✓ MPS (Apple Silicon) available"
else
    echo "   ⚠ No GPU detected, will use CPU (slow)"
fi

echo
echo "6. Checking output directory..."
if [ -f "config.local.sh" ]; then
    source config.local.sh
    if [ ! -z "$OUTPUT_FOLDER" ]; then
        mkdir -p "$OUTPUT_FOLDER" 2>/dev/null
        if [ -d "$OUTPUT_FOLDER" ]; then
            echo "   ✓ Output directory: $OUTPUT_FOLDER"
        else
            echo "   ✗ Cannot create output directory: $OUTPUT_FOLDER"
            errors=$((errors + 1))
        fi
    fi
fi

echo
echo "==================================="
if [ $errors -eq 0 ]; then
    echo "✓ Setup validation passed!"
    echo "Ready to train. Run: ./train.sh"
else
    echo "✗ Found $errors error(s)"
    echo "Please fix the issues above before training"
    exit 1
fi
