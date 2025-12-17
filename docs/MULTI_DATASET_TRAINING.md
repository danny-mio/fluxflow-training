# Multi-Dataset Pipeline Training

This guide explains how to configure and use multiple datasets within a single training pipeline, allowing you to train different phases with different data sources.

## Overview

Multi-dataset support allows you to:
- Define multiple named datasets (local or webdatasets)
- Assign specific datasets to different pipeline steps
- Switch between datasets during training
- Mix local and remote (webdataset) data sources
- Override batch size and workers per dataset

## Configuration

### Basic Structure

```yaml
training:
  pipeline:
    # 1. Define multiple datasets
    datasets:
      dataset_name_1:
        type: local  # or 'webdataset'
        # ... dataset-specific config
      dataset_name_2:
        type: webdataset
        # ... dataset-specific config
    
    # 2. Set default dataset (optional)
    default_dataset: dataset_name_1
    
    # 3. Assign datasets to steps
    steps:
      - name: step1
        dataset: dataset_name_1  # Use specific dataset
        # ... other step config
      
      - name: step2
        dataset: dataset_name_2  # Switch dataset
        # ... other step config
      
      - name: step3
        # No dataset specified - uses default_dataset
        # ... other step config
```

### Dataset Types

#### Local Dataset

```yaml
datasets:
  my_local_data:
    type: local
    image_folder: /path/to/images
    captions_file: /path/to/captions.txt
    batch_size: 4        # Optional: override step batch_size
    workers: 8           # Optional: override step workers
```

**Required fields**:
- `image_folder`: Path to folder containing images
- `captions_file`: Path to text file with captions

#### WebDataset

```yaml
datasets:
  my_webdataset:
    type: webdataset
    webdataset_url: "pipe:aws s3 cp s3://bucket/data/{00000..00099}.tar -"
    webdataset_token: hf_your_token_here
    webdataset_image_key: png           # Optional, default: "png"
    webdataset_label_key: json          # Optional, default: "json"
    webdataset_caption_key: prompt      # Optional, default: "prompt"
    webdataset_size: 100000             # Optional, default: 10000
    webdataset_samples_per_shard: 1000  # Optional, default: 1000
    batch_size: 4                       # Optional
    workers: 8                          # Optional
```

**Required fields**:
- `webdataset_url`: URL pattern for webdataset shards
- `webdataset_token`: Authentication token (e.g., HuggingFace token)

## Use Cases

### 1. Progressive Training (Low-Res → High-Res)

Train on low-resolution data first for speed, then switch to high-resolution:

```yaml
datasets:
  lowres_dataset:
    type: local
    image_folder: /data/256x256
    captions_file: /data/captions_lowres.txt
    batch_size: 8
  
  highres_dataset:
    type: local
    image_folder: /data/1024x1024
    captions_file: /data/captions_highres.txt
    batch_size: 2

steps:
  - name: vae_lowres
    dataset: lowres_dataset
    train_vae: true
    n_epochs: 10
  
  - name: vae_highres
    dataset: highres_dataset
    train_vae: true
    n_epochs: 5
```

### 2. Domain-Specific Training

Train different components on different domains:

```yaml
datasets:
  faces_dataset:
    type: local
    image_folder: /data/faces
    captions_file: /data/captions_faces.txt
  
  landscapes_dataset:
    type: local
    image_folder: /data/landscapes
    captions_file: /data/captions_landscapes.txt

steps:
  - name: vae_faces
    dataset: faces_dataset
    train_vae: true
    n_epochs: 10
  
  - name: flow_combined
    dataset: landscapes_dataset
    train_diff: true
    n_epochs: 20
    freeze: [compressor, expander]
```

### 3. Local + Remote Data

Combine local datasets with cloud-hosted webdatasets:

```yaml
datasets:
  local_curated:
    type: local
    image_folder: /data/curated
    captions_file: /data/curated.txt
    batch_size: 2
  
  remote_large:
    type: webdataset
    webdataset_url: "https://huggingface.co/datasets/my-dataset/data-{000..999}.tar"
    webdataset_token: ${HF_TOKEN}
    webdataset_size: 1000000
    batch_size: 8

steps:
  - name: pretrain_remote
    dataset: remote_large
    train_vae: true
    n_epochs: 50
  
  - name: finetune_local
    dataset: local_curated
    train_vae: true
    n_epochs: 10
```

## Dataset Selection Rules

1. **Explicit dataset**: If `step.dataset` is specified, use that dataset
2. **Default dataset**: If `step.dataset` is None and `default_dataset` is set, use default
3. **Fallback**: If neither is set, use the original args-based dataset initialization

## Batch Size and Workers

Both datasets and steps can specify `batch_size` and `workers`. The configuration resolution follows this priority:

```
Priority (highest to lowest):
1. step.batch_size / step.workers (most specific)
2. dataset.batch_size / dataset.workers (dataset-level override)
3. defaults.batch_size / defaults.workers (pipeline defaults)
4. args.batch_size / args.workers (global training config)
5. Hard-coded defaults (batch_size=2, workers=8)
```

**Recommendation**: Set `batch_size` in the dataset config for dataset-specific requirements, and override at the step level only when a specific step needs different batching:

```yaml
datasets:
  high_res:
    type: local
    image_folder: /data/highres
    captions_file: /data/captions.txt
    batch_size: 2  # High-res images need smaller batches

steps:
  - name: warmup
    dataset: high_res
    # Uses batch_size=2 from dataset
  
  - name: main_training
    dataset: high_res
    batch_size: 4  # Override to 4 for this step only
```

## Validation

The pipeline validator checks:
- ✅ Dataset types are valid ('local' or 'webdataset')
- ✅ Required fields are present for each type
- ✅ `default_dataset` exists in `datasets` dict
- ✅ Step `dataset` references exist in `datasets` dict
- ✅ WebDataset has token and URL
- ✅ Local dataset has image_folder and captions_file

## Example: Complete Multi-Dataset Pipeline

See `examples/multi_dataset_pipeline.yaml` for a complete working example.

## API Usage

Access dataset configuration in Python:

```python
from fluxflow_training.training.pipeline_config import parse_pipeline_config

# Parse configuration
config = parse_pipeline_config(yaml_dict["training"]["pipeline"])

# Access datasets
for name, dataset in config.datasets.items():
    print(f"Dataset: {name}, Type: {dataset.type}")

# Check step datasets
for step in config.steps:
    dataset_name = step.dataset or config.default_dataset
    if dataset_name:
        dataset = config.datasets[dataset_name]
        print(f"Step {step.name} uses {dataset_name} ({dataset.type})")
```

## Migration from Single Dataset

If you have existing single-dataset configurations, they will continue to work. Multi-dataset is opt-in:

```yaml
# Old style (still works)
training:
  pipeline:
    steps:
      - name: my_step
        train_vae: true
        batch_size: 4

# Uses dataset from args (--image_folder, --captions_file)
```

## Limitations

- Dataset switching requires full dataloader recreation
- Samplers are recreated per step (resume state may be lost)
- Mixed local/webdataset in same step not supported

## Resume Behavior

When training is interrupted and resumed mid-epoch:

- **Skip mechanism**: Dataloader wrapped to yield None for skipped batches without fetching data
- **batch_idx preservation**: Batch counter increments correctly via enumerate(), preserving checkpoint/logging intervals  
- **Minimal overhead**: Skipped batches return None without downloading tar files or decoding images
- **Works for all dataset types**: Both local and webdataset benefit from this optimization

How it works:
```python
# FastForwardDataLoader yields (None, None) for first N batches
# Training loop skips None batches with continue
for batch_idx, (imgs, input_ids) in enumerate(dataloader_wrapper):
    if imgs is None:  # Skipped batch
        continue  # batch_idx still increments
    # Real training starts here
```

Note: For WebDatasets, this avoids network downloads. For local datasets with ResumableDimensionSampler, 
both the sampler's skip logic AND the wrapper's skip logic apply sequentially.

## Troubleshooting

### "Dataset 'X' not found"

Ensure the dataset name in `step.dataset` matches a key in `datasets`:

```yaml
datasets:
  my_data:  # ← Name must match
    type: local
    ...

steps:
  - dataset: my_data  # ← Reference here
```

### "default_dataset not found in datasets"

Ensure `default_dataset` references an existing dataset:

```yaml
datasets:
  dataset1: ...
  dataset2: ...

default_dataset: dataset1  # ← Must be 'dataset1' or 'dataset2'
```

### WebDataset authentication errors

Ensure `webdataset_token` is valid and has access to the dataset URL.
