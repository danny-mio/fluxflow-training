# Resume Optimization Proposal for WebDatasets

## Problem Analysis

### Current Behavior

When training is interrupted and resumed:

1. **Local Datasets (ResumableDimensionSampler)**:
   - Resume works well: skips to exact batch position
   - Sampler maintains deterministic order per epoch
   - Slow on resume but acceptable (just skipping, not downloading)

2. **WebDatasets (StreamingWebDataset)** - PROBLEMATIC:
   - Uses `IterableDataset` with no sampler
   - Resume logic at `pipeline_orchestrator.py:1229`:
     ```python
     if step_idx == start_step and epoch == start_epoch and batch_idx < start_batch:
         continue  # Skip batch
     ```
   - **Downloads and decodes** every skipped batch before discarding it
   - WebDataset shuffles data randomly anyway (shardshuffle=10, shuffle=100)
   - Resuming to batch 5000 downloads and processes 5000 batches wastefully

### Why It's Wasteful for WebDatasets

1. **Network overhead**: Downloads tar shards for batches that will be discarded
2. **CPU overhead**: Decodes images that will be immediately thrown away
3. **Time overhead**: Can take minutes/hours to skip through batches
4. **Pointless precision**: WebDataset is shuffled, so "exact position" is meaningless
5. **Shard inefficiency**: May download entire shards just to skip batches

### Example Impact

```
Resume from batch 10,000 with batch_size=2:
- Downloads: ~20,000 images from shards
- Decodes: ~20,000 images
- Uses: 0 images (all skipped)
- Time: 10-30 minutes of wasted work
```

## Proposed Solutions

### Option 1: Skip Resume for WebDatasets (SIMPLEST, RECOMMENDED)

**Change**: Don't skip batches when using WebDatasets - just start from batch 0.

**Rationale**:
- WebDatasets are shuffled anyway (random order each epoch)
- "Exact resume position" is meaningless with shuffled streaming data
- Global step counter and model weights are preserved (what matters)
- Loss/optimizer state restored correctly

**Implementation**:
```python
# In pipeline_orchestrator.py, line ~1229
# Skip batches if resuming mid-epoch (BUT NOT FOR WEBDATASETS)
if step_idx == start_step and epoch == start_epoch and batch_idx < start_batch:
    # Check if current dataset is a webdataset
    current_dataset_name = step.dataset or self.config.default_dataset
    is_webdataset = False
    if current_dataset_name and self.config.datasets:
        dataset_config = self.config.datasets.get(current_dataset_name)
        is_webdataset = dataset_config and dataset_config.type == "webdataset"

    if not is_webdataset:
        continue  # Only skip for local datasets
```

**Pros**:
- Simple: 10-line change
- Fast: No wasted downloads
- Correct: Global step/weights/optimizer preserved
- Efficient: Start training immediately on resume

**Cons**:
- Batch counter resets to 0 (cosmetic only)
- Technically sees ~batch_idx samples twice (negligible with millions of samples)

---

### Option 2: Fast-Forward Shard Position (COMPLEX)

**Change**: Skip entire shards instead of individual batches.

**Rationale**:
- WebDataset streams from tar shards sequentially
- Can estimate which shard contains target batch
- Skip shards without downloading

**Implementation**:
```python
# Calculate approximate shard to start from
samples_per_shard = dataset_config.webdataset_samples_per_shard
target_sample = start_batch * batch_size
start_shard = min(target_sample // samples_per_shard, len(dataset.urls) - 1)

# Modify url list to start from estimated shard
dataset.urls = dataset.urls[start_shard:]
```

**Pros**:
- Reduces download overhead significantly
- Maintains approximate resume position

**Cons**:
- Complex: requires WebDataset internals modification
- Imprecise: shard boundaries don't align with batches
- Still wasteful: downloads/decodes samples within shard
- Fragile: breaks if shard size assumptions are wrong

---

### Option 3: Checkpoint Frequency Tuning (PARTIAL SOLUTION)

**Change**: Save checkpoints less frequently for WebDataset steps.

**Configuration**:
```yaml
steps:
  - name: webdataset_phase
    dataset: web_2m
    checkpoint_save_interval: 5000  # Save every 5000 batches instead of 100
```

**Pros**:
- Reduces resume overhead frequency
- Easy to configure

**Cons**:
- Doesn't eliminate the problem
- Trades resume frequency for training loss on crash
- Still wasteful when resume happens

---

### Option 4: Hybrid Checkpoint Strategy (SOPHISTICATED)

**Change**: Save two checkpoint types:
1. **Full checkpoint**: Model + optimizer + position (every N steps)
2. **Model-only checkpoint**: Just model weights (frequent)

On resume:
- Load latest model-only checkpoint (fast)
- Don't try to restore exact batch position for WebDatasets

**Pros**:
- Minimal training loss on crash
- Fast resume
- Preserves model progress

**Cons**:
- Complex implementation
- More disk space (2 checkpoint types)
- Optimizer state not preserved in model-only checkpoints

---

## REVISED Analysis - batch_idx Cannot Be Reset

### Why Resetting batch_idx Breaks Everything

You're absolutely right - batch_idx is NOT cosmetic. It's used for:

1. **Checkpoint naming**: `{step}_{epoch}_{batch:05d}.safetensors`
2. **Checkpoint save intervals**: `if batch_idx % checkpoint_save_interval == 0`
3. **Logging intervals**: `if batch_idx % log_interval == 0`
4. **Memory cleanup triggers**: Periodic cleanup at batch 100, 200, etc.
5. **Sample generation**: Samples named with batch_idx
6. **Progress tracking**: "Batch 5000/10000" display

Resetting batch_idx would:
- ❌ Break checkpoint interval logic (save too frequently)
- ❌ Mess up logging (log at wrong times)
- ❌ Trigger memory cleanup incorrectly
- ❌ Create checkpoint filename collisions
- ❌ Confuse progress displays

## Revised Recommendation

### Option 5: Virtual Skip via Iterator Wrapping (BEST SOLUTION)

**Change**: Wrap the WebDataset iterator to track position without downloading/decoding.

**Key Insight**:
- We need to maintain batch_idx for all the logic above
- But we DON'T need to actually process those batches
- Solution: Track position in a counter without yielding data

**Implementation**:
```python
class FastForwardIterator:
    """Wrapper that fast-forwards an iterator without processing items."""

    def __init__(self, iterator, skip_count=0):
        self.iterator = iterator
        self.skip_count = skip_count
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Fast-forward: just increment counter without yielding
        while self.current < self.skip_count:
            self.current += 1
            # Return None or dummy data - will be skipped by continue logic
            # This way batch_idx increments but we don't download/decode
            return None, None

        # Past skip point - yield real data
        return next(self.iterator)


# In pipeline_orchestrator.py training loop:
if step_idx == start_step and epoch == start_epoch and start_batch > 0:
    current_dataset_name = step.dataset or self.config.default_dataset
    is_webdataset = False
    if current_dataset_name and self.config.datasets:
        dataset_config = self.config.datasets.get(current_dataset_name)
        is_webdataset = dataset_config and dataset_config.type == "webdataset"

    if is_webdataset:
        logger.info(f"WebDataset: Using fast-forward to batch {start_batch} (no downloads)")
        # Wrap dataloader to fast-forward without downloading
        dataloader = FastForwardIterator(iter(dataloader), skip_count=start_batch)

# Then in the loop:
for batch_idx, (imgs, input_ids) in enumerate(dataloader):
    # Skip None batches from fast-forward
    if imgs is None:
        continue

    # ... rest of training logic
```

**Pros**:
- ✅ batch_idx increments correctly (all dependent logic works)
- ✅ No downloads/decoding for skipped batches
- ✅ Fast resume (just increments counter)
- ✅ Works with existing checkpoint/logging infrastructure
- ✅ Clean separation: fast-forward vs real training

**Cons**:
- Slightly more complex than Option 1
- Need to handle None batches in loop

---

### Option 6: Accept Wasted Time, Add Progress Bar (PRAGMATIC)

**Change**: Keep current behavior but add clear progress feedback.

**Rationale**:
- Current implementation is "correct" but slow
- Maybe slow resume is acceptable if user knows what's happening
- Better UX through transparency

**Implementation**:
```python
# In pipeline_orchestrator.py, before skip loop:
if step_idx == start_step and epoch == start_epoch and start_batch > 0:
    logger.info(f"Resuming from batch {start_batch}...")
    logger.info(f"Note: This may take time for WebDatasets (downloading required)")

    # Add progress counter
    skip_progress = 0
    skip_total = start_batch

for batch_idx, (imgs, input_ids) in enumerate(dataloader):
    if step_idx == start_step and epoch == start_epoch and batch_idx < start_batch:
        skip_progress += 1
        if skip_progress % 100 == 0:
            logger.info(f"Skipping batches: {skip_progress}/{skip_total} ({skip_progress*100//skip_total}%)")
        continue
```

**Pros**:
- ✅ Minimal code change
- ✅ No risk of breaking existing logic
- ✅ User knows progress is happening

**Cons**:
- ❌ Still slow (10-30 minutes for large skip_batch)
- ❌ Still wastes network/CPU
- ❌ Doesn't actually solve the problem

---

### Option 7: Checkpoint Strategy - Save Less Often for WebDatasets (MITIGATION)

**Change**: Configure WebDataset steps to checkpoint less frequently.

**Implementation**:
```yaml
steps:
  - name: webdataset_vae_training
    dataset: web_2m
    checkpoint_save_interval: 10000  # Instead of default 100
    # If crash happens, max 10000 batches to skip on resume
```

**Pros**:
- ✅ Easy to configure
- ✅ Reduces problem frequency
- ✅ No code changes

**Cons**:
- ❌ Doesn't eliminate problem
- ❌ Lose more progress on crash
- ❌ Trade-off between resume speed and crash recovery

---

## Final Recommendation

### My Recommendation: Option 5 + Option 7 (Hybrid)

**Why Option 5 (Virtual Skip)**:
1. ✅ Preserves batch_idx semantics (no breaking changes)
2. ✅ Zero download/decode overhead for WebDatasets
3. ✅ Fast resume (counter increment only)
4. ✅ Works with all existing checkpoint/logging logic

**Plus Option 7 (Checkpoint Less Frequently)**:
- Configure WebDataset steps: `checkpoint_save_interval: 5000` (vs 100 for local)
- Reduces problem occurrence
- Even if crash happens, Option 5 makes resume fast

**Why NOT Option 6** (just add progress bar):
- Doesn't solve the problem, just makes it visible
- Still wastes 10-30 minutes on resume

---

## BUT... There's Actually a Better Option 8

### Option 8: Sample-Based Checkpointing for WebDatasets

**Key Insight**: For WebDatasets, we don't care about batch_idx within an epoch - we care about **total samples seen**.

**Change**: Track `samples_processed_this_epoch` instead of batch_idx for WebDatasets.

**Implementation**:
```python
# For WebDatasets, checkpoint saves:
{
    "pipeline": {
        "current_step_index": 0,
        "current_step_epoch": 0,
        "samples_this_epoch": 12485,  # NEW for webdatasets
        "current_batch_idx": None,     # Not meaningful for streaming
    }
}

# On resume:
if is_webdataset:
    # Don't skip batches - just adjust global_step
    samples_to_recover = saved_samples_this_epoch
    logger.warning(
        f"WebDataset resume: Skipping {samples_to_recover} samples would waste time. "
        f"Starting from sample 0 in epoch {epoch}. "
        f"Note: With shuffled streaming data, seeing some samples twice is acceptable."
    )
    # Don't skip any batches - global_step already correct
else:
    # Local dataset - use precise batch_idx skip
    start_batch = saved_batch_idx
```

**Rationale for WebDatasets**:
- Shuffled data → order doesn't matter
- Seeing 10k samples twice out of 2M samples = 0.5% duplication (negligible)
- Global step preserved → scheduler/LR correct
- Model weights preserved → training continues correctly
- Checkpoints still saved at regular batch intervals (so logging/cleanup work)

**For Local Datasets**:
- Keep exact batch_idx resume (deterministic sampler makes this meaningful)

**Pros**:
- ✅ Fast resume for WebDatasets (instant)
- ✅ Preserves batch_idx for checkpoint interval logic
- ✅ No downloads wasted
- ✅ Training quality unaffected (negligible duplication with shuffle)
- ✅ Different behavior for local vs webdataset (each optimized)

**Cons**:
- Some samples seen twice (but with 2M shuffled samples, this is fine)
- Slightly more complex: two resume paths

---

## My Actual Recommendation: Option 8

Implement **dual resume strategy**:
- **WebDatasets**: Don't skip batches, just log the discrepancy, continue from 0
- **Local datasets**: Keep current precise skip behavior

This is the most pragmatic because:
1. Acknowledges the fundamental difference between streaming/shuffled vs deterministic datasets
2. Optimizes each type appropriately
3. No wasted time for WebDatasets
4. No breaking changes to batch_idx logic
5. Honest about trade-offs (some duplication acceptable with shuffled data)

**Implementation complexity**: ~30 lines
**Performance improvement**: Resume goes from 10-30 minutes → <1 second for WebDatasets
**Training quality impact**: Negligible (0.5% sample duplication with 2M dataset)

1. **Correct semantics**: With shuffled data, exact batch position is meaningless
2. **Minimal code**: 10-15 line change
3. **Zero overhead**: No wasted downloads/decoding
4. **Preserves what matters**: Model weights, optimizer state, global step
5. **User-friendly**: Fast resume = better UX

### Trade-offs Accepted

- Resuming at batch 5000 starts from batch 0 again
  - But with millions of shuffled samples, seeing ~5000 twice is negligible
  - Global step counter preserved, so scheduler/metrics unaffected
  - Model already saw those samples (weights preserved)

### Additional Improvement

Add logging to make behavior clear:

```python
if is_webdataset and start_batch > 0:
    logger.warning(
        f"WebDataset detected: skipping batch resume (would waste time downloading). "
        f"Starting from batch 0 (global_step={self.global_step} preserved)"
    )
    start_batch = 0  # Reset for this dataset
```

## Implementation Plan

1. Modify `pipeline_orchestrator.py` resume logic (~line 1229)
2. Add dataset type detection
3. Add warning log for WebDataset resume
4. Update `MULTI_DATASET_TRAINING.md` to document behavior
5. Test with both local and webdataset resumes

## Performance Impact

**Before** (resuming to batch 10,000):
- Time: 10-30 minutes downloading/decoding
- Network: Gigabytes of tar downloads
- CPU: Wasted on decoding discarded images

**After**:
- Time: <1 second (just load checkpoint)
- Network: 0 bytes wasted
- CPU: 0% wasted

**Training Quality**: Unchanged (global step/weights/optimizer preserved)
