# Research Plan: Fine-tuning Hallo3 with TalkVid Dataset

**Research Question:** Does fine-tuning Hallo3 on the high-quality TalkVid dataset improve talking head video generation quality compared to the original pretrained model?

**Model:** Hallo3 (based on CogVideoX-5B image-to-video diffusion model)  
**Dataset:** TalkVid (181,752 video clips with face detection, audio embeddings, and quality scores)  
**Cluster:** NEU Explorer — Slurm, H200/H100 GPUs

---

## Current Status (as of 2026-07-20)

### ✅ Completed

| Task | Details |
|------|---------|
| Environment setup | conda env at `/home/li.qianyi/envs/hallo`, Python 3.10, all deps installed |
| Data format investigation | Mapped TalkVid `clips_flat/` structure to Hallo3 input format |
| Data conversion script | `scripts/convert_talkvid_to_hallo3.py` — converts 42 pilot clips |
| Meta index generation | `data/talkvid.json` — 42 samples indexed |
| Training config | `configs/sft_s1.yaml` updated with data path, AdamW optimizer fix |
| **Pipeline validation** | 100-iter Stage 1 training ran without errors on H200 |
| **Checkpoint saved** | `stage-1/train-stage-1-07-20-02-38/100/mp_rank_00_model_states.pt` |
| **Inference verified** | Generated videos from fine-tuned checkpoint (2 reference images × 4 audio clips) |

### Key engineering fixes resolved
- `FusedEmaAdam` not compiled for H200 → replaced with `AdamW`
- SAT `latest` metadata file required for checkpoint loading
- `nproc_per_node` must match actual GPU count (`1` for single-GPU runs)
- `pyav==14.0.1` broken package → use `requirements_fixed.txt`

---

## Experiment Phases

### Phase 1 — Pipeline Validation ✅ DONE
**Goal:** Confirm training and inference pipelines work end-to-end  
**Data:** 42 video clips  
**Training:** 100 iterations, 1× H200  
**Result:** Checkpoint saved, videos generated — pipeline confirmed working

---

### Phase 2 — Small Dataset Full Training (NEXT)
**Goal:** Verify that TalkVid data actually improves generation quality with sufficient training  
**Data:** 42 video clips (same as Phase 1)  
**Training:** 30,000 iterations, 4× H200 (~15 hours)  
**Comparison:** Side-by-side with original Hallo3 baseline using same inputs

#### Steps
```bash
# 1. Update train_iters
sed -i 's/train_iters: 100/train_iters: 30000/' configs/sft_s1.yaml

# 2. Submit Stage 1 training (sbatch)
sbatch run_stage1.sh   # 4× H200, logs/stage1_<jobid>.log

# 3. After Stage 1 completes, run Stage 2
#    Update configs/sft_s2.yaml with same data path, then:
sbatch run_stage2.sh

# 4. Run inference with fine-tuned checkpoint
bash scripts/inference_long_batch.sh \
    my_inference/input.txt \
    my_inference/outputs/talkvid_42_30k/

# 5. Run inference with original baseline for comparison
#    (restore load path in inference.yaml to ./pretrained_models/hallo3)
bash scripts/inference_long_batch.sh \
    my_inference/input.txt \
    my_inference/outputs/baseline/
```

#### Milestone
- [ ] Stage 1 training completes (30k iter, loss visibly decreasing)
- [ ] Stage 2 training completes
- [ ] Generated videos show qualitative improvement over baseline

---

### Phase 3 — Full TalkVid Dataset Training
**Goal:** Validate whether scale matters — train on the complete 181,752-clip TalkVid dataset  
**Data:** Full TalkVid (quality-filtered subset using `dover_scores`)  
**Training:** 30,000 iterations, multi-GPU

#### Prerequisites before Phase 3
1. **Update conversion script** to use real captions from `filtered_video_clips_with_captions.json`  
   (currently writes placeholder `"A person talking."` — full training should use `description` field)

2. **Add `--caption_json` argument** to `scripts/convert_talkvid_to_hallo3.py`:
   ```python
   caption = caption_map.get(video_id, "A person talking.")
   ```

3. **Run full data conversion** on cluster:
   ```bash
   python scripts/convert_talkvid_to_hallo3.py \
       --clips_flat /scratch/li.qianyi/TalkVid/clips_flat_full \
       --output /scratch/li.qianyi/hallo3_data_full \
       --dataset_name talkvid_full \
       --caption_json /scratch/li.qianyi/TalkVid/filtered_video_clips_with_captions.json

   python hallo3/extract_meta_info.py \
       -r /scratch/li.qianyi/hallo3_data_full \
       -n talkvid_full
   ```

4. **(Optional) Quality filtering** using `dover_scores` field:
   ```python
   high_quality = [item for item in data if item["dover_scores"] > 0.5]
   ```

#### Milestone
- [ ] Full dataset converted and indexed
- [ ] Stage 1 + Stage 2 training complete
- [ ] Inference results generated for comparison

---

## Evaluation Plan

### Qualitative (Primary)
Compare generated videos across three conditions using **identical inputs** (same reference image + same audio):

| Condition | Checkpoint |
|-----------|-----------|
| Baseline | Original Hallo3 pretrained weights |
| Fine-tuned (42 clips) | Phase 2 checkpoint |
| Fine-tuned (full TalkVid) | Phase 3 checkpoint |

Evaluation criteria:
- **Lip sync accuracy** — does mouth movement match speech timing?
- **Face naturalness** — no artifacts, identity preserved
- **Temporal stability** — no jitter or flickering between frames

### Quantitative (Optional, later)
| Metric | Description | Target |
|--------|-------------|--------|
| Sync-C | Audio-visual sync confidence (SyncNet) | Higher is better |
| Sync-D | Audio-visual sync distance | Lower is better |
| FID | Fréchet Inception Distance (image quality) | Lower is better |
| FVD | Fréchet Video Distance (video quality) | Lower is better |

---

## File Structure Reference

```
hallo3/
├── configs/
│   ├── sft_s1.yaml              # Stage 1 fine-tune config (train_iters, data path, optimizer)
│   ├── sft_s2.yaml              # Stage 2 fine-tune config
│   └── inference.yaml           # load: path to checkpoint for inference
├── scripts/
│   ├── convert_talkvid_to_hallo3.py   # TalkVid → Hallo3 format converter
│   └── inference_long_batch.sh        # Batch inference script
├── data/
│   └── talkvid.json             # Meta index for 42-clip training set
├── my_inference/
│   ├── input.txt                # Inference inputs (text@@image@@audio)
│   ├── images/                  # Reference face images
│   ├── audios/                  # Drive audio files (H1–H4)
│   └── outputs/
│       ├── baseline/            # Original Hallo3 outputs
│       └── finetuned_100iter/   # Phase 1 outputs (sanity check) ✅
└── stage-1/                     # Training checkpoints (Stage 1)
```

---

## Cluster Quick Reference

```bash
# Activate environment
conda activate /home/li.qianyi/envs/hallo
cd /scratch/li.qianyi/hallo3

# Check available GPUs
sinfo -o "%P %G %N %t" | grep idle

# Interactive session (single GPU)
srun -p sharing --gres=gpu:h200:1 --mem=60G --pty bash

# Submit batch job
sbatch run_stage1.sh

# Run inference (single GPU interactive)
CUDA_VISIBLE_DEVICES="0" bash scripts/inference_long_batch.sh \
    my_inference/input.txt \
    my_inference/outputs/<output_dir>/
```

---

## Decision Points

| After | Decision |
|-------|---------|
| Phase 2 inference | If quality improvement visible → proceed to Phase 3; if not → investigate data quality / training config |
| Phase 3 inference | Quantitative evaluation; write up findings |
