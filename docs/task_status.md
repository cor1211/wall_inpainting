# Task Checklist

## Objective
Create a high-quality, color-neutral dataset for wall inpainting LoRA training.
Focus: **Quality over Quantity**.

## Progress

### Phase 1: Core Modules ✅
- [x] `sam2_segmenter.py` - Segmentation logic (Reverted to FastSAM/WallSegmenter for stability)
- [x] `color_augmentor.py` - Unlimited random color augmentation (HSV/LAB realistic)
- [x] `prepare_dataset_v2.py` - Pipeline with `--proposal-only` and `--include-ceiling`
- [x] `tools/mask_editor.py` - Fast review editor with shortcuts (Space/Del)
- [x] `docs/dataset_pipeline.md` - High-quality hybrid workflow documentation

### Phase 2: High-Quality Dataset Creation
- [x] Switch backend to FastSAM (No heavy downloads)
- [ ] **Step 1:** Generate Proposals (`--proposal-only --include-ceiling`)
      *Command:* `python prepare_dataset_v2.py --input ... --output ... --proposal-only --include-ceiling`
- [ ] **Step 2:** Manual Verification
      *Tool:* `python tools/mask_editor.py --masks ...`
- [ ] **Step 3:** Final Dataset Generation
      *Command:* `python prepare_dataset_v2.py --input ... --output ... --colors-per-image 10`

### Phase 3: Integration & Training
- [ ] Modify training script to use new dataset
- [ ] Modify inference script for validation
- [ ] Verify neutral caption effectiveness
