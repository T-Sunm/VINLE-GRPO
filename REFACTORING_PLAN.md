# Kế Hoạch Refactor: vqa-nle → VINLE-GRPO

## 📋 Tổng quan

Tài liệu này mô tả chi tiết kế hoạch refactor code từ repository `vqa-nle` sang `VINLE-GRPO` với cấu trúc mới được tối ưu hóa cho nghiên cứu học thuật về GRPO cho VLMs.

**Mục tiêu**:
- Tách biệt rõ ràng external dependencies (ms-swift, smile) khỏi research code
- Tổ chức code theo hướng thí nghiệm, dễ reproduce
- Tối ưu hóa cho paper submission (ICISN 2026)
- Giữ lại backward compatibility với old results

---

## 🔍 Phân tích Cấu trúc Hiện tại

### Repository `vqa-nle` (Current)

```
vqa-nle/
├── src/                           # Research code (mixed)
│   ├── data_loader/              # 16 files - Data preparation (nhiều variants)
│   ├── evaluation/               # 8 files - Evaluation pipeline
│   ├── inference/                # 10 files - Inference scripts
│   ├── rewards/                  # 4 files - Custom rewards
│   ├── trainers/                 # 1 file - Custom GRPO trainer
│   └── training/                 # 4 files - Training scripts
│
├── ms-swift/                      # External (modified) - 1000+ files
├── smile-metric-qna-eval/        # External (modified) - 18 files
├── CPSRank/                      # External - old baseline
├── VLM-R1/                       # External - old baseline
├── uqlm/                         # External - old baseline
│
├── configs/                       # 2 files - Global configs
├── data/                          # Dataset storage
│   ├── raw/                      # ViVQA-X (symlink)
│   └── processed/                # Preprocessed data
│
├── notebooks/                    # 15 notebooks - Analysis
├── models/                       # Model checkpoints (symlink)
└── tests/                        # 4 test files
```

**Vấn đề**:
1. ❌ **External repos lẫn lộn**: `ms-swift`, `smile`, `CPSRank`, `VLM-R1`, `uqlm` nằm cùng root
2. ❌ **Không có experiment tracking structure**: Wandb runs khó map với configs
3. ❌ **Data loader phức tạp**: 16 variants không rõ ràng (curriculum, reasoning, sft, vqax, vlmr1, etc.)
4. ❌ **Scripts thiếu tổ chức**: Training scripts nằm rải rác trong `src/training/` và `ms-swift/examples/`
5. ❌ **Configs không đầy đủ**: Chỉ 2 files, thiếu per-experiment configs

### Repository `VINLE-GRPO` (Target)

```
VINLE-GRPO/
├── external/                      # External dependencies (with modifications)
│   ├── ms-swift/                 # GRPO training framework
│   └── smile/                    # SMILE evaluation metric
│
├── src/                           # Research code (clean)
│   ├── data/                     # Data preparation for ViVQA-X
│   ├── rewards/                  # Custom reward functions
│   ├── evaluation/               # Evaluation pipeline
│   ├── inference/                # Inference scripts
│   └── utils/                    # Shared utilities
│
├── scripts/                       # Executable scripts
│   ├── train/                    # Training wrappers
│   ├── eval/                     # Evaluation scripts
│   ├── data/                     # Data preprocessing
│   └── setup/                    # Environment setup
│
├── configs/                       # Configuration files
│   ├── experiments/              # Per-experiment configs
│   ├── models/                   # Model-specific configs
│   └── ms-swift/                 # ms-swift wrapper configs
│
├── experiments/                   # Experiment tracking
│   ├── exp001_grpo_baseline/
│   ├── exp002_grpo_ours/        # Main paper results
│   └── exp003_ablation_study/
│
├── data/                         # Datasets
│   ├── raw/                      # ViVQA-X (symlink)
│   └── processed/                # GRPO-formatted JSONL
│
├── docs/                         # Documentation
│   └── paper/                    # Paper materials
│
└── notebooks/                    # Analysis notebooks
```

**Ưu điểm**:
1. ✅ **External repo isolation**: Rõ ràng trong `external/`
2. ✅ **Experiment-driven**: Mọi thứ map với experiments (configs, checkpoints, results)
3. ✅ **Clean separation**: Scripts riêng, configs riêng, source code riêng
4. ✅ **Paper-ready**: Docs structure support paper submission

---

## 🎯 Kế Hoạch Refactor (Step-by-Step)

### **Phase 1: Setup Infrastructure** (1-2 giờ)

#### Step 1.1: Verify Current VINLE-GRPO Structure
```bash
cd /home/vlai-vqa-nle/minhtq/VINLE-GRPO
tree -L 2 -I 'wandb|__pycache__|.git'
```

**Tasks**:
- [x] Verify `external/ms-swift` exists ✅ (đã copy từ conversation trước)
- [x] Verify `external/smile` exists ✅ (đã copy từ conversation trước)
- [ ] Check configs structure
- [ ] Check scripts structure
- [ ] Check experiments structure

#### Step 1.2: Create Missing Directories
```bash
cd VINLE-GRPO

# Ensure all directories exist
mkdir -p experiments/{exp001_grpo_baseline,exp002_grpo_ours,exp003_ablation_study}/{checkpoints,results,logs}
mkdir -p data/{raw,processed/{grpo,sft}}
mkdir -p configs/{experiments,models,ms-swift}
mkdir -p scripts/{train,eval,data,setup}
mkdir -p src/{data,rewards,evaluation,inference,utils}
```

**Deliverables**:
- [ ] All directories created
- [ ] `.gitkeep` files added to empty directories

---

### **Phase 2: Migrate Source Code** (3-4 giờ)

#### Step 2.1: Data Processing (`src/data/`)

**Mapping**:
```
vqa-nle/src/data_loader/          → VINLE-GRPO/src/data/
├── dataset_loader_msswift.py     → dataset_loader.py (main)
├── dataset_loader_standard_vivqax.py → vivqa_processor.py
├── convert_dataset.py            → data_utils.py (utilities)
└── image_preprocessing.py        → data_utils.py (merge)
```

**Actions**:
1. **Consolidate loaders**: 
   - Keep only GRPO-relevant loaders (`dataset_loader_msswift.py` → `dataset_loader.py`)
   - Archive old variants (curriculum, vlmr1, etc.) to `docs/legacy/`
   
2. **Simplify**:
   - Remove unnecessary variants
   - Focus on `ViVQA-X → GRPO format` conversion only
   
3. **Create clean API**:
   ```python
   # src/data/dataset_loader.py
   from .vivqa_processor import ViVQAProcessor
   from .data_utils import preprocess_image, convert_to_grpo_format
   
   def load_vivqa_for_grpo(split='train', output_dir='data/processed/grpo'):
       """Main entry point for data loading"""
       pass
   ```

**Deliverables**:
- [ ] `src/data/dataset_loader.py` (main)
- [ ] `src/data/vivqa_processor.py` (ViVQA-specific)
- [ ] `src/data/data_utils.py` (utilities)
- [ ] `src/data/__init__.py` (exports)

---

#### Step 2.2: Rewards (`src/rewards/`)

**Mapping**:
```
vqa-nle/src/rewards/               → VINLE-GRPO/src/rewards/
├── base_rewards.py               → (keep as is)
├── outcome_rewards.py            → accuracy_reward.py + format_reward.py
├── explaination_rewards.py       → explanation_reward.py
└── length_rewards.py             → explanation_reward.py (merge if used)
```

**Actions**:
1. **Rename for clarity**:
   - `outcome_rewards.py` → split into `accuracy_reward.py` + `format_reward.py`
   - `explaination_rewards.py` → `explanation_reward.py` (fix typo)

2. **Create registry**:
   ```python
   # src/rewards/reward_registry.py
   from .accuracy_reward import AccuracyReward
   from .format_reward import FormatReward
   from .explanation_reward import ExplanationReward
   
   REWARD_REGISTRY = {
       'accuracy': AccuracyReward,
       'format': FormatReward,
       'explanation': ExplanationReward,
   }
   ```

3. **Clean up**:
   - Remove unused rewards (length if not in paper)
   - Ensure all rewards follow same interface

**Deliverables**:
- [ ] `src/rewards/accuracy_reward.py`
- [ ] `src/rewards/format_reward.py`
- [ ] `src/rewards/explanation_reward.py`
- [ ] `src/rewards/reward_registry.py`
- [ ] `src/rewards/__init__.py`

---

#### Step 2.3: Evaluation (`src/evaluation/`)

**Mapping**:
```
vqa-nle/src/evaluation/            → VINLE-GRPO/src/evaluation/
├── calculate_scores.py           → calculate_scores.py (main)
├── nlg_metrics.py                → metrics/nlg.py
├── shared_models.py              → evaluators/shared_models.py
├── text_preprocessing.py         → metrics/preprocessing.py
└── OEA_calculate_score.py        → (archive if not used)
```

**Actions**:
1. **Restructure**:
   ```
   src/evaluation/
   ├── calculate_scores.py        # Main CLI entry
   ├── evaluators/
   │   ├── __init__.py
   │   ├── accuracy_evaluator.py
   │   ├── smile_evaluator.py
   │   └── bertscore_evaluator.py
   └── metrics/
       ├── __init__.py
       ├── nlg.py
       └── preprocessing.py
   ```

**Deliverables**:
- [ ] `src/evaluation/calculate_scores.py`
- [ ] `src/evaluation/evaluators/` (3 files)
- [ ] `src/evaluation/metrics/` (2 files)

---

#### Step 2.4: Inference (`src/inference/`)

**Mapping**:
```
vqa-nle/src/inference/             → VINLE-GRPO/src/inference/
├── run_inference_grpo.py         → run_inference_grpo.py
├── generate_text.py              → inference_utils.py
└── (others)                      → (archive)
```

**Actions**:
1. **Keep minimal**:
   - Main: `run_inference_grpo.py`
   - Utils: `inference_utils.py`, `batch_inference.py`


**Deliverables**:
- [ ] `src/inference/run_inference_grpo.py`
- [ ] `src/inference/inference_utils.py`
- [ ] `src/inference/batch_inference.py`
- [ ] `src/inference/__init__.py`

---

#### Step 2.5: Utilities (`src/utils/`)

**New module** - Extract common utilities:

```python
src/utils/
├── __init__.py
├── config_utils.py         # Load/merge configs
├── logging_utils.py        # Logging setup
└── visualization.py        # Plot results
```

**Actions**:
1. Extract config loading from scattered files
2. Create unified logging setup
3. Add visualization utilities for notebooks

**Deliverables**:
- [ ] `src/utils/config_utils.py`
- [ ] `src/utils/logging_utils.py`
- [ ] `src/utils/visualization.py`

---

### **Phase 3: Scripts & Configs** (2-3 giờ)

#### Step 3.1: Training Scripts (`scripts/train/`)

**Mapping**:
```
vqa-nle/ms-swift/examples/train/   → VINLE-GRPO/scripts/train/
├── grpo/grpo_our.sh              → run_grpo_vintern.sh
├── grpo/grpo_internvl3_*.sh      → run_grpo_internvl.sh
├── sft/sft_Vintern3B.sh          → run_sft_vintern.sh
└── merge_lora/merge_lora.sh      → merge_lora.sh
```

**Actions**:
1. **Create thin wrappers** that:
   - Load experiment configs from `configs/experiments/`
   - Call ms-swift commands in `external/ms-swift/`
   - Save outputs to `experiments/{exp_name}/`

2. **Template**:
   ```bash
   #!/bin/bash
   # scripts/train/run_grpo_vintern.sh
   
   EXP_NAME=${1:-"exp002_grpo_ours"}
   CONFIG="configs/experiments/${EXP_NAME}.yaml"
   
   # Load config and run ms-swift
   python -m external.ms-swift.swift.cli.sft grpo \
       --config $CONFIG \
       --output_dir experiments/$EXP_NAME/checkpoints
   ```

**Deliverables**:
- [ ] `scripts/train/run_grpo_vintern.sh`
- [ ] `scripts/train/run_grpo_internvl.sh`
- [ ] `scripts/train/run_sft_vintern.sh`
- [ ] `scripts/train/merge_lora.sh`
- [ ] `scripts/train/README.md` (usage docs)

---

#### Step 3.2: Evaluation Scripts (`scripts/eval/`)

**New scripts**:
```bash
scripts/eval/
├── run_inference.sh       # Wrapper for src/inference/run_inference_grpo.py
└── run_evaluation.sh      # Wrapper for src/evaluation/calculate_scores.py
```

**Actions**:
1. Create unified inference wrapper
2. Create unified evaluation wrapper
3. Add GPU selection logic

**Deliverables**:
- [ ] `scripts/eval/run_inference.sh`
- [ ] `scripts/eval/run_evaluation.sh`

---

#### Step 3.3: Data Scripts (`scripts/data/`)

**Mapping**:
```
vqa-nle/ (scattered)               → VINLE-GRPO/scripts/data/
└── (manual commands)             → prepare_vivqa_grpo.py
                                   → link_datasets.sh
```

**Actions**:
1. Create automated data preparation script
2. Create symlink setup script

**Deliverables**:
- [ ] `scripts/data/prepare_vivqa_grpo.py`
- [ ] `scripts/data/prepare_vivqa_sft.py`
- [ ] `scripts/data/link_datasets.sh`

---

#### Step 3.4: Configs (`configs/`)

**Structure**:
```
configs/
├── experiments/
│   ├── grpo_baseline.yaml         # Exp001: GRPO w/o explanation
│   ├── grpo_with_explanation.yaml # Exp002: GRPO ours (full)
│   └── template.yaml              # Template for new experiments
│
├── models/
│   ├── internvl_1b.yaml
│   └── internvl_3b.yaml
│
└── ms-swift/
    ├── sft_args.yaml              # SFT defaults
    └── grpo_args.yaml             # GRPO defaults
```

**Actions**:
1. **Extract hardcoded params** from old scripts to YAML configs
2. **Create experiment configs** matching paper experiments:
   - `grpo_baseline.yaml`: Format + Accuracy rewards only
   - `grpo_with_explanation.yaml`: Full rewards (our method)

3. **Template example**:
   ```yaml
   # configs/experiments/grpo_with_explanation.yaml
   experiment:
     name: "exp002_grpo_ours"
     description: "GRPO with full rewards (accuracy + format + explanation)"
   
   model:
     base: "5CD-AI/Vintern-3B-v1"
     lora_rank: 32
     lora_alpha: 64
   
   training:
     max_steps: 1000
     learning_rate: 1e-5
     num_generations: 4
     beta: 0.04
     temperature: 0.9
   
   rewards:
     enabled: ["accuracy", "format", "explanation"]
     weights:
       accuracy: 1.0
       format: 0.5
       explanation: 0.5
   
   data:
     train: "data/processed/grpo/ViVQA-X_train_grpo.jsonl"
     val: "data/processed/grpo/ViVQA-X_val_grpo.jsonl"
   ```

**Deliverables**:
- [ ] `configs/experiments/grpo_baseline.yaml`
- [ ] `configs/experiments/grpo_with_explanation.yaml`
- [ ] `configs/experiments/template.yaml`
- [ ] `configs/models/internvl_3b.yaml`
- [ ] `configs/ms-swift/grpo_args.yaml`

---

### **Phase 4: Experiments Tracking** (1-2 giờ)

#### Step 4.1: Map Old Results to New Structure

**Actions**:
1. **Identify paper experiments**:
   - Exp001: GRPO baseline (no explanation reward)
   - Exp002: GRPO ours (full rewards) ← **Main paper results**
   - Exp003: Ablation studies

2. **Map old wandb runs** to experiments:
   ```bash
   # Find checkpoints from wandb
   cd vqa-nle/wandb
   ls -lt run-* | head -5  # Find recent runs
   
   # Copy to VINLE-GRPO structure
   cp -r wandb/run-20251224_*/files/checkpoint-* \
         VINLE-GRPO/experiments/exp002_grpo_ours/checkpoints/
   ```

3. **Copy inference results**:
   ```bash
   # Map old results
   cp vqa-nle/src/inference/results/grpo/*.jsonl \
      VINLE-GRPO/experiments/exp002_grpo_ours/results/
   ```

**Deliverables**:
- [ ] `experiments/exp001_grpo_baseline/` (checkpoints, results, logs)
- [ ] `experiments/exp002_grpo_ours/` (checkpoints, results, logs)
- [ ] `experiments/README.md` (mapping docs)

---

#### Step 4.2: Create Experiment README

```markdown
# experiments/exp002_grpo_ours/README.md

## Experiment 002: GRPO with Full Rewards (Our Method)

**Paper Section**: Table 1, Row 4  
**Config**: `configs/experiments/grpo_with_explanation.yaml`  
**Training Script**: `scripts/train/run_grpo_vintern.sh`

### Training Details
- Model: Vintern-3B
- LoRA rank: 32
- Steps: 1000
- Rewards: accuracy + format + explanation
- Training time: ~8 hours on 1x A100

### Results (ViVQA-X Test)
- Accuracy: 62.65%
- SMILE: 60.42
- BERTScore: 52.81

### Checkpoints
- `checkpoints/checkpoint-500/`
- `checkpoints/checkpoint-1000/` (final)

### Wandb Run
- Run ID: `run-20251224_222906-d3gad2x5`
- Link: [wandb.ai/...](https://wandb.ai/...)
```

**Deliverables**:
- [ ] README for each experiment

---

### **Phase 5: Documentation & Testing** (2-3 giờ)

#### Step 5.1: Update Main README

**Actions**:
1. Update installation instructions
2. Add quickstart with new structure
3. Update file paths in examples

**Deliverables**:
- [ ] Updated `VINLE-GRPO/README.md`

---

#### Step 5.2: Create Migration Docs

```markdown
# docs/MIGRATION.md

## Migrating from vqa-nle

### Quick Reference
| Old Path | New Path |
|----------|----------|
| `vqa-nle/src/data_loader/dataset_loader_msswift.py` | `src/data/dataset_loader.py` |
| `vqa-nle/src/rewards/outcome_rewards.py` | `src/rewards/accuracy_reward.py` |
| `vqa-nle/ms-swift/examples/train/grpo/grpo_our.sh` | `scripts/train/run_grpo_vintern.sh` |

### Breaking Changes
- Data loaders consolidated to single API
- Reward functions renamed for clarity
- Scripts moved to `scripts/` directory
```

**Deliverables**:
- [ ] `docs/MIGRATION.md`
- [ ] `docs/ARCHITECTURE.md` (explain new structure)

---

#### Step 5.3: Test New Structure

**Checklist**:
```bash
# 1. Test data preparation
python -m src.data.dataset_loader

# 2. Test training (dry run)
bash scripts/train/run_grpo_vintern.sh exp002_grpo_ours --dry_run

# 3. Test inference
python -m src.inference.run_inference_grpo \
    --model experiments/exp002_grpo_ours/checkpoints/final \
    --limit 10

# 4. Test evaluation
python -m src.evaluation.calculate_scores \
    --input experiments/exp002_grpo_ours/results/predictions.jsonl
```

**Deliverables**:
- [ ] All tests passing
- [ ] `tests/test_refactoring.py` (integration test)

---

### **Phase 6: Cleanup & Archive** (1 giờ)

#### Step 6.1: Archive Old Code

**Actions**:
1. Create `docs/legacy/` for old variants:
   ```bash
   mkdir -p VINLE-GRPO/docs/legacy
   
   # Archive old loaders
   cp vqa-nle/src/data_loader/dataset_loader_curriculum*.py \
      VINLE-GRPO/docs/legacy/
   
   # Archive old baselines
   cp -r vqa-nle/{CPSRank,VLM-R1,uqlm} \
      VINLE-GRPO/docs/legacy/
   ```

2. Add `.gitignore` rules:
   ```gitignore
   # VINLE-GRPO/.gitignore
   
   # Experiments
   experiments/*/checkpoints/
   experiments/*/wandb/
   
   # Data
   data/raw/*
   data/processed/*
   !data/processed/.gitkeep
   
   # Legacy
   docs/legacy/
   ```

**Deliverables**:
- [ ] Legacy code archived
- [ ] `.gitignore` updated

---

#### Step 6.2: Final Verification

**Checklist**:
- [ ] All paths in README are correct
- [ ] All scripts are executable (`chmod +x scripts/**/*.sh`)
- [ ] All imports work (`python -c "import src"`)
- [ ] Git submodules work (`git submodule status`)
- [ ] Configs are valid YAML
- [ ] Paper PDF is in `docs/paper/`

---

## 📊 Migration Checklist Summary

### Phase 1: Infrastructure ⏱️ 1-2h
- [ ] 1.1 Verify current structure
- [ ] 1.2 Create missing directories

### Phase 2: Source Code ⏱️ 3-4h
- [ ] 2.1 Migrate `src/data/` (4 files)
- [ ] 2.2 Migrate `src/rewards/` (5 files)
- [ ] 2.3 Migrate `src/evaluation/` (restructure)
- [ ] 2.4 Migrate `src/inference/` (3 files)
- [ ] 2.5 Create `src/utils/` (3 files)

### Phase 3: Scripts & Configs ⏱️ 2-3h
- [ ] 3.1 Create training scripts (4 files)
- [ ] 3.2 Create evaluation scripts (2 files)
- [ ] 3.3 Create data scripts (3 files)
- [ ] 3.4 Create configs (7 files)

### Phase 4: Experiments ⏱️ 1-2h
- [ ] 4.1 Map old results to new structure
- [ ] 4.2 Create experiment READMEs

### Phase 5: Documentation ⏱️ 2-3h
- [ ] 5.1 Update main README
- [ ] 5.2 Create migration docs
- [ ] 5.3 Test new structure

### Phase 6: Cleanup ⏱️ 1h
- [ ] 6.1 Archive old code
- [ ] 6.2 Final verification

**Total Estimated Time**: 10-15 hours

---

## 🚀 Next Steps

### Immediate (Bước tiếp theo)

1. **Review this plan**: Kiểm tra kế hoạch có hợp lý không?
2. **Confirm phases**: Có phase nào cần điều chỉnh?
3. **Start Phase 1**: Bắt đầu setup infrastructure

### Questions to Answer

- [ ] Có cần giữ lại old baselines (CPSRank, VLM-R1) không?
- [ ] Wandb runs nào là paper results?
- [ ] Có experiment nào cần re-run với new structure?
- [ ] Có cần support multiple backbones (Vintern + InternVL)?

---

## 📝 Notes

- **Backward compatibility**: Giữ old `vqa-nle` repo intact, không xóa
- **Incremental migration**: Migrate từng phase, test sau mỗi phase
- **Document everything**: Mọi thay đổi phải có docs
- **Test thoroughly**: Đảm bảo reproduce được paper results

**Tác giả**: Gemini 2.0 Flash (Thinking - Experimental)  
**Ngày**: 2025-12-29  
**Version**: 1.0
