# Milestone 3: Model Architecture Design and E2E Pipeline Verification
## ✅ COMPLETED - March 16, 2026

---

## 📋 Deliverables

### 1. **LaTeX Report** 
- File: `Milestone_TeX_Files/milestone_3.tex`
- Status: ✅ Complete with real metrics, ASR values, sample outputs
- Sections covered:
  - Objective
  - Dataset organization and directory structure
  - Preprocessing pipeline description
  - Architecture design (DistilBERT + CLS pooling + linear head)
  - Data flow diagram (Mermaid)
  - Input format compatibility
  - Architecture suitability and limitations
  - E2E pipeline verification methodology
  - Metrics and evaluation results
  - Sample predictions with probabilities
  - Per-class performance tables
  - Confusion matrices
  - Attack Success Rate (ASR) metrics

### 2. **Model Implementation**
- `models/guardrail_classifier.py` ✅ - DistilBERT architecture with label mappings
- `models/train.py` ✅ - Training loop with class-weighted loss, AdamW optimizer, ASR metrics
- `models/evaluate.py` ✅ - Evaluation pipeline with per-class metrics and ASR calculation

### 3. **Pipeline Scripts**
- `scripts/run_e2e_subset.py` ✅ - Orchestration for stratified subset sampling and E2E verification

### 4. **Training Results (100-epoch run)**
- Model: tiny-random-distilbert (proof-of-concept)
- Output directory: `milestone_3_outputs/train100_final/`
- Artifacts:
  - ✅ `best_model.pt` - Trained checkpoint
  - ✅ `training_metrics.json` - Loss, accuracy, F1, per-class metrics, ASR per epoch
  - ✅ `validation_metrics.json` - Validation set metrics with ASR breakdown
  - ✅ `test_metrics.json` - Test set metrics with ASR breakdown
  - ✅ `validation_samples.json` - 12 sample predictions with probabilities
  - ✅ `test_samples.json` - 12 test predictions with probabilities
  - ✅ `tokenizer/` - Saved DistilBERT tokenizer

---

## 📊 Key Metrics (2 decimal places)

| Metric | Validation | Test |
|--------|-----------|------|
| **Accuracy** | 0.33 | 0.33 |
| **Macro-F1** | 0.17 | 0.17 |
| **Jailbreak ASR** | 0.0 | 0.0 |
| **Harmful ASR** | 1.0 | 1.0 |
| **Overall ASR** | 0.43 | 0.43 |

### Per-Class Breakdown
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------| 
| Benign | 0.0 | 0.0 | 0.0 | 5 |
| Jailbreak | 0.33 | 1.0 | 0.5 | 4 |
| Harmful | 0.0 | 0.0 | 0.0 | 3 |

---

## 🏗️ Architecture Summary

**Model**: DistilBERT + Task-Specific Head
```
Input Prompts (max 192 tokens)
    ↓
DistilBERT Encoder (contextual embeddings)
    ↓
CLS Token Pooling [B, 768]
    ↓
Dropout (0.2)
    ↓
Linear Head (768 → 3 logits)
    ↓
Model Output: [B, 3] logits + softmax probabilities
```

**Training Config**:
- Epochs: 100
- Batch size: 8
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup: 10%
- Loss: Class-weighted CrossEntropyLoss
- Optimizer: AdamW
- Scheduler: Linear warmup + decay

---

## 🎯 ASR Metrics Explanation

**Attack Success Rate (ASR)** measures the **ratio of attack prompts that were NOT correctly detected**.

- **Jailbreak ASR = 0.0**: All 4 jailbreak prompts correctly detected
- **Harmful ASR = 1.0**: All 3 harmful prompts missed (misclassified as benign/jailbreak)
- **Overall ASR = 0.43**: 3 out of 7 attack prompts failed detection

This metric is critical for safety-critical guardrail systems.

---

## 📁 Project Organization

```
DS_and_AI_Lab_Project/
├── Milestone_PDFs/
│   ├── Milestone_1_Analysis_of_existing_solutions.pdf
│   └── Milestone_2_Dataset_Info_Preparation.pdf
├── Milestone_TeX_Files/
│   ├── milestone_2.tex
│   └── milestone_3.tex                    [UPDATED with real metrics]
├── Logs/
│   ├── milestone_2.log
│   ├── milestone3_run.log
│   ├── run_full.log
│   └── upload_log.txt
├── datasets/                              [Preprocessed splits]
├── models/                                [Architecture code]
│   ├── guardrail_classifier.py
│   ├── train.py
│   ├── evaluate.py
│   └── __init__.py
├── scripts/                               [Orchestration]
│   ├── run_e2e_subset.py
│   └── __init__.py
├── milestone_3_outputs/
│   └── train100_final/                   [100-epoch training artifacts]
│       ├── best_model.pt
│       ├── training_metrics.json
│       ├── validation_metrics.json       [with ASR]
│       ├── test_metrics.json            [with ASR]
│       ├── validation_samples.json
│       ├── test_samples.json
│       └── tokenizer/
└── requirements_milestone3.txt
```

---

## ✨ Features Implemented

### Core
- ✅ DistilBERT-based classifier architecture
- ✅ Class-weighted cross-entropy loss
- ✅ AdamW optimizer with warmup scheduler
- ✅ Stratified train/val/test split preservation
- ✅ Checkpoint saving based on validation macro-F1

### Evaluation
- ✅ Accuracy and Macro-F1 metrics
- ✅ Per-class precision, recall, F1
- ✅ **NEW**: Attack Success Rate (ASR) metrics
  - Per-class ASR (jailbreak_asr, harmful_asr)
  - Overall ASR (combined attack detection rate)
- ✅ Confusion matrices
- ✅ Sample predictions with softmax probabilities

### Quality
- ✅ All metrics rounded to 2 decimal places
- ✅ Deterministic random seed (seed=42)
- ✅ Reproducible E2E pipeline
- ✅ Clean project structure with organized folders

---

## 📝 Notes on Current Results

The tiny-random-distilbert model on a small deterministic subset is a **proof-of-concept**:
- Shows all components work end-to-end ✅
- Demonstrates metrics calculation (accuracy, ASR, per-class) ✅
- Reveals model bias (jailbreak detection=100%, harmful detection=0%) 

**Production deployment** will use:
- `distilbert-base-uncased` (pre-trained weights)
- Complete dataset (full train/val/test)
- Hyperparameter tuning via cross-validation
- Threshold tuning for production safety trade-offs

---

## 🎓 Rubric Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Architecture design & justification | ✅ | Section 3 of milestone_3.tex |
| Data flow diagram | ✅ | Mermaid diagram in section 4 |
| Input format compatibility | ✅ | Section 5 of milestone_3.tex |
| Suitability & limitations | ✅ | Section 6 of milestone_3.tex |
| E2E pipeline verification | ✅ | Section 7, training_metrics.json |
| Output examples & format | ✅ | validation_samples.json, test_samples.json |
| Loss function & metrics | ✅ | Section 8 with full metric tables |
| ASR metrics | ✅ | ASR subsection with per-class breakdown |

---

## 🚀 How to Use

### Run Training
```bash
cd d:\DS_and_AI_Lab_Project
.\.venv\Scripts\python.exe models\train.py \
  --train-data milestone_3_outputs\subset_e2e_20260317_025358\subset_data\train_subset.json \
  --val-data milestone_3_outputs\subset_e2e_20260317_025358\subset_data\validation_subset.json \
  --output-dir milestone_3_outputs\my_run \
  --epochs 100 \
  --batch-size 8
```

### Run Evaluation
```bash
.\.venv\Scripts\python.exe models\evaluate.py \
  --checkpoint milestone_3_outputs\train100_final\best_model.pt \
  --dataset milestone_3_outputs\subset_e2e_20260317_025358\subset_data\validation_subset.json \
  --output-metrics val_metrics.json \
  --output-samples val_samples.json
```

### Compile LaTeX
```bash
cd Milestone_TeX_Files
pdflatex milestone_3.tex
```
(Requires LaTeX distribution: MiKTeX, TeX Live, etc.)

---

## 📦 Dependencies

```
torch>=2.10.0
transformers>=5.3.0
scikit-learn>=1.8.0
datasets>=2.19.0
numpy>=1.26.0
tqdm>=4.66.0
```

All installed in `.venv` ✅

---

## ✅ Status: READY FOR SUBMISSION

- [x] Architecture code complete and tested
- [x] Training pipeline working with 100 epochs completed
- [x] Evaluation with ASR metrics implemented
- [x] All metrics rounded to 2 decimal places
- [x] LaTeX report finalized with real data
- [x] Project organized into clean folder structure
- [x] All artifacts saved and reproducible
- [x] Logs organized in dedicated folder

**Next step**: Compile milestone_3.tex to PDF (on system with LaTeX) or submit .tex source as-is.

---

*Generated: 2026-03-16*  
*Model: tiny-random-distilbert (PoC) | Production: distilbert-base-uncased*
