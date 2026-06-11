# Copy-Move Verifier — Six-Model Comparison

Compares the **existing** verifier against **five more similar models** on the
copy-move verification task, with a proper **train / validation / test** split,
and selects the best one.

## Data protocol
| Split | Source | Pairs |
|-------|--------|-------|
| **Train** | `../../Data/New` triplets (the *training images*) | 6 400 |
| **Validation** | 20 % held-out of New | 1 600 |
| **Test** | `../../Data/Old` triplets (the *test images*) | 8 000 |

Genuine pair = (anchor, positive); impostor pair = (anchor, negative).
The frozen `copy_move` Siamese model (`../../models/siamese/copy_move`,
checkpoint `ckpt-5004`) supplies the 128-d embeddings and is **never retrained**.
Pair feature = `[(a−b)², a·b, a, b, |a−b|]` (640-d), standardised on the train split.

## The six models (all trained on the same features)
| Key | Model |
|-----|-------|
| M1_Logistic | linear verifier (logistic regression) |
| M2_Shallow  | 1 hidden layer (64) |
| **M3_DeepBN** | 256-128-64 + BatchNorm — **EXISTING** model |
| M4_Wide     | 512-256 wide MLP |
| M5_Tanh     | 128-64 tanh MLP |
| M6_DeepNarrow | 128-128-128-128 (RMSprop) |

## Results (TEST = Data/Old)
| Model | Train acc | Val acc | **Test acc** | Precision | Recall | F1 | FN | ROC-AUC |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| M1 Logistic | 0.997 | 0.977 | 0.9948 | 0.993 | 0.997 | 0.995 | 13 | 0.9989 |
| M2 Shallow  | 1.000 | 0.991 | 0.9982 | 0.997 | 0.999 | 0.998 | 3 | 0.9999 |
| **M3 Deep+BN (existing)** | 1.000 | 0.994 | **0.9982** | 0.997 | 1.000 | 0.998 | **2** | **1.0000** |
| M4 Wide     | 1.000 | 0.993 | 0.9971 | 0.995 | 1.000 | 0.997 | 2 | 0.9995 |
| M5 Tanh     | 0.997 | 0.988 | 0.9975 | 0.997 | 0.999 | 0.998 | 6 | 0.9999 |
| M6 DeepNarrow | 0.999 | 0.990 | 0.9970 | 0.996 | 0.998 | 0.997 | 9 | 0.9998 |

### 🏆 Best model: **M3 Deep+BN (the existing model)**
Tied-highest test accuracy (0.9982), the **highest ROC-AUC (1.0000)** and the
**fewest false negatives (2)**. Saved as `model/best_model.keras`.
All six models exceed **99% test accuracy**.

## Folders
* **`code/`** — execution files: `compare_models.py` (this comparison),
  `evaluate_verification.py` (extracts & caches the frozen-model embeddings),
  `analyze_verification.py` (single-model verification report).
* **`result/`** — every CSV + PNG:
  * `comparison_summary.csv`, `all_metrics.json`, `BEST_MODEL.txt`
  * `history_<model>.csv` (per-epoch train/val loss & acc) ×6
  * **Bar graphs:** `bar_accuracy.png` (train/val/test acc), `bar_prf.png`
    (precision/recall/F1), `bar_FN.png` (false negatives)
  * **Line graphs:** `line_training_loss.png`, `line_validation_loss.png`,
    `line_training_accuracy.png`, `line_validation_accuracy.png`,
    plus per-model `curves_<model>.png`
  * **Confusion matrices:** `confusion_<model>.png` ×6
  * **ROC:** `roc_all_models.png` (all six overlaid, on the test set)
* **`model/`** — every trained verifier (`M1..M6_*.keras`) + `best_model.keras`.
  The frozen feature extractor is the existing TF checkpoint at
  `../../models/siamese/copy_move/` (ckpt-5004).

## Reproduce
```
# from src/  (TF1 graph step needs eager disabled; run once to cache embeddings)
C:\tfv\Scripts\python.exe evaluate_verification.py
# then the comparison (Keras, eager)
C:\tfv\Scripts\python.exe comparison\code\compare_models.py
```
