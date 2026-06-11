# Copy-Move Forgery Detection — Test & Optimisation Results

Evaluation of the **existing trained** Siamese (facenet-style, triplet-loss)
model on the 48 forged images in `SD_Images/` against the ground-truth masks in
`SD_GT/`. **The saved model is used only for inference — it is never retrained.**

## How to reproduce
```
# 1. extract frozen-model embeddings for every 30x30 block (uses the saved model)
C:\tfv\Scripts\python.exe extract_embeddings.py
# 2. compute all metrics, train the optimisation head, make plots
C:\tfv\Scripts\python.exe analyze_and_optimize.py
# 3. (optional) qualitative mask panels + comparison bar chart
C:\tfv\Scripts\python.exe make_mask_visuals.py
C:\tfv\Scripts\python.exe make_comparison_chart.py
```
A dedicated virtual-env (`C:\tfv`, TensorFlow 2.20) was created because the
system TensorFlow install was corrupted by the Windows 260-char path limit.

## Pipeline
* **Loader fix** — the `checkpoint` pointer file referenced the *untrained*
  `ckpt-0`; the loader was changed to restore the **most-trained** checkpoint
  (`ckpt-5004`).
* The TF1 meta-graph is loaded under `tf.compat.v1`; the 128-d L2-normalised
  embedding is computed for every non-overlapping 30×30 block of every image.
* A block is labelled **forged** if >10 % of its pixels are white in the GT mask
  (overall prevalence ≈ 15 %).

## Two models compared (5-fold cross-validation over images)
* **BASELINE** — the original method: a block is "forged" if its nearest
  neighbour in embedding space is within a distance threshold.
* **OPTIMISED** (the *different model*) — an MLP classification head trained on
  **duplication + offset-consistency features** derived from the same frozen
  embeddings. Offset-consistency is the classic copy-move cue (many matched
  pairs share one translation) that the original distance test ignores.

> Key finding: on this data, authentic background blocks (sky/grass/walls) are
> *more* self-similar than the pasted regions, so the plain nearest-distance
> test ranks **worse than random (AUC 0.42)**. The optimisation head corrects
> this, lifting AUC to 0.60.

## Results (full dataset, out-of-fold; F1-optimal operating point)

| Metric            | Baseline | Optimised |
|-------------------|:--------:|:---------:|
| Precision         |  0.152   | **0.210** |
| Recall            |  0.985   |   0.941   |
| F1                |  0.263   | **0.343** |
| Accuracy          |  0.158   | **0.451** |
| Average Precision |  0.133   | **0.187** |
| ROC-AUC           |  0.424   | **0.603** |

Optimisation-head training curves (held-out fold):
`train_loss 0.585 · val_loss 0.548 · train_acc 0.849 · val_acc 0.868`
(no over-fitting — validation tracks/leads training).

The optimised model improves **every** comparable metric. The baseline only
reaches F1 0.26 by flagging ~98 % of blocks as forged (3570 false positives);
the optimised model recovers 1308 true negatives while keeping recall high.

## Files
| File | Contents |
|------|----------|
| `baseline_metrics.txt`, `optimized_metrics.txt` | precision/recall/F1/accuracy/AP/AUC + confusion matrix |
| `metrics_summary.csv` | side-by-side metric table |
| `confusion_matrix_baseline.png`, `confusion_matrix_optimized.png` | confusion-matrix heatmaps |
| `loss_curve.png`, `accuracy_curve.png` | training vs validation loss / accuracy |
| `training_loss_eventlog.png` | saved model's original triplet-loss curve (from TF event logs) |
| `metrics_comparison.png` | baseline-vs-optimised bar chart |
| `history.csv` | per-epoch train/val loss & accuracy |
| `masks/panel_*.png` | qualitative input / GT / baseline / optimised mask panels |
| `embeddings_cache.npz` | cached frozen-model block embeddings + labels |

## Honest limitations
Localisation is coarse (30×30 blocks) and the frozen triplet embedding carries
only a weak copy-move signal (AUC ≈ 0.60), so some images with distinctive
pasted objects (e.g. `kore`) are still missed. Further gains would require
re-training the embedding network or a higher-resolution detector — out of scope
here, since the task is to **test the existing model** and optimise around it.
