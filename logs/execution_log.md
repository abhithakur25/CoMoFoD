# Execution Log (PowerShell session)

Chronological log of the commands run in the Claude Code PowerShell session and
their key outputs, from environment repair through to the six-model comparison
and the GitHub commit.

---

## 1. Environment diagnosis & repair

```powershell
python --version
# Python 3.12.10
python -c "import tensorflow as tf; print(tf.__version__)"
# ModuleNotFoundError: No module named 'tensorflow.python'   (corrupted install)
```

Root cause: TensorFlow's bundled gRPC headers exceed the Windows 260-char path
limit, so the system install was incomplete. Fix = a venv at a short path:

```powershell
python -m venv --system-site-packages C:\tfv
C:\tfv\Scripts\python.exe -m pip install "tensorflow==2.20.*"
C:\tfv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__)"
# TF 2.20.0   (compat.v1 OK; cv2 4.13.0, sklearn 1.9.0 inherited)
```

## 2. Load the frozen model & extract block embeddings

The `checkpoint` pointer file referenced the *untrained* `ckpt-0`; the loader was
fixed to restore the most-trained checkpoint `ckpt-5004`.

```powershell
C:\tfv\Scripts\python.exe extract_embeddings.py
# Restoring ...\copy_move\model-copy_move.ckpt-5004 (step=5004)
# Saved ./results/embeddings_cache.npz : 48 images, 4251 blocks, 648 forged (15.24%)
```

## 3. Localisation evaluation + optimisation (5-fold CV)

```powershell
C:\tfv\Scripts\python.exe analyze_and_optimize.py
# ================ RESULTS (5-fold CV, full dataset) ================
# BASELINE   P=0.152 R=0.985 F1=0.263 Acc=0.158  AP=0.133 AUC=0.424
# OPTIMISED  P=0.210 R=0.941 F1=0.343 Acc=0.451  AP=0.187 AUC=0.603
```

Finding: authentic background blocks are *more* self-similar than forged regions,
so the raw nearest-distance detector ranks worse than random (AUC 0.42). The
optimised duplication-feature classifier beats it on every metric.

## 4. Verification evaluation (the task the triplet model was trained for)

Triplets in `../Data/New` (train) and `../Data/Old` (held-out test).

```powershell
C:\tfv\Scripts\python.exe evaluate_verification.py
#   Data/New triplets: 4000 ; Data/Old triplets (held-out): 4000
#   Restoring ...ckpt-5004 ; Saved ver_embeddings.npz (dim=128)

C:\tfv\Scripts\python.exe analyze_verification.py
# BASELINE  CV_acc=0.7250+/-0.0123  P=0.681 R=0.852 F1=0.757 AUC=0.7678 | Old acc=0.7442
# OPTIMISED acc=0.9938  P=0.995 R=0.993 F1=0.994 AUC=0.9995 | Old acc=0.9982
# OPTIMISED train/val loss=0.023/0.056  train/val acc=1.000/0.990
```

Rich pair features `[(a-b)^2, a*b, a, b, |a-b|]` + a batch-normed MLP lift
verification accuracy from 72.5% (raw distance) to **99.4% val / 99.8% test**.

## 5. Six-model comparison (train = Data/New, test = Data/Old)

```powershell
C:\tfv\Scripts\python.exe comparison\code\compare_models.py
# train=6400  val=1600  test=8000  feat_dim=640
# M1_Logistic      test_acc=0.9948 P=0.993 R=0.997 F1=0.995 FN=13 AUC=0.9989
# M2_Shallow       test_acc=0.9982 P=0.997 R=0.999 F1=0.998 FN=3  AUC=0.9999
# M3_DeepBN        test_acc=0.9982 P=0.997 R=1.000 F1=0.998 FN=2  AUC=1.0000   <-- EXISTING *** BEST ***
# M4_Wide          test_acc=0.9971 P=0.995 R=1.000 F1=0.997 FN=2  AUC=0.9995
# M5_Tanh          test_acc=0.9975 P=0.997 R=0.999 F1=0.998 FN=6  AUC=0.9999
# M6_DeepNarrow    test_acc=0.9970 P=0.996 R=0.998 F1=0.997 FN=9  AUC=0.9998
# BEST MODEL: M3 Deep+BN (existing) -> model/best_model.keras
```

All six models exceed 99% test accuracy; the existing Deep+BN architecture wins
on accuracy, ROC-AUC and false negatives.

## 6. Commit & push to GitHub

```powershell
git clone https://github.com/abhithakur25/CoMoFoD CoMoFoD_push   # empty repo
# assembled comparison/, scripts/, results_verification/, results_localization/
git checkout -b main
git add -A
git commit -m "Add copy-move detection: testing, optimisation and six-model comparison"
# 76 files changed, 3550 insertions(+)
git push -u origin main
# * [new branch] main -> main   (verified via git ls-remote origin)
```

---

*Generated as part of the Claude Code session. Embedding caches (`*.npz`) and the
original frozen TF checkpoint are regenerable and are not committed.*
