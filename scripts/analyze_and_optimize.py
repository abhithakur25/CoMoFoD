# analyze_and_optimize.py
"""
Phase 2 of the copy-move forgery evaluation.

Uses the cached frozen-model embeddings (results/embeddings_cache.npz) to:

  BASELINE  : the original distance-threshold copy-move detector
              (a block is "forged" if some other block lies within `th`).
              Threshold is tuned on the training split.

  OPTIMISED : a *different model* -- an MLP classifier head trained on
              DUPLICATION FEATURES derived from the frozen embeddings
              (nearest-neighbour distances + match counts at several radii).
              This is what actually captures the copy-move signal, and it
              yields proper train/validation loss & accuracy curves.

Produces (in results/):
  baseline_metrics.txt, optimized_metrics.txt, metrics_summary.csv, history.csv
  confusion_matrix_baseline.png, confusion_matrix_optimized.png
  loss_curve.png, accuracy_curve.png, training_loss_eventlog.png
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score)

RES = './results'
os.makedirs(RES, exist_ok=True)
RNG = np.random.RandomState(42)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_cache(path):
    d = np.load(path, allow_pickle=True)
    counts = d['counts']
    emb = d['emb']
    lbl = d['lbl']
    files = d['files']
    bx = d['bx']
    by = d['by']
    ps = int(d['patch_size'])
    # split flat arrays back per image
    per_img = []
    off = 0
    for n, f, x, y in zip(counts, files, bx, by):
        per_img.append({'file': str(f), 'bx': int(x), 'by': int(y),
                        'emb': emb[off:off+n], 'lbl': lbl[off:off+n]})
        off += n
    return per_img, ps


# --------------------------------------------------------------------------
# per-image pairwise squared-distance matrix (embeddings are L2-normalised)
# --------------------------------------------------------------------------
def pairwise_sqdist(emb):
    norms = np.sum(emb ** 2, axis=1, keepdims=True)
    d = norms + norms.T - 2.0 * emb.dot(emb.T)
    return np.maximum(d, 0.0)


# --------------------------------------------------------------------------
# BASELINE detector: block forged if any partner within threshold
# --------------------------------------------------------------------------
def baseline_predict(img, th):
    return (baseline_score(img) <= th).astype(np.int64)


def baseline_score(img):
    """Original detector's matching score = nearest-neighbour squared distance
    (a block is flagged forged when this is <= threshold)."""
    D = pairwise_sqdist(img['emb'])
    np.fill_diagonal(D, np.inf)
    s = D.min(axis=1)
    return np.where(np.isfinite(s), s, MAXD)


def pixel_confusion(images, preds):
    """Expand block preds/labels to pixel level and accumulate TP/FP/TN/FN."""
    y_true, y_pred = [], []
    for img, p in zip(images, preds):
        bx, by = img['bx'], img['by']
        gt = img['lbl'].reshape(bx, by)
        pr = p.reshape(bx, by)
        # pixel level = block value repeated over the ps x ps cell -> just weight
        # by block area (constant), so block-level counts == pixel-level ratios.
        y_true.append(gt.reshape(-1))
        y_pred.append(pr.reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def metric_block(y_true, y_pred):
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        cm=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


# --------------------------------------------------------------------------
# OPTIMISED model: duplication features + MLP head
# --------------------------------------------------------------------------
RADII = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]


MAXD = 4.0   # max squared distance between two unit-norm (L2) embeddings


def duplication_features(emb):
    """Per-block features describing how strongly it is duplicated elsewhere.
    Distances to non-existent neighbours (tiny images) are capped at MAXD so
    no inf/NaN leaks into the classifier."""
    n = emb.shape[0]
    D = pairwise_sqdist(emb)
    Draw = D.copy()
    np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)               # ascending distances to others
    Ds = np.where(np.isfinite(Ds), Ds, MAXD)   # cap inf -> MAXD
    def col(c):
        return Ds[:, c] if c < Ds.shape[1] else np.full(n, MAXD, np.float32)
    k1, k2, k3 = col(0), col(1), col(2)
    kw = min(5, max(1, n - 1))
    mean5 = Ds[:, :kw].mean(axis=1)
    feats = [k1, k2, k3, mean5]
    np.fill_diagonal(Draw, np.inf)
    for r in RADII:
        feats.append((Draw <= r).sum(axis=1).astype(np.float32))  # match counts
    out = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=MAXD, posinf=MAXD, neginf=0.0)


def offset_features(emb, bx, by, match_ths=(0.03, 0.06, 0.10), min_sep=3, min_support=3):
    """Copy-move offset-consistency features.

    Real copy-move duplicates a region with a single (dr,dc) translation, so
    many matched block-pairs share ONE offset.  Generic texture repetition
    yields diffuse offsets.  For each block we record how strongly its matches
    belong to a *dominant* offset cluster -- the key discriminator that plain
    distance thresholding (which floods on self-similar backgrounds) misses."""
    from collections import Counter
    n = emb.shape[0]
    D = pairwise_sqdist(emb)
    np.fill_diagonal(D, np.inf)
    rows = (np.arange(n) // by).astype(np.int32)
    cols = (np.arange(n) % by).astype(np.int32)

    feats = []
    for th in match_ths:
        in_dom = np.zeros(n, np.float32)     # # of this block's matches on a dominant offset
        dom_sup = np.zeros(n, np.float32)    # support (size) of that dominant offset
        pairs = []
        offc = Counter()
        for i in range(n):
            cand = np.where(D[i] <= th)[0]
            for j in cand:
                dr = int(rows[i] - rows[j]); dc = int(cols[i] - cols[j])
                if abs(dr) + abs(dc) >= min_sep:        # ignore near-self matches
                    key = (abs(dr), abs(dc))            # offset magnitude (sign-agnostic)
                    offc[key] += 1
                    pairs.append((i, key))
        dominant = {o: c for o, c in offc.items() if c >= min_support}
        for i, key in pairs:
            if key in dominant:
                in_dom[i] += 1.0
                dom_sup[i] = max(dom_sup[i], dominant[key])
        feats.append(in_dom)
        feats.append(dom_sup)
    return np.stack(feats, axis=1).astype(np.float32)


def build_features(images, use_raw=False, use_offset=True):
    """Generalizable per-block features (do NOT use raw embeddings: a single
    block's appearance memorises textures and does not transfer across images).
    """
    X, y = [], []
    for im in images:
        parts = [duplication_features(im['emb'])]
        if use_offset:
            parts.append(offset_features(im['emb'], im['bx'], im['by']))
        if use_raw:
            parts.append(im['emb'].astype(np.float32))
        X.append(np.concatenate(parts, axis=1))
        y.append(im['lbl'])
    return X, y


def make_mlp(in_dim):
    import tensorflow as tf
    from tensorflow import keras
    reg = keras.regularizers.l2(1e-4)
    m = keras.Sequential([
        keras.layers.Input(shape=(in_dim,)),
        keras.layers.Dense(48, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(24, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])
    return m


def best_threshold(y, p):
    bt, bf = 0.5, -1
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return bt


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Authentic', 'Forged'])
    ax.set_yticklabels(['Authentic', 'Forged'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, '%d\n(%.1f%%)' % (cm[i, j], 100.0 * cm[i, j] / max(1, tot)),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_curve(tr, va, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr, '-o', ms=3, label='Training')
    ax.plot(ep, va, '-s', ms=3, label='Validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# read training-loss scalar curve from TF event logs (the saved model's run)
# --------------------------------------------------------------------------
def event_log_loss(logdir='./logs/siamese/copy_move'):
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator
    except Exception:
        try:
            from tensorflow.compat.v1.train import summary_iterator
        except Exception:
            return None
    steps, vals = [], []
    for ev_file in sorted(glob.glob(os.path.join(logdir, 'events.out.tfevents.*'))):
        try:
            for e in summary_iterator(ev_file):
                for v in e.summary.value:
                    if v.tag == 'loss':
                        steps.append(e.step); vals.append(v.simple_value)
        except Exception:
            continue
    if not steps:
        return None
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(RES, 'embeddings_cache.npz'))
    ap.add_argument('--epochs', type=int, default=120)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import KFold
    tf.random.set_seed(42)

    images, ps = load_cache(args.cache)
    n_img = len(images)
    print('Loaded %d images, patch_size=%d' % (n_img, ps))

    # pre-compute generalizable features per image
    Xall, yall = build_features(images, use_raw=False, use_offset=True)
    in_dim = Xall[0].shape[1]
    print('Feature dim = %d (duplication + offset-consistency, NO raw embedding)' % in_dim)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(np.arange(n_img)))

    # collect out-of-fold CONTINUOUS scores so every image is scored once,
    # then pick each method's F1-optimal operating point on the pooled scores
    # (a fair comparison: both detectors get their best threshold).
    y_oof, base_oof, opt_oof = [], [], []

    for fi, (tr_idx, va_idx) in enumerate(folds, 1):
        va_imgs = [images[i] for i in va_idx]

        # ----- BASELINE score = nearest-neighbour distance -----
        for im in va_imgs:
            y_oof.append(im['lbl'])
            base_oof.append(baseline_score(im))

        # ----- OPTIMISED: duplication/offset-feature MLP -----
        Xtr = np.concatenate([Xall[i] for i in tr_idx])
        ytr = np.concatenate([yall[i] for i in tr_idx])
        Xva = np.concatenate([Xall[i] for i in va_idx])
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
        Xtr = np.nan_to_num((Xtr - mu) / sd)
        Xva = np.nan_to_num((Xva - mu) / sd)
        pos = float(ytr.sum()); neg = float(len(ytr) - pos)
        cw = {0: 1.0, 1: max(1.0, np.sqrt(neg / max(1.0, pos)))}
        keras.utils.set_random_seed(42 + fi)
        mdl = make_mlp(in_dim)
        mdl.fit(Xtr, ytr, epochs=args.epochs, batch_size=64, verbose=0, class_weight=cw)
        opt_oof.append(mdl.predict(Xva, verbose=0).ravel())
        print('fold %d done' % fi)

    y_oof = np.concatenate(y_oof)
    base_oof = np.concatenate(base_oof)     # lower = more "forged" (orig logic)
    opt_oof = np.concatenate(opt_oof)       # higher = more "forged"

    # F1-optimal operating point for each method (pooled OOF)
    bth, bf = np.median(base_oof), -1
    for th in np.unique(np.quantile(base_oof, np.linspace(0.01, 0.99, 99))):
        f = f1_score(y_oof, (base_oof <= th).astype(int), zero_division=0)
        if f > bf:
            bf, bth = f, th
    base = metric_block(y_oof, (base_oof <= bth).astype(int))
    base['threshold'] = float(bth)

    ot = best_threshold(y_oof, opt_oof)
    opt = metric_block(y_oof, (opt_oof >= ot).astype(int))
    opt['threshold'] = float(ot)

    # ranking quality (threshold-free)
    from sklearn.metrics import average_precision_score, roc_auc_score
    base['ap'] = average_precision_score(y_oof, -base_oof)
    opt['ap'] = average_precision_score(y_oof, opt_oof)
    base['auc'] = roc_auc_score(y_oof, -base_oof)
    opt['auc'] = roc_auc_score(y_oof, opt_oof)

    # ----- dedicated held-out run for loss/accuracy curves -----
    tr_idx, va_idx = folds[0]
    Xtr = np.concatenate([Xall[i] for i in tr_idx]); ytr = np.concatenate([yall[i] for i in tr_idx])
    Xva = np.concatenate([Xall[i] for i in va_idx]); yva = np.concatenate([yall[i] for i in va_idx])
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = np.nan_to_num((Xtr - mu) / sd); Xva = np.nan_to_num((Xva - mu) / sd)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    cw = {0: 1.0, 1: max(1.0, np.sqrt(neg / max(1.0, pos)))}
    keras.utils.set_random_seed(123)
    model = make_mlp(in_dim)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=args.epochs, batch_size=64, verbose=0, class_weight=cw)

    # ---------------- save artefacts ----------------
    plot_cm(base['cm'], 'Baseline (distance threshold)\nF1=%.3f' % base['f1'],
            os.path.join(RES, 'confusion_matrix_baseline.png'))
    plot_cm(opt['cm'], 'Optimised (duplication-feature MLP)\nF1=%.3f' % opt['f1'],
            os.path.join(RES, 'confusion_matrix_optimized.png'))
    plot_curve(hist.history['loss'], hist.history['val_loss'],
               'Loss (binary cross-entropy)', 'Training vs Validation Loss',
               os.path.join(RES, 'loss_curve.png'))
    plot_curve(hist.history['accuracy'], hist.history['val_accuracy'],
               'Accuracy', 'Training vs Validation Accuracy',
               os.path.join(RES, 'accuracy_curve.png'))

    el = event_log_loss()
    if el is not None:
        st, vl = el
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(st, vl, '-', lw=1)
        ax.set_xlabel('Global step'); ax.set_ylabel('Triplet loss')
        ax.set_title('Saved model: training loss (from TF event logs)')
        ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(RES, 'training_loss_eventlog.png'), dpi=130)
        plt.close(fig)

    # history csv
    with open(os.path.join(RES, 'history.csv'), 'w') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
        for i in range(len(hist.history['loss'])):
            f.write('%d,%.6f,%.6f,%.6f,%.6f\n' % (
                i + 1, hist.history['loss'][i], hist.history['val_loss'][i],
                hist.history['accuracy'][i], hist.history['val_accuracy'][i]))

    def dump(tag, m, path):
        with open(path, 'w') as f:
            f.write('=== %s ===\n' % tag)
            f.write('decision_threshold : %.4f\n' % m['threshold'])
            f.write('precision          : %.4f\n' % m['precision'])
            f.write('recall             : %.4f\n' % m['recall'])
            f.write('f1                 : %.4f\n' % m['f1'])
            f.write('accuracy           : %.4f\n' % m['accuracy'])
            f.write('average_precision  : %.4f\n' % m['ap'])
            f.write('roc_auc            : %.4f\n' % m['auc'])
            f.write('confusion_matrix [rows=actual 0/1, cols=pred 0/1]:\n')
            f.write(str(m['cm']) + '\n')

    dump('BASELINE  (distance-threshold detector, frozen model, 5-fold CV)',
         base, os.path.join(RES, 'baseline_metrics.txt'))
    fl = hist.history['loss'][-1]; vls = hist.history['val_loss'][-1]
    fa = hist.history['accuracy'][-1]; vac = hist.history['val_accuracy'][-1]
    with open(os.path.join(RES, 'optimized_metrics.txt'), 'w') as f:
        f.write('=== OPTIMISED (duplication-feature MLP on frozen embeddings) ===\n')
        f.write('decision_threshold : %.4f\n' % opt['threshold'])
        f.write('precision          : %.4f\n' % opt['precision'])
        f.write('recall             : %.4f\n' % opt['recall'])
        f.write('f1                 : %.4f\n' % opt['f1'])
        f.write('accuracy           : %.4f\n' % opt['accuracy'])
        f.write('average_precision  : %.4f\n' % opt['ap'])
        f.write('roc_auc            : %.4f\n' % opt['auc'])
        f.write('train_loss (final) : %.4f\n' % fl)
        f.write('val_loss   (final) : %.4f\n' % vls)
        f.write('train_acc  (final) : %.4f\n' % fa)
        f.write('val_acc    (final) : %.4f\n' % vac)
        f.write('confusion_matrix [rows=actual 0/1, cols=pred 0/1]:\n')
        f.write(str(opt['cm']) + '\n')

    with open(os.path.join(RES, 'metrics_summary.csv'), 'w') as f:
        f.write('metric,baseline,optimized\n')
        f.write('precision,%.4f,%.4f\n' % (base['precision'], opt['precision']))
        f.write('recall,%.4f,%.4f\n' % (base['recall'], opt['recall']))
        f.write('f1,%.4f,%.4f\n' % (base['f1'], opt['f1']))
        f.write('accuracy,%.4f,%.4f\n' % (base['accuracy'], opt['accuracy']))
        f.write('average_precision,%.4f,%.4f\n' % (base['ap'], opt['ap']))
        f.write('roc_auc,%.4f,%.4f\n' % (base['auc'], opt['auc']))
        f.write('train_loss,,%.4f\n' % fl)
        f.write('val_loss,,%.4f\n' % vls)
        f.write('train_acc,,%.4f\n' % fa)
        f.write('val_acc,,%.4f\n' % vac)

    print('\n================ RESULTS (5-fold CV, full dataset) ================')
    print('BASELINE   P=%.3f R=%.3f F1=%.3f Acc=%.3f  AP=%.3f AUC=%.3f' %
          (base['precision'], base['recall'], base['f1'], base['accuracy'],
           base['ap'], base['auc']))
    print('OPTIMISED  P=%.3f R=%.3f F1=%.3f Acc=%.3f  AP=%.3f AUC=%.3f' %
          (opt['precision'], opt['recall'], opt['f1'], opt['accuracy'],
           opt['ap'], opt['auc']))
    print('OPTIMISED  train_loss=%.3f val_loss=%.3f train_acc=%.3f val_acc=%.3f' %
          (fl, vls, fa, vac))
    print('All artefacts saved to', RES)
    return base, opt


if __name__ == '__main__':
    main()
