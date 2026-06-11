# evaluate_verification.py
"""
Verification-protocol evaluation (the methodology used by the source repo
github.com/abhithakur25/knuckleRecognitionCode -> validate_on_fkp.py +
facenet.calculate_roc).

The copy-move model is a TRIPLET network: every training sample is a triplet
(anchor, positive, negative) where the positive is the genuine copy-move match
and the negative is an unrelated patch.  The task the model was actually trained
for is VERIFICATION: "are these two patches a genuine match?".  We evaluate it
exactly as the knuckle repo does:

  genuine  pairs = (anchor, positive)   -> should have small embedding distance
  impostor pairs = (anchor, negative)   -> should have large embedding distance

BASELINE  : raw L2 embedding distance + best threshold, scored with 10-fold
            cross-validation (facenet.calculate_roc, re-implemented here to avoid
            the legacy `scipy.misc` import in facenet.py).
OPTIMISED : a learned verifier (MLP on the |a-b|^2 embedding-difference vector)
            -- a *different model* on top of the frozen embeddings, giving proper
            train/validation loss & accuracy curves.

The model is used purely for inference; it is never retrained.
Outputs -> results/verification/
"""

import os
import re
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score,
                             average_precision_score, roc_auc_score)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tfv1 = tf.compat.v1

OUT = './results/verification'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.RandomState(42)


# ----------------------------------------------------------------------
# model loading (highest-step checkpoint, as in extract_embeddings.py)
# ----------------------------------------------------------------------
def pick_checkpoint(model_dir):
    files = os.listdir(model_dir)
    meta = [f for f in files if f.endswith('.meta')][0]
    best_step, best = -1, None
    for f in files:
        m = re.match(r'(model-.+\.ckpt-(\d+))\.index$', f)
        if m and int(m.group(2)) > best_step:
            best_step, best = int(m.group(2)), m.group(1)
    return os.path.join(model_dir, meta), os.path.join(model_dir, best), best_step


def find_embeddings_tensor(graph):
    try:
        return graph.get_tensor_by_name("embeddings:0")
    except Exception:
        for op in graph.get_operations():
            for o in op.outputs:
                if 'embedd' in o.name.lower():
                    return o
    return None


# ----------------------------------------------------------------------
# triplet loaders
# ----------------------------------------------------------------------
def load_triplets_new(data_dir, n):
    files = os.listdir(data_dir)
    RNG.shuffle(files)
    A, P, N = [], [], []
    for f in files:
        if len(A) >= n:
            break
        try:
            a = np.load(os.path.join(data_dir, f), allow_pickle=True)
        except Exception:
            continue
        if a.shape == (3, 30, 30):
            A.append(a[0]); P.append(a[1]); N.append(a[2])
    return np.array(A), np.array(P), np.array(N)


def load_triplets_old(data_dir, n):
    """Data/Old files are (M,3,30,30) stacks named by the real forged images."""
    files = sorted(os.listdir(data_dir))
    A, P, N = [], [], []
    for f in files:
        try:
            arr = np.load(os.path.join(data_dir, f), allow_pickle=True)
        except Exception:
            continue
        if arr.ndim == 4 and arr.shape[1:] == (3, 30, 30):
            for t in arr:
                A.append(t[0]); P.append(t[1]); N.append(t[2])
        if len(A) >= n:
            break
    idx = RNG.permutation(len(A))[:n]
    A, P, N = np.array(A), np.array(P), np.array(N)
    return A[idx], P[idx], N[idx]


def embed(sess, emb_t, image_ph, phase_ph, patches, bs=512):
    """patches: (K,30,30) uint8 -> (K,dim) embeddings."""
    x = patches.astype(np.float32) / 255.0
    x = x.reshape(-1, 30, 30, 1)
    out = []
    for i in range(0, x.shape[0], bs):
        feed = {image_ph: x[i:i+bs]}
        try:
            feed[phase_ph] = False
            out.append(sess.run(emb_t, feed_dict=feed))
        except Exception:
            out.append(sess.run(emb_t, feed_dict={image_ph: x[i:i+bs]}))
    return np.concatenate(out, 0).astype(np.float32)


# ----------------------------------------------------------------------
# verification metrics (re-implemented from facenet.py)
# ----------------------------------------------------------------------
def sqdist(a, b):
    return np.sum((a - b) ** 2, axis=1)


def calc_acc(threshold, dist, issame):
    pred = np.less(dist, threshold)
    tp = np.sum(pred & issame); fp = np.sum(pred & ~issame)
    tn = np.sum(~pred & ~issame); fn = np.sum(~pred & issame)
    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (fp + tn == 0) else fp / (fp + tn)
    return tpr, fpr, (tp + tn) / dist.size


def calculate_roc(thresholds, dist, issame, nrof_folds=10):
    kf = KFold(n_splits=nrof_folds, shuffle=False)
    idx = np.arange(len(issame))
    acc = np.zeros(nrof_folds)
    best_thrs = np.zeros(nrof_folds)
    for fi, (tr, te) in enumerate(kf.split(idx)):
        acc_tr = [calc_acc(t, dist[tr], issame[tr])[2] for t in thresholds]
        bt = thresholds[int(np.argmax(acc_tr))]
        best_thrs[fi] = bt
        acc[fi] = calc_acc(bt, dist[te], issame[te])[2]
    return acc, best_thrs


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap='Greens')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Impostor', 'Genuine']); ax.set_yticklabels(['Impostor', 'Genuine'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, '%d\n(%.1f%%)' % (cm[i, j], 100.0*cm[i, j]/max(1, tot)),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
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


def metrics_at(threshold, dist, issame):
    pred = (dist < threshold)
    return dict(
        accuracy=accuracy_score(issame, pred),
        precision=precision_score(issame, pred, zero_division=0),
        recall=recall_score(issame, pred, zero_division=0),
        f1=f1_score(issame, pred, zero_division=0),
        cm=confusion_matrix(issame, pred, labels=[False, True]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='./models/siamese/copy_move')
    ap.add_argument('--data_new', default='../Data/New')
    ap.add_argument('--data_old', default='../Data/Old')
    ap.add_argument('--n', type=int, default=4000, help='triplets to sample')
    ap.add_argument('--epochs', type=int, default=60)
    args = ap.parse_args()

    print('Loading triplets...')
    A, P, N = load_triplets_new(args.data_new, args.n)
    print('  Data/New triplets:', A.shape[0])
    Ao, Po, No = load_triplets_old(args.data_old, args.n)
    print('  Data/Old triplets (held-out):', Ao.shape[0])

    g = tfv1.Graph()
    with g.as_default(), tfv1.Session() as sess:
        image_ph = tfv1.placeholder(tf.float32, shape=(None, 30, 30, 1), name='image')
        phase_ph = tfv1.placeholder(tf.bool, name='phase_train')
        lr_ph = tfv1.placeholder(tf.float32, name='learning_rate')
        al_ph = tfv1.placeholder(tf.float32, name='dynamic_alpha_placeholder')
        input_map = {'image': image_ph, 'phase_train': phase_ph,
                     'learning_rate': lr_ph, 'dynamic_alpha_placeholder': al_ph}
        meta, ckpt, step = pick_checkpoint(args.model)
        print('Restoring %s (step=%d)' % (ckpt, step))
        saver = tfv1.train.import_meta_graph(meta, input_map=input_map)
        saver.restore(sess, ckpt)
        emb_t = find_embeddings_tensor(g)

        ea = embed(sess, emb_t, image_ph, phase_ph, A)
        ep_ = embed(sess, emb_t, image_ph, phase_ph, P)
        en = embed(sess, emb_t, image_ph, phase_ph, N)
        eao = embed(sess, emb_t, image_ph, phase_ph, Ao)
        epo = embed(sess, emb_t, image_ph, phase_ph, Po)
        eno = embed(sess, emb_t, image_ph, phase_ph, No)

    # The TF1 graph requires eager DISABLED, but the Keras verifier needs eager
    # ENABLED -- they cannot coexist in one process.  So we only extract & cache
    # embeddings here; analyze_verification.py does ROC + the learned verifier.
    cache = os.path.join(OUT, 'ver_embeddings.npz')
    np.savez_compressed(cache, ea=ea, ep=ep_, en=en, eao=eao, epo=epo, eno=eno)
    print('Saved embeddings ->', cache,
          '| New triplets=%d  Old triplets=%d  dim=%d' % (len(ea), len(eao), ea.shape[1]))
    return


def _unused_make_pairs():
    def make_pairs(ea, ep_, en):
        e1 = np.concatenate([ea, ea], 0)
        e2 = np.concatenate([ep_, en], 0)
        issame = np.concatenate([np.ones(len(ea), bool), np.zeros(len(ea), bool)])
        order = RNG.permutation(len(issame))
        return e1[order], e2[order], issame[order]

    e1, e2, issame = make_pairs(ea, ep_, en)
    e1o, e2o, issame_o = make_pairs(eao, epo, eno)

    thresholds = np.arange(0.0, 4.0, 0.01)

    # ================= BASELINE: raw-distance verification =================
    dist = sqdist(e1, e2)
    acc_cv, best_thrs = calculate_roc(thresholds, dist, issame, nrof_folds=10)
    gth = float(np.mean(best_thrs))
    base = metrics_at(gth, dist, issame)
    base['cv_accuracy'] = float(np.mean(acc_cv))
    base['cv_accuracy_std'] = float(np.std(acc_cv))
    base['auc'] = roc_auc_score(issame, -dist)
    base['ap'] = average_precision_score(issame, -dist)
    base['threshold'] = gth
    # held-out (Old)
    disto = sqdist(e1o, e2o)
    base_old = metrics_at(gth, disto, issame_o)

    # ================= OPTIMISED: learned MLP verifier =====================
    from tensorflow import keras
    keras.utils.set_random_seed(42)
    feat = (e1 - e2) ** 2                      # |a-b|^2 difference vector (128-d)
    feat_o = (e1o - e2o) ** 2
    mu = feat.mean(0); sd = feat.std(0) + 1e-8
    Xtr_all = (feat - mu) / sd
    Xo = (feat_o - mu) / sd
    y = issame.astype(np.float32)

    # train/val split (within New)
    ntr = int(0.8 * len(y))
    perm = RNG.permutation(len(y))
    tr_i, va_i = perm[:ntr], perm[ntr:]
    Xtr, ytr = Xtr_all[tr_i], y[tr_i]
    Xva, yva = Xtr_all[va_i], y[va_i]

    reg = keras.regularizers.l2(1e-4)
    model = keras.Sequential([
        keras.layers.Input(shape=(Xtr.shape[1],)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=args.epochs, batch_size=128, verbose=0)

    pva = model.predict(Xva, verbose=0).ravel()
    yhat = (pva >= 0.5).astype(int)
    opt = dict(
        accuracy=accuracy_score(yva, yhat),
        precision=precision_score(yva, yhat, zero_division=0),
        recall=recall_score(yva, yhat, zero_division=0),
        f1=f1_score(yva, yhat, zero_division=0),
        cm=confusion_matrix(yva, yhat, labels=[0, 1]),
        auc=roc_auc_score(yva, pva),
        ap=average_precision_score(yva, pva),
    )
    # held-out Old
    po = model.predict(Xo, verbose=0).ravel()
    yho = (po >= 0.5).astype(int)
    opt_old = dict(accuracy=accuracy_score(issame_o, yho),
                   f1=f1_score(issame_o, yho, zero_division=0),
                   auc=roc_auc_score(issame_o, po))

    fl = hist.history['loss'][-1]; vls = hist.history['val_loss'][-1]
    fa = hist.history['accuracy'][-1]; vac = hist.history['val_accuracy'][-1]

    # ----------------------------- artefacts -----------------------------
    plot_cm(base['cm'], 'Baseline verification (raw distance)\nacc=%.3f' % base['cv_accuracy'],
            os.path.join(OUT, 'confusion_matrix_baseline.png'))
    plot_cm(opt['cm'], 'Optimised verifier (MLP)\nacc=%.3f' % opt['accuracy'],
            os.path.join(OUT, 'confusion_matrix_optimized.png'))
    plot_curve(hist.history['loss'], hist.history['val_loss'],
               'Loss (binary cross-entropy)', 'Verifier: Training vs Validation Loss',
               os.path.join(OUT, 'loss_curve.png'))
    plot_curve(hist.history['accuracy'], hist.history['val_accuracy'],
               'Accuracy', 'Verifier: Training vs Validation Accuracy',
               os.path.join(OUT, 'accuracy_curve.png'))

    with open(os.path.join(OUT, 'history.csv'), 'w') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
        for i in range(len(hist.history['loss'])):
            f.write('%d,%.6f,%.6f,%.6f,%.6f\n' % (
                i+1, hist.history['loss'][i], hist.history['val_loss'][i],
                hist.history['accuracy'][i], hist.history['val_accuracy'][i]))

    with open(os.path.join(OUT, 'baseline_metrics.txt'), 'w') as f:
        f.write('=== BASELINE  (raw embedding-distance verification, 10-fold CV) ===\n')
        f.write('cv_accuracy        : %.4f +/- %.4f\n' % (base['cv_accuracy'], base['cv_accuracy_std']))
        f.write('threshold (mean)   : %.4f\n' % base['threshold'])
        f.write('precision          : %.4f\n' % base['precision'])
        f.write('recall             : %.4f\n' % base['recall'])
        f.write('f1                 : %.4f\n' % base['f1'])
        f.write('accuracy@threshold : %.4f\n' % base['accuracy'])
        f.write('roc_auc            : %.4f\n' % base['auc'])
        f.write('average_precision  : %.4f\n' % base['ap'])
        f.write('held-out(Old) acc  : %.4f   f1 %.4f\n' % (base_old['accuracy'], base_old['f1']))
        f.write('confusion_matrix [rows=actual imp/gen, cols=pred imp/gen]:\n')
        f.write(str(base['cm']) + '\n')

    with open(os.path.join(OUT, 'optimized_metrics.txt'), 'w') as f:
        f.write('=== OPTIMISED  (learned MLP verifier on frozen embeddings) ===\n')
        f.write('accuracy           : %.4f\n' % opt['accuracy'])
        f.write('precision          : %.4f\n' % opt['precision'])
        f.write('recall             : %.4f\n' % opt['recall'])
        f.write('f1                 : %.4f\n' % opt['f1'])
        f.write('roc_auc            : %.4f\n' % opt['auc'])
        f.write('average_precision  : %.4f\n' % opt['ap'])
        f.write('train_loss (final) : %.4f\n' % fl)
        f.write('val_loss   (final) : %.4f\n' % vls)
        f.write('train_acc  (final) : %.4f\n' % fa)
        f.write('val_acc    (final) : %.4f\n' % vac)
        f.write('held-out(Old) acc  : %.4f   f1 %.4f   auc %.4f\n' %
                (opt_old['accuracy'], opt_old['f1'], opt_old['auc']))
        f.write('confusion_matrix [rows=actual imp/gen, cols=pred imp/gen]:\n')
        f.write(str(opt['cm']) + '\n')

    with open(os.path.join(OUT, 'metrics_summary.csv'), 'w') as f:
        f.write('metric,baseline,optimized\n')
        f.write('accuracy,%.4f,%.4f\n' % (base['cv_accuracy'], opt['accuracy']))
        f.write('precision,%.4f,%.4f\n' % (base['precision'], opt['precision']))
        f.write('recall,%.4f,%.4f\n' % (base['recall'], opt['recall']))
        f.write('f1,%.4f,%.4f\n' % (base['f1'], opt['f1']))
        f.write('roc_auc,%.4f,%.4f\n' % (base['auc'], opt['auc']))
        f.write('average_precision,%.4f,%.4f\n' % (base['ap'], opt['ap']))
        f.write('held_out_old_acc,%.4f,%.4f\n' % (base_old['accuracy'], opt_old['accuracy']))

    print('\n=============== VERIFICATION RESULTS ===============')
    print('Pairs: %d genuine + %d impostor (New)' % (len(ea), len(en)))
    print('BASELINE  cv_acc=%.4f+/-%.4f  P=%.3f R=%.3f F1=%.3f AUC=%.3f  | Old acc=%.3f'
          % (base['cv_accuracy'], base['cv_accuracy_std'], base['precision'],
             base['recall'], base['f1'], base['auc'], base_old['accuracy']))
    print('OPTIMISED acc=%.4f  P=%.3f R=%.3f F1=%.3f AUC=%.3f  | Old acc=%.3f'
          % (opt['accuracy'], opt['precision'], opt['recall'], opt['f1'],
             opt['auc'], opt_old['accuracy']))
    print('OPTIMISED train/val loss=%.3f/%.3f  train/val acc=%.3f/%.3f'
          % (fl, vls, fa, vac))
    print('Artefacts ->', OUT)


if __name__ == '__main__':
    main()
