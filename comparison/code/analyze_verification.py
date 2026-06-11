# analyze_verification.py
"""
Phase 2 of the verification evaluation.  Loads cached embeddings from
results/verification/ver_embeddings.npz and computes:

  BASELINE  : raw L2 embedding-distance verification, 10-fold CV (the repo's
              facenet.calculate_roc methodology).
  OPTIMISED : a learned MLP verifier on the |a-b|^2 embedding-difference vector
              (a *different model*), with train/validation loss & accuracy curves.

Reports genuine-vs-impostor accuracy / precision / recall / F1 / confusion matrix
for both, on a held-out split (New) and on the independent Old set.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score,
                             average_precision_score, roc_auc_score)

OUT = './results/verification'
RNG = np.random.RandomState(42)


def sqdist(a, b):
    return np.sum((a - b) ** 2, axis=1)


def calc_acc(th, dist, issame):
    pred = np.less(dist, th)
    return (np.sum(pred == issame)) / dist.size


def calculate_roc(thresholds, dist, issame, nrof_folds=10):
    kf = KFold(n_splits=nrof_folds, shuffle=False)
    idx = np.arange(len(issame)); acc = np.zeros(nrof_folds); bts = np.zeros(nrof_folds)
    for fi, (tr, te) in enumerate(kf.split(idx)):
        at = [calc_acc(t, dist[tr], issame[tr]) for t in thresholds]
        bt = thresholds[int(np.argmax(at))]; bts[fi] = bt
        acc[fi] = calc_acc(bt, dist[te], issame[te])
    return acc, bts


def metrics_pred(issame, pred):
    return dict(accuracy=accuracy_score(issame, pred),
                precision=precision_score(issame, pred, zero_division=0),
                recall=recall_score(issame, pred, zero_division=0),
                f1=f1_score(issame, pred, zero_division=0),
                cm=confusion_matrix(issame, pred, labels=[False, True]))


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


def make_pairs(ea, ep_, en):
    e1 = np.concatenate([ea, ea], 0)
    e2 = np.concatenate([ep_, en], 0)
    issame = np.concatenate([np.ones(len(ea), bool), np.zeros(len(ea), bool)])
    o = RNG.permutation(len(issame))
    return e1[o], e2[o], issame[o]


def main():
    d = np.load(os.path.join(OUT, 'ver_embeddings.npz'))
    e1, e2, issame = make_pairs(d['ea'], d['ep'], d['en'])
    e1o, e2o, issame_o = make_pairs(d['eao'], d['epo'], d['eno'])
    print('New pairs=%d  Old pairs=%d  dim=%d' % (len(issame), len(issame_o), e1.shape[1]))

    thresholds = np.arange(0.0, 4.0, 0.005)

    # ================= BASELINE =================
    dist = sqdist(e1, e2)
    acc_cv, bts = calculate_roc(thresholds, dist, issame, 10)
    gth = float(np.mean(bts))
    base = metrics_pred(issame, dist < gth)
    base.update(cv_accuracy=float(acc_cv.mean()), cv_std=float(acc_cv.std()),
                threshold=gth, auc=roc_auc_score(issame, -dist),
                ap=average_precision_score(issame, -dist))
    disto = sqdist(e1o, e2o)
    base_old = metrics_pred(issame_o, disto < gth)
    base_old['auc'] = roc_auc_score(issame_o, -disto)

    # ================= OPTIMISED: learned MLP verifier =================
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(42)

    def pair_feats(a, b):
        # rich learned-metric features (not just squared diff)
        return np.concatenate([(a - b) ** 2, a * b, a, b,
                               np.abs(a - b)], axis=1).astype(np.float32)

    feat = pair_feats(e1, e2)
    feat_o = pair_feats(e1o, e2o)
    mu = feat.mean(0); sd = feat.std(0) + 1e-8
    X = (feat - mu) / sd; Xo = (feat_o - mu) / sd
    y = issame.astype(np.float32)

    ntr = int(0.8 * len(y)); perm = RNG.permutation(len(y))
    tr_i, va_i = perm[:ntr], perm[ntr:]
    Xtr, ytr, Xva, yva = X[tr_i], y[tr_i], X[va_i], y[va_i]

    reg = keras.regularizers.l2(1e-4)
    def block(units):
        return [keras.layers.Dense(units, kernel_regularizer=reg),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.3)]
    model = keras.Sequential([keras.layers.Input(shape=(X.shape[1],))]
                             + block(256) + block(128) + block(64)
                             + [keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    cb = [keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
          keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=150, batch_size=128, verbose=0, callbacks=cb)

    pva = model.predict(Xva, verbose=0).ravel()
    opt = metrics_pred(yva.astype(bool), pva >= 0.5)
    opt.update(auc=roc_auc_score(yva, pva), ap=average_precision_score(yva, pva))
    po = model.predict(Xo, verbose=0).ravel()
    opt_old = metrics_pred(issame_o, po >= 0.5)
    opt_old['auc'] = roc_auc_score(issame_o, po)

    fl, vls = hist.history['loss'][-1], hist.history['val_loss'][-1]
    fa, vac = hist.history['accuracy'][-1], hist.history['val_accuracy'][-1]

    # ----------------------------- artefacts -----------------------------
    plot_cm(base['cm'], 'Baseline verification (raw distance)\nCV acc=%.3f' % base['cv_accuracy'],
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
        f.write('cv_accuracy        : %.4f +/- %.4f\n' % (base['cv_accuracy'], base['cv_std']))
        f.write('threshold (mean)   : %.4f\n' % base['threshold'])
        f.write('precision          : %.4f\n' % base['precision'])
        f.write('recall             : %.4f\n' % base['recall'])
        f.write('f1                 : %.4f\n' % base['f1'])
        f.write('accuracy@threshold : %.4f\n' % base['accuracy'])
        f.write('roc_auc            : %.4f\n' % base['auc'])
        f.write('average_precision  : %.4f\n' % base['ap'])
        f.write('held-out(Old) acc  : %.4f   f1 %.4f   auc %.4f\n'
                % (base_old['accuracy'], base_old['f1'], base_old['auc']))
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
        f.write('held-out(Old) acc  : %.4f   f1 %.4f   auc %.4f\n'
                % (opt_old['accuracy'], opt_old['f1'], opt_old['auc']))
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
    print('BASELINE  CV_acc=%.4f+/-%.4f  P=%.3f R=%.3f F1=%.3f AUC=%.4f | Old acc=%.4f'
          % (base['cv_accuracy'], base['cv_std'], base['precision'], base['recall'],
             base['f1'], base['auc'], base_old['accuracy']))
    print('OPTIMISED acc=%.4f  P=%.3f R=%.3f F1=%.3f AUC=%.4f | Old acc=%.4f'
          % (opt['accuracy'], opt['precision'], opt['recall'], opt['f1'],
             opt['auc'], opt_old['accuracy']))
    print('OPTIMISED train/val loss=%.3f/%.3f  train/val acc=%.3f/%.3f'
          % (fl, vls, fa, vac))
    print('Artefacts ->', OUT)


if __name__ == '__main__':
    main()
