# compare_models.py
"""
Model comparison for the copy-move VERIFICATION task.

Protocol (proper train / validation / test):
  * TRAIN      : pairs built from ../Data/New triplets  (the training images)
  * VALIDATION : 20% held-out split of the New pairs
  * TEST       : pairs built from ../Data/Old triplets  (the test images)

The frozen `copy_move` Siamese model (checkpoint ckpt-5004) supplies the 128-d
embeddings (cached by evaluate_verification.py); it is never retrained.  On top
of those frozen embeddings we compare SIX similar verifier models:

  M1_Logistic     linear verifier (logistic regression as a 1-layer NN)
  M2_Shallow      one hidden layer (64)
  M3_DeepBN       256-128-64 with BatchNorm     <-- EXISTING model (99% verifier)
  M4_Wide         512-256 wide MLP
  M5_Tanh         128-64 tanh MLP
  M6_DeepNarrow   128-128-128-128 deep-narrow MLP (RMSprop)

All six produce per-epoch train/val loss & accuracy, a confusion matrix and an
ROC curve on the TEST set, so they can be charted uniformly.

Outputs:
  ../result/  : every CSV + PNG
  ../model/   : every trained model (.keras) + the BEST one copied as best_model.keras
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score, roc_curve, auc,
                             average_precision_score)

import tensorflow as tf
from tensorflow import keras

# ---------------- paths ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # .../comparison
SRC = os.path.dirname(BASE)                        # .../src
RESULT = os.path.join(BASE, 'result')
MODELDIR = os.path.join(BASE, 'model')
EMB = os.path.join(SRC, 'results', 'verification', 'ver_embeddings.npz')
for d in (RESULT, MODELDIR):
    os.makedirs(d, exist_ok=True)

RNG = np.random.RandomState(42)
keras.utils.set_random_seed(42)

COLORS = ['#4f81bd', '#c0504d', '#9bbb59', '#8064a2', '#4bacc6', '#f79646']


# ---------------- data ----------------
def pair_feats(a, b):
    return np.concatenate([(a - b) ** 2, a * b, a, b, np.abs(a - b)],
                          axis=1).astype(np.float32)


def make_pairs(ea, ep_, en, rng):
    e1 = np.concatenate([ea, ea], 0)
    e2 = np.concatenate([ep_, en], 0)
    issame = np.concatenate([np.ones(len(ea), bool), np.zeros(len(ea), bool)])
    o = rng.permutation(len(issame))
    return e1[o], e2[o], issame[o].astype(np.float32)


def load_data():
    d = np.load(EMB)
    e1, e2, y = make_pairs(d['ea'], d['ep'], d['en'], RNG)       # New  -> train/val
    e1t, e2t, yt = make_pairs(d['eao'], d['epo'], d['eno'], RNG)  # Old  -> test
    X = pair_feats(e1, e2)
    Xt = pair_feats(e1t, e2t)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    X = (X - mu) / sd; Xt = (Xt - mu) / sd
    ntr = int(0.8 * len(y)); perm = RNG.permutation(len(y))
    tr, va = perm[:ntr], perm[ntr:]
    return (X[tr], y[tr]), (X[va], y[va]), (Xt, yt), X.shape[1]


# ---------------- models ----------------
def build(name, in_dim):
    reg = keras.regularizers.l2(1e-4)
    inp = keras.layers.Input(shape=(in_dim,))
    if name == 'M1_Logistic':
        out = keras.layers.Dense(1, activation='sigmoid')(inp)
        opt = keras.optimizers.Adam(1e-3)
    elif name == 'M2_Shallow':
        x = keras.layers.Dense(64, activation='relu', kernel_regularizer=reg)(inp)
        x = keras.layers.Dropout(0.3)(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        opt = keras.optimizers.Adam(1e-3)
    elif name == 'M3_DeepBN':                       # EXISTING model
        x = inp
        for u in (256, 128, 64):
            x = keras.layers.Dense(u, kernel_regularizer=reg)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.Dropout(0.3)(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        opt = keras.optimizers.Adam(1e-3)
    elif name == 'M4_Wide':
        x = keras.layers.Dense(512, activation='relu', kernel_regularizer=reg)(inp)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
        x = keras.layers.Dropout(0.3)(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        opt = keras.optimizers.Adam(1e-3)
    elif name == 'M5_Tanh':
        x = keras.layers.Dense(128, activation='tanh', kernel_regularizer=reg)(inp)
        x = keras.layers.Dense(64, activation='tanh', kernel_regularizer=reg)(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        opt = keras.optimizers.Adam(1e-3)
    elif name == 'M6_DeepNarrow':
        x = inp
        for _ in range(4):
            x = keras.layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
            x = keras.layers.Dropout(0.2)(x)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        opt = keras.optimizers.RMSprop(1e-3)
    else:
        raise ValueError(name)
    m = keras.Model(inp, out)
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return m


DISPLAY = {
    'M1_Logistic': 'M1 Logistic (linear)',
    'M2_Shallow': 'M2 Shallow MLP',
    'M3_DeepBN': 'M3 Deep+BN  (existing)',
    'M4_Wide': 'M4 Wide MLP',
    'M5_Tanh': 'M5 Tanh MLP',
    'M6_DeepNarrow': 'M6 Deep-Narrow',
}
ORDER = ['M1_Logistic', 'M2_Shallow', 'M3_DeepBN', 'M4_Wide', 'M5_Tanh', 'M6_DeepNarrow']


# ---------------- plotting ----------------
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap='Blues')
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


def plot_model_curves(h, disp, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(h['loss']) + 1)
    ax[0].plot(ep, h['loss'], '-o', ms=3, label='Training loss')
    ax[0].plot(ep, h['val_loss'], '-s', ms=3, label='Validation loss')
    ax[0].set_title('%s — Loss' % disp); ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Binary cross-entropy'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, h['accuracy'], '-o', ms=3, label='Training acc')
    ax[1].plot(ep, h['val_accuracy'], '-s', ms=3, label='Validation acc')
    ax[1].set_title('%s — Accuracy' % disp); ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy'); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def combined_line(histories, key, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, k in enumerate(ORDER):
        ax.plot(range(1, len(histories[k][key]) + 1), histories[k][key],
                color=COLORS[i], label=DISPLAY[k])
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def combined_roc(rocs, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for i, k in enumerate(ORDER):
        fpr, tpr, a = rocs[k]
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.8,
                label='%s (AUC=%.4f)' % (DISPLAY[k], a))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curves — TEST set (Data/Old)'); ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def grouped_bar(metrics, keys, klabels, title, path, ylim=None, annotate='%.3f'):
    x = np.arange(len(ORDER)); n = len(keys); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, mk in enumerate(keys):
        vals = [metrics[m][mk] for m in ORDER]
        b = ax.bar(x + (j - (n-1)/2) * w, vals, w, label=klabels[j], color=COLORS[j % len(COLORS)])
        for bb in b:
            ax.text(bb.get_x()+bb.get_width()/2, bb.get_height(),
                    annotate % bb.get_height(), ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in ORDER], rotation=15, fontsize=8)
    ax.set_title(title); ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------- main ----------------
def main():
    (Xtr, ytr), (Xva, yva), (Xt, yt), in_dim = load_data()
    print('train=%d  val=%d  test=%d  feat_dim=%d' % (len(ytr), len(yva), len(yt), in_dim))

    metrics, histories, rocs = {}, {}, {}
    for name in ORDER:
        keras.utils.set_random_seed(42)
        m = build(name, in_dim)
        cb = [keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
              keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=120, batch_size=128,
                  verbose=0, callbacks=cb)
        histories[name] = h.history

        tr_acc = m.evaluate(Xtr, ytr, verbose=0)[1]
        va_acc = m.evaluate(Xva, yva, verbose=0)[1]
        prob = m.predict(Xt, verbose=0).ravel()
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(yt, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(yt, prob); a = auc(fpr, tpr)
        rocs[name] = (fpr, tpr, a)
        metrics[name] = dict(
            train_acc=float(tr_acc), val_acc=float(va_acc),
            test_acc=float(accuracy_score(yt, pred)),
            precision=float(precision_score(yt, pred, zero_division=0)),
            recall=float(recall_score(yt, pred, zero_division=0)),
            f1=float(f1_score(yt, pred, zero_division=0)),
            FN=int(fn), FP=int(fp), TP=int(tp), TN=int(tn),
            roc_auc=float(a),
            avg_precision=float(average_precision_score(yt, prob)),
            train_loss=float(h.history['loss'][-1]),
            val_loss=float(h.history['val_loss'][-1]),
        )
        # save model + per-model artefacts
        m.save(os.path.join(MODELDIR, name + '.keras'))
        plot_cm(cm, '%s — TEST\nacc=%.4f' % (DISPLAY[name], metrics[name]['test_acc']),
                os.path.join(RESULT, 'confusion_%s.png' % name))
        plot_model_curves(h.history, DISPLAY[name],
                          os.path.join(RESULT, 'curves_%s.png' % name))
        with open(os.path.join(RESULT, 'history_%s.csv' % name), 'w') as f:
            f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
            for i in range(len(h.history['loss'])):
                f.write('%d,%.6f,%.6f,%.6f,%.6f\n' % (
                    i+1, h.history['loss'][i], h.history['val_loss'][i],
                    h.history['accuracy'][i], h.history['val_accuracy'][i]))
        print('%-16s test_acc=%.4f P=%.3f R=%.3f F1=%.3f FN=%d AUC=%.4f'
              % (name, metrics[name]['test_acc'], metrics[name]['precision'],
                 metrics[name]['recall'], metrics[name]['f1'],
                 metrics[name]['FN'], metrics[name]['roc_auc']))

    # ---------- combined line graphs ----------
    combined_line(histories, 'loss', 'Training loss', 'Training Loss (all models)',
                  os.path.join(RESULT, 'line_training_loss.png'))
    combined_line(histories, 'val_loss', 'Validation loss', 'Validation Loss (all models)',
                  os.path.join(RESULT, 'line_validation_loss.png'))
    combined_line(histories, 'accuracy', 'Training accuracy', 'Training Accuracy (all models)',
                  os.path.join(RESULT, 'line_training_accuracy.png'))
    combined_line(histories, 'val_accuracy', 'Validation accuracy', 'Validation Accuracy (all models)',
                  os.path.join(RESULT, 'line_validation_accuracy.png'))

    # ---------- combined ROC ----------
    combined_roc(rocs, os.path.join(RESULT, 'roc_all_models.png'))

    # ---------- bar graphs ----------
    grouped_bar(metrics, ['train_acc', 'val_acc', 'test_acc'],
                ['Train acc', 'Val acc', 'Test acc'],
                'Training / Validation / Test Accuracy', os.path.join(RESULT, 'bar_accuracy.png'),
                ylim=(0, 1.08))
    grouped_bar(metrics, ['precision', 'recall', 'f1'],
                ['Precision', 'Recall', 'F1'],
                'Precision / Recall / F1 (TEST)', os.path.join(RESULT, 'bar_prf.png'),
                ylim=(0, 1.08))
    grouped_bar(metrics, ['FN'], ['False Negatives'],
                'False Negatives on TEST set', os.path.join(RESULT, 'bar_FN.png'),
                annotate='%.0f')

    # ---------- summary csv ----------
    cols = ['train_acc', 'val_acc', 'test_acc', 'precision', 'recall', 'f1',
            'FN', 'FP', 'TP', 'TN', 'roc_auc', 'avg_precision', 'train_loss', 'val_loss']
    with open(os.path.join(RESULT, 'comparison_summary.csv'), 'w') as f:
        f.write('model,' + ','.join(cols) + '\n')
        for k in ORDER:
            f.write(DISPLAY[k] + ',' + ','.join(
                ('%.4f' % metrics[k][c]) if isinstance(metrics[k][c], float) else str(metrics[k][c])
                for c in cols) + '\n')

    # ---------- choose best (by test accuracy, tie-break F1 then AUC) ----------
    best = max(ORDER, key=lambda k: (metrics[k]['test_acc'], metrics[k]['f1'], metrics[k]['roc_auc']))
    import shutil
    shutil.copy(os.path.join(MODELDIR, best + '.keras'),
                os.path.join(MODELDIR, 'best_model.keras'))
    with open(os.path.join(RESULT, 'BEST_MODEL.txt'), 'w') as f:
        f.write('BEST MODEL: %s (%s)\n' % (best, DISPLAY[best]))
        f.write('test_accuracy=%.4f  f1=%.4f  roc_auc=%.4f  FN=%d\n' % (
            metrics[best]['test_acc'], metrics[best]['f1'],
            metrics[best]['roc_auc'], metrics[best]['FN']))
        f.write('existing model = M3_DeepBN (test_acc=%.4f)\n' % metrics['M3_DeepBN']['test_acc'])
    json.dump(metrics, open(os.path.join(RESULT, 'all_metrics.json'), 'w'), indent=2)

    print('\n================ COMPARISON (TEST = Data/Old) ================')
    for k in ORDER:
        tag = '  <-- EXISTING' if k == 'M3_DeepBN' else ''
        star = '  *** BEST ***' if k == best else ''
        print('%-22s acc=%.4f F1=%.4f AUC=%.4f FN=%d%s%s'
              % (DISPLAY[k], metrics[k]['test_acc'], metrics[k]['f1'],
                 metrics[k]['roc_auc'], metrics[k]['FN'], tag, star))
    print('\nBEST MODEL: %s -> model/best_model.keras' % DISPLAY[best])
    print('Artefacts -> result/  ;  models -> model/')


if __name__ == '__main__':
    main()
