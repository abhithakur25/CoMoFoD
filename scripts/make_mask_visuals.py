# make_mask_visuals.py
"""
Visual qualitative output: for a handful of representative images produce a
panel  [ input | ground-truth | baseline prediction | optimised prediction ].
Re-uses the cached embeddings + the same models as analyze_and_optimize.py.
Saved to results/masks/.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analyze_and_optimize as A

RES = './results'
OUT = os.path.join(RES, 'masks')
os.makedirs(OUT, exist_ok=True)
IMG_DIR, GT_DIR = './SD_Images', './SD_GT'


def main():
    images, ps = A.load_cache(os.path.join(RES, 'embeddings_cache.npz'))
    Xall, yall = A.build_features(images, use_raw=False, use_offset=True)

    # train optimised model on ALL images (we only need qualitative masks here)
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(7)
    X = np.concatenate(Xall); y = np.concatenate(yall)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xn = np.nan_to_num((X - mu) / sd)
    pos = float(y.sum()); neg = float(len(y) - pos)
    cw = {0: 1.0, 1: max(1.0, np.sqrt(neg / max(1.0, pos)))}
    mdl = A.make_mlp(Xn.shape[1])
    mdl.fit(Xn, y, epochs=120, batch_size=64, verbose=0, class_weight=cw)

    pick = ['extension_copy_gcs100.png', 'kore_copy_gcs100.png',
            'tapestry_copy_gcs100.png', 'malawi_copy_gcs100.png',
            'red_tower_copy_gcs100.png', 'sweets_copy_gcs100.png']

    for im in images:
        f = im['file']
        if f not in pick:
            continue
        bx, by = im['bx'], im['by']
        img = cv2.imread(os.path.join(IMG_DIR, f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(GT_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if gt is None:
            gt = np.zeros_like(img)

        # baseline mask (nearest-distance threshold 0.62 from CV)
        base_blk = A.baseline_predict(im, 0.62).reshape(bx, by)
        # optimised mask
        feat = np.concatenate([A.duplication_features(im['emb']),
                               A.offset_features(im['emb'], bx, by)], axis=1)
        feat = np.nan_to_num((feat - mu) / sd)
        opt_blk = (mdl.predict(feat, verbose=0).ravel() >= 0.15).astype(int).reshape(bx, by)

        def up(m):
            return np.repeat(np.repeat(m, ps, 0), ps, 1) * 255

        fig, ax = plt.subplots(1, 4, figsize=(14, 4))
        ax[0].imshow(img[:bx*ps, :by*ps], cmap='gray'); ax[0].set_title('Input')
        ax[1].imshow(gt[:bx*ps, :by*ps], cmap='gray'); ax[1].set_title('Ground truth')
        ax[2].imshow(up(base_blk), cmap='gray'); ax[2].set_title('Baseline')
        ax[3].imshow(up(opt_blk), cmap='gray'); ax[3].set_title('Optimised')
        for a in ax:
            a.axis('off')
        fig.suptitle(f)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'panel_' + f), dpi=110)
        plt.close(fig)
        print('saved panel for', f)
    print('mask panels in', OUT)


if __name__ == '__main__':
    main()
