# extract_embeddings.py
"""
Phase 1 of the copy-move forgery evaluation.

Loads the EXISTING trained Siamese (facenet-style) model and uses it purely as a
frozen feature extractor (NO training of the model happens here).  For every
non-overlapping ps x ps block of every test image it computes the L2-normalised
128-d embedding and the ground-truth block label (forged / authentic) taken from
the matching mask in SD_GT.

Everything is cached to  results/embeddings_cache.npz  so the analysis /
optimisation script can run repeatedly without reloading TensorFlow.
"""

import os
import sys
import math
import argparse
import numpy as np
import cv2

# ----- TF1 compatibility (model is a TF1 facenet meta-graph) -----
import tensorflow as tf
try:
    tf.compat.v1.disable_eager_execution()
    tfv1 = tf.compat.v1
except Exception:
    tfv1 = tf

def pick_checkpoint(model_dir):
    """Return (meta_path, ckpt_path) using the HIGHEST-step checkpoint
    (the most-trained one), ignoring the stale `checkpoint` pointer file."""
    import re
    files = os.listdir(model_dir)
    meta = [f for f in files if f.endswith('.meta')][0]
    best_step, best = -1, None
    for f in files:
        m = re.match(r'(model-.+\.ckpt-(\d+))\.index$', f)
        if m:
            step = int(m.group(2))
            if step > best_step:
                best_step, best = step, m.group(1)
    if best is None:
        raise RuntimeError('no .ckpt-*.index found in ' + model_dir)
    return os.path.join(model_dir, meta), os.path.join(model_dir, best), best_step


def find_embeddings_tensor(graph):
    try:
        return graph.get_tensor_by_name("embeddings:0")
    except Exception:
        pass
    for op in graph.get_operations():
        for out in op.outputs:
            if 'embedd' in out.name.lower():
                return out
    return None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='./models/siamese/copy_move')
    p.add_argument('--images', default='./SD_Images')
    p.add_argument('--gt', default='./SD_GT')
    p.add_argument('--out', default='./results/embeddings_cache.npz')
    p.add_argument('--patch_size', type=int, default=30)
    p.add_argument('--gt_block_thresh', type=float, default=0.10,
                   help='fraction of forged pixels in a block to label it forged')
    return p.parse_args()


def main():
    args = get_args()
    ps = args.patch_size
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    files = sorted([f for f in os.listdir(args.images)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    if not files:
        print("No images found in", args.images)
        sys.exit(1)

    g = tfv1.Graph()
    with g.as_default():
        with tfv1.Session() as sess:
            image_ph = tfv1.placeholder(tf.float32, shape=(None, ps, ps, 1), name='image')
            phase_ph = tfv1.placeholder(tf.bool, name='phase_train')
            lr_ph = tfv1.placeholder(tf.float32, name='learning_rate')
            alpha_ph = tfv1.placeholder(tf.float32, name='dynamic_alpha_placeholder')
            input_map = {'image': image_ph, 'phase_train': phase_ph,
                         'learning_rate': lr_ph, 'dynamic_alpha_placeholder': alpha_ph}

            print("Loading model from", args.model)
            meta_path, ckpt_path, step = pick_checkpoint(args.model)
            print("Meta graph :", meta_path)
            print("Checkpoint : %s  (step=%d, most-trained)" % (ckpt_path, step))
            saver = tfv1.train.import_meta_graph(meta_path, input_map=input_map)
            saver.restore(sess, ckpt_path)
            emb_t = find_embeddings_tensor(g)
            if emb_t is None:
                print("ERROR: embeddings tensor not found")
                sys.exit(2)
            print("Embeddings tensor:", emb_t.name, "shape", emb_t.shape)

            all_emb = []          # list of (n_blocks_i, dim)
            all_lbl = []          # list of (n_blocks_i,)
            meta = []             # (filename, blocks_x, blocks_y)

            for k, fname in enumerate(files, 1):
                ipath = os.path.join(args.images, fname)
                gpath = os.path.join(args.gt, fname)
                img = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(gpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("  skip (no image):", fname)
                    continue
                if gt is None:
                    print("  warn (no GT, labels=0):", fname)
                    gt = np.zeros_like(img)
                if gt.shape != img.shape:
                    gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                bx = img.shape[0] // ps
                by = img.shape[1] // ps
                if bx == 0 or by == 0:
                    print("  skip (too small):", fname)
                    continue

                gt_bin = (gt > 127).astype(np.float32)
                patches = np.empty((bx * by, ps, ps, 1), dtype=np.float32)
                labels = np.empty((bx * by,), dtype=np.int64)
                idx = 0
                for i in range(bx):
                    for j in range(by):
                        patch = img[ps*i:ps*(i+1), ps*j:ps*(j+1)].astype(np.float32) / 255.0
                        patches[idx, :, :, 0] = patch
                        frac = gt_bin[ps*i:ps*(i+1), ps*j:ps*(j+1)].mean()
                        labels[idx] = 1 if frac > args.gt_block_thresh else 0
                        idx += 1

                feed = {image_ph: patches}
                try:
                    feed[phase_ph] = False
                except Exception:
                    pass
                try:
                    emb = sess.run(emb_t, feed_dict=feed)
                except Exception:
                    # some graphs don't require phase_train to be fed
                    emb = sess.run(emb_t, feed_dict={image_ph: patches})
                emb = np.asarray(emb, dtype=np.float32).reshape(bx * by, -1)
                all_emb.append(emb)
                all_lbl.append(labels)
                meta.append((fname, bx, by))
                print("[%d/%d] %s  blocks=%dx%d  forged_blocks=%d/%d"
                      % (k, len(files), fname, bx, by, int(labels.sum()), labels.size))

    # pack: store per-image arrays plus concatenated index
    np.savez_compressed(
        args.out,
        emb=np.concatenate(all_emb, axis=0),
        lbl=np.concatenate(all_lbl, axis=0),
        counts=np.array([e.shape[0] for e in all_emb], dtype=np.int64),
        files=np.array([m[0] for m in meta]),
        bx=np.array([m[1] for m in meta], dtype=np.int64),
        by=np.array([m[2] for m in meta], dtype=np.int64),
        patch_size=np.array(ps),
    )
    total = sum(e.shape[0] for e in all_emb)
    pos = int(np.concatenate(all_lbl).sum())
    print("Saved %s : %d images, %d blocks, %d forged (%.2f%%)"
          % (args.out, len(meta), total, pos, 100.0 * pos / max(1, total)))


if __name__ == '__main__':
    main()
