import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.reader(open('./results/metrics_summary.csv')))[1:]
names, base, opt = [], [], []
for r in rows:
    if r[1] == '' or r[0] in ('train_loss', 'val_loss'):   # skip non-comparable
        continue
    names.append(r[0]); base.append(float(r[1])); opt.append(float(r[2]))

x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - w/2, base, w, label='Baseline (distance threshold)', color='#c0504d')
b2 = ax.bar(x + w/2, opt, w, label='Optimised (MLP head)', color='#4f81bd')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
ax.set_title('Copy-Move Forgery Detection: Baseline vs Optimised (5-fold CV)')
ax.legend()
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.01,
                '%.2f' % bb.get_height(), ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig('./results/metrics_comparison.png', dpi=130)
print('saved results/metrics_comparison.png')
