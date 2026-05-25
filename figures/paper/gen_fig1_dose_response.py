#!/usr/bin/env python3
"""Generate dose-response curve for CKA coefficient α (Figure 1)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

alphas = [0.5, 1.0, 2.0, 3.0, 4.0]
sft_acc =  [60.73, 61.49, 64.97, 64.14, 61.94]
grpo_acc = [61.18, 60.88, 66.11, 63.46, 62.77]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

ax.plot(alphas, sft_acc, 'o-', color='#1f77b4', linewidth=2.0, markersize=8,
        label='After SFT', zorder=5)
ax.plot(alphas, grpo_acc, 's--', color='#a63131', linewidth=2.0, markersize=8,
        label='After GRPO', zorder=5)

ax.fill_between(alphas, sft_acc, grpo_acc, alpha=0.10, color='gray', zorder=1)

ax.annotate(r'$\alpha{=}2.0$ (peak)', xy=(2.0, 66.11),
            xytext=(2.6, 67.5), fontsize=10, color='#555555',
            arrowprops=dict(arrowstyle='-', color='#999999', lw=0.8))

ax.text(1.5, 59.8, r'$|\Delta| \leq 1.14\;pp$',
        fontsize=11, fontstyle='italic', color='#555555')

ax.set_xlabel(r'CKA coefficient $\alpha$', fontsize=12)
ax.set_ylabel('GSM8K Accuracy (%)', fontsize=12)
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas], fontsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.set_ylim(59, 69)

ax.legend(fontsize=9, frameon=False)

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
doc_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), 'docs', 'figures', 'paper')
os.makedirs(doc_dir, exist_ok=True)

for d in [out_dir, doc_dir]:
    plt.savefig(os.path.join(d, 'fig1_dose_response.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(d, 'fig1_dose_response.png'), format='png', dpi=300, bbox_inches='tight')

print(f"Saved to {out_dir}/ and {doc_dir}/")
plt.close()
