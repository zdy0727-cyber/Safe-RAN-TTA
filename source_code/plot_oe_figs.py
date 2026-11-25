# -*- coding: utf-8 -*-
"""Optical Engineering 规格结果图：单栏 3.3125 in、字体 8 pt、600 dpi、灰度可辨、白底无网格"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join('/private/zdy/deepseek_road', 'runs', 'figs_oe')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'hatch.linewidth': 0.6,
    'savefig.dpi': 600,
})
W = 3.3125
INK = '#111111'
GRAY = '#9a9a9a'
MID = '#5f5f5f'


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', pad_inches=0.01,
                facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---------- Fig.3  TTA gain collapse ----------
def fig3():
    settings = ['CtD1', 'CtD2', 'CtD3', 'Ts→Xj', 'Xj→Ts']
    src = np.array([0.1705, 0.0727, 0.0918, 0.4563, 0.3857])
    tent = np.array([0.1817, 0.0694, 0.0901, 0.4584, 0.3528])
    deyo = np.array([0.1707, 0.0705, 0.0954, 0.4562, 0.3640])
    ours = np.array([0.1705, 0.1633, 0.0918, 0.4563, 0.3857])
    g = lambda v: (v - src) * 100
    fig, ax = plt.subplots(figsize=(W, 2.35))
    x = np.arange(len(settings)); w = 0.26
    ax.axhline(0, color=INK, lw=0.8)
    b1 = ax.bar(x - w, g(tent), w, facecolor='white', edgecolor=INK, hatch='/////', lw=0.7, label='TENT')
    b2 = ax.bar(x, g(deyo), w, facecolor='#d0d0d0', edgecolor=INK, hatch='....', lw=0.7, label='DeYO')
    b3 = ax.bar(x + w, g(ours), w, facecolor=INK, edgecolor=INK, lw=0.7, label='Ours')
    ax.set_xticks(x); ax.set_xticklabels(settings)
    ax.set_ylabel('F1 gain vs. source (points)')
    ax.set_ylim(-5.0, 12.0)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16), handlelength=1.0, columnspacing=0.9)
    for i, v in enumerate(g(ours)):
        if v > 0.5:
            ax.text(i + w, v + 0.35, '+%.1f' % v, ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig3_gain_collapse.png')


# ---------- Fig.4  CtD2 comparison ----------
def fig4():
    names = ['Source', 'TENT', 'SAR', 'EATA', 'DeYO', 'CoTTA', 'Ours']
    vals = [0.0727, 0.0694, 0.0683, 0.0693, 0.0705, 0.0728, 0.1633]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(names))
    bars = ax.bar(x[:-1], vals[:-1], 0.62, facecolor='white', edgecolor=INK, hatch='/////', lw=0.7)
    ax.bar(x[-1], vals[-1], 0.62, facecolor=INK, edgecolor=INK, lw=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, '%.3f' % v, ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('macro-F1 (CtD2)')
    ax.set_ylim(0, 0.19)
    ax.annotate('+9.3', xy=(6, 0.1633), xytext=(4.1, 0.178), fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.7, color=INK))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig4_ctd2_compare.png')


# ---------- Fig.5  risk-accuracy trade-off ----------
def fig5():
    pts = [('argmax', 51.33, 0.1267, 's'), ('0.1', 46.33, 0.0415, '^'),
           ('0.2', 50.20, 0.0771, '*'), ('0.5', 50.73, 0.1311, 'o'), ('1.0', 50.53, 0.1846, 'd')]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for lab, a, e, mk in pts:
        filled = (lab == '0.2')
        ax.scatter(a, e, s=70 if filled else 42, marker=mk,
                   facecolor=INK if filled else 'white', edgecolor=INK, lw=0.8, zorder=3)
        ax.annotate('λ=' + lab if lab != 'argmax' else 'argmax', (a, e),
                    textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax.set_xlabel('Accuracy (%)'); ax.set_ylabel('Escalation cost')
    ax.set_xlim(45, 53); ax.set_ylim(0.02, 0.20)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig5_risk_pareto.png')


# ---------- Fig.6  calibration sensitivity ----------
def fig6():
    ks = ['20', '60', '100']
    src = [0.0764, 0.0727, 0.0703]
    ours = [0.1186, 0.1633, 0.1316]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(ks)); w = 0.34
    ax.bar(x - w/2, src, w, facecolor='white', edgecolor=INK, hatch='/////', lw=0.7, label='Source')
    ax.bar(x + w/2, ours, w, facecolor=INK, edgecolor=INK, lw=0.7, label='Ours')
    for i in range(len(ks)):
        ax.text(x[i] - w/2, src[i] + 0.003, '%.3f' % src[i], ha='center', fontsize=8)
        ax.text(x[i] + w/2, ours[i] + 0.003, '%.3f' % ours[i], ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(['%s / class' % k for k in ks])
    ax.set_ylabel('macro-F1 (CtD2)'); ax.set_ylim(0, 0.19)
    ax.legend(frameon=False, loc='upper left', ncol=2, handlelength=1.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, 'fig6_calib.png')


if __name__ == '__main__':
    fig3(); fig4(); fig5(); fig6()
