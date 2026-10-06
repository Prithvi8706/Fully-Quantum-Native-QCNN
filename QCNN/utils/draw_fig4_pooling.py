#!/usr/bin/env python3
"""Draw the paper's coherent-pooling figure (figs_final/fig4.png, Fig. 3 in the paper).

Two retained/compressed pairs are shown. For each pair the gates follow
QuantumNativePooling.quantum_unitary_pooling: CRY and CRZ controlled on the
compressed qubit and targeting the retained qubit, a trainable R_Y on the
retained qubit, and the fixed R_Y(epsilon) on the compressed qubit. The last two
act on different wires and commute, so their drawing order is cosmetic.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

INK = 'black'
KEEP_FILL, KEEP_EDGE = '#E3F1E3', '#3C8D40'
DISC_FILL, DISC_EDGE = '#FBF0D9', '#C9A23B'
NOTE_FILL, NOTE_EDGE = '#FAF0D7', '#C9A23B'
LW = 0.9
BOX_W, BOX_H = 1.55, 1.15
GATE_X = (3.8, 5.8, 7.8)           # CRY, CRZ, R_Y on the retained wire
DISC_X = 9.8                       # R_Y(epsilon) on the compressed wire
WIRE_X0, WIRE_X1 = 2.4, 13.4
FONT = 21
NOTE_FONT = 17


def _box(ax, x, y, label, fill, edge):
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0,rounding_size=0.08',
        facecolor=fill, edgecolor=edge, lw=LW * 1.4, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=FONT, zorder=4)


def _pair(ax, y_keep, y_disc, keep, disc):
    for y in (y_keep, y_disc):
        ax.plot([WIRE_X0, WIRE_X1], [y, y], color=INK, lw=LW, zorder=1)
    ax.text(WIRE_X0 - 0.2, y_keep, rf'$q_{keep}$ (ret.)', ha='right', va='center', fontsize=FONT)
    ax.text(WIRE_X0 - 0.2, y_disc, rf'$q_{disc}$ (comp.)', ha='right', va='center', fontsize=FONT)
    for x in GATE_X[:2]:
        ax.plot([x, x], [y_disc, y_keep - BOX_H / 2], color=INK, lw=LW, zorder=2)
        ax.add_patch(Circle((x, y_disc), 0.14, facecolor=INK, edgecolor=INK, zorder=5))
    for x, label in zip(GATE_X, (r'$R_Y$', r'$R_Z$', r'$R_Y$')):
        _box(ax, x, y_keep, label, KEEP_FILL, KEEP_EDGE)
    _box(ax, DISC_X, y_disc, r'$R_Y$', DISC_FILL, DISC_EDGE)
    ax.text(WIRE_X1 + 0.15, y_keep, rf'$|\psi^{{\prime}}_{keep}\rangle$ (active)', ha='left', va='center', fontsize=FONT)
    ax.text(WIRE_X1 + 0.15, y_disc, r'$\it{discarded}$', ha='left', va='center', fontsize=FONT)


def draw(out, dpi):
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = True
    fig = plt.figure(figsize=(4.6 * 2, 2.4 * 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-2.0, 18.4)
    ax.set_ylim(-1.3, 10.0)
    ax.axis('off')

    _pair(ax, 9.0, 7.0, 'a', 'b')
    _pair(ax, 4.6, 2.6, 'c', 'd')

    ax.add_patch(FancyBboxPatch(
        (-1.6, -1.15), 19.6, 2.45, boxstyle='square,pad=0',
        facecolor=NOTE_FILL, edgecolor=NOTE_EDGE, lw=LW * 1.4, ls=(0, (4, 3)), zorder=1))
    ax.text(-1.0, 0.65,
            r'$\mathbf{Pooling\ unitary:}\quad W_{ab}(\Theta_p)=R_Y^{(a)}(\gamma)\,R_Y^{(b)}(\epsilon)\,'
            r'\mathrm{CRZ}_{b\to a}(\beta)\,\mathrm{CRY}_{b\to a}(\alpha)$',
            ha='left', va='center', fontsize=NOTE_FONT)
    ax.text(-1.0, -0.5, 'No measurement; coherence preserved.', ha='left', va='center',
            fontsize=NOTE_FONT, style='italic')

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Draw the coherent unitary pooling figure')
    ap.add_argument('--output', default='figs_final/fig4.png')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    draw(args.output, args.dpi)
    print('saved {}'.format(args.output))


if __name__ == '__main__':
    main()
