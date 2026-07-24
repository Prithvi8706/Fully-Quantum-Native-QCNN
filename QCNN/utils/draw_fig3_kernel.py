#!/usr/bin/env python3
"""Draw Fig. 3 of the paper: the 2x2 quantum convolutional kernel.

The CNOT pattern is taken straight from the ordering in
QuantumNativeConvolution.quantum_conv2d_kernel, so the figure cannot drift from
the implementation. Each qubit's SU(2) rotation block is collapsed into a single
labelled V_q box, matching the paper's existing figure style.

Geometry and colours reproduce the original figure: wire spacing 92 px, boxes
87x57 px filled #DBE8F7 with a #BFCBD8 edge, CNOT columns spaced 123 px, all at
300 dpi.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from QCNN.config.Qconfig import QuantumNativeConfig  # noqa: E402

PX = 92.0                      # one wire spacing, in px of the original figure
BOX_W, BOX_H = 87 / PX, 57 / PX
BOX_CX = 177.5 / PX
CNOT_X0, CNOT_DX = 331 / PX, 123 / PX
WIRE_X0, LABEL_X = 110 / PX, 68 / PX
TAIL = 1.67                    # wire length after the last CNOT

FILL, EDGE, INK = '#DBE8F7', '#BFCBD8', 'black'
R_CTRL, R_TARG = 7.5 / PX, 22.0 / PX
LW = 0.6                       # matches the original's 2-3 px strokes at 300 dpi
FONT = 7.5


def cnot_pairs(entanglement):
    """Intra-window CNOTs in application order, mirroring quantum_conv2d_kernel."""
    if entanglement == 'none':
        return []
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)]   # 4 edges + first diagonal
    if entanglement == 'full':
        pairs.append((1, 2))                            # second diagonal -> all-to-all
    elif entanglement != 'one_diagonal':
        raise ValueError("Unknown conv_entanglement '{}'.".format(entanglement))
    return pairs


def draw(pairs, out, dpi):
    n = 4
    ys = [-i for i in range(n)]
    x_end = CNOT_X0 + CNOT_DX * (len(pairs) - 1) + TAIL if pairs else BOX_CX + 2
    x_lo, x_hi = LABEL_X - 0.45, x_end + 0.05
    y_lo, y_hi = ys[-1] - 0.62, ys[0] + 0.62

    fig_w = (x_hi - x_lo) * PX / 300.0
    fig_h = (y_hi - y_lo) * PX / 300.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis('off')

    lw = LW
    for i, y in enumerate(ys):
        ax.plot([WIRE_X0, x_end], [y, y], color=INK, lw=lw, zorder=1,
                solid_capstyle='butt')
        ax.text(LABEL_X, y, 'q{}'.format(i), ha='center', va='center',
                fontsize=FONT, style='italic', family='serif', color=INK)

        box = FancyBboxPatch(
            (BOX_CX - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle='round,pad=0,rounding_size=0.045',
            facecolor=FILL, edgecolor=EDGE, lw=lw, zorder=3)
        ax.add_patch(box)
        ax.text(BOX_CX, y, 'V_q{}'.format(i), ha='center', va='center',
                fontsize=FONT, style='italic', family='serif', color=INK, zorder=4)

    for k, (c, t) in enumerate(pairs):
        x = CNOT_X0 + CNOT_DX * k
        yc, yt = ys[c], ys[t]
        ax.plot([x, x], [yc, yt], color=INK, lw=lw, zorder=2,
                solid_capstyle='butt')
        ax.add_patch(Circle((x, yc), R_CTRL, facecolor=INK, edgecolor=INK,
                            zorder=5))
        ax.add_patch(Circle((x, yt), R_TARG, facecolor='white', edgecolor=INK,
                            lw=lw, zorder=5))
        ax.plot([x - R_TARG, x + R_TARG], [yt, yt], color=INK, lw=lw, zorder=6,
                solid_capstyle='butt')
        ax.plot([x, x], [yt - R_TARG, yt + R_TARG], color=INK, lw=lw, zorder=6,
                solid_capstyle='butt')

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Draw the 2x2 convolutional kernel (paper Fig. 3)')
    ap.add_argument('--output', default='figs_final/fig3.png')
    ap.add_argument('--entanglement', default=None,
                    choices=['full', 'one_diagonal', 'none'],
                    help='Defaults to conv_entanglement in Qconfig.')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    ent = args.entanglement or QuantumNativeConfig().conv_entanglement
    pairs = cnot_pairs(ent)
    draw(pairs, args.output, args.dpi)

    print('entanglement : {}'.format(ent))
    print('CNOTs        : {}  {}'.format(
        len(pairs), ' '.join('q{}->q{}'.format(a, b) for a, b in pairs)))
    print('saved        : {}'.format(args.output))


if __name__ == '__main__':
    main()
