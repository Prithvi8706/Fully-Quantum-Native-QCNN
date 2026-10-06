#!/usr/bin/env python3
"""Render the journal's architecture overview from the frozen FQCNN path.

The overview intentionally names only global amplitude encoding.  The previous
artwork mentioned a correlation-injection feature map that is not present in
the validated circuit, so keeping this small deterministic renderer next to the
other figure utilities makes that distinction reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


def draw(output: Path, dpi: int = 600) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.0, 2.35), facecolor="white")
    ax = fig.add_axes([0.005, 0.03, 0.99, 0.94])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.35)
    ax.axis("off")

    boxes = [
        (0.10, "Input", r"$x\in\mathbb{R}^{d}$", "#dedede"),
        (2.25, "Encoding", r"$U_{\mathrm{enc}}(x)$", "#a9d7e7"),
        (4.45, "Convolution", r"$U_{\mathrm{conv}}(\Theta_k)$", "#8be28b"),
        (6.75, "Pooling", r"$U_{\mathrm{pool}}(\Theta_p)$", "#f4c477"),
        (8.95, "Classification", r"$U_{\mathrm{cls}}(\Theta_c)$", "#efa9b9"),
        (11.55, "Output", r"$\hat{y}=\mathrm{sign}(f_\Theta)$", "#dedede"),
    ]
    widths = [1.55, 1.75, 1.95, 1.70, 2.05, 2.05]
    y = 1.31
    h = 0.68
    edge = "#777777"

    # The dashed enclosure covers the coherent variational path only.
    ax.add_patch(
        Rectangle(
            (2.08, 0.88), 9.25, 1.10, fill=False, linestyle=(0, (5, 4)),
            linewidth=1.0, edgecolor="#a6a6a6", zorder=0,
        )
    )
    ax.text(
        6.70, 2.08, "coherent unitary path before terminal readout",
        ha="center", va="bottom", fontsize=10.5, family="serif", color="#555555",
    )

    centers = []
    for (x, title, equation, color), width in zip(boxes, widths):
        centers.append((x + width / 2, width))
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, h, boxstyle="round,pad=0.015,rounding_size=0.10",
                facecolor=color, edgecolor=edge, linewidth=1.0, zorder=2,
            )
        )
        ax.text(x + width / 2, y + 0.43, title, ha="center", va="center",
                fontsize=16, fontweight="bold", family="serif", zorder=3)
        ax.text(x + width / 2, y + 0.19, equation, ha="center", va="center",
                fontsize=12.5, family="serif", zorder=3)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + widths[i] + 0.06
        x1 = boxes[i + 1][0] - 0.06
        ax.add_patch(
            FancyArrowPatch(
                (x0, y + h / 2), (x1, y + h / 2), arrowstyle="-|>",
                mutation_scale=13, linewidth=1.0, color="black", zorder=1,
            )
        )

    notes = [
        (0.88, ""),
        (3.12, "global amplitude\nencoding"),
        (5.42, "shared SU(2)\n2$\\times$2 kernel"),
        (7.60, "coherent CRY/CRZ\nno measurement"),
        (9.98, "variational head\n$\\langle Z_0\\rangle$ readout"),
        (12.58, ""),
    ]
    for x, note in notes:
        if note:
            ax.text(x, 0.56, note, ha="center", va="top", fontsize=10.5,
                    family="serif", color="#333333")

    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FQCNN Fig. 1")
    parser.add_argument("--output", type=Path, default=Path("figs_final/fig1.png"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    draw(args.output, dpi=args.dpi)
    print(f"saved {args.output} at {args.dpi} dpi")


if __name__ == "__main__":
    main()
