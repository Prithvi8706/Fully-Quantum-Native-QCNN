#!/usr/bin/env python3
"""Draw Fig. 4 of the paper: test accuracy of every arm on the four tasks.

Bars are the five-seed mean and error bars the population standard deviation,
read directly from Results/evidence/q1_comparison.json so the figure cannot
drift from the reported tables.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from QCNN.utils.plotstyle import apply_paper_style  # noqa: E402

TASKS = (
    ('mnist_0v1', 'MNIST\n0-vs-1'),
    ('mnist_3v5', 'MNIST\n3-vs-5'),
    ('fashion_mnist_0v6', 'Fashion-MNIST\n0-vs-6'),
    ('kmnist_2v3', 'KMNIST\n2-vs-3'),
)
ARMS = (
    ('proposed', 'FQCNN'),
    ('logistic', 'Logistic'),
    ('mlp', 'MLP (2-unit)'),
    ('ttn', 'TTN'),
)


def draw(evidence: Path, output: Path, dpi: int) -> None:
    tasks = json.loads(evidence.read_text(encoding='utf-8'))['tasks']
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(TASKS))
    width = 0.2
    for i, (arm, label) in enumerate(ARMS):
        acc = [tasks[task]['arms'][arm]['metrics']['accuracy'] for task, _ in TASKS]
        ax.bar(x + (i - 1.5) * width, [a['mean'] for a in acc], width,
               yerr=[a['std'] for a in acc], capsize=3, label=label,
               error_kw={'elinewidth': 1})
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in TASKS])
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0.65, 1.02)
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FQCNN Fig. 4")
    parser.add_argument("--evidence", type=Path, default=Path("Results/evidence/q1_comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("figs_final/q1_comparison.png"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    draw(args.evidence, args.output, dpi=args.dpi)
    print(f"saved {args.output} at {args.dpi} dpi")


if __name__ == "__main__":
    main()
