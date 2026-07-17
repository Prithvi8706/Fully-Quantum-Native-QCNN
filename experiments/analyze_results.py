#!/usr/bin/env python3
"""
analyze_results.py — publication statistics + figure for the FQCNN ablation study.

Reads the per-seed metrics written by run_experiments
(``Results/experiments/<dataset>/<config>/seed_<s>.json``) and produces:

  - Results/experiments/significance.csv : paired significance tests (Wilcoxon signed-rank
    + paired t) with effect sizes, comparing the 'proposed' model against every ablation
    variant AND every baseline, per dataset and metric.
  - Results/Graphs/fig_ablation.png : grouped bar chart (accuracy per config per dataset)
    with 95% CI error bars and a chance line.

Runnable standalone against an existing Results/experiments/ tree, or imported and called
via ``run_analysis()`` at the end of the study.

    python -m experiments.analyze_results
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

EXP_ROOT = os.path.join("Results", "experiments")
FIG_PATH = os.path.join("Results", "Graphs", "fig_ablation.png")

# The reference model every comparison is made against.
REFERENCE = "proposed"
# Metrics to test (accuracy is primary; f1/roc_auc for completeness).
TEST_METRICS = ("accuracy", "f1", "roc_auc")
# Configs whose split differs from 'proposed' (different image_size/qubits) → seed-paired
# but not split-identical; flagged in the significance output.
DIFFERENT_SPLIT = {"enc_feature_map"}


# ── data loading ────────────────────────────────────────────────────────────────

def _load_seed_metrics(cfg_dir: str) -> dict:
    """Return {seed:int -> metrics dict} for a config directory."""
    out = {}
    if not os.path.isdir(cfg_dir):
        return out
    for fn in os.listdir(cfg_dir):
        if fn.startswith("seed_") and fn.endswith(".json"):
            seed = int(fn[len("seed_"):-len(".json")])
            with open(os.path.join(cfg_dir, fn)) as f:
                out[seed] = json.load(f)
    return out


def _list_datasets(exp_root: str) -> list[str]:
    return sorted(d for d in os.listdir(exp_root)
                  if os.path.isdir(os.path.join(exp_root, d)))


def _list_configs(ds_dir: str) -> list[str]:
    """Config dirs under a dataset (both ablation configs and baseline_* dirs)."""
    return sorted(d for d in os.listdir(ds_dir)
                  if os.path.isdir(os.path.join(ds_dir, d)))


# ── statistics ──────────────────────────────────────────────────────────────────

def _paired_stats(ref_vals: dict, var_vals: dict, metric: str) -> dict | None:
    """
    Paired comparison of `metric` between the reference and a variant, aligned by seed.

    Returns a dict of statistics, or None if there are no usable paired seeds.
    delta = mean(variant) - mean(reference)  (positive → variant better).
    """
    from scipy import stats

    seeds = sorted(set(ref_vals) & set(var_vals))
    a, b = [], []  # a = reference, b = variant
    for s in seeds:
        ra, rb = ref_vals[s].get(metric), var_vals[s].get(metric)
        if ra is None or rb is None:
            continue
        if not (np.isfinite(ra) and np.isfinite(rb)):
            continue
        a.append(float(ra)); b.append(float(rb))
    n = len(a)
    if n == 0:
        return None
    a, b = np.array(a), np.array(b)
    diff = b - a
    mean_ref, mean_var = float(a.mean()), float(b.mean())
    delta = float(diff.mean())

    # paired t-test
    if n > 1 and np.any(diff != 0):
        t_p = float(stats.ttest_rel(b, a).pvalue)
    else:
        t_p = float("nan")

    # Wilcoxon signed-rank (needs ≥1 non-zero diff; meaningful for n≳6)
    if n > 1 and np.any(diff != 0):
        try:
            w_p = float(stats.wilcoxon(b, a).pvalue)
        except ValueError:
            w_p = float("nan")
    else:
        w_p = float("nan")

    # 95% CI on the paired difference + paired Cohen's d
    if n > 1:
        sd = float(np.std(diff, ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if sd > 0 else 0.0
        cohens_d = float(delta / sd) if sd > 0 else 0.0
    else:
        half, cohens_d = 0.0, 0.0

    return {
        "mean_proposed": round(mean_ref, 4),
        "mean_variant": round(mean_var, 4),
        "delta": round(delta, 4),
        "ci_delta_low": round(delta - half, 4),
        "ci_delta_high": round(delta + half, 4),
        "p_wilcoxon": w_p,
        "p_ttest": t_p,
        "significant_0.05": int(np.isfinite(w_p) and w_p < 0.05),
        "cohens_d": round(cohens_d, 4),
        "n_pairs": n,
    }


def write_significance_csv(exp_root: str = EXP_ROOT) -> str:
    """Compute proposed-vs-{ablations,baselines} paired tests → significance.csv."""
    path = os.path.join(exp_root, "significance.csv")
    fields = ["dataset", "comparison", "metric", "mean_proposed", "mean_variant",
              "delta", "ci_delta_low", "ci_delta_high", "p_wilcoxon", "p_ttest",
              "significant_0.05", "cohens_d", "n_pairs", "note"]
    rows = []
    for ds in _list_datasets(exp_root):
        ds_dir = os.path.join(exp_root, ds)
        ref = _load_seed_metrics(os.path.join(ds_dir, REFERENCE))
        if not ref:
            continue
        for cfg in _list_configs(ds_dir):
            if cfg == REFERENCE:
                continue
            var = _load_seed_metrics(os.path.join(ds_dir, cfg))
            if not var:
                continue
            is_baseline = cfg.startswith("baseline_")
            note = []
            if cfg in DIFFERENT_SPLIT:
                note.append("seed-paired only: different image_size/split than proposed")
            if is_baseline:
                note.append("classical/quantum baseline")
            for metric in TEST_METRICS:
                st = _paired_stats(ref, var, metric)
                if st is None:
                    continue
                rows.append({
                    "dataset": ds,
                    "comparison": f"proposed_vs_{cfg}",
                    "metric": metric,
                    "note": "; ".join(note),
                    **st,
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(rows)} significance rows to {path}")
    return path


# ── figure ──────────────────────────────────────────────────────────────────────

def make_ablation_figure(exp_root: str = EXP_ROOT, out_path: str = FIG_PATH) -> str:
    """Grouped bar chart: accuracy per config per dataset with 95% CI error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = _list_datasets(exp_root)
    # Fixed, meaningful config order; baselines grouped at the end.
    ablations = ["proposed", "pool_none", "pool_measurement", "ent_one_diagonal",
                 "ent_none", "kernel_ry", "enc_feature_map"]
    baselines = ["baseline_logistic", "baseline_mlp", "baseline_cong",
                 "baseline_hur", "baseline_ttn"]

    # Only keep configs that actually exist somewhere.
    present = set()
    for ds in datasets:
        present |= set(_list_configs(os.path.join(exp_root, ds)))
    configs = [c for c in ablations + baselines if c in present]

    # Colorblind-safe palette (Wong): ablations vs baselines visually separated.
    abl_color = "#0072B2"      # blue for ablation variants
    prop_color = "#009E73"     # green highlights the proposed model
    base_color = "#E69F00"     # orange for baselines

    def color_for(c):
        if c == "proposed":
            return prop_color
        return base_color if c.startswith("baseline_") else abl_color

    def acc_mean_ci(ds, cfg):
        vals = [m.get("accuracy") for m in
                _load_seed_metrics(os.path.join(exp_root, ds, cfg)).values()]
        vals = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if vals.size == 0:
            return np.nan, 0.0
        mean = float(vals.mean())
        if vals.size > 1:
            from scipy import stats
            sd = float(np.std(vals, ddof=1))
            half = float(stats.t.ppf(0.975, vals.size - 1) * sd / np.sqrt(vals.size)) if sd > 0 else 0.0
        else:
            half = 0.0
        return mean, half

    n_ds, n_cfg = len(datasets), len(configs)
    group_w = 0.8
    bar_w = group_w / max(n_cfg, 1)
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * n_ds * n_cfg / 3), 6))
    for j, cfg in enumerate(configs):
        means, errs = [], []
        for ds in datasets:
            m, e = acc_mean_ci(ds, cfg)
            means.append(m); errs.append(e)
        offset = (j - (n_cfg - 1) / 2) * bar_w
        label = cfg.replace("baseline_", "").replace("_", " ")
        ax.bar(x + offset, means, bar_w, yerr=errs, capsize=2,
               color=color_for(cfg), edgecolor="black", linewidth=0.4,
               label=label, error_kw={"linewidth": 0.8})

    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("v", " vs ") for d in datasets])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("FQCNN ablations & baselines — accuracy (mean ± 95% CI)")
    # Legend below the axis so it never overlaps the bars.
    ax.legend(ncol=min(len(configs) + 1, 7), fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote ablation figure to {out_path}")
    return out_path


def run_analysis(exp_root: str = EXP_ROOT) -> None:
    write_significance_csv(exp_root)
    make_ablation_figure(exp_root)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FQCNN ablation stats + figure")
    ap.add_argument("--exp-root", default=EXP_ROOT)
    args = ap.parse_args()
    run_analysis(args.exp_root)
