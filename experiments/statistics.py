"""Paired statistics for arm comparisons (UPGRADE_PLAN.md 5.3, roadmap M2.4/M5.3).

The governing rule is that no single-run number appears in the manuscript again.
Every comparison therefore carries a distribution across seeds and a paired test
against the reference arm, with a family-wise correction over the arms compared.

Two paired tests, because they answer different questions:

- **McNemar** works *within* one split, on per-example predictions. It asks
  whether two arms disagree asymmetrically on the same test items, and it is the
  right test when the arms saw identical data.
- **Wilcoxon signed-rank** works *across* seeds, on per-seed accuracies. It asks
  whether one arm is consistently ahead once training randomness is resampled.

Holm-Bonferroni then controls the family-wise error rate over the arms compared
against the reference, which matters because T5 makes four comparisons at once.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260725


def mean_std(values) -> dict:
    """Mean and population standard deviation of a per-seed sample."""
    array = np.asarray(values, dtype=float)
    return {
        'n': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=0)),
    }


def bootstrap_ci(values, confidence: float = DEFAULT_CONFIDENCE,
                 n_resamples: int = DEFAULT_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap CI for the mean.

    Seeded, so the interval a run reports is the interval a rerun reports.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError('bootstrap needs at least one observation')
    if array.size == 1:
        return {'confidence': confidence, 'low': float(array[0]), 'high': float(array[0]),
                'n_resamples': 0}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        'confidence': confidence,
        'low': float(np.quantile(means, alpha)),
        'high': float(np.quantile(means, 1.0 - alpha)),
        'n_resamples': int(n_resamples),
    }


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test on paired per-example correctness.

    ``correct_a`` / ``correct_b`` are boolean arrays over the *same* test items.
    Only the discordant pairs carry information: b01 items that A got right and
    B got wrong, and b10 the reverse. Under the null those split like a fair
    coin, so the exact binomial test is used rather than the chi-square
    approximation, which is unreliable when discordance is small -- and here it
    is often zero, because Theorem 1 predicts some arms never disagree at all.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('paired arrays must have the same shape')

    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    discordant = b01 + b10
    if discordant == 0:
        # The arms agree on every item; there is nothing to test.
        return {'b01': 0, 'b10': 0, 'n_discordant': 0, 'p_value': 1.0, 'exact': True}

    p = float(stats.binomtest(b01, discordant, 0.5).pvalue)
    return {'b01': b01, 'b10': b10, 'n_discordant': discordant,
            'p_value': p, 'exact': True}


def wilcoxon_across_seeds(values_a, values_b) -> dict:
    """Wilcoxon signed-rank over paired per-seed scores."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('paired arrays must have the same shape')

    differences = a - b
    if np.allclose(differences, 0.0):
        # Every seed ties. scipy raises on an all-zero difference vector.
        return {'statistic': 0.0, 'p_value': 1.0, 'n_pairs': int(a.size),
                'all_tied': True}
    try:
        result = stats.wilcoxon(a, b, zero_method='wilcox')
    except ValueError as exc:               # too few non-tied pairs
        return {'statistic': None, 'p_value': None, 'n_pairs': int(a.size),
                'all_tied': False, 'error': str(exc)}
    return {'statistic': float(result.statistic), 'p_value': float(result.pvalue),
            'n_pairs': int(a.size), 'all_tied': False}


def holm_bonferroni(p_values, alpha: float = 0.05) -> dict:
    """Holm step-down correction over a family of comparisons.

    Returns adjusted p-values (monotone, capped at 1) and rejection flags, in
    the caller's original order. ``None`` entries are passed through unadjusted
    and never rejected, so a test that could not run cannot silently count as a
    negative result.
    """
    raw = list(p_values)
    testable = [(i, p) for i, p in enumerate(raw) if p is not None]
    adjusted = [None] * len(raw)
    rejected = [False] * len(raw)

    order = sorted(testable, key=lambda item: item[1])
    m = len(order)
    running = 0.0
    for rank, (index, p) in enumerate(order):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)        # enforce monotonicity
        adjusted[index] = float(running)
        rejected[index] = bool(running <= alpha)

    return {'alpha': alpha, 'n_tests': m, 'adjusted': adjusted, 'rejected': rejected}


def compare_arms(reference: str, arms: dict, alpha: float = 0.05) -> dict:
    """Full T5-style comparison of every arm against a reference arm.

    ``arms`` maps arm name to ``{'accuracies': [...per seed...],
    'correct': array over pooled test items}``. Every arm must share the
    reference's test items in the same order for McNemar to be meaningful.
    """
    if reference not in arms:
        raise ValueError('reference arm {!r} is not in the comparison'.format(reference))

    summary = {}
    for name, data in arms.items():
        summary[name] = {
            'accuracy': mean_std(data['accuracies']),
            'ci95': bootstrap_ci(data['accuracies']),
        }

    others = [name for name in arms if name != reference]
    ref = arms[reference]
    for name in others:
        summary[name]['vs_reference'] = {
            'reference': reference,
            'delta_accuracy': (summary[name]['accuracy']['mean']
                               - summary[reference]['accuracy']['mean']),
            'mcnemar': mcnemar(ref['correct'], arms[name]['correct']),
            'wilcoxon': wilcoxon_across_seeds(ref['accuracies'], arms[name]['accuracies']),
        }

    for test_name in ('mcnemar', 'wilcoxon'):
        p_values = [summary[name]['vs_reference'][test_name]['p_value'] for name in others]
        correction = holm_bonferroni(p_values, alpha=alpha)
        for i, name in enumerate(others):
            summary[name]['vs_reference'][test_name]['p_holm'] = correction['adjusted'][i]
            summary[name]['vs_reference'][test_name]['significant'] = correction['rejected'][i]

    return {'reference': reference, 'alpha': alpha, 'arms': summary}
