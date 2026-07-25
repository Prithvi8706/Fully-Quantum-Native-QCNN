"""The paired statistics that decide every T5 claim (UPGRADE_PLAN.md 5.3).

These functions turn raw runs into the sentences the manuscript will assert, so
each is pinned against a case whose answer is known independently of the code.
"""
import numpy as np
import pytest

from experiments.statistics import (
    bootstrap_ci,
    compare_arms,
    holm_bonferroni,
    mcnemar,
    mean_std,
    wilcoxon_across_seeds,
)


def test_mean_and_population_std():
    stats_ = mean_std([0.90, 0.92, 0.94, 0.96, 0.98])
    assert stats_['n'] == 5
    assert stats_['mean'] == pytest.approx(0.94)
    assert stats_['std'] == pytest.approx(np.std([0.90, 0.92, 0.94, 0.96, 0.98]))


def test_bootstrap_brackets_the_mean_and_is_deterministic():
    values = [0.90, 0.92, 0.94, 0.96, 0.98]
    ci = bootstrap_ci(values)
    assert ci['low'] < np.mean(values) < ci['high']
    assert bootstrap_ci(values) == ci


def test_bootstrap_of_identical_values_is_a_point_interval():
    ci = bootstrap_ci([0.97] * 6)
    assert ci['low'] == pytest.approx(0.97)
    assert ci['high'] == pytest.approx(0.97)


def test_bootstrap_of_a_single_observation_degenerates_gracefully():
    ci = bootstrap_ci([0.5])
    assert ci['low'] == ci['high'] == 0.5
    assert ci['n_resamples'] == 0


def test_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError, match='at least one'):
        bootstrap_ci([])


def test_mcnemar_on_identical_predictions_cannot_reject():
    """Theorem 1 predicts exactly this for the measurement arm."""
    correct = np.array([True, False, True, True, False])
    result = mcnemar(correct, correct)
    assert result['n_discordant'] == 0
    assert result['p_value'] == 1.0


def test_mcnemar_counts_discordant_pairs_in_the_right_direction():
    a = np.array([True, True, True, False])
    b = np.array([False, False, True, False])
    result = mcnemar(a, b)
    assert result['b01'] == 2       # A right, B wrong
    assert result['b10'] == 0
    assert result['n_discordant'] == 2


def test_mcnemar_detects_a_lopsided_disagreement():
    a = np.array([True] * 20 + [False] * 2)
    b = np.array([False] * 20 + [False] * 2)
    assert mcnemar(a, b)['p_value'] < 0.001


def test_mcnemar_is_symmetric_in_its_p_value():
    a = np.array([True, True, False, True, False, False])
    b = np.array([False, True, True, True, False, True])
    assert mcnemar(a, b)['p_value'] == pytest.approx(mcnemar(b, a)['p_value'])


def test_mcnemar_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='same shape'):
        mcnemar(np.array([True, False]), np.array([True]))


def test_wilcoxon_on_all_tied_seeds_reports_a_tie_rather_than_raising():
    """scipy raises on an all-zero difference vector; a tie is a real outcome."""
    values = [0.9, 0.91, 0.92, 0.93, 0.94]
    result = wilcoxon_across_seeds(values, values)
    assert result['all_tied'] is True
    assert result['p_value'] == 1.0


def test_wilcoxon_detects_a_consistent_advantage():
    a = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    b = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
    assert wilcoxon_across_seeds(a, b)['p_value'] < 0.05


def test_holm_is_more_conservative_than_raw_but_less_than_bonferroni():
    raw = [0.01, 0.02, 0.03, 0.04]
    result = holm_bonferroni(raw, alpha=0.05)
    assert result['adjusted'][0] == pytest.approx(0.04)      # 4 * 0.01
    for adjusted, plain in zip(result['adjusted'], raw):
        assert adjusted >= plain
        assert adjusted <= min(1.0, len(raw) * plain) + 1e-12


def test_holm_adjusted_values_are_monotone():
    result = holm_bonferroni([0.04, 0.005, 0.2, 0.01])
    ordered = sorted(zip([0.04, 0.005, 0.2, 0.01], result['adjusted']))
    values = [adjusted for _, adjusted in ordered]
    assert values == sorted(values)


def test_holm_caps_adjusted_values_at_one():
    result = holm_bonferroni([0.5, 0.6, 0.7])
    assert all(adjusted <= 1.0 for adjusted in result['adjusted'])


def test_holm_passes_untestable_comparisons_through_without_rejecting():
    """A test that could not run must never count as a negative result."""
    result = holm_bonferroni([0.001, None, 0.5])
    assert result['n_tests'] == 2
    assert result['adjusted'][1] is None
    assert result['rejected'][1] is False
    assert result['rejected'][0] is True


def test_compare_arms_produces_a_full_t5_row_set():
    rng = np.random.default_rng(0)
    items = 200
    reference_correct = rng.random(items) < 0.95

    arms = {
        'unitary': {'accuracies': [0.95, 0.94, 0.96, 0.95, 0.95],
                    'correct': reference_correct},
        # Predicted by Theorem 1 to be bit-identical to the reference.
        'measurement': {'accuracies': [0.95, 0.94, 0.96, 0.95, 0.95],
                        'correct': reference_correct.copy()},
        'none': {'accuracies': [0.70, 0.68, 0.72, 0.69, 0.71],
                 'correct': rng.random(items) < 0.70},
    }
    result = compare_arms('unitary', arms)

    assert result['reference'] == 'unitary'
    assert 'vs_reference' not in result['arms']['unitary']

    tie = result['arms']['measurement']['vs_reference']
    assert tie['delta_accuracy'] == pytest.approx(0.0)
    assert tie['mcnemar']['n_discordant'] == 0
    assert tie['wilcoxon']['all_tied'] is True
    assert tie['mcnemar']['significant'] is False

    floor = result['arms']['none']['vs_reference']
    assert floor['delta_accuracy'] < -0.2
    assert floor['mcnemar']['p_holm'] < 0.05
    assert floor['mcnemar']['significant'] is True


def test_compare_arms_requires_the_reference_to_exist():
    with pytest.raises(ValueError, match='reference arm'):
        compare_arms('missing', {'a': {'accuracies': [1.0], 'correct': np.array([True])}})
