"""Paired bootstrap CI used by eval/compare.py."""
import numpy as np

from eval.compare import paired_bootstrap


def test_identical_runs_have_a_zero_interval():
    a = np.random.default_rng(0).uniform(0.0, 1.0, 89)
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0)


def test_a_constant_shift_is_exact_and_the_ci_brackets_the_mean():
    a = np.random.default_rng(1).uniform(0.0, 1.0, 89)
    d, lo, hi = paired_bootstrap(a, a + 0.1)
    assert abs(d - 0.1) < 1e-12 and abs(lo - 0.1) < 1e-12 and abs(hi - 0.1) < 1e-12
    b = a + np.random.default_rng(2).normal(0.05, 0.2, 89)
    d, lo, hi = paired_bootstrap(a, b)
    assert lo < d < hi
    # Pairing: the CI of a noisy difference is about 2 * 1.96 * sd / sqrt(n) wide, not that of two unpaired means.
    assert abs((hi - lo) - 2 * 1.96 * np.std(b - a) / np.sqrt(89)) < 0.02


def test_the_resampling_is_seeded():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    assert paired_bootstrap(a, b) == paired_bootstrap(a, b)
