"""Boundary pins for stationary_distribution in app/services/regime.py.

The solver answers pi @ P = pi, sum(pi) = 1 by least squares, clips negative
noise and renormalises. These tests sit on the chains whose answer is known
in closed form: a doubly-stochastic chain (uniform), a chain with an
absorbing state (a unit vector), the identity (every vector is stationary,
so the call must still return a probability vector) and a seeded random
chain, where only the fixed-point identity itself can be checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.regime import STATE_ORDER, stationary_distribution

N = len(STATE_ORDER)


def test_a_doubly_stochastic_chain_is_stationary_at_the_uniform_vector():
    circulant = np.array([[0.5, 0.25, 0.25],
                          [0.25, 0.5, 0.25],
                          [0.25, 0.25, 0.5]])
    np.testing.assert_allclose(stationary_distribution(circulant), np.ones(N) / N, atol=1e-9)


@pytest.mark.parametrize("absorbing", range(N))
def test_an_absorbing_state_ends_up_with_all_the_mass(absorbing):
    p = np.full((N, N), 0.1)
    np.fill_diagonal(p, 0.8)                 # every row leaks 0.1 into each other state
    p[absorbing] = np.eye(N)[absorbing]      # ... except the absorbing one, which stays put
    np.testing.assert_allclose(stationary_distribution(p), np.eye(N)[absorbing], atol=1e-9)


def test_the_identity_chain_returns_a_probability_vector_rather_than_raising():
    pi = stationary_distribution(np.eye(N))
    assert pi.shape == (N,)
    assert np.all(pi >= 0)
    assert pi.sum() == pytest.approx(1.0)


def test_the_result_is_a_fixed_point_of_a_seeded_random_row_stochastic_chain():
    rng = np.random.default_rng(42)
    p = rng.random((N, N))
    p /= p.sum(axis=1, keepdims=True)
    pi = stationary_distribution(p)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-9)
    assert np.all(pi >= 0)
    assert pi.sum() == pytest.approx(1.0)
