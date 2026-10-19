import numpy as np
import pytest
from experimentlab.core import *


def test_broken_logging_is_blocked_despite_naive_win():
    r = analyse(*simulate(effect=0, broken=True), theta=2.0)
    assert r["raw"]["p"] < 0.001 and r["raw"]["effect"] > 0
    assert r["decision"] == "INVALID_RANDOMISATION"


def test_cuped_reduces_variance_and_preserves_effect():
    r = analyse(*simulate(n=12000), theta=2.0)
    assert r["variance_reduction"] > 0.6 and abs(r["cuped"]["effect"] - 0.4) < 0.06
    assert r["decision"] == "SHIP"


def test_early_win_cannot_ship():
    assert (
        analyse(*simulate(n=800), theta=2.0)["decision"] == "CONTINUE_TO_FIXED_HORIZON"
    )


def test_srm_and_power():
    assert srm([0, 1] * 100) == 1
    assert sample_size(1, 0.1) > sample_size(1, 0.2)
    with pytest.raises(ValueError):
        sample_size(1, 0)


def test_bh_known_values():
    assert np.allclose(bh_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


def test_null_constant_and_invalid_values():
    assert difference([1, 1], [1, 1])["p"] == 1
    with pytest.raises(ValueError):
        difference([1, np.nan], [2, 3])


def test_guardrail_requires_evidence_of_safety():
    pre, outcome, z, guard = simulate(n=12000)
    guard = guard + 0.12 * z
    result = analyse(pre, outcome, z, guard, theta=2.0)
    assert result["decision"] == "REJECT" and not result["guardrail_safe"]
