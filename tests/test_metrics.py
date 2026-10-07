"""Tests for the reported label-agreement criteria in nutpred.metrics.

These pin the behaviour that distinguishes the criteria from each other --
the zero conventions and the EU basis -- rather than re-deriving arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from nutpred.metrics import (
    all_masks,
    eu_mask,
    eu_row,
    fda_mask,
    has_eu_row,
    r2_manual,
    round_declared,
    smape,
    w20_mask,
    w20_v2_mask,
    within_20pct,
)

GRAMS = np.array(["g", "g", "g", "g"], dtype=object)


def test_r2_and_smape_basics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_manual(y, y) == pytest.approx(1.0)
    assert smape(y, y) == pytest.approx(0.0)
    # constant truth has no variance, so R2 is undefined rather than 0
    assert np.isnan(r2_manual(np.ones(4), np.arange(4.0)))


def test_within_20pct_matches_w20_on_nonzero():
    y = np.array([100.0, 100.0, 100.0])
    p = np.array([85.0, 100.0, 125.0])
    assert within_20pct(y, p) == pytest.approx(2 / 3)
    assert w20_mask(y, p).tolist() == [True, True, False]


def test_w20_declared_zero_needs_exact_zero():
    y = np.array([0.0, 0.0])
    p = np.array([0.0, 0.4])
    assert w20_mask(y, p).tolist() == [True, False]


def test_w20_v2_forgives_a_zero_that_would_round_to_zero():
    """A declared 0 passes when the prediction is under the per-serving threshold."""
    y = np.array([0.0, 0.0, 0.0, 0.0])
    # Calcium threshold is 26 mg per serving; at a 50 g serving that is
    # 52 mg/100 g. 40 is under it, 200 is not.
    p = np.array([0.0, 40.0, 200.0, 40.0])
    ss = np.array([50.0, 50.0, 50.0, 50.0])
    unresolvable = np.array(["g", "g", "g", "serving"], dtype=object)

    v2 = w20_v2_mask(y, p, "Calcium(mg)", ss, GRAMS)
    assert v2.tolist() == [True, True, False, True]

    # W20 alone forgives none of the non-zero predictions
    assert w20_mask(y, p).tolist() == [True, False, False, False]

    # an unreadable serving unit falls back to the exact-zero rule
    fallback = w20_v2_mask(y, p, "Calcium(mg)", ss, unresolvable)
    assert fallback.tolist() == [True, True, False, False]


def test_w20_v2_threshold_is_a_strict_inequality():
    """A per-serving amount exactly AT the threshold is not a declarable zero.

    21 CFR 101.9(c) describes amounts at or below the threshold as declarable
    zero, so the value at the boundary is still a declarable non-zero. The
    manuscript states this explicitly; fda_mask() uses the same strict test.
    """
    y = np.array([0.0, 0.0, 0.0])
    ss = np.array([100.0, 100.0, 100.0])   # per-100 g == per-serving
    units = np.array(["g", "g", "g"], dtype=object)
    p = np.array([25.9, 26.0, 26.1])       # calcium threshold is 26 mg
    assert w20_v2_mask(y, p, "Calcium(mg)", ss, units).tolist() == [True, False, False]


def test_w20_v2_thresholds_are_the_fda_table():
    """The W20-Z thresholds are 21 CFR 101.9(c) values, not a separate copy."""
    from nutpred.metrics import FDA_ZERO_PER_SERVING
    assert FDA_ZERO_PER_SERVING["Calcium(mg)"] == 26.0    # 2% of 1300 mg DV
    assert FDA_ZERO_PER_SERVING["Iron(mg)"] == 0.36       # 2% of 18 mg DV
    assert FDA_ZERO_PER_SERVING["Fiber(g)"] == 0.5


def test_eu_row_assignment():
    assert eu_row("Calcium(mg)") == "mineral"
    assert eu_row("Fiber(g)") == "carb"
    assert eu_row("Total fat(g)") == "fat"
    # Table 1 has no row for energy or cholesterol
    assert not has_eu_row("Energy(kcal)")
    assert not has_eu_row("Cholesterol(mg)")
    with pytest.raises(ValueError):
        eu_mask(np.array([1.0]), np.array([1.0]), "Energy(kcal)")


def test_eu_declared_and_reference_bases_differ_at_the_upper_edge():
    """The +45% headroom attaches to the declared value, not the reference.

    y=100, p=150: on the reference basis the estimate is +50% and fails. On
    the declared basis the band is centred on the declaration of 150, which
    admits a reference down to 97.5 -- so 100 passes.
    """
    y = np.array([100.0])
    p = np.array([150.0])
    assert eu_mask(y, p, "Calcium(mg)", basis="declared")[0]
    assert not eu_mask(y, p, "Calcium(mg)", basis="reference")[0]

    # and they agree well inside the band
    p_mid = np.array([105.0])
    assert eu_mask(y, p_mid, "Calcium(mg)", basis="declared")[0]
    assert eu_mask(y, p_mid, "Calcium(mg)", basis="reference")[0]


def test_eu_rejects_an_unknown_basis():
    with pytest.raises(ValueError):
        eu_mask(np.array([1.0]), np.array([1.0]), "Calcium(mg)", basis="nope")


def test_round_declared_follows_table_4():
    # macros: nearest 1 g at or above 10 g, nearest 0.1 g below, 0 at/below 0.5 g
    got = round_declared(np.array([12.4, 0.64, 0.5, 0.4]), "Fiber(g)")
    assert got.tolist() == pytest.approx([12.0, 0.6, 0.0, 0.0])
    # calcium is one of the named 3-significant-figure nutrients
    assert round_declared(np.array([123.456]), "Calcium(mg)")[0] == pytest.approx(123.0)
    # a negative prediction cannot be declared; it declares 0
    assert round_declared(np.array([-5.0]), "Fiber(g)")[0] == pytest.approx(0.0)


def test_fda_floor_and_ceiling_are_different_rules():
    y = np.array([100.0])
    # fibre is a floor nutrient: under-delivery fails, over-delivery passes
    assert fda_mask(y, np.array([85.0]), "Fiber(g)")[0]
    assert not fda_mask(y, np.array([70.0]), "Fiber(g)")[0]
    assert fda_mask(y, np.array([200.0]), "Fiber(g)")[0]
    # sodium is a ceiling nutrient: the asymmetry runs the other way
    assert fda_mask(y, np.array([115.0]), "Sodium(mg)")[0]
    assert not fda_mask(y, np.array([130.0]), "Sodium(mg)")[0]
    assert fda_mask(y, np.array([10.0]), "Sodium(mg)")[0]


def test_all_masks_returns_every_rate_criterion():
    y = np.array([0.0, 50.0, 100.0])
    p = np.array([0.0, 55.0, 130.0])
    ss = np.array([30.0, 30.0, 30.0])
    m = all_masks(y, p, "Calcium(mg)", ss, np.array(["g"] * 3, dtype=object))
    assert set(m) == {"W20", "W20_v2", "EU", "FDA"}
    for k, v in m.items():
        assert v.dtype == bool and v.shape == y.shape, k
