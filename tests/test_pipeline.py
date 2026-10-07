"""Tests for the split, the hurdle threshold, embedding I/O and the optimizer.

Each test pins one property the reported pipeline depends on: test rows never
choose anything, and every entry point agrees on which rows are test rows.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from nutpred.metrics import F1_THRESHOLD_GRID, select_threshold, zero_upper_bound_per100g
from nutpred.preprocess import _expand_umap_list_column, parse_vector_cell, rank_vocabulary
from nutpred.split import SPLIT_TARGETS, assign_train_test


def _targets(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({t: rng.uniform(0.1, 10, n) for t in SPLIT_TARGETS})
    for t, pz in zip(SPLIT_TARGETS, (0.3, 0.25, 0.1)):
        df.loc[rng.random(n) < pz, t] = 0.0
    return df


# --- split ------------------------------------------------------------------

def test_split_is_deterministic_and_80_20():
    df = _targets(500)
    a, b = assign_train_test(df, seed=42), assign_train_test(df, seed=42)
    assert a.equals(b)
    assert (a == "test").sum() == 100 and (a == "train").sum() == 400


def test_split_ignores_index_labels_and_extra_columns():
    """Two entry points loading the same table must get the same rows."""
    df = _targets(300)
    relabelled = df.set_index(np.arange(1000, 1300))
    relabelled["data_type"] = "train"          # a stale marker is never read
    assert (assign_train_test(df).to_numpy() == assign_train_test(relabelled).to_numpy()).all()


def test_split_keeps_zero_patterns_in_proportion():
    df = _targets(2000)
    s = assign_train_test(df)
    pattern = df[SPLIT_TARGETS].le(0).astype(int).astype(str).agg("".join, axis=1)
    share = pattern[s == "test"].value_counts(normalize=True)
    overall = pattern.value_counts(normalize=True)
    assert (share - overall).abs().max() < 0.01


def test_split_excludes_rows_missing_a_target():
    df = _targets(200)
    df.loc[[3, 7], "Iron(mg)"] = np.nan
    s = assign_train_test(df)
    assert s.loc[[3, 7]].eq("excluded").all()
    assert (s == "excluded").sum() == 2


# --- hurdle threshold ---------------------------------------------------------

def test_select_threshold_f1_scans_the_grid():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = select_threshold(y, p, method="f1")
    assert thr in F1_THRESHOLD_GRID
    assert 0.3 < thr <= 0.7                    # any cut between the classes is perfect
    assert select_threshold(y, p, method="fixed", fixed_threshold=0.42) == 0.42
    with pytest.raises(ValueError):
        select_threshold(y, p, method="nope")


def test_hurdle_does_not_look_at_test_labels():
    """Shuffling the test labels must not change a single test prediction."""
    pytest.importorskip("xgboost")
    from nutpred.predict import _train_hurdle

    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 4))
    y = np.where(X[:, 0] > 0.3, 0.0, np.exp(X[:, 1]))
    Xtr, Xte, ytr, yte = X[:200], X[200:], y[:200], y[200:]
    kw = dict(Xtr=Xtr, Xte=Xte, Xall=X, y_tr=ytr, model_type="xgboost", cv=3,
              random_state=42, zero_threshold_method="f1", zero_threshold=0.5,
              zero_eps=0.0, pos_log1p=False, n_trials=2)
    a = _train_hurdle(y_te=yte, **kw)
    b = _train_hurdle(y_te=rng.permutation(yte)[::-1].copy(), **kw)
    assert np.array_equal(a[0], b[0])                      # test predictions
    assert a[4]["Threshold"] == b[4]["Threshold"]


# --- embedding cells ----------------------------------------------------------

def test_vector_cells_parse_plain_and_numpy2_reprs():
    assert parse_vector_cell("[1.5, -2.0]") == [1.5, -2.0]
    assert parse_vector_cell("[np.float64(1.5), np.float32(-2.0)]") == [1.5, -2.0]
    assert parse_vector_cell(float("nan")) is None
    assert parse_vector_cell("[]") is None


def test_umap_expansion_never_fills_zeros_silently():
    df = pd.DataFrame({"umap_2": ["[np.float64(1.0), np.float64(2.0)]", None]})
    _expand_umap_list_column(df, "umap_2", 2)
    assert df.loc[0, ["umap_2_1", "umap_2_2"]].tolist() == [1.0, 2.0]
    assert df.loc[1, ["umap_2_1", "umap_2_2"]].isna().all()

    # a wrong-length cell (e.g. a column-shifted row) is missing, not zero-padded
    shifted = pd.DataFrame({"umap_2": ["[1.0, 2.0]"] * 30 + ["23"]})
    _expand_umap_list_column(shifted, "umap_2", 2)
    assert shifted.iloc[-1][["umap_2_1", "umap_2_2"]].isna().all()
    # ...but a whole column of them is a mistake, not a few bad rows
    with pytest.raises(ValueError):
        _expand_umap_list_column(pd.DataFrame({"umap_2": ["[1.0, 2.0, 3.0]"]}), "umap_2", 2)
    with pytest.raises(ValueError):
        _expand_umap_list_column(pd.DataFrame({"umap_2": ["[a, b]"]}), "umap_2", 2)


def test_umap_is_fit_on_train_rows_only():
    """Moving the test rows must not move the train rows' coordinates."""
    pytest.importorskip("umap")
    from nutpred.prepare import fit_umap_list_columns

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    mask = np.arange(120) < 90
    a = fit_umap_list_columns(X, [2], random_state=0, train_mask=mask, n_neighbors=10)["UMAP_2"]
    X2 = X.copy()
    X2[~mask] += 50.0
    b = fit_umap_list_columns(X2, [2], random_state=0, train_mask=mask, n_neighbors=10)["UMAP_2"]
    assert np.allclose(a[mask], b[mask])


def test_embedding_dims_known_models():
    from nutpred.prepare import DEFAULT_UMAP_DIMS, EMBEDDING_DIMS
    assert EMBEDDING_DIMS["text-embedding-3-small"] == 1536
    assert DEFAULT_UMAP_DIMS == (20,)


# --- vocabulary ---------------------------------------------------------------

def test_train_vocabulary_uses_the_shared_split():
    df = _targets(100)
    df["mapped_list"] = [["SUGAR"]] * 100
    test = assign_train_test(df).eq("test").to_numpy()
    # an ingredient that occurs only in test rows must not enter the vocabulary
    df.loc[test, "mapped_list"] = pd.Series([["SUGAR", "SECRET"]] * int(test.sum()),
                                            index=df.index[test])
    vocab, audit = rank_vocabulary(df, k=5, scope="train")
    assert vocab == ["SUGAR"]
    assert audit["n_train_rows"] == 80


# --- misc fixes ---------------------------------------------------------------

@pytest.mark.parametrize("mod", ["nutpred.prepare", "nutpred.viz", "nutpred.cleaning",
                                 "nutpred.preprocess"])
def test_star_import(mod):
    m = importlib.import_module(mod)
    for name in m.__all__:
        assert isinstance(name, str) and hasattr(m, name), name


def test_filter_valid_nutrition_data_runs():
    from nutpred.preprocess import filter_valid_nutrition_data
    df = pd.DataFrame({
        "Energy(kcal)": [400.0, 2000.0], "Carbohydrate(g)": [60.0, 60.0],
        "Protein(g)": [5.0, 5.0], "Total fat(g)": [15.0, 15.0],
        "Saturated fatty acids(g)": [3.0, 3.0], "Trans fatty acids(g)": [0.0, 0.0],
        "Cholesterol(mg)": [0.0, 0.0],
    })
    assert len(filter_valid_nutrition_data(df)) == 1      # 2000 kcal row dropped


def test_zero_upper_bound_converts_per_serving_to_per_100g():
    # 26 mg per 50 g serving == 52 mg per 100 g
    assert zero_upper_bound_per100g(np.array([50.0]), 26.0)[0] == pytest.approx(52.0)
    assert np.isnan(zero_upper_bound_per100g(np.array([0.0]), 26.0)[0])


@pytest.mark.parametrize("solver", ["osqp", "clarabel"])
def test_optimizer_solves_with_each_solver(solver):
    cp = pytest.importorskip("cvxpy")
    from nutpred.build_knowledge_guided_features import _run_optimization
    if getattr(cp, solver.upper()) not in cp.installed_solvers():
        pytest.skip(f"{solver} not installed")
    rng = np.random.default_rng(0)
    X_all = rng.uniform(0, 50, size=(4, 8))                     # 4 ingredients x 8 nutrients
    Y = np.array([0.6, 0.3, 0.1, 0.0]) @ X_all
    _, failed = _run_optimization([[0, 1, 2, 3]], X_all, Y[None, :], solver_name=solver,
                                  top_list=list("abcd"), use_yf=False)
    assert failed == []
