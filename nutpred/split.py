"""The train/test split, defined once.

run.py and the feature builders (``build_ingredient_features --vocab-scope
train`` and ``--embed``) all call :func:`assign_train_test`, so the vocabulary
and the UMAP reducer are fit on exactly the rows run.py later trains on. No
split is read from the input file.

The split is the one the reported results use: 80/20, ``random_state=42``,
over rows whose three targets are all present, stratified on each row's
zero/non-zero pattern across the targets. A row whose calcium is non-zero,
iron zero and fiber non-zero has pattern ``"010"``; stratifying on it keeps
every pattern -- including rare ones such as fiber = 0 -- in the same
proportion in train and test, which the hurdle model's zero classifier needs.

The pattern is built in ``SPLIT_TARGETS`` order (calcium, iron, fiber). That
order is not cosmetic: it fixes the class labels the stratifier sorts, and so
which rows land in test. It is the order the reported split was made in.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_TARGETS = ["Calcium(mg)", "Iron(mg)", "Fiber(g)"]
SPLIT_COL = "SampleType"
TRAIN, TEST, EXCLUDED = "train", "test", "excluded"


def assign_train_test(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    targets: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Label every row ``"train"``, ``"test"`` or ``"excluded"``.

    Rows missing any target are ``"excluded"``: they can be neither trained on
    nor scored. The result depends only on the row order and the target
    values, not on the index labels, so the same table split by two different
    entry points gives the same rows.
    """
    targets = list(targets or SPLIT_TARGETS)
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise KeyError(f"split needs target columns {missing}")

    y = df[targets].apply(pd.to_numeric, errors="coerce")
    eligible = y.notna().all(axis=1).to_numpy()
    pos = np.flatnonzero(eligible)
    if pos.size < 10:
        raise ValueError(f"only {pos.size} rows have all of {targets}; cannot split")

    pattern = (
        y.iloc[pos].le(0).astype(int).astype(str).agg("".join, axis=1).to_numpy()
    )
    # A pattern seen once cannot be stratified; fall back to an unstratified
    # split rather than fail (tiny smoke-test subsamples hit this).
    counts = pd.Series(pattern).value_counts()
    stratify = pattern if int(counts.min()) >= 2 else None
    _, te = train_test_split(pos, test_size=test_size, random_state=int(seed), stratify=stratify)

    out = np.full(len(df), EXCLUDED, dtype=object)
    out[pos] = TRAIN
    out[te] = TEST
    return pd.Series(out, index=df.index, name=SPLIT_COL)


def train_mask(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2) -> np.ndarray:
    """Boolean mask of the train rows of :func:`assign_train_test`."""
    return (assign_train_test(df, seed=seed, test_size=test_size) == TRAIN).to_numpy()
