import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def r2_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return np.nan
    y_bar = np.mean(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_bar) ** 2)
    if sst <= 1e-12:
        return np.nan
    return float(1 - (sse / sst))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(y_true, dtype=float)
    nz = denom != 0
    if not np.any(nz):
        return 0.0
    out[nz] = np.abs(y_true[nz] - y_pred[nz]) / denom[nz]
    return float(100 * np.mean(out))


def within_20pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Within 20% relative error over NON-ZERO rows only.

    Zero-declared rows are dropped from the denominator entirely. This is the
    training harness's historical metric; the paper's headline W20 is
    within_20pct_including_zero(), which keeps them. On these zero-inflated
    targets the two are far apart -- report which one you mean.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt != 0)
    if not np.any(mask):
        return float("nan")
    pct_err = np.abs(yp[mask] - yt[mask]) / np.abs(yt[mask])
    return float(np.mean(pct_err <= 0.2))


def within_20pct_including_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """W20 as the paper reports it: zero-declared rows included in the denominator.

    A row declaring 0 counts as a hit only when the prediction is also 0 --
    relative error is undefined at zero. This is NOT within_20pct(), which
    drops those rows entirely; with 30-35% zero-inflated targets the two differ
    substantially. Delegates to w20_mask() so there is one rule.
    """
    hit = w20_mask(y_true, y_pred)
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(hit[valid]))


def _safe_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr


# Candidate cut-offs for the F1 search: 0.01, 0.02, ..., 0.99.
F1_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1",
    fixed_threshold: float = 0.5,
) -> float:
    """Probability cut-off for the hurdle model's zero classifier.

    Call this with TRAINING labels and the classifier's probabilities on the
    training rows -- never with test data, or the threshold is tuned on the
    rows it is then scored on.

    ``"f1"`` scans F1_THRESHOLD_GRID and keeps the first cut-off with the
    highest F1 of the zero class (label 1); this is the reported setting.
    ``"youden"`` maximises TPR - FPR on the ROC curve. ``"fixed"`` returns
    ``fixed_threshold``.
    """
    if method == "fixed":
        return float(fixed_threshold)
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if method == "youden":
        fpr, tpr, thr = _safe_roc(y_true, y_prob)
        return float(thr[int(np.argmax(tpr - fpr))])
    if method == "f1":
        best_thr, best = float(fixed_threshold), -np.inf
        for thr in F1_THRESHOLD_GRID:
            score = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
            if score > best:
                best, best_thr = float(score), float(thr)
        return best_thr
    raise ValueError(f"Unknown threshold method {method!r}; use fixed, youden or f1")


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = _safe_roc(y_true, y_prob)
    return float(auc(fpr, tpr))


def zero_upper_bound_per100g(serving_g: np.ndarray, threshold_per_serving: float) -> np.ndarray:
    """Per-100 g amount below which a declared 0 is interval-compatible.

    The label rule is per serving and predictions are per 100 g, so the
    per-serving threshold is scaled by 100 / serving size. A serving size that
    is missing or not positive gives NaN, which no prediction can fall under.
    """
    s = np.asarray(serving_g, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, float(threshold_per_serving) * 100.0 / s, np.nan)


@dataclass
class PredictionMetrics:
    r2: float
    rmse: float
    mae: float
    within20: float
    interval_aware_rmse: float
    interval_aware_mae: float
    interval_aware_within20: float
    zero_label_n: int
    zero_label_pred_zero_or_near_zero: int
    zero_label_pred_low_positive_compatible: int
    zero_label_pred_above_threshold: int
    # The paper's headline W20, kept alongside the harness's zero-excluding
    # `within20` so a run's own output can be compared with the manuscript.
    within20_incl_zero: float = float("nan")


def evaluate_predictions(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    *args: Any,
    nutrient: Optional[str] = None,
    target: Optional[str] = None,
    model_label: Optional[str] = None,
    model: Optional[str] = None,
    serving_g: Optional[np.ndarray] = None,
    threshold_per_serving: Optional[float] = None,
    **kwargs: Any,
) -> PredictionMetrics:
    """
    Regression metrics with optional interval-aware evaluation for zero labels.

    Supports ``evaluate_predictions(y_true, y_pred, nutrient, model_label, ...)``
    or keyword-only ``y_true=..., y_pred=..., target=..., model=...``.
    """
    if len(args) >= 2:
        nutrient = nutrient or args[0]
        model_label = model_label or args[1]
    name = nutrient or target or kwargs.get("nutrient") or kwargs.get("target") or ""
    _ = model_label or model or kwargs.get("model_label") or kwargs.get("model") or name

    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size != yp.size:
        raise ValueError(f"y_true and y_pred length mismatch: {yt.size} vs {yp.size}")

    finite = np.isfinite(yt) & np.isfinite(yp)
    serv = None
    if serving_g is not None:
        serv = np.asarray(serving_g, dtype=float).ravel()
        if serv.size != yt.size:
            raise ValueError(f"serving_g length mismatch: {serv.size} vs {yt.size}")
        finite &= np.isfinite(serv)

    yt_f = yt[finite]
    yp_f = yp[finite]
    serv_f = serv[finite] if serv is not None else None

    nanf = float("nan")
    if yt_f.size == 0:
        return PredictionMetrics(
            r2=nanf,
            rmse=nanf,
            mae=nanf,
            within20=nanf,
            interval_aware_rmse=nanf,
            interval_aware_mae=nanf,
            interval_aware_within20=nanf,
            zero_label_n=0,
            zero_label_pred_zero_or_near_zero=0,
            zero_label_pred_low_positive_compatible=0,
            zero_label_pred_above_threshold=0,
        )

    r2 = float(r2_score(yt_f, yp_f))
    mse = float(mean_squared_error(yt_f, yp_f))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt_f, yp_f))
    w20 = float(within_20pct(yt_f, yp_f))
    w20_iz = float(within_20pct_including_zero(yt_f, yp_f))

    if threshold_per_serving is None or serv_f is None:
        zmask = yt_f <= 0
        return PredictionMetrics(
            r2=r2,
            rmse=rmse,
            mae=mae,
            within20=w20,
            within20_incl_zero=w20_iz,
            interval_aware_rmse=nanf,
            interval_aware_mae=nanf,
            interval_aware_within20=nanf,
            zero_label_n=int(np.sum(zmask)),
            zero_label_pred_zero_or_near_zero=0,
            zero_label_pred_low_positive_compatible=0,
            zero_label_pred_above_threshold=0,
        )

    upper = zero_upper_bound_per100g(serv_f, threshold_per_serving)
    zero_mask = yt_f <= 0
    nz_mask = yt_f > 0

    sq = np.zeros_like(yt_f, dtype=float)
    sq[nz_mask] = (yt_f[nz_mask] - yp_f[nz_mask]) ** 2
    neg = zero_mask & (yp_f < 0)
    compat = zero_mask & (yp_f >= 0) & (yp_f < upper)
    # Everything else declared 0 is above the bound; a row with no usable
    # serving size (NaN bound) is scored against 0, i.e. exact-zero.
    above = zero_mask & ~neg & ~compat
    upper = np.where(np.isfinite(upper), upper, 0.0)
    sq[neg] = yp_f[neg] ** 2
    sq[compat] = 0.0
    sq[above] = (yp_f[above] - upper[above]) ** 2
    interval_rmse = float(np.sqrt(np.mean(sq)))

    abs_e = np.zeros_like(yt_f, dtype=float)
    abs_e[nz_mask] = np.abs(yt_f[nz_mask] - yp_f[nz_mask])
    abs_e[neg] = np.abs(yp_f[neg])
    abs_e[compat] = 0.0
    abs_e[above] = np.abs(yp_f[above] - upper[above])
    interval_mae = float(np.mean(abs_e))

    ok = np.zeros_like(yt_f, dtype=bool)
    if np.any(nz_mask):
        rel = np.abs(yp_f[nz_mask] - yt_f[nz_mask]) / np.maximum(np.abs(yt_f[nz_mask]), 1e-12)
        ok[nz_mask] = rel <= 0.2
    ok[compat] = True
    interval_w20 = float(np.mean(ok))

    atol = 1e-5
    zl = zero_mask
    zero_n = int(np.sum(zl))
    n_near = int(np.sum(zl & (np.abs(yp_f) <= atol)))
    n_low = int(np.sum(zl & (yp_f > atol) & (yp_f < upper)))
    n_above = int(np.sum(zl & ((yp_f >= upper) | (yp_f < -atol))))

    return PredictionMetrics(
        r2=r2,
        rmse=rmse,
        mae=mae,
        within20=w20,
        within20_incl_zero=w20_iz,
        interval_aware_rmse=interval_rmse,
        interval_aware_mae=interval_mae,
        interval_aware_within20=interval_w20,
        zero_label_n=zero_n,
        zero_label_pred_zero_or_near_zero=n_near,
        zero_label_pred_low_positive_compatible=n_low,
        zero_label_pred_above_threshold=n_above,
    )


@dataclass
class ClassifierMetrics:
    target: str
    model: str
    accuracy: float
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    target: str = "",
    model: str = "",
) -> ClassifierMetrics:
    yt = np.asarray(y_true, dtype=int).ravel()
    yhat = np.asarray(y_pred, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if yt.size == 0:
        nanf = float("nan")
        return ClassifierMetrics(
            target=target,
            model=model,
            accuracy=nanf,
            balanced_accuracy=nanf,
            f1=nanf,
            precision=nanf,
            recall=nanf,
            roc_auc=nanf,
        )
    acc = float(accuracy_score(yt, yhat))
    bacc = float(balanced_accuracy_score(yt, yhat))
    uniq = len(np.unique(yt))
    if uniq < 2:
        nanf = float("nan")
        return ClassifierMetrics(
            target=target,
            model=model,
            accuracy=acc,
            balanced_accuracy=bacc,
            f1=nanf,
            precision=nanf,
            recall=nanf,
            roc_auc=nanf,
        )
    f1 = float(f1_score(yt, yhat, zero_division=0))
    prec = float(precision_score(yt, yhat, zero_division=0))
    rec = float(recall_score(yt, yhat, zero_division=0))
    try:
        aucv = float(roc_auc_score(yt, p))
    except ValueError:
        aucv = float("nan")
    return ClassifierMetrics(
        target=target,
        model=model,
        accuracy=acc,
        balanced_accuracy=bacc,
        f1=f1,
        precision=prec,
        recall=rec,
        roc_auc=aucv,
    )

# ===========================================================================
# Reported label-agreement criteria
# ===========================================================================
# Everything above this line is the training harness's own scoring. Everything
# below is the set of criteria the paper reports, kept here so there is exactly
# one definition of each.
#
#   W20       Within 20% relative error. A row declaring 0 counts as a hit only
#             when the prediction is exactly 0 -- relative error is undefined at
#             zero, so nothing weaker is defensible without a tolerance.
#
#   W20_v2    "W20-Z" in the manuscript: label-zero-adjusted accuracy. Same as
#             W20 on non-zero rows; a row declaring 0 counts as a hit when the
#             prediction is small enough that a real label could legitimately
#             have printed "0" -- per-serving amount strictly below the
#             21 CFR 101.9(c) zero-declaration threshold. Rows whose serving
#             unit cannot be resolved to a mass fall back to the exact-0 rule.
#
#   EU        EU Regulation tolerance bands (Guidance Document Table 1) on a
#             DECLARED-VALUE basis: the prediction is rounded to label precision
#             by Table 4 and the band is centred on that declaration, which is
#             the direction the regulation governs. basis="reference" recovers
#             the superseded reading that centred the band on y_true.
#
#   FDA       21 CFR 101.9 compliance, which is TWO rules, not one. (g)(4)(ii)
#             puts protein, carbohydrate, fibre and the minerals under a Class
#             II FLOOR of 80% of declared; (g)(5) puts calories, total fat,
#             saturates, cholesterol, sodium and sugars under a CEILING of 120%.
#             A row declaring 0 carries no ratio information either way, so it
#             is scored on the (c) rounding rule instead.
#
# R2 for reported numbers is r2_manual() above.
# ---------------------------------------------------------------------------
# EU Guidance Table 4 -- rounding guidelines for the nutrition declaration.
# European Commission (December 2012), section 6 and Table 4.

# Tiered rows, as (bound, step, inclusive) in the nutrient's own per-100 g
# units, ordered coarsest-first. ``step`` is the rounding increment and
# ``inclusive`` picks >= over >. Bounds are on |value|. Anything matching no
# tier is below the row's declare-as-zero floor and becomes 0.
#
# The mixed inclusivity is the table's, not ours: the top tier reads ">= 10 g"
# while the middle reads "< 10 g and > 0.5 g", so exactly 0.5 g falls in the
# "<= 0.5 g" branch and declares as 0 g.
ROUND_TIERS: dict[str, Tuple[Tuple[float, float, bool], ...]] = {
    # Fat, Carbohydrate, sugars, Protein, fibre, polyols, starch -- g/100 g
    "macro": ((10.0, 1.0, True), (0.5, 0.1, False)),
    # Saturates, mono-unsaturates, polyunsaturates -- g/100 g
    "fatty_acid": ((10.0, 1.0, True), (0.1, 0.1, False)),
    # Sodium -- the table's 1 g / 0.1 g / 0.01 g / 0.005 g, carried in MG.
    "sodium_mg": ((1000.0, 100.0, True), (5.0, 10.0, False)),
    # Salt -- 1 g / 0.1 g / 0.01 g / 0.0125 g, carried in MG. Not a column we
    # hold; present so the table is fully represented.
    "salt_mg": ((1000.0, 100.0, True), (12.5, 10.0, False)),
    # Energy -- nearest whole unit, no zero branch.
    "energy": ((0.0, 1.0, True),),
}

# Significant-figure rows. The named eight take 3; every other vitamin or
# mineral takes 2.
SIGFIG_ROW: dict[str, int] = {"vitamin_mineral_3sf": 3, "vitamin_mineral_2sf": 2}

# Which Table 4 row each modelled nutrient falls in. A target
# absent from this map has no row -- see has_rounding_row().
ROUNDING_ROW: dict[str, str] = {
    "Energy(kcal)": "energy",
    "Total fat(g)": "macro",
    "Carbohydrate(g)": "macro",
    "Total sugar(g)": "macro",
    "Protein(g)": "macro",
    "Fiber(g)": "macro",
    "Saturated fatty acids(g)": "fatty_acid",
    "Sodium(mg)": "sodium_mg",
    # Calcium is one of the eight named 3-sig-fig nutrients; iron is not, so it
    # falls in "all other vitamins and minerals" at 2 significant figures.
    "Calcium(mg)": "vitamin_mineral_3sf",
    "Iron(mg)": "vitamin_mineral_2sf",
}

# Short human-readable form, for figure annotation.
ROUND_ROW_TEXT: dict[str, str] = {
    "macro": "1 g / 0.1 g, 0 below 0.5 g",
    "fatty_acid": "1 g / 0.1 g, 0 below 0.1 g",
    "sodium_mg": "100 mg / 10 mg, 0 below 5 mg",
    "salt_mg": "100 mg / 10 mg, 0 below 12.5 mg",
    "energy": "nearest 1 kcal",
    "vitamin_mineral_3sf": "3 significant figures",
    "vitamin_mineral_2sf": "2 significant figures",
}

# Declare-as-zero floor per row, in the nutrient's own per-100 g units. None
# where the row has no zero branch (energy, and the sig-fig rows -- Table 4
# gives vitamins and minerals no "may be declared 0" concentration).
ZERO_FLOOR: dict[str, Optional[float]] = {
    "macro": 0.5,
    "fatty_acid": 0.1,
    "sodium_mg": 5.0,
    "salt_mg": 12.5,
    "energy": None,
    "vitamin_mineral_3sf": None,
    "vitamin_mineral_2sf": None,
}


def rounding_row(target: Optional[str]) -> Optional[str]:
    """Which Table 4 row ``target`` falls in, or None if the table has no row."""
    return ROUNDING_ROW.get(target) if target is not None else None


def has_rounding_row(target: Optional[str]) -> bool:
    """False for anything Table 4 does not cover (e.g. cholesterol)."""
    return rounding_row(target) is not None


def _decimal_scale_back(counts: np.ndarray, decimals: np.ndarray | int) -> np.ndarray:
    """``counts * 10**-decimals``, taking the branch that lands on the right double.

    Every step in Table 4 is a power of ten, so a rounded value is an integer
    count of steps and the only question is how to get back to the value. It
    matters: ``68 * 0.1`` is 6.800000000000001, which is NOT the nearest double
    to 6.8, and a fibre row whose true value sits exactly 2 g away then fails a
    tolerance the guidance says it meets. Dividing an integer by an integer
    power of ten gives the correctly rounded double; multiplying by an integer
    power of ten is exact. So divide when the step is a fraction and multiply
    when it is not, and never multiply by a fractional step.
    """
    d = np.asarray(decimals)
    pos = d > 0
    out = np.where(pos,
                   counts / np.power(10.0, np.where(pos, d, 0)),
                   counts * np.power(10.0, np.where(pos, 0, -d)))
    return out


def _round_half_away(values: np.ndarray, step: float) -> np.ndarray:
    """Round to a multiple of ``step``, halves away from zero.

    numpy's round is half-to-even, which is not what "rounding to the nearest
    0.1 g" means on a label, so the half case is done by hand. The 1e-9 nudge
    absorbs the binary representation error that otherwise sends values such as
    0.15 (stored just under) down instead of up.
    """
    if step <= 0:
        return np.zeros_like(values)
    counts = np.floor(np.abs(values) / step + 0.5 + 1e-9)
    decimals = int(round(-np.log10(step)))
    return np.sign(values) * _decimal_scale_back(counts, decimals)


def _round_sigfig(values: np.ndarray, digits: int) -> np.ndarray:
    """Round to ``digits`` significant figures, halves away from zero.

    Exact zero stays zero; significant figures are undefined there and a
    declared 0 is already at label precision.
    """
    out = np.zeros_like(values)
    nz = np.isfinite(values) & (values != 0)
    mag = np.floor(np.log10(np.abs(values[nz])))
    decimals = (digits - 1 - mag).astype(int)
    counts = np.floor(np.abs(values[nz]) * np.power(10.0, decimals) + 0.5 + 1e-9)
    out[nz] = np.sign(values[nz]) * _decimal_scale_back(counts, decimals)
    out[~np.isfinite(values)] = np.nan
    return out


def round_declared(values: np.ndarray, target: Optional[str]) -> np.ndarray:
    """Round ``values`` to the precision at which ``target`` may be declared.

    Values are per 100 g, in the target's own units (g for macros, mg for
    sodium/calcium/iron). Negative inputs are clipped to zero first: a label
    cannot declare a negative amount, so a negative prediction declares "0".

    Raises for a target Table 4 does not cover -- call has_rounding_row() first.
    """
    row = rounding_row(target)
    if row is None:
        raise ValueError(
            f"EU Table 4 has no rounding row for {target!r}; "
            "check has_rounding_row() before rounding."
        )
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    v = np.where(finite, np.maximum(v, 0.0), np.nan)

    if row in SIGFIG_ROW:
        return _round_sigfig(v, SIGFIG_ROW[row])

    tiers: Sequence[Tuple[float, float, bool]] = ROUND_TIERS[row]
    out = np.zeros_like(v)
    remaining = finite.copy()
    mag = np.where(finite, np.abs(v), 0.0)
    for bound, step, inclusive in tiers:
        sel = remaining & (mag >= bound if inclusive else mag > bound)
        out[sel] = _round_half_away(v[sel], step)
        remaining &= ~sel
    # Anything left matched no tier, i.e. it is at or below the row's
    # declare-as-zero floor, and ``out`` is already 0 there.
    out[~finite] = np.nan
    return out


# Serving-size units that can be read as a mass (ml taken as g). Rows with any
# other unit cannot be converted to a per-serving amount, so the per-serving
# zero rules in w20_v2_mask() and fda_mask() fall back to an exact-zero test.
# The thresholds themselves are FDA_ZERO_PER_SERVING, further down.
RESOLVABLE_SERVING_UNITS = {"g", "grm", "gm", "ml", "mlt"}


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ((y - p) ** 2).sum() / ss_tot)


def w20_mask(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    z = y == 0
    hit = np.empty(len(y), dtype=bool)
    hit[z] = np.abs(p[z]) <= 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        hit[~z] = np.abs(p[~z] - y[~z]) / y[~z] <= 0.20
    return hit


def w20_v2_mask(
    y: np.ndarray,
    p: np.ndarray,
    target: str,
    serving_size: np.ndarray | None = None,
    serving_size_unit: np.ndarray | None = None,
) -> np.ndarray:
    """W20-Z: w20, but zero-declared rows are scored against the label rule.

    A declared 0 is a hit when the predicted per-serving amount falls strictly
    below the 21 CFR 101.9(c) zero-declaration threshold for that nutrient --
    i.e. when the prediction would itself have been printed as "0" on a real
    label. Needs serving_size and serving_size_unit; rows whose unit is not a
    resolvable mass fall back to the exact-zero test.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    hit = w20_mask(y, p)
    thr = FDA_ZERO_PER_SERVING.get(target)
    if thr is None or serving_size is None or serving_size_unit is None:
        return hit
    unit = pd.Series(serving_size_unit).astype(str).str.strip().str.lower().to_numpy()
    resolvable = np.isin(unit, list(RESOLVABLE_SERVING_UNITS))
    ss = pd.to_numeric(pd.Series(serving_size), errors="coerce").to_numpy(dtype=float)
    per_serving = np.abs(p) * ss / 100.0
    z = y == 0
    # Strict "<": 21 CFR 101.9(c) allows "0" only for amounts LESS THAN the
    # threshold, so an amount exactly at it is still a declarable non-zero.
    # Matches fda_mask() and the manuscript.
    roundable = z & resolvable & np.isfinite(per_serving) & (per_serving < thr)
    return hit | roundable


# ---------------------------------------------------------------------------
# EU tolerance -- Guidance Document Table 1, in full.
#
# European Commission (December 2012), "Guidance Document for Competent
# Authorities for the Control of Compliance with EU Legislation ... with Regard
# to the Setting of Tolerances for Nutrient Values Declared on a Label",
# Table 1 ("tolerances for foods other than food supplements including
# measurement uncertainty"):
#
#     Vitamins                                  -35% / +50%
#     Minerals                                  -35% / +45%
#     Carbohydrate, Sugars, Protein, Fibre      <10 g/100 g:   +/- 2 g
#                                               10-40 g/100 g: +/- 20%
#                                               >40 g/100 g:   +/- 8 g
#     Fat                                       <10 g/100 g:   +/- 1.5 g
#                                               10-40 g/100 g: +/- 20%
#                                               >40 g/100 g:   +/- 8 g
#     Saturates, Mono-, Polyunsaturates         <4 g/100 g:    +/- 0.8 g
#                                               >=4 g/100 g:   +/- 20%
#     Sodium                                    <0.5 g/100 g:  +/- 0.15 g
#                                               >=0.5 g/100 g: +/- 20%
#     Salt                                      <1.25 g/100 g: +/- 0.375 g
#                                               >=1.25 g/100 g:+/- 20%
#
# Energy and Cholesterol have no Table 1 row: energy is calculated from the
# macronutrients under Reg. 1169/2011 Annex XIV and inherits their accuracy,
# and cholesterol is not a declarable nutrient in the EU scheme. Call
# has_eu_row() before scoring rather than letting them fall through to a band
# that does not apply.
MINERAL_BAND = (-0.35, 0.45)
VITAMIN_BAND = (-0.35, 0.50)

# Tiered rows as (upper_bound_exclusive, kind, value) in the nutrient's own
# per-100 g units; the last tier's bound is +inf. kind is "abs" (a half-width
# in native units) or "rel" (a fraction of the amount the tier was read at).
EU_TIERS: dict[str, tuple[tuple[float, str, float], ...]] = {
    "carb": ((10.0, "abs", 2.0), (40.0, "rel", 0.20), (np.inf, "abs", 8.0)),
    "fat": ((10.0, "abs", 1.5), (40.0, "rel", 0.20), (np.inf, "abs", 8.0)),
    "saturates": ((4.0, "abs", 0.8), (np.inf, "rel", 0.20)),
    # The guidance states sodium and salt in grams; our columns are mg.
    "sodium": ((500.0, "abs", 150.0), (np.inf, "rel", 0.20)),
    "salt": ((1250.0, "abs", 375.0), (np.inf, "rel", 0.20)),
}

EU_ROW: dict[str, str] = {
    "Carbohydrate(g)": "carb",
    "Total sugar(g)": "carb",
    "Protein(g)": "carb",
    "Fiber(g)": "carb",
    "Total fat(g)": "fat",
    "Saturated fatty acids(g)": "saturates",
    "Sodium(mg)": "sodium",
    "Calcium(mg)": "mineral",
    "Iron(mg)": "mineral",
}

EU_ROW_TEXT: dict[str, str] = {
    "mineral": "-35% / +45%",
    "vitamin": "-35% / +50%",
    "carb": "+/-2 g / +/-20% / +/-8 g",
    "fat": "+/-1.5 g / +/-20% / +/-8 g",
    "saturates": "+/-0.8 g / +/-20%",
    "sodium": "+/-150 mg / +/-20%",
    "salt": "+/-375 mg / +/-20%",
}

# Table 4 negligible amounts, used only by the superseded reference basis. On
# the declared basis the zero case needs no special rule: rounding already maps
# a small prediction to a declared 0 and the lowest tier is absolute.
EU_NEGLIGIBLE: dict[str, float] = {
    "carb": 0.5, "fat": 0.5, "saturates": 0.1, "sodium": 5.0, "salt": 12.5,
}

EU_BASES = ("declared", "reference")


def eu_row(target: str | None) -> str | None:
    """Which Table 1 row ``target`` falls in, or None if the table has no row."""
    return EU_ROW.get(target) if target is not None else None


def has_eu_row(target: str | None) -> bool:
    """False for Energy and Cholesterol, which Table 1 does not cover."""
    return eu_row(target) is not None


def eu_halfwidth(amount: np.ndarray, row: str) -> np.ndarray:
    """Tolerance half-width for a tiered row, in the nutrient's own units.

    Tiers are selected on ``|amount|`` so a boundary means the same thing on
    either side of zero. ``amount`` is the declared value on the declared basis
    and ``y_true`` on the reference basis -- that choice is the whole
    difference between the two readings on these rows.
    """
    yt = np.abs(np.asarray(amount, dtype=float))
    out = np.full(yt.shape, np.nan)
    lower = 0.0
    for upper, kind, value in EU_TIERS[row]:
        sel = (yt >= lower) & (yt < upper) if np.isfinite(upper) else (yt >= lower)
        out[sel] = value if kind == "abs" else value * yt[sel]
        lower = upper
    return out


def declared_value(p: np.ndarray, target: str) -> np.ndarray:
    """The prediction as it would be printed on a label (EU Table 4 rounding).

    Targets Table 4 does not cover are returned unrounded so the declared-basis
    band can still be evaluated on them.
    """
    if not has_rounding_row(target):
        return np.asarray(p, dtype=float)
    return np.asarray(round_declared(p, target), dtype=float)


def _band_about(centre: np.ndarray, other: np.ndarray,
                band: tuple[float, float]) -> np.ndarray:
    """``other`` within a relative ``band`` centred on ``centre``.

    A centre of 0 carries no relative tolerance -- it asserts absence -- so it
    is met only when ``other`` is ~0.
    """
    c = np.asarray(centre, float)
    o = np.asarray(other, float)
    ok = np.isfinite(c) & np.isfinite(o)
    out = np.zeros(c.shape, dtype=bool)
    lo, hi = band
    nz = ok & (c != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (o[nz] - c[nz]) / c[nz]
    out[nz] = (rel >= lo) & (rel <= hi)
    z = ok & (c == 0)
    out[z] = np.abs(o[z]) <= 1e-8
    return out


def eu_mask(y: np.ndarray, p: np.ndarray, target: str,
            basis: str = "declared") -> np.ndarray:
    """Pass mask under EU Table 1 for ``target``.

    basis="declared" (canonical, the reported definition)
        The prediction is what would be printed, so it is rounded to label
        precision by Table 4 and becomes the declared value ``d``; ``y_true``
        is the analytical result checked against that declaration. The band is
        centred on ``d``. This is the direction the regulation actually
        governs.

    basis="reference" (superseded)
        The band is centred on ``y_true``. It hands the +45% headroom to the
        wrong side and is kept only for comparison; do not report it.

    Raises for a target Table 1 does not cover -- call has_eu_row() first.
    """
    if basis not in EU_BASES:
        raise ValueError(f"basis must be one of {EU_BASES}, got {basis!r}")
    row = eu_row(target)
    if row is None:
        raise ValueError(
            f"EU Table 1 has no tolerance row for {target!r}; "
            "check has_eu_row() before scoring."
        )
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    band = MINERAL_BAND if row == "mineral" else VITAMIN_BAND

    if basis == "declared":
        d = declared_value(p, target)
        if row in ("mineral", "vitamin"):
            return _band_about(d, y, band)
        ok = np.isfinite(y) & np.isfinite(d)
        out = np.zeros(d.shape, dtype=bool)
        # The tier is read off the declared amount, and the lowest tier is
        # absolute, so a declared 0 needs no separate convention.
        out[ok] = np.abs(y[ok] - d[ok]) <= eu_halfwidth(d[ok], row)
        return out

    if row in ("mineral", "vitamin"):
        return _band_about(y, p, band)
    ok = np.isfinite(y) & np.isfinite(p)
    out = np.zeros(y.shape, dtype=bool)
    out[ok] = np.abs(p[ok] - y[ok]) <= eu_halfwidth(y[ok], row)
    zero = ok & (y == 0)
    if zero.any():
        out[zero] = np.abs(p[zero]) <= EU_NEGLIGIBLE[row]
    return out


# 21 CFR 101.9(g) gives two different compliance criteria, and which one applies
# depends on the nutrient, not on the direction of the error:
#
#   (g)(4)(ii) Class II -- naturally occurring vitamins, minerals, protein, total
#              carbohydrate, dietary fiber, other carbohydrate, unsaturated fat
#              and potassium: the composite must be at least 80% of the declared
#              value. A FLOOR; over-delivery is not a violation.
#   (g)(5)     calories, total sugars, added sugars, total fat, saturated fat,
#              trans fat, cholesterol and sodium: the composite must not be more
#              than 20% in excess of the declared value. A CEILING; under-
#              delivery is not a violation.
#
# Class I (added nutrients, >=100%) is not used: nothing in the branded data
# distinguishes an added nutrient from an indigenous one, so every floor
# nutrient is scored under the more lenient Class II criterion.
FDA_CEILING = frozenset({
    "Energy(kcal)", "Total fat(g)", "Saturated fatty acids(g)",
    "Cholesterol(mg)", "Sodium(mg)", "Total sugar(g)",
})
FDA_FLOOR = frozenset({
    "Carbohydrate(g)", "Protein(g)", "Fiber(g)", "Calcium(mg)", "Iron(mg)",
})

# 21 CFR 101.9(c): an amount LESS THAN these may be declared as zero on a US
# label. These are PER SERVING in the nutrient's own units -- the one place the
# per-serving basis matters, since the 80%/120% criteria are ratios and so are
# scale-invariant. Fibre is (c)(6)(i), 0.5 g. Calcium and iron come from
# (c)(8)(iii): under 2% of the Daily Value (1300 mg, 18 mg), i.e. 26 mg and
# 0.36 mg. These are US FDA values; W20-Z (w20_v2_mask) uses the same table.
FDA_ZERO_PER_SERVING = {
    "Energy(kcal)": 5.0,
    "Total fat(g)": 0.5,
    "Saturated fatty acids(g)": 0.5,
    "Cholesterol(mg)": 2.0,
    "Sodium(mg)": 5.0,
    "Carbohydrate(g)": 0.5,
    "Total sugar(g)": 0.5,
    "Fiber(g)": 0.5,
    "Protein(g)": 0.5,
    "Calcium(mg)": 26.0,
    "Iron(mg)": 0.36,
}


def fda_mask(
    y: np.ndarray,
    p: np.ndarray,
    target: str,
    serving_size: np.ndarray | None = None,
    serving_size_unit: np.ndarray | None = None,
) -> np.ndarray:
    """21 CFR 101.9 compliance, per nutrient.

    Declared non-zero: the (g)(4)(ii) floor for FDA_FLOOR nutrients, the (g)(5)
    ceiling for FDA_CEILING nutrients.

    Declared zero: the label printed "0" because the true amount was under the
    (c) threshold, so the estimate is compared the same way -- it passes when it
    would also print "0" on that food's label. Our values are per 100 g and the
    thresholds are per serving, so the estimate is converted with the food's own
    serving size; where the serving unit is not a resolvable mass or volume the
    row falls back to an exact-zero test rather than guessing.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if target in FDA_CEILING:
        hit = p <= 1.2 * y
    elif target in FDA_FLOOR:
        hit = p >= 0.8 * y
    else:
        raise ValueError(f"21 CFR 101.9 gives no compliance criterion for {target!r}")

    z = y == 0
    if not z.any():
        return hit
    # A declared zero carries no ratio information: 1.2*0 and 0.8*0 are both 0,
    # which would pass every non-negative estimate under the ceiling and fail
    # every positive one under the floor. Score it on the rounding rule instead.
    thr = FDA_ZERO_PER_SERVING.get(target)
    hit = hit.copy()
    hit[z] = np.abs(p[z]) <= 1e-8
    if thr is None or serving_size is None or serving_size_unit is None:
        return hit
    unit = pd.Series(serving_size_unit).astype(str).str.strip().str.lower().to_numpy()
    ss = pd.to_numeric(pd.Series(serving_size), errors="coerce").to_numpy(dtype=float)
    resolvable = np.isin(unit, list(RESOLVABLE_SERVING_UNITS)) & np.isfinite(ss) & (ss > 0)
    per_serving = np.abs(p) * ss / 100.0
    hit[z] = hit[z] | (resolvable[z] & (per_serving[z] < thr))
    return hit


def all_masks(
    y: np.ndarray,
    p: np.ndarray,
    target: str,
    serving_size: np.ndarray | None = None,
    serving_size_unit: np.ndarray | None = None,
    eu_basis: str = "declared",
) -> dict[str, np.ndarray]:
    """Per-row boolean masks for every rate metric (everything except R2).

    FDA_nonzero is FDA restricted to rows declaring the nutrient non-zero; it
    is NaN-free but its denominator differs, so callers must score it on its
    own subset rather than averaging it over all rows.

    ``eu_basis`` selects the EU reading; see eu_mask(). The default is the
    canonical declared-value basis.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {
        "W20": w20_mask(y, p),
        "W20_v2": w20_v2_mask(y, p, target, serving_size, serving_size_unit),
        "EU": eu_mask(y, p, target, basis=eu_basis),
        "FDA": fda_mask(y, p, target, serving_size, serving_size_unit),
    }
