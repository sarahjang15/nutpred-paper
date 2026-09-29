#!/usr/bin/env python3
"""Render docs/architecture.png -- how the package fits together.

    python docs/make_architecture_diagram.py

Kept in the repo so the picture can be regenerated when the code moves,
rather than drifting out of date as a binary nobody can edit.

Boxes size themselves to their content: heights come from the number of body
lines, so editing the text cannot silently overflow a box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

# Canvas is in points-like units so spacing is easy to reason about.
W, H = 1000.0, 660.0

INK = "#1b1e21"
MUTED = "#59636e"
FAINT = "#8b949d"

PALETTE = {
    "input": ("#e7eef7", "#6f92bb"),
    "build": ("#e8f1e8", "#6e9f6e"),
    "core": ("#f2ebf7", "#8f74ac"),
    "out": ("#faf0e3", "#c39a58"),
    "grey": ("#f5f6f7", "#a8b0b8"),
}

TITLE_H = 19.0          # space the bold title occupies
LINE_H = 13.2           # one body line
PAD_X = 11.0
PAD_TOP = 11.0
PAD_BOT = 9.0


def box_height(lines: list[str], has_title: bool = True) -> float:
    return PAD_TOP + (TITLE_H if has_title else 0.0) + len(lines) * LINE_H + PAD_BOT


def box(ax, x, y, w, title, lines, kind, title_size=10.0, body_size=8.1, lw=1.3):
    """Draw a box whose TOP-LEFT is (x, y). Returns (height, bottom_y)."""
    face, edge = PALETTE[kind]
    h = box_height(lines, bool(title))
    ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                                boxstyle="round,pad=0,rounding_size=7",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    cy = y - PAD_TOP
    if title:
        ax.text(x + PAD_X, cy, title, fontsize=title_size, fontweight="bold",
                color=INK, va="top", ha="left", zorder=3)
        cy -= TITLE_H
    for ln in lines:
        ax.text(x + PAD_X, cy, ln, fontsize=body_size, color=MUTED,
                va="top", ha="left", family="monospace", zorder=3)
        cy -= LINE_H
    return h, y - h


def label(ax, x, y, text, color, size=8.3):
    ax.text(x, y, text, fontsize=size, fontweight="bold", color=color,
            ha="left", va="bottom")


def arrow(ax, p, q, color, lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color, zorder=5, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.9, 9.2), dpi=190)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(W / 2, H - 14, "nutpred — code architecture", fontsize=17,
            fontweight="bold", color=INK, ha="center", va="top")
    ax.text(W / 2, H - 40,
            "Predicting Calcium / Fiber / Iron in packaged foods from the "
            "macronutrient panel and the ingredient list",
            fontsize=9.6, color=MUTED, ha="center", va="top")

    # ── row 1 ────────────────────────────────────────────────────────────
    y1 = H - 84.0
    col1_x, col1_w = 18.0, 252.0
    col2_x, col2_w = 298.0, 414.0
    col3_x, col3_w = 740.0, 242.0

    label(ax, col1_x, y1 + 6, "YOU SUPPLY", PALETTE["input"][1])
    label(ax, col2_x, y1 + 6, "STEP 1 — OPTIONAL PREPROCESSING (two CLIs)", PALETTE["build"][1])
    label(ax, col3_x, y1 + 6, "SUPPORT MODULES", PALETTE["grey"][1])

    h1, b1 = box(ax, col1_x, y1, col1_w, "Three input files", [
        "--food-df       panel, ingredients",
        "                text, data_type",
        "--thesaurus-df  surface form ->",
        "                descriptor",
        "--ingnut-df     ingredient nutrients",
        "",
        "schema: data/README.md",
    ], "input")

    h2, b2 = box(ax, col2_x, y1, col2_w,
                 "python -m nutpred.build_ingredient_features", [
                     "--all | --binary | --score | --embed",
                     "",
                     "preprocess.build_ingredient_feature_blocks()",
                     "   thesaurus -> top-K vocab -> binary_* score_*",
                     "   --vocab-scope train  ranks on train rows only",
                     "",
                     "prepare.build_embedding_columns()",
                     "   OpenAI embed (cached) -> UMAP -> umap_<d>",
                 ], "build", title_size=10.0, body_size=8.0)

    h2b, b2b = box(ax, col2_x, b2 - 20.0, col2_w,
                   "python -m nutpred.build_knowledge_guided_features", [
                       "--method opt | yf_opt | sum",
                       "constrained ingredient-ratio optimization (CVXPY)",
                       "over reference nutrient profiles, label order and",
                       "predicted yield factors      -> <target>_ing_pred",
                   ], "build", title_size=10.0, body_size=8.0)
    b2 = b2b

    h3, b3 = box(ax, col3_x, y1, col3_w, "nutpred/", [
        "cleaning.py    ingredient text",
        "constants.py   column groups",
        "prepare.py     validity filters,",
        "               embed cache, UMAP",
        "preprocess.py  mapping, vocabulary,",
        "               binary/score blocks",
    ], "grey", title_size=9.8, body_size=7.9)

    # ── row 2: run.py container ──────────────────────────────────────────
    y2 = min(b1, b2, b3) - 46.0
    run_x, run_w = 18.0, 694.0
    inner_w = 330.0
    gap = 14.0
    ix1 = run_x + 16.0
    ix2 = ix1 + inner_w + gap

    label(ax, run_x + 190, y2 + 6, "STEP 2 — PIPELINE", PALETTE["core"][1])

    steps = [
        ("1  preprocess.py", [
            "load_inputs()",
            "ensure_mapped_list_column()",
            "filter_rows()  drop_outliers_nutrients()",
            "ensure_umap_columns()  expands umap_<d>",
        ]),
        ("2  feature sets", [
            "nut8 + the requested blocks",
            "any <target>_ing_pred columns present",
            "are appended to every set and",
            "median-imputed where uncovered",
        ]),
        ("3  predict.py", [
            "train_tree_models()",
            "  Optuna TPE per target x feature set",
            "  --hurdle: zero classifier, then a",
            "  regressor on the non-zero rows",
        ]),
        ("4  metrics.py  /  viz.py", [
            "R2  RMSE  MAE  SMAPE",
            "Within20Pct           non-zero rows only",
            "Within20Pct_inclZero  the paper's W20",
            "heatmaps, scatterplots, SHAP",
        ]),
    ]
    step_h = max(box_height(l) for _, l in steps)

    head_h = 30.0
    inner_top = y2 - head_h
    row_gap = 34.0
    note_lines = 3
    run_h = head_h + step_h * 2 + row_gap + 18.0 + note_lines * 12.4 + 12.0

    ax.add_patch(FancyBboxPatch((run_x, y2 - run_h), run_w, run_h,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#fcfcfd", edgecolor="#3d4753",
                                linewidth=1.8, zorder=1))
    ax.text(run_x + 16, y2 - 9, "run.py", fontsize=12.5, fontweight="bold",
            color=INK, va="top", ha="left", family="monospace")
    ax.text(run_x + 86, y2 - 11,
            "--hurdle  --model-type  --ing-feature  --aux-cols  --n-trials  --shap",
            fontsize=7.9, color=MUTED, va="top", ha="left", family="monospace")

    pos = {}
    for i, (t, l) in enumerate(steps):
        bx = ix1 if i % 2 == 0 else ix2
        by = inner_top - (step_h + row_gap) * (i // 2)
        box(ax, bx, by, inner_w, t, l, "core", title_size=9.7, body_size=7.9)
        pos[i] = (bx, by, by - step_h)

    ny = inner_top - step_h * 2 - row_gap - 16.0
    for k, line in enumerate([
        "metrics.py also holds the reported label-agreement criteria, one definition each:",
        "W20 · W20_v2 · EU (Guidance Table 1, declared-value basis) · FDA 21 CFR 101.9",
        "The hurdle exists because the three targets are heavily zero-inflated on real labels.",
    ]):
        ax.text(run_x + 16, ny - k * 12.4, line, fontsize=7.9,
                color=MUTED if k < 2 else FAINT, va="top", ha="left")

    # ── right column: outputs, then exclusions ───────────────────────────
    out_x, out_w = 740.0, 242.0
    label(ax, out_x, y2 + 6, "OUTPUTS", PALETTE["out"][1])
    ho, bo = box(ax, out_x, y2, out_w, "{test_,complete_}<outdir>/", [
        "config.yaml          run parameters",
        "metrics_<mode>.csv",
        "within20_metrics_<mode>.csv",
        "df_with_preds_<mode>.csv",
        "ingredient_parsing_<mode>.csv",
        "execution_summary_<mode>.txt",
        "plots/  heatmaps, scatterplots,",
        "        shap/  (with --shap)",
    ], "out", title_size=9.7, body_size=7.9)

    hx, bx_ = box(ax, out_x, bo - 30.0, out_w, "Not in this repository", [
        "no food data (synthetic examples",
        "   only -- bring your own)",
        "no figures or stored run outputs",
        "cohort construction from raw FDC",
        "   is documented, not executable:",
        "   docs/data_provenance_methods.md",
    ], "grey", title_size=9.5, body_size=7.8)

    # ── arrows ───────────────────────────────────────────────────────────
    ci, cb, cc, co = (PALETTE["input"][1], PALETTE["build"][1],
                      PALETTE["core"][1], PALETTE["out"][1])
    mid1 = y1 - h1 / 2
    arrow(ax, (col1_x + col1_w, mid1), (col2_x, mid1), ci, lw=1.7)

    # builder -> run.py
    arrow(ax, (col2_x + col2_w / 2, b2), (col2_x + col2_w / 2, y2), cb, lw=1.7)
    ax.text(col2_x + col2_w / 2 + 8, (b2 + y2) / 2 - 2,
            "model input with binary_ / score_ / umap_ / ing_pred columns",
            fontsize=7.8, color=MUTED, va="center", ha="left")

    # inputs -> run.py directly (features already present)
    arrow(ax, (col1_x + 34, b1), (col1_x + 34, y2), ci, lw=1.4, ls=(0, (4, 3)))
    ax.text(col1_x + 44, (b1 + y2) / 2 + 4,
            "or straight in, if those\ncolumns already exist",
            fontsize=7.6, color=FAINT, va="center", ha="left", style="italic")

    # 1 -> 2, 3 -> 4, 2 -> 3
    y_r1 = inner_top - step_h / 2
    y_r2 = inner_top - step_h - row_gap - step_h / 2
    arrow(ax, (ix1 + inner_w, y_r1), (ix2, y_r1), cc, lw=1.5)
    arrow(ax, (ix2 + inner_w * 0.5, inner_top - step_h),
          (ix1 + inner_w * 0.5, inner_top - step_h - row_gap), cc, lw=1.5, rad=0.0)
    arrow(ax, (ix1 + inner_w, y_r2), (ix2, y_r2), cc, lw=1.5)

    # run.py -> outputs
    arrow(ax, (run_x + run_w, y2 - run_h * 0.42), (out_x, y2 - ho * 0.55), co, lw=1.7, rad=-0.1)

    fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.22)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
