#!/usr/bin/env python3
"""Build the ingredient feature blocks a model input needs.

One CLI over ``nutpred.preprocess`` and ``nutpred.prepare``, covering the three
ingredient-derived feature groups the reported model uses:

  --binary   binary_<ingredient>  presence flag, one per vocabulary entry
  --score    score_<ingredient>   positional rank, max_score - position
  --embed    umap_<d>             ingredient-text embedding reduced to d dims
  --all      all three (the reported configuration)

At least one must be given. ``--binary`` and ``--score`` share the same
thesaurus mapping and vocabulary, so asking for both costs no more than one.

``--embed`` calls the OpenAI embeddings API. Only DISTINCT ingredient strings
are sent, and results are cached to ``data/embedding_cache``, so a re-run is
free; it still needs ``OPENAI_API_KEY`` (or ``--api-key``) the first time.

A NOTE ON THE SPLIT. ``--embed`` fits UMAP over every row it is given. The
reported pipeline fits on train rows only and transforms test through that
same reducer. Reducing a frame that still contains test rows, and splitting
afterwards, leaks test structure into the representation -- so build the
embedding on your train rows and project test through the fitted reducer,
rather than running this over a combined frame. ``--vocab-scope train`` does
the equivalent for the binary/score vocabulary.

Run it as a module::

    python -m nutpred.build_ingredient_features --all --out-csv ...

Examples
--------
The reported configuration, top-100 vocabulary ranked on train rows,
embeddings reduced to 20 dimensions::

    python -m nutpred.build_ingredient_features --all \\
      --in-csv data/example/food_df.example.csv \\
      --thesaurus-xlsx data/example/thesaurus.example.xlsx \\
      --top-k 100 --vocab-scope train \\
      --out-csv data/model_input_features.csv

Ingredient blocks only, vocabulary taken from the ingnut reference table::

    python -m nutpred.build_ingredient_features --binary --score \\
      --vocab-csv data/example/ingnut_df.example.csv \\
      --out-csv data/model_input_binscore.csv

The embedding dimensionality ablation::

    python -m nutpred.build_ingredient_features --embed \\
      --embed-dims 2,10,20 --out-csv data/model_input_umap.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from nutpred.prepare import DEFAULT_MODEL, build_embedding_columns
from nutpred.preprocess import build_ingredient_feature_blocks

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_dims(raw: str) -> list[int]:
    dims = [int(x) for x in str(raw).replace(" ", "").split(",") if x]
    if not dims or any(d < 1 for d in dims):
        raise SystemExit(f"--embed-dims must be positive integers, got {raw!r}")
    return dims


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    sel = ap.add_argument_group("feature groups (choose at least one)")
    sel.add_argument("--all", action="store_true",
                     help="binary + score + embed, the reported configuration")
    sel.add_argument("--binary", action="store_true",
                     help="binary_<ingredient> presence flags")
    sel.add_argument("--score", action="store_true",
                     help="score_<ingredient> positional ranks")
    sel.add_argument("--embed", action="store_true",
                     help="umap_<d> ingredient-embedding columns")

    io = ap.add_argument_group("input / output")
    io.add_argument("--in-csv", type=Path,
                    default=ROOT / "data/example/food_df.example.csv")
    io.add_argument("--out-csv", type=Path, required=True)
    io.add_argument("--out-vocab-csv", type=Path, default=None,
                    help="also write the ranked vocabulary with its counts")
    io.add_argument("--thesaurus-xlsx", type=Path,
                    default=ROOT / "data/example/thesaurus.example.xlsx",
                    help="ingredient thesaurus workbook (see data/README.md)")

    voc = ap.add_argument_group("vocabulary (--binary / --score)")
    voc.add_argument("--top-k", type=int, default=100)
    voc.add_argument("--vocab-scope", choices=["all", "train"], default="all",
                     help="rank over all rows, or train rows only (no leakage)")
    voc.add_argument("--vocab-csv", type=Path, default=None,
                     help="take the vocabulary from this table's 'ing' column "
                          "instead of ranking by corpus frequency")
    voc.add_argument("--split-col", default="data_type",
                     help="train/test marker, for --vocab-scope train")
    voc.add_argument("--ingredient-cap", type=int, default=20,
                     help="keep only the first N parsed ingredients; 0 disables")
    voc.add_argument("--max-score", type=int, default=20,
                     help="score = max_score - position within the filtered list")
    voc.add_argument("--no-exact-k", action="store_true",
                     help="do not pad the binary block to the full vocabulary width")

    emb = ap.add_argument_group("embedding (--embed)")
    emb.add_argument("--embed-dims", type=str, default="20",
                     help="UMAP dimensionalities, comma-separated (default 20, "
                          "the reported setting)")
    emb.add_argument("--embed-model", type=str, default=DEFAULT_MODEL)
    emb.add_argument("--api-key", type=str, default=None,
                     help="falls back to the OPENAI_API_KEY environment variable")
    emb.add_argument("--ingredient-col", type=str, default="ingredients",
                     help="column holding the raw ingredient text")
    emb.add_argument("--random-state", type=int, default=42)

    args = ap.parse_args()

    want_binary = args.all or args.binary
    want_score = args.all or args.score
    want_embed = args.all or args.embed
    if not (want_binary or want_score or want_embed):
        ap.error("choose at least one of --all / --binary / --score / --embed")

    cap = args.ingredient_cap if args.ingredient_cap and args.ingredient_cap > 0 else None

    log(f"Loading {args.in_csv} ...")
    df = pd.read_csv(args.in_csv, low_memory=False)
    log(f"  {df.shape[0]} rows x {df.shape[1]} cols")
    log(f"Requested: {', '.join(n for n, w in (('binary', want_binary), ('score', want_score), ('embed', want_embed)) if w)}")

    out = df

    # --- binary / score -----------------------------------------------------
    if want_binary or want_score:
        thesaurus = pd.read_excel(args.thesaurus_xlsx)
        log(f"Thesaurus: {len(thesaurus)} rows")

        top_list = None
        if args.vocab_csv is not None:
            vocab_df = pd.read_csv(args.vocab_csv)
            if "ing" not in vocab_df.columns:
                raise SystemExit(f"{args.vocab_csv} has no 'ing' column")
            top_list = vocab_df["ing"].astype(str).tolist()
            log(f"  vocabulary supplied: {len(top_list)} entries from {args.vocab_csv}")

        out, binary_df, score_df, info = build_ingredient_feature_blocks(
            out, thesaurus,
            top_k=args.top_k,
            top_list=top_list,
            vocab_scope=args.vocab_scope,
            split_col=args.split_col,
            ingredient_cap=cap,
            max_score=args.max_score,
            exact_k=not args.no_exact_k,
        )

        m = info["thesaurus_match"]
        log(f"Thesaurus match: {m['rows_with_mapped']}/{m['n_rows']} rows "
            f"({m['row_match_pct']:.1f}%) have >=1 mapped token; "
            f"{m['tokens_mapped']}/{m['tokens_parsed']} tokens matched "
            f"({m['token_match_pct']:.1f}%)")
        if "n_train_rows" in info:
            log(f"Vocabulary ranked from {info['n_train_rows']} train rows only")
        if "overlap_with_all_rows" in info:
            log(f"Leakage audit: train-only top-{info['k']} vocabulary overlaps "
                f"{info['overlap_with_all_rows']}/{info['k']} with an all-rows one")
        if info.get("binary_cols_filled"):
            log(f"  padded {info['binary_cols_filled']} never-occurring binary cols to zero "
                f"(e.g. {info['never_occurring'][:6]})")

        blocks = [out.reset_index(drop=True)]
        if want_binary:
            blocks.append(binary_df.reset_index(drop=True))
            log(f"  binary: {info['binary_cols']} cols "
                f"({info['constant_zero_binary']} constant-zero)")
        if want_score:
            blocks.append(score_df.reset_index(drop=True))
            log(f"  score:  {info['score_cols']} cols "
                f"({info['constant_zero_score']} constant-zero)")
        out = pd.concat(blocks, axis=1)

        if args.out_vocab_csv is not None and "vocab_table" in info:
            args.out_vocab_csv.parent.mkdir(parents=True, exist_ok=True)
            info["vocab_table"].to_csv(args.out_vocab_csv, index=False)
            log(f"Wrote {args.out_vocab_csv} ({len(info['vocab_table'])} rows)")

    # --- embeddings ---------------------------------------------------------
    if want_embed:
        dims = _parse_dims(args.embed_dims)
        log(f"Embedding with {args.embed_model}, reducing to {dims} ...")
        try:
            out, einfo = build_embedding_columns(
                out,
                dims=dims,
                ingredient_col=args.ingredient_col,
                model=args.embed_model,
                api_key=args.api_key,
                random_state=args.random_state,
            )
        except ValueError as exc:
            if "API key" not in str(exc):
                raise
            raise SystemExit(
                f"{exc}\n"
                "--embed needs a key only when the embeddings are not already "
                "cached in data/embedding_cache. Set OPENAI_API_KEY, pass "
                "--api-key, or drop --embed to build the ingredient blocks alone."
            ) from None
        log(f"  {einfo['n_distinct_texts']} distinct strings -> "
            f"{einfo['embedding_dim']}-d embeddings -> "
            f"{', '.join('umap_%d' % d for d in einfo['umap_dims'])}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    log(f"Wrote {args.out_csv} ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
