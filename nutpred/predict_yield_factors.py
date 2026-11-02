#!/usr/bin/env python3
"""CLI: per-ingredient processing yield factors for ``--method yf_opt``.

``build_knowledge_guided_features --method yf_opt`` needs ``--yf-column``: a
column holding, for each product, one yield factor per entry of its mapped
ingredient list. This module writes that column.

A yield factor is the mass an ingredient retains in the finished product divided
by the mass that went in, so 0 < YF <= 1. They are not in any public table at
ingredient level, so the reported analysis estimated them with an LLM from the
product's description, brand, category, declared panel and ordered ingredient
list. The prompts below are the ones that produced the reported values.

**The three target nutrients are never sent.** Only the eight base panel
nutrients go into the prompt; calcium, dietary fiber and iron would leak the
quantity the model is meant to predict. The code asserts this rather than
trusting the caller.

Values are clipped into ``(--min-yield-factor, --max-yield-factor]``. The lower
bound matters: the prompt asks for 0 < YF <= 1, but a model can still return 0,
and a zero yield factor divides the reference profile by the clip floor instead,
inflating that ingredient by six orders of magnitude. The reported run clipped
only the upper bound and carries 78 such zeros; the default floor here rejects
them at the source.

Responses are cached per product under ``--cache-dir``, so a re-run costs
nothing and an interrupted run resumes.

    python -m nutpred.predict_yield_factors \
        --food-df data/example/food_df.example.csv \
        --thesaurus-df data/example/thesaurus.example.xlsx \
        --api-key-file ~/openai_key.txt \
        --out-csv food_df_yf.csv

    python -m nutpred.build_knowledge_guided_features \
        --food-df food_df_yf.csv --method yf_opt --yf-column pred_yield_factors \
        --thesaurus-df ... --ingnut-df ... --out-csv food_df_kg.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nutpred import constants as C

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4o"
DEFAULT_CACHE_DIR = ROOT / "data/yf_cache"

SYSTEM_PROMPT = """You are a food science expert specializing in food processing and yield factor prediction. Your task is to predict yield factors for ingredients in processed food products.

YIELD FACTOR DEFINITION:
- Yield factor (YF) = (ingredient weight in final product) / (original weight in raw ingredients)
- YF < 1.0: Weight loss during processing (cooking, evaporation, degradation)
- YF = 1.0: No weight change (already processed ingredients)

CRITICAL RULES FOR YF CALCULATION:
1. YIELD VALUES MUST NEVER EXCEED 1.0 (YF <= 1.0)
2. INGREDIENT STATE INTERPRETATION:
   - If ingredient name has NO state descriptors (like "DRIED", "FLOUR", "POWDER"), assume RAW/FRESH state
   - Examples: "CORN" -> raw corn, "WHEAT FLOUR" -> dried flour state

FOOD PROCESSING CONTEXT:
- Deep frying: Oil absorption increases fat content, but YF for oil <= 1.0
- Baking/cooking: Water loss, nutrient degradation (YF < 1.0)
- Drying: Concentration of nutrients (YF <= 1.0)
- Raw ingredients: Significant loss during processing (YF < 1.0)
- Processed ingredients (flour, oil, salt): Minimal change (YF ~ 1.0)
"""


def build_prompt(description: str, panel: Dict[str, Any], ingredients: List[str],
                 max_yf: float, brand_name: str = "", brand_owner: str = "",
                 category: str = "") -> str:
    """The user prompt. ``panel`` must hold base-panel nutrients only."""
    leaked = [k for k in panel if k in set(C.TARGET_NUT3)]
    if leaked:
        raise ValueError(f"target nutrients must never be sent to the model: {leaked}")
    target_str = "\n".join(f"- {k}: {v}" for k, v in panel.items())
    ing_str = "\n".join(f"  {i + 1}. {ing}" for i, ing in enumerate(ingredients))
    return f"""Analyze this snack product and assign one yield factor to each ingredient.

PRODUCT: FOOD PRODUCT: {description}
BRAND: BRAND: {brand_name}
BRAND OWNER: BRAND OWNER: {brand_owner}
CATEGORY: {category}

TARGET NUTRITION (final product per 100g):
{target_str}

INGREDIENTS (in order of predominance):
{ing_str}

TASK:
1. Infer the most likely processing method.
2. Assign one yield factor (YF) per ingredient.

IMPORTANT DOMAIN ANCHORS:
- Potato chips / kettle chips / root-vegetable chips:
  raw potato or similar raw slice ingredients usually end up around YF 0.20-0.35 after frying because of major water loss.
- Tortilla / corn chips:
  main corn base is usually lower than 1.0; often around 0.55-0.85 depending on whether it behaves like hydrated masa versus a drier processed corn base.
- Pretzels / crackers / baked flour snacks:
  flour/starch bases are usually much higher, often around 0.90-0.98.
- Oils, salts, sugars, acids, flavors, dried seasonings:
  usually near 0.95-1.0.

RAW VS PROCESSED:
- Processed/dry ingredients (FLOUR, POWDER, STARCH, MEAL, OIL, SALT, SUGAR, DEXTROSE, DRIED) usually retain mass.
- Raw/high-moisture ingredients lose much more mass during frying, baking, roasting, or drying.

CRITICAL RULES:
- Yield factors are (final weight) / (raw weight).
- 0 < YF <= {min(max_yf, 1.0):.2f}.
- Do not assign implausibly high YF to raw fried potatoes or similar raw chip bases.
- Return short output only. Do not include long reasoning per ingredient.

Respond ONLY with valid JSON in this format:
{{
  "analysis": "1 short sentence",
  "processing_method": "deep_frying/kettle_cooking/air_frying/baking/extrusion/roasting/popping/drying/fermentation",
  "confidence": "High/Medium/Low",
  "yield_factors": {{
    "POTATO": 0.25,
    "OIL, SUNFLOWER": 0.98,
    "SODIUM CHLORIDE": 1.0
  }}
}}"""


def _json_block(text: str) -> str:
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


def _resolve_key(api_key_file: Optional[Path]) -> str:
    if api_key_file:
        key = Path(api_key_file).expanduser().read_text().strip().splitlines()[0].strip()
        if key:
            return key
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "no API key: pass --api-key-file <file> or set OPENAI_API_KEY. "
            "No key is distributed with this repository."
        )
    return key


def clamp(raw: Any, lo: float, hi: float) -> Optional[float]:
    """Coerce one model-returned value into (lo, hi]; None when unusable."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or v <= 0:        # NaN, zero or negative: the prompt forbids these
        return None
    return float(min(max(v, lo), hi))


def predict_for_row(client, description: str, panel: Dict[str, Any],
                    ingredients: List[str], *, model: str, min_yf: float,
                    max_yf: float, brand_name: str = "", brand_owner: str = "",
                    category: str = "", retries: int = 3) -> Dict[str, Any]:
    """One product. Returns {'yield_factors': {ing: yf}, 'evidence', 'status'}."""
    prompt = build_prompt(description, panel, ingredients, max_yf,
                          brand_name, brand_owner, category)
    last = ""
    for attempt in range(1, retries + 1):
        try:
            rsp = client.chat.completions.create(
                model=model, temperature=0.3,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}])
            payload = json.loads(_json_block(rsp.choices[0].message.content or ""))
            out: Dict[str, float] = {}
            for name, val in (payload.get("yield_factors") or {}).items():
                if isinstance(val, dict):
                    val = val.get("yield_factor")
                v = clamp(val, min_yf, max_yf)
                if v is not None:
                    out[str(name).strip().upper()] = round(v, 3)
            return {"yield_factors": out, "evidence": payload,
                    "status": "success", "attempts": attempt}
        except Exception as exc:                      # noqa: BLE001 - logged, then retried
            last = str(exc)
            logger.warning("row failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return {"yield_factors": {}, "evidence": {"error": last},
            "status": "api_error", "attempts": retries}


def align(tokens: List[str], predicted: Dict[str, float], default: float) -> List[float]:
    """One yield factor per mapped token, in the order the optimizer reads them.

    The model keys its answer by ingredient name, so a name it renames, merges or
    drops must not shift the rest of the list: an unmatched token takes
    ``default`` in place rather than being skipped.
    """
    out = []
    for t in tokens:
        key = str(t).strip().upper()
        v = predicted.get(key)
        if v is None:                                  # tolerate light renaming
            for k, kv in predicted.items():
                if k.startswith(key) or key.startswith(k):
                    v = kv
                    break
        out.append(float(v) if v is not None else float(default))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--food-df", type=Path,
                    default=ROOT / "data/example/food_df.example.csv")
    ap.add_argument("--thesaurus-df", type=Path,
                    default=ROOT / "data/example/thesaurus.example.xlsx")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--yf-col", default="pred_yield_factors",
                    help="name of the column written (pass it to --yf-column)")
    ap.add_argument("--api-key-file", type=Path, default=None,
                    help="file whose first line is your OpenAI key; falls back to "
                         "OPENAI_API_KEY. Needed only on a cache miss.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--min-yield-factor", type=float, default=0.05,
                    help="reject values at or below this; the reported run had no "
                         "lower bound and carries 78 zeros (default 0.05)")
    ap.add_argument("--max-yield-factor", type=float, default=1.0)
    ap.add_argument("--ingredient-cap", type=int, default=20,
                    help="map only the first N parsed ingredients, as the reported "
                         "setting does; 0 disables")
    ap.add_argument("--unmatched-yf", type=float, default=1.0,
                    help="value for a mapped token the model did not return")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N products, for a smoke test")
    ap.add_argument("--evidence-json", type=Path, default=None,
                    help="also write the models' per-product rationale here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nutpred.preprocess import ensure_mapped_list_column

    food = pd.read_csv(args.food_df, low_memory=False)
    if args.limit:
        food = food.head(args.limit).copy()
    thes = pd.read_excel(args.thesaurus_df)
    cap = args.ingredient_cap if args.ingredient_cap and args.ingredient_cap > 0 else None
    food = ensure_mapped_list_column(food, thes, ingredient_cap=cap)
    logger.info("%d products, thesaurus %d rows", len(food), len(thes))

    panel_cols = [c for c in C.NUT8 if c in food.columns]
    missing = [c for c in C.NUT8 if c not in food.columns]
    if missing:
        logger.warning("base panel columns absent, not sent: %s", missing)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    client = None
    yf_lists: List[List[float]] = []
    evidence: List[Dict[str, Any]] = []
    stats = {"cached": 0, "called": 0, "failed": 0, "no_ingredients": 0, "clamped": 0}

    for _, row in food.iterrows():
        tokens = row.get("mapped_list", [])
        if not isinstance(tokens, list) or not tokens:
            yf_lists.append([])
            stats["no_ingredients"] += 1
            continue
        panel = {c: row[c] for c in panel_cols}
        key = hashlib.sha256(json.dumps(
            [str(row.get("description", "")), str(row.get("brand_name", "")),
             str(row.get("brand_owner", "")), str(row.get("branded_food_category", "")),
             [str(t) for t in tokens], {k: str(v) for k, v in panel.items()},
             args.model, args.max_yield_factor],
            sort_keys=True).encode()).hexdigest()[:40]
        cache_file = args.cache_dir / f"{key}.json"
        if cache_file.is_file():
            res = json.loads(cache_file.read_text())
            stats["cached"] += 1
        else:
            if client is None:
                from openai import OpenAI
                client = OpenAI(api_key=_resolve_key(args.api_key_file))
            res = predict_for_row(
                client, str(row.get("description", "")), panel, [str(t) for t in tokens],
                model=args.model, min_yf=args.min_yield_factor,
                max_yf=args.max_yield_factor,
                brand_name=str(row.get("brand_name", "")),
                brand_owner=str(row.get("brand_owner", "")),
                category=str(row.get("branded_food_category", "")))
            cache_file.write_text(json.dumps(res))
            stats["called"] += 1
            if res["status"] != "success":
                stats["failed"] += 1
        vals = align([str(t) for t in tokens], res["yield_factors"], args.unmatched_yf)
        stats["clamped"] += sum(1 for t in tokens
                                if str(t).strip().upper() not in res["yield_factors"])
        yf_lists.append([round(v, 3) for v in vals])
        evidence.append({"description": row.get("description", ""),
                         **{k: v for k, v in res.get("evidence", {}).items()
                            if k != "yield_factors"}})

    food[args.yf_col] = [str(v) for v in yf_lists]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    food.to_csv(args.out_csv, index=False)
    if args.evidence_json:
        args.evidence_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))

    flat = [v for lst in yf_lists for v in lst]
    logger.info("cached %d | called %d | failed %d | products with no mapped "
                "ingredient %d | tokens falling back to --unmatched-yf %d",
                stats["cached"], stats["called"], stats["failed"],
                stats["no_ingredients"], stats["clamped"])
    if flat:
        s = pd.Series(flat)
        logger.info("yield factors: n=%d min=%.3f median=%.3f max=%.3f | "
                    "at or below 0: %d (must be 0)",
                    len(s), s.min(), s.median(), s.max(), int((s <= 0).sum()))
    logger.info("wrote %s (column %r -> pass as --yf-column)", args.out_csv, args.yf_col)


if __name__ == "__main__":
    main()
