import re
import logging
from typing import List

logger = logging.getLogger(__name__)

_TOKEN_NORMALIZE_REMOVE = re.compile(r'\b(?:select|added)\b', flags=re.I)


# ---- Split on top-level commas/periods (ignore those inside () and []) --------
def split_top_level(text: str) -> List[str]:
    parts, buf = [], []
    depth_par = depth_brk = 0
    for c in text:
        if c == '(':
            depth_par += 1; buf.append(c)
        elif c == ')':
            depth_par = max(depth_par - 1, 0); buf.append(c)
        elif c == '[':
            depth_brk += 1; buf.append(c)
        elif c == ']':
            depth_brk = max(depth_brk - 1, 0); buf.append(c)
        elif (c in {',','.'}) and depth_par == 0 and depth_brk == 0:
            part = ''.join(buf).strip()
            if part: parts.append(part)
            buf = []
        else:
            buf.append(c)
    last = ''.join(buf).strip()
    if last: parts.append(last)
    return parts

# ==== Patterns =================================================================

# Global leading "Ingredients:" or "<flavor> Ingredients:" at the very start
_GLOBAL_ING_HEADER = re.compile(r'(?ix)^\s*(?:[\w&/+\- ]+\s+){0,5}?ingredients?\s*:\s*')

# Mid-string "<flavor> Ingredients:" after '.' or ';'
_SECTION_HEADER_ANYWHERE = re.compile(r'(?ix)(?:^|(?<=[.;]))\s*[\w&/+\- ]+\s+ingredients?\s*:\s*')

# Leading "made with ... and " clause (delete entirely)
_MADE_WITH_DELETE = re.compile(r'(?ix)^\s*made\s+with\s+[^,.;()]+?\s+and\s+')

# Token-level prefixes to strip but keep the trailing items
#   e.g., "contains less than 2% of: X", "contains one or more of the following: X", "contains: X"
_CONTAINS_PREFIX = re.compile(
    r'''(?ix)^
        \s*
        (?:may\s+)?contains?              # "contain/contains/may contain"
        (?:\s+\d+%?\s*(?:or\s+less|or\s+more)?)?   # optional "2% or less"
        (?:\s+of)?                         # optional "of"
        (?:\s+(?:one\s+or\s+more\s+)?of\s+the\s+following)?   # optional "of the following"
        \s*:\s*
    '''
)

# Another common header: "and less than 2% of (the following):"
_LESS_THAN_PREFIX = re.compile(
    r'''(?ix)^
        \s*
        (?:and\s+)?less\s+than\s+\d+%?\s+of
        (?:\s+the\s+following)?
        \s*:\s*
    '''
)

# "less than X of each of the following: ..." → keep only the list after the lead-in
_LESS_THAN_EACH_PATTERN = re.compile(
    r'''(?ix)
        (?:and\s+)?less\s+than\s+\d+\s+of\s+each\s+of\s+the\s+following\s*:?\s*
        (.+)
    '''
)

# ==== Helpers ==================================================================

def _strip_nested_parens_and_brackets(s: str) -> str:
    """Remove nested (...) and [...] by iteratively stripping innermost groups."""
    prev = None
    out = s
    while prev != out:
        prev = out
        out = re.sub(r'\([^()]*\)', '', out)
    prev = None
    while prev != out:
        prev = out
        out = re.sub(r'\[[^\[\]]*\]', '', out)
    return out

def _clean_less_than_each_pattern(s: str) -> str:
    m = _LESS_THAN_EACH_PATTERN.match(s)
    if m:
        cleaned = m.group(1).strip()
        logger.info(f"Cleaned 'less than X of each' pattern: '{s}' -> '{cleaned}'")
        return cleaned
    return s


def clean_ingredient_text(text) -> List[str]:
    """
    Turn an INGREDIENTS string into a clean, de-duplicated list (preserve first occurrence).

    Rules:
      • Drop global or mid-string "<flavor> Ingredients:" headers.
      • Drop a leading "made with ... and " clause entirely (do not keep).
      • Keep items after "contains ... (of the following):" by stripping the header only.
      • Collapse nested parentheses/brackets to keep only the head phrase.
      • Canonicalize some common variants and de-duplicate.
    """
    text = str(text).strip()
    if not text:
        return []

    #logger.info(f"Cleaning ingredient text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    text = text.lower()

    # (0) Delete leading "made with ... and " clause if present (company-specific pattern)
    text2 = _MADE_WITH_DELETE.sub('', text)
    if text2 != text:
        logger.info("Removed leading 'made with ... and' clause.")
        text = text2

    # (1) Drop a global "Ingredients:" / "<flavor> Ingredients:" at the very start
    text = _GLOBAL_ING_HEADER.sub('', text)

    # (2) Remove mid-string "<flavor> Ingredients:" headers (e.g., ".Strawberry Ingredients:")
    text = _SECTION_HEADER_ANYWHERE.sub(' ', text)

    # (3) Normalize braces for the splitter
    text = text.replace('{', '(').replace('}', ')')

    # (4) Split by top-level separators
    parts = split_top_level(text)

    cleaned: List[str] = []
    for part in parts:
        ing = part.strip()
        if not ing:
            continue

        # Preserve placeholder tildes like "~~" as separate tokens.
        tilde_compact = ing.replace(" ", "")
        if re.fullmatch(r'~+', tilde_compact):
            cleaned.extend(["~"] * len(tilde_compact))
            continue

        # Keep only the list after "less than X of each of the following:"
        ing = _clean_less_than_each_pattern(ing)

        # Strip token-level headers like "contains ...", "and less than ... of (the following):"
        ing_before = ing
        ing = _CONTAINS_PREFIX.sub('', ing)
        ing = _LESS_THAN_PREFIX.sub('', ing)
        if ing != ing_before and not ing:
            # If stripping left the token empty, skip it.
            continue

        # Remove explicit "(contains ...)" clauses inside tokens
        ing = re.sub(r'\(\s*(?:may\s+)?contains?.*?\)', '', ing, flags=re.I)

        # Remove nested parentheses/brackets entirely (robust against nesting)
        ing = _strip_nested_parens_and_brackets(ing)

        # Leading connectors
        ing = re.sub(r'^(?:and\/or|and or|and|or|andor)\s+', '', ing)

        # Descriptive lead-ins — keep "made with" only if it’s *not* the head clause we removed already
        ing = re.sub(r'^\s*(?:made\s+of|made\s+from)\s*:\s*', '', ing)
        ing = re.sub(r'^\s*(?:includes?|including|containing)\s*:\s*', '', ing)
        ing = re.sub(r'^\s*if[^,;]*', '', ing)

        # Fallback numeric prefix like "2% or less of ..."
        ing = re.sub(r'^\s*\d+%?\s*(?:or\s+less|or\s+more)?\s*(?:of\s*:?)\s*', '', ing)

        # Normalize characters & spaces
        ing = re.sub(r'[^a-z0-9\s\-&/#]', '', ing).strip()
        ing = re.sub(r'\s+', ' ', ing)
        ing = ing.replace('andor', 'and/or')
        ing = _TOKEN_NORMALIZE_REMOVE.sub('', ing)
        ing = re.sub(r'\s+', ' ', ing).strip()

        # Canonicalize some variants
        if ing in ('natural flavors', 'natural flavoring'):
            ing = 'natural flavor'
        if ing == 'thiamine mononitrate':
            ing = 'thiamin mononitrate'

        # Trim trailing connectors like "... and"
        ing = re.sub(r'(?:\s+(?:and|or|and\/or))+$', '', ing)

        if ing:
            cleaned.append(ing)

    # De-duplicate while preserving order
    seen, unique = set(), []
    for ing in cleaned:
        if ing and ing not in seen:
            seen.add(ing)
            unique.append(ing)

    #logger.info(f"Cleaning complete. Input parts={len(parts)}, Output unique ingredients={len(unique)}")
    return unique


__all__ = [clean_ingredient_text]