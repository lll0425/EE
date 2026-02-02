import re

def parse_distance_cm(token: str) -> float:
    """
    Robustly parse things like '1cm', '1.0cm', '0.76cm', '  1.12CM ' -> float centimeters.
    Raises ValueError if it can't parse.
    """
    s = token.strip().lower()
    # remove trailing 'cm' if present
    if s.endswith('cm'):
        s = s[:-2]
    # keep only digits and a single dot (in case of odd naming)
    m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s*$', s)
    if not m:
        raise ValueError(f"Cannot parse distance from token '{token}'")
    return float(m.group(1))

def build_dom_to_idx(distance_tokens):
    """
    distance_tokens: iterable of folder-name strings like ['0cm','0.76cm','1cm','1.12cm','1.2cm', ...]
    Returns a dict {token: idx}, sorted by numeric centimeters; ties break by token name for stability.
    """
    uniq = sorted(set(distance_tokens),
                  key=lambda t: (parse_distance_cm(t), t.lower()))
    return {tok: i for i, tok in enumerate(uniq)}