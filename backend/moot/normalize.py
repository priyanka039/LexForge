# ─────────────────────────────────────────────
# moot/normalize.py
# Post-STT normalization for Indian legal speech.
#
# STT models mangle legal vocabulary in predictable
# ways ("Lord Ship", "right petition", "suo moto",
# "A I R 1978 SC 597"). Sarvam saarika has no
# hotwords parameter, so we repair the transcript
# here BEFORE any agent sees it — citations must be
# normalized or the citation checker flags phantom
# errors and the judge reasons over garbage.
#
# Pure regex. Zero latency. Conservative by design:
# only fixes unambiguous, well-known mistakes.
# ─────────────────────────────────────────────

import re

# (pattern, replacement) — applied case-insensitively, in order.
_PHRASE_FIXES = [
    # Forms of address
    (r"\blord\s*ships?\b",                       "Lordship"),
    (r"\byour\s+lord\s+ship\b",                  "Your Lordship"),
    (r"\bmy\s+lord'?s\b",                        "My Lords"),
    (r"\byour\s+honor\b",                        "Your Honour"),
    (r"\bhon'?ourable\s+court\b",                "Hon'ble Court"),
    (r"\bhonorable\s+court\b",                   "Hon'ble Court"),

    # Latin maxims
    (r"\blocus\s+stand(?:ee|y|i)\b",             "locus standi"),
    (r"\bsuo\s+mot[oa]\b",                       "suo motu"),
    (r"\bres\s+judicat[ae]?\b",                  "res judicata"),
    (r"\brace\s+judicata\b",                     "res judicata"),
    (r"\bprima\s+faci[ea]\b",                    "prima facie"),
    (r"\bobiter\s+dict[ao]\b",                   "obiter dicta"),
    (r"\bratio\s+decident[i]?\b",                "ratio decidendi"),
    (r"\bratio\s+decidendi\b",                   "ratio decidendi"),
    (r"\baudi\s+alter[ae]m\s+part[ae]m\b",       "audi alteram partem"),
    (r"\bultra\s+wires\b",                       "ultra vires"),
    (r"\bamicus\s+curi(?:ae|a|e)\b",             "amicus curiae"),
    (r"\bhabeas\s+corpus\b",                     "habeas corpus"),
    (r"\bhabeus\s+corpus\b",                     "habeas corpus"),
    (r"\bsub\s+judic[ea]\b",                     "sub judice"),

    # Procedure vocabulary
    (r"\brigh?t\s+petition\b",                   "writ petition"),
    (r"\brit\s+petition\b",                      "writ petition"),
    (r"\bwrit\s+of\s+mand[ae]mus\b",             "writ of mandamus"),
    (r"\bcertiora?ri\b",                         "certiorari"),
    (r"\bspecial\s+leave\s+petition\b",          "Special Leave Petition"),
    (r"\bcause\s+of\s+auction\b",                "cause of action"),
    (r"\bmain?tain[ae]bility\b",                 "maintainability"),

    # Statute abbreviations spelled out letter by letter
    (r"\bc\s*r\s*p\s*c\b",                       "CrPC"),
    (r"\bi\s*p\s*c\b",                           "IPC"),
    (r"\bc\s*p\s*c\b",                           "CPC"),

    # Landmark case names
    (r"\bm[ae]n[ae]ka\s+gandh[iy]\b",            "Maneka Gandhi"),
    (r"\bkes[ah]?[av]+ananda\s+bhart?at?h?i\b",  "Kesavananda Bharati"),
    (r"\bkeshavananda\s+bharti\b",               "Kesavananda Bharati"),
    (r"\bputtaswam[iy]\b",                       "Puttaswamy"),
    (r"\bolga\s+tell?is\b",                      "Olga Tellis"),
    (r"\bbachan\s+sing\b",                       "Bachan Singh"),
    (r"\bmach?hi\s+sing\b",                      "Machhi Singh"),
    (r"\bd\s*k\s+basu\b",                        "D.K. Basu"),
    (r"\bvishakha?\b",                           "Vishaka"),
    (r"\bnavtej\s+singh?\s+joh?ar\b",            "Navtej Singh Johar"),
]

# Spelled-out numbers for Articles / Sections counsel commonly speaks.
_NUMBER_WORDS = {
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "twenty one": "21", "twenty-one": "21",
    "twenty two": "22", "twenty-two": "22", "thirty two": "32",
    "thirty-two": "32", "one hundred and thirty six": "136",
    "one thirty six": "136", "two twenty six": "226",
    "two two six": "226", "two hundred and twenty six": "226",
    "one forty two": "142", "one hundred and forty two": "142",
    "three hundred": "300", "four eighty two": "482",
    "four hundred and eighty two": "482",
}

# Reporter fragments dictated letter by letter: "A I R" → "AIR"
_REPORTER_FIXES = [
    (r"\ba\s*\.?\s*i\s*\.?\s*r\b\.?",            "AIR"),
    (r"\bs\s*\.?\s*c\s*\.?\s*c\b\.?",            "SCC"),
    (r"\bs\s*\.?\s*c\s*\.?\s*r\b\.?",            "SCR"),
]


def _fix_reporter_casing(text: str) -> str:
    """air 1978 sc 597 → AIR 1978 SC 597 ; (2017) 10 scc 1 → (2017) 10 SCC 1"""
    text = re.sub(
        r"\bair\s+(\d{4})\s+sc\s+(\d+)",
        lambda m: f"AIR {m.group(1)} SC {m.group(2)}",
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\((\d{4})\)\s+(\d+)\s+scc\s+(\d+)",
        lambda m: f"({m.group(1)}) {m.group(2)} SCC {m.group(3)}",
        text, flags=re.IGNORECASE,
    )
    return text


def _fix_article_section_numbers(text: str) -> str:
    """'Article twenty one' → 'Article 21' ; 'Section four eighty two' → 'Section 482'"""
    # Longest phrases first so "twenty one" wins over "twenty".
    for words in sorted(_NUMBER_WORDS, key=len, reverse=True):
        digits = _NUMBER_WORDS[words]
        text = re.sub(
            rf"\b(article|section)\s+{words}\b",
            lambda m, d=digits: f"{m.group(1).capitalize()} {d}",
            text, flags=re.IGNORECASE,
        )
    # Normalize casing of "article 21" → "Article 21"
    text = re.sub(
        r"\b(article|section)\s+(\d+[A-Za-z\-]*)\b",
        lambda m: f"{m.group(1).capitalize()} {m.group(2)}",
        text,
    )
    return text


def legal_normalize(text: str) -> str:
    """Repair an STT transcript of Indian courtroom speech. Idempotent."""
    if not text:
        return text
    for pattern, repl in _REPORTER_FIXES:
        # Only when followed by a year — a stray "air" in prose stays untouched.
        text = re.sub(pattern + r"(?=\s*\d{4})", repl, text, flags=re.IGNORECASE)
    for pattern, repl in _PHRASE_FIXES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = _fix_article_section_numbers(text)
    text = _fix_reporter_casing(text)
    return text.strip()
