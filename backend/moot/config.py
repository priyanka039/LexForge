# ─────────────────────────────────────────────
# moot/config.py
# All Moot Chamber configuration in one place.
# Voice + LLM providers are resolved at runtime
# from environment keys — nothing here is a
# hard dependency.
# ─────────────────────────────────────────────

import os

# ═══ LLM PROVIDER CHAIN ═══════════════════════
# Tried in order. A provider is skipped if its
# key is missing or the call fails. Ollama is the
# final offline fallback (no key required).
ANTHROPIC_MODEL = os.getenv("MOOT_ANTHROPIC_MODEL", "claude-haiku-4-5")
OPENAI_MODEL    = os.getenv("MOOT_OPENAI_MODEL",    "gpt-4o-mini")
GEMINI_MODEL    = os.getenv("MOOT_GEMINI_MODEL",    "gemini-2.5-flash")

LLM_TIMEOUT_SECONDS = 25.0

# ═══ SESSION ══════════════════════════════════
REDIS_URL                  = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS        = 6 * 60 * 60      # sessions expire after 6 hours
MAX_HISTORY_IN_PROMPT      = 8                # exchanges shown to the judge
WEAKNESS_TRIGGER_INTERVAL  = 5                # fire weakness agent every N counsel exchanges
WEAKNESS_MIN_EXCHANGES     = 3

# ═══ SCORING WEIGHTS ══════════════════════════
SCORE_WEIGHTS = {
    "structure":      0.25,
    "authority":      0.25,
    "responsiveness": 0.20,
    "precision":      0.15,
    "coherence":      0.15,
}

# ═══ JUDGE TTS VOICES (Sarvam bulbul:v3) ══════
# Each personality gets a distinct male voice.
JUDGE_VOICE_MAP = {
    "verma":        "ratan",    # deep, measured — the philosopher
    "mehta":        "amit",     # clipped, impatient — the technocrat
    "krishnaswamy": "anand",    # warm but probing — the activist
    "sinha":        "kabir",    # heavy, skeptical — the skeptic
    "kaul":         "rohan",    # conversational — the pragmatist
}
DEFAULT_JUDGE_VOICE = "aditya"

# ═══ COURT-LEVEL ADDRESS FORMS ════════════════
COURT_ADDRESS_FORMS = {
    "district": {
        "judge":  "Your Honour",
        "court":  "this Court",
        "name":   "District and Sessions Court",
    },
    "high_court": {
        "judge":  "Your Lordship",
        "court":  "this Hon'ble Court",
        "name":   "High Court",
    },
    "supreme": {
        "judge":  "My Lords",
        "court":  "this Hon'ble Court",
        "name":   "Supreme Court of India",
    },
}

# ═══ CITATION PATTERNS (Indian reporters) ═════
CITATION_PATTERNS = [
    r"AIR\s+\d{4}\s+(?:SC|SCC|Bom|Del|Cal|Mad|All|Ker|Raj|MP|Guj|AP|Pat|Ori)\s+\d+",
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
    r"\d{4}\s+SCC\s+\(\d+\)\s+\d+",
    r"\d{4}\s+SCR\s+\(\d+\)\s+\d+",
    r"\(\d{4}\)\s+\d+\s+SCR\s+\d+",
    r"ILR\s+\d{4}\s+\w+\s+\d+",
    r"\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+",
    r"\(\d{4}\)\s+\d+\s+(?:Bom|Del|Cal|Mad|All|Ker|Raj|MP|Guj|AP)\s*(?:LJ|LR|HC)?\s+\d+",
]

# A "loose" citation mention — something that looks like the mooter is citing
# a reporter but may have the format wrong. Used to flag suspect citations.
CITATION_LOOSE_PATTERN = (
    r"\b(?:AIR|SCC|SCR|ILR|SCALE|MANU|CriLJ|Cri\s?LJ|JT|LW|MLJ|CTC|KLT|GLR|BomCR)\b[^.,;]{0,40}"
)

# ═══ LANGUAGES (mirrors Sarvam support) ═══════
SUPPORTED_SESSION_LANGUAGES = [
    {"code": "en-IN", "label": "English"},
    {"code": "hi-IN", "label": "हिन्दी (Hindi)"},
    {"code": "mr-IN", "label": "मराठी (Marathi)"},
    {"code": "ta-IN", "label": "தமிழ் (Tamil)"},
    {"code": "te-IN", "label": "తెలుగు (Telugu)"},
    {"code": "kn-IN", "label": "ಕನ್ನಡ (Kannada)"},
    {"code": "ml-IN", "label": "മലയാളം (Malayalam)"},
    {"code": "bn-IN", "label": "বাংলা (Bengali)"},
    {"code": "gu-IN", "label": "ગુજરાતી (Gujarati)"},
    {"code": "pa-IN", "label": "ਪੰਜਾਬੀ (Punjabi)"},
]

LANGUAGE_NAMES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
}


def language_directive(lang_code: str) -> str:
    """
    Prompt suffix that makes every agent reason and answer in the
    session language. Case names, citations, and section numbers stay
    in their conventional English/Latin form — exactly how Indian
    courts mix languages in practice.
    """
    if not lang_code or lang_code == "en-IN":
        return (
            "\n\nLANGUAGE: Respond in formal Indian courtroom English."
        )
    name = LANGUAGE_NAMES.get(lang_code, "the user's language")
    return (
        f"\n\nLANGUAGE: Respond entirely in {name}, in its native script, "
        f"in the formal register a judge of an Indian court would use when "
        f"proceedings are conducted in {name}. Keep case names, citations "
        f"(e.g. AIR 1978 SC 597), Article/Section numbers, and established "
        f"Latin maxims in their standard form — do not translate those. "
        f"Counsel may mix English legal phrases into their {name} speech; "
        f"this is normal Indian courtroom practice."
    )
