"""Dangerous-action patterns shared by lightweight and semantic probes."""
import re

DANGER_PATTERNS = [
    ("exfiltration", re.compile(
        r"\b(send|upload|exfiltrate|transmit|share|post|leak)\b.{0,80}"
        r"(https?://|all (the )?(files|data|documents|history|conversation))",
        re.I)),
    ("deletion", re.compile(
        r"\b(delete|erase|wipe|remove|destroy|shred)\b.{0,40}"
        r"(all |every |the )?(files|data|documents|directory|disk|records)",
        re.I)),
    ("execution", re.compile(
        r"\b(run|execute|invoke|spawn)\b.{0,40}"
        r"(command|code|script|binary|program|shell)", re.I)),
    ("privilege_escalation", re.compile(
        r"\b(sudo|administrator|root access|elevate|bypass (the )?"
        r"(restrictions?|permissions?|safety))\b", re.I)),
    ("persistence", re.compile(
        r"\b(from now on|always|permanently|never again)\b.{0,60}",
        re.I)),
    ("authority_override", re.compile(
        r"\b(ignore|disregard|override|overrule)\b.{0,60}"
        r"(previous|prior|above|user|system|instructions?)", re.I)),
    ("jailbreak", re.compile(
        r"\b(you are now (DAN|unrestricted)|no restrictions|"
        r"developer mode|jailbreak)\b", re.I)),
]


def danger_taxonomy(text):
    """Return list of (category, matched_text) hits on extracted text."""
    hits = []
    if not text:
        return hits
    for cat, pat in DANGER_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((cat, m.group(0)))
    return hits
