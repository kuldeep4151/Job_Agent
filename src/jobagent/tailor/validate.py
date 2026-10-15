"""Anti-fabrication check for the one free-text part of tailoring: the summary.

Bullets and skills can't be invented (tailoring only picks and orders existing ones by ID/name),
so the summary is where a model could slip in a tool, number or employer the candidate never had.
Every number and every capitalised/technical term in it must already appear in the profile.
"""

import re

from jobagent.profile import Profile

MAX_SUMMARY_WORDS = 90

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./&'-]*[A-Za-z0-9+#]|[A-Za-z0-9]")
_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|my|me)\b")


def profile_text(profile: Profile) -> str:
    parts = [profile.name, profile.headline, profile.summary, profile.location]
    parts += profile.all_skills() + list(profile.bullets().values())
    for role in profile.experience:
        parts += [role.title, role.company, role.location, role.start, role.end]
    for edu in profile.education:
        parts += [edu.institution, edu.degree, edu.start, edu.end, edu.score]
    parts += [p.name for p in profile.projects] + profile.certifications + profile.achievements
    return "\n".join(parts)


def _needs_support(token: str, sentence_start: bool) -> bool:
    """Numbers, CamelCase/ACRONYMS/tech tokens, and capitalised words mid-sentence."""
    if any(ch.isdigit() for ch in token):
        return True
    if any(ch.isupper() for ch in token[1:]):  # PyTorch, FAISS, LLMs, ML/DL
        return True
    return token[0].isupper() and not sentence_start


def _in_source(term: str, source: str) -> bool:
    """Whole-word, case-insensitive, singular or plural ("LLM" matches "LLMs")."""
    for form in (term, term.removesuffix("s")):
        if form and re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}s?(?![A-Za-z0-9])", source, re.IGNORECASE):
            return True
    return False


def _supported(token: str, source: str) -> bool:
    base = re.sub(r"'(s|m|re|ve|ll|d|t)$", "", token)  # "Kafka's", "I'm", "We've" → the word itself
    if _in_source(base, source):
        return True
    parts = [part for part in re.split(r"[/&]", base) if part]  # "ML/DL" passes as "ML" + "DL"
    return len(parts) > 1 and all(_in_source(part, source) for part in parts)


def _same_dashes(text: str) -> str:
    return text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")


def unsupported_terms(summary: str, profile: Profile, extra_source: str = "") -> list[str]:
    """Numbers and capitalised/technical terms in `summary` found neither in the profile nor `extra_source`."""
    source = _same_dashes(f"{profile_text(profile)}\n{extra_source}")
    summary = _same_dashes(summary)  # "40-55%" written with a hyphen is the resume's "40–55%"
    problems: list[str] = []
    for sentence in re.split(r"(?<=[.!?;:])\s+", summary.strip()):
        for index, token in enumerate(_TOKEN.findall(sentence)):
            if _needs_support(token, sentence_start=index == 0) and not _supported(token, source):
                if token not in problems:
                    problems.append(token)
    return problems


_YEARS_CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)


def unsupported_experience_claims(summary: str, profile: Profile) -> list[str]:
    """"5+ years" passes the number check if a "5" appears anywhere, so years claims need an exact match."""
    source = profile_text(profile)
    return [
        claim.group(0)
        for claim in _YEARS_CLAIM.finditer(summary)
        if not re.search(rf"(?<![\d.]){re.escape(claim.group(1))}\s*\+?\s*(?:years?|yrs?)\b", source, re.I)
    ]


def check_summary(summary: str, profile: Profile) -> list[str]:
    """Problems that make a tailored summary unusable (empty list = OK)."""
    problems = []
    claims = unsupported_experience_claims(summary, profile)
    if claims:
        problems.append(f"summary claims experience not in your profile: {', '.join(claims)}")
    words = len(summary.split())
    if not words:
        problems.append("summary is empty")
    if words > MAX_SUMMARY_WORDS:
        problems.append(f"summary is {words} words (max {MAX_SUMMARY_WORDS})")
    if _FIRST_PERSON.search(summary):
        problems.append("summary uses first person")
    terms = unsupported_terms(summary, profile)
    if terms:
        problems.append(f"summary mentions things not in your profile: {', '.join(terms)}")
    return problems
