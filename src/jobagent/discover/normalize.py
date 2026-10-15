"""Turn postings from any source into Job rows, plus the parsing helpers the filters rely on."""

import html
import math
import re
from datetime import date, datetime, timedelta, timezone

from selectolax.lexbor import LexborHTMLParser

from jobagent.models import Job

IST = timezone(timedelta(hours=5, minutes=30))

# Canonical city → spellings seen on Indian job boards.
CITY_ALIASES: dict[str, list[str]] = {
    "bengaluru": ["bengaluru", "bangalore"],
    "mumbai": ["mumbai", "bombay", "navi mumbai", "thane"],
    "delhi": ["delhi", "new delhi", "ncr"],
    "gurugram": ["gurugram", "gurgaon"],
    "noida": ["noida", "greater noida"],
    "hyderabad": ["hyderabad", "secunderabad"],
    "chennai": ["chennai", "madras"],
    "kolkata": ["kolkata", "calcutta"],
    "pune": ["pune"],
    "ahmedabad": ["ahmedabad"],
    "kochi": ["kochi", "cochin"],
    "jaipur": ["jaipur"],
    "chandigarh": ["chandigarh", "mohali"],
    "indore": ["indore"],
    "coimbatore": ["coimbatore"],
    "trivandrum": ["trivandrum", "thiruvananthapuram"],
}


def city_aliases(city: str) -> list[str]:
    key = city.strip().lower()
    for canonical, aliases in CITY_ALIASES.items():
        if key == canonical or key in aliases:
            return aliases
    return [key]


def mentions_city(text: str, city: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in city_aliases(city))


def mentions_any_known_city(text: str) -> bool:
    return any(mentions_city(text, canonical) for canonical in CITY_ALIASES)


_COMPANY_NOISE = re.compile(
    r"\b(private|pvt|limited|ltd|llp|inc|incorporated|corp|corporation|co|india|the)\b"
)


def norm_company(name: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return " ".join(_COMPANY_NOISE.sub(" ", text).split())


def norm_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#]", " ", title.lower()).split())


def dedupe_key(job: Job) -> str:
    """Same company + same title = same opening, whichever board or city it was listed on."""
    company = norm_company(job.company)
    if not company:
        return f"{job.source}:{job.external_id}"
    return f"{company}|{norm_title(job.title)}"


_YEARS = r"(?:yrs?|years?)"
_NUM = r"(\d{1,2}(?:\.\d)?)"
_RANGE = re.compile(rf"{_NUM}\s*(?:-|–|to)\s*{_NUM}\s*\+?\s*{_YEARS}", re.I)
_PLUS = re.compile(
    rf"{_NUM}\s*(?:\+|plus)\s*{_YEARS}|(?:minimum|min\.?|at\s*least)\s*(?:of\s*)?{_NUM}\s*{_YEARS}", re.I
)


def parse_experience(text: str | None, *, require_keyword: bool = False) -> tuple[float | None, float | None]:
    """(min, max) years from "3-5 Yrs", "2 to 4 years", "5+ years", "minimum 3 years".

    With require_keyword (free-text JDs), a match only counts if "exp…" appears within ~80 chars,
    so "a company with 20 years of history" is ignored.
    """
    if not text:
        return None, None

    def near_keyword(m: re.Match) -> bool:
        window = text[max(0, m.start() - 80) : m.end() + 80].lower()
        return not require_keyword or "exp" in window

    for m in _RANGE.finditer(text):
        low, high = float(m[1]), float(m[2])
        if near_keyword(m) and low <= high <= 40:
            return low, high
    for m in _PLUS.finditer(text):
        value = float(m[1] or m[2])
        if near_keyword(m) and value <= 40:
            return value, None
    if not require_keyword and re.search(r"\bfresher", text, re.I):
        return 0.0, 1.0
    return None, None


def _clean(value):
    """pandas gives NaN for missing values; normalise to None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def salary_lpa(amount, interval, currency) -> float | None:
    amount, interval, currency = _clean(amount), _clean(interval), _clean(currency)
    if amount is None or currency not in (None, "INR"):
        return None
    if interval == "yearly":
        return round(float(amount) / 100_000, 1)
    if interval == "monthly":
        return round(float(amount) * 12 / 100_000, 1)
    return None


def html_to_text(content: str) -> str:
    text = LexborHTMLParser(html.unescape(content)).text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


FREE_MAIL = {"gmail.com", "yahoo.com", "yahoo.co.in", "outlook.com", "hotmail.com", "rediffmail.com", "ymail.com",
             "icloud.com", "live.com", "protonmail.com"}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_NOT_EMAIL_TLDS = {"png", "jpg", "jpeg", "gif", "svg", "webp"}


def extract_emails(text: str) -> list[str]:
    found: list[str] = []
    for match in _EMAIL.findall(text or ""):
        email = match.lower().strip(".")
        if email.rsplit(".", 1)[-1] not in _NOT_EMAIL_TLDS and email not in found:
            found.append(email)
    return found


def _split_list(value) -> list[str]:
    value = _clean(value)
    if not value:
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    return [item.strip() for item in items if item and item.strip()]


def _as_datetime(value) -> datetime | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=IST)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=IST)
    return datetime.fromisoformat(str(value)).replace(tzinfo=IST)


def finalize(job: Job) -> Job:
    """Fill fields derived from the description, then the dedupe key."""
    if job.exp_min is None:
        job.exp_min, job.exp_max = parse_experience(job.description, require_keyword=True)
    for email in extract_emails(job.description):
        if email not in job.emails:
            job.emails.append(email)
    job.dedupe_key = dedupe_key(job)
    return job


def from_jobspy(row: dict, query_city: str | None) -> Job:
    experience_text = _clean(row.get("experience_range"))
    exp_min, exp_max = parse_experience(experience_text)
    interval, currency = row.get("interval"), row.get("currency")
    return finalize(
        Job(
            dedupe_key="",
            source=str(row["site"]),
            external_id=str(row["id"]),
            url=str(row["job_url"]),
            apply_url=_clean(row.get("job_url_direct")),
            title=str(row["title"]),
            company=str(_clean(row.get("company")) or ""),
            location=str(_clean(row.get("location")) or ""),
            query_city=query_city,
            is_remote=bool(_clean(row.get("is_remote")) or False),
            experience_text=experience_text,
            exp_min=exp_min,
            exp_max=exp_max,
            salary_min_lpa=salary_lpa(row.get("min_amount"), interval, currency),
            salary_max_lpa=salary_lpa(row.get("max_amount"), interval, currency),
            description=str(_clean(row.get("description")) or ""),
            emails=[e.lower() for e in _split_list(row.get("emails"))],
            listed_skills=_split_list(row.get("skills")),
            posted_at=_as_datetime(row.get("date_posted")),
            company_website=_clean(row.get("company_url_direct")),  # Indeed/Glassdoor: the company's own site
        )
    )


def from_greenhouse(job: dict, board: str) -> Job:
    posted = job.get("first_published") or job.get("updated_at")
    return finalize(
        Job(
            dedupe_key="",
            source="greenhouse",
            external_id=f"gh-{board}-{job['id']}",
            url=job["absolute_url"],
            title=job["title"],
            company=job.get("company_name") or board,
            location=(job.get("location") or {}).get("name", ""),
            is_remote="remote" in (job.get("location") or {}).get("name", "").lower(),
            description=html_to_text(job.get("content") or ""),
            posted_at=datetime.fromisoformat(posted) if posted else None,
        )
    )


def from_lever(posting: dict, board: str) -> Job:
    parts = [posting.get("descriptionPlain", "")]
    for section in posting.get("lists", []):
        parts.append(f"{section.get('text', '')}\n{html_to_text(section.get('content', ''))}")
    parts.append(posting.get("additionalPlain", ""))
    created = posting.get("createdAt")
    return finalize(
        Job(
            dedupe_key="",
            source="lever",
            external_id=f"lv-{board}-{posting['id']}",
            url=posting["hostedUrl"],
            apply_url=posting.get("applyUrl"),
            title=posting["text"],
            company=board,
            location=(posting.get("categories") or {}).get("location", ""),
            is_remote=posting.get("workplaceType") == "remote",
            description="\n\n".join(p for p in parts if p).strip(),
            posted_at=datetime.fromtimestamp(int(created) / 1000, tz=timezone.utc) if created else None,
        )
    )
