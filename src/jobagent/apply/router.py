"""Where a job is really applied to, decided by code from its links (no clicks, no logins).

Greenhouse and Lever have standard public forms the agent can fill; everything else (Workday,
Taleo, SuccessFactors, LinkedIn Easy Apply, custom career sites…) goes to the "Needs you" queue
with the link, the tailored resume and copy-ready answers.
"""

import re
from urllib.parse import urlparse

import httpx

from jobagent.models import Job

AUTO_ATS = {"greenhouse", "lever"}

_ATS_HOSTS = [
    ("greenhouse", r"(^|\.)greenhouse\.io$|^grnh\.se$"),
    ("lever", r"^jobs\.(eu\.)?lever\.co$"),
    ("workday", r"myworkdayjobs\.com$|myworkdaysite\.com$"),
    ("successfactors", r"successfactors\.(com|eu)$|^jobs\.sap\.com$"),
    ("taleo", r"taleo\.net$"),
    ("smartrecruiters", r"smartrecruiters\.com$"),
    ("icims", r"icims\.com$"),
    ("darwinbox", r"darwinbox\.(in|com)$"),
    ("keka", r"keka\.com$"),
    ("ashby", r"ashbyhq\.com$"),
    ("zoho recruit", r"zohorecruit\.(com|in)$"),
    ("jazzhr", r"applytojob\.com$"),
    ("avature", r"avature\.net$"),
    ("linkedin", r"(^|\.)linkedin\.com$"),
    ("naukri", r"(^|\.)naukri\.com$"),
    ("indeed", r"(^|\.)indeed\.com$"),
    ("glassdoor", r"(^|\.)glassdoor\.(com|co\.in)$"),
]
_SHORTENERS = {"grnh.se", "lnkd.in", "bit.ly", "tinyurl.com", "t.co", "rebrand.ly"}

HOW_TO_APPLY = {
    "workday": "Workday: sign in or create an account on the company's Workday site, then apply.",
    "taleo": "Taleo: needs an account on the company's careers site.",
    "successfactors": "SuccessFactors: needs an account on the company's careers site.",
    "icims": "iCIMS: needs an account on the company's careers site.",
    "darwinbox": "Darwinbox: apply on the company's Darwinbox careers page.",
    "linkedin": "LinkedIn: open the job and apply there (LinkedIn is never automated, to protect your account).",
    "naukri": "Naukri sends this one to the company's site: open it and click 'Apply on company site'.",
    "indeed": "Indeed: open the job and follow its apply button.",
    "glassdoor": "Glassdoor: open the job and follow its apply button.",
}
DEFAULT_HOW = "Company careers site: open the link and apply."


def classify(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return next((ats for ats, pattern in _ATS_HOSTS if re.search(pattern, host)), "company site")


def resolve(url: str, client: httpx.Client) -> str:
    """Follow a short link (grnh.se/…, lnkd.in/…) to where it really goes; the original on any error."""
    try:
        return str(client.get(url, follow_redirects=True).url)
    except httpx.HTTPError:
        return url


def route(job: Job, client: httpx.Client) -> tuple[str, str]:
    """(ats, url to apply at). Board jobs keep their own form; others use the company link if known."""
    if job.source in AUTO_ATS:
        return job.source, job.apply_url or job.url
    url = job.apply_url or job.url
    if urlparse(url).netloc.lower().removeprefix("www.") in _SHORTENERS:
        url = resolve(url, client)
    return classify(url), url


def how_to_apply(ats: str) -> str:
    return HOW_TO_APPLY.get(ats, DEFAULT_HOW if ats == "company site" else f"{ats.title()}: open the link and apply.")
