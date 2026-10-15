"""Greenhouse and Lever boards of companies listed in config/targets.yaml (public JSON, no key)."""

import re
from datetime import datetime, timedelta, timezone

import httpx

from jobagent.config import Search, Targets
from jobagent.discover.jobspy_src import Batch
from jobagent.discover.normalize import from_greenhouse, from_lever, mentions_city
from jobagent.models import Job

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
LEVER_URL = "https://api.lever.co/v0/postings/{board}"

# Words too generic to say a title is in the user's field ("Senior Sales Engineer" vs "Senior Engineer").
_GENERIC = {"senior", "sr", "junior", "jr", "lead", "principal", "staff", "i", "ii", "iii", "and", "of", "the"}


def _role_words(search: Search) -> set[str]:
    words = {w for role in search.roles for w in re.findall(r"[a-z0-9+#.]+", role.lower())}
    return words - _GENERIC


def wanted(job: Job, search: Search, now: datetime) -> bool:
    """Boards list every department and country, so keep only fresh, Indian, in-field postings."""
    location = job.location.lower()
    in_india = "india" in location or any(mentions_city(location, city) for city in search.cities)
    title_words = set(re.findall(r"[a-z0-9+#.]+", job.title.lower()))
    fresh = job.posted_at is not None and job.posted_at >= now - timedelta(hours=search.hours_old)
    return in_india and bool(title_words & _role_words(search)) and fresh


def discover(targets: Targets, search: Search, batch: Batch) -> None:
    now = datetime.now(timezone.utc)
    sources = [
        ("greenhouse", board, GREENHOUSE_URL, {"content": "true"}, lambda d: d["jobs"], from_greenhouse)
        for board in targets.greenhouse
    ] + [
        ("lever", board, LEVER_URL, {"mode": "json"}, lambda d: d, from_lever)
        for board in targets.lever
    ]
    with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0 (personal job search)"}) as client:
        for source, board, url, params, listing, convert in sources:
            try:
                response = client.get(url.format(board=board), params=params)
                response.raise_for_status()
                postings = listing(response.json())
            except Exception as e:
                batch.errors.append(f"{source} board '{board}': {type(e).__name__}: {e}"[:300])
                continue
            for posting in postings:
                try:
                    job = convert(posting, board)
                except Exception as e:
                    batch.errors.append(f"{source} board '{board}': bad posting: {e}"[:300])
                    continue
                if wanted(job, search, now):
                    batch.jobs.append(job)
                    batch.counts[source] += 1
