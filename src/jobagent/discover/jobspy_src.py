"""Naukri / LinkedIn / Indeed / Glassdoor via python-jobspy: public listings, no login."""

import random
import time
from collections import Counter
from dataclasses import dataclass, field

from jobspy import scrape_jobs

from jobagent.config import Search
from jobagent.discover.normalize import from_jobspy
from jobagent.models import Job


@dataclass
class Batch:
    """Everything one discovery pass found, across sources."""

    jobs: list[Job] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


def queries(search: Search, smoke: bool = False) -> list[tuple[str, str, bool]]:
    """(role, location, remote) combinations. Smoke = first role in the first city only."""
    roles = search.roles[:1] if smoke else search.roles
    cities = search.cities[:1] if smoke else search.cities
    combos = [(role, city, False) for role in roles for city in cities]
    if search.include_remote and not smoke:
        combos += [(role, "India", True) for role in roles]
    return combos


def discover(search: Search, batch: Batch, smoke: bool = False) -> None:
    results_wanted = 5 if smoke else search.results_per_query
    for role, location, remote in queries(search, smoke):
        for site in search.sites:
            label = f"{site} '{role}' in {location}{' (remote)' if remote else ''}"
            try:
                frame = scrape_jobs(
                    site_name=[site],
                    search_term=role,
                    location=location,
                    is_remote=remote,
                    results_wanted=results_wanted,
                    hours_old=search.hours_old,
                    country_indeed="India",
                    fetch_description=True,
                    verbose=0,
                )
            except Exception as e:  # one failing site must not stop the others
                batch.errors.append(f"{label}: {type(e).__name__}: {e}"[:300])
                continue
            for row in frame.to_dict("records"):
                try:
                    batch.jobs.append(from_jobspy(row, query_city=None if remote else location))
                    batch.counts[site] += 1
                except Exception as e:
                    batch.errors.append(f"{label}: bad row {row.get('id')}: {e}"[:300])
            if site == "linkedin":
                time.sleep(random.uniform(3, 6))  # LinkedIn returns 429 on bursts from one IP
