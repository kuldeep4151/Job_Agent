"""What happened on an application attempt, plus the screenshot/HTML kept as evidence (all sites)."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from playwright.sync_api import Page

from jobagent import settings

SHOTS_DIR = settings.LOG_DIR / "applications"


@dataclass
class Outcome:
    status: Literal["applied", "dry_run", "already_applied", "company_site", "closed", "manual", "failed", "logged_out"]
    note: str
    screenshot: Path | None = None
    qa: list[dict] = field(default_factory=list)


def save_evidence(page: Page, job_id: int | str, tag: str, html: bool = False) -> Path:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SHOTS_DIR / f"job-{job_id}-{tag}-{int(time.time())}.png"
    page.screenshot(path=str(path), full_page=False)
    if html:
        path.with_suffix(".html").write_text(page.content())
    return path


def qa_json(outcome: Outcome) -> str | None:
    return json.dumps(outcome.qa, ensure_ascii=False) if outcome.qa else None
