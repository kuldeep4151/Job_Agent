"""Typed access to config/config.yaml, config/targets.yaml and config/answers.yaml."""

from pydantic import BaseModel

from jobagent import settings


class Search(BaseModel):
    roles: list[str]
    cities: list[str]
    include_remote: bool = True
    hours_old: int = 24
    results_per_query: int = 15
    sites: list[str] = ["naukri", "linkedin", "indeed", "glassdoor"]


class Blocklist(BaseModel):
    companies: list[str] = []


class Matching(BaseModel):
    min_score: int = 70
    min_skill_overlap: int = 2
    experience_tolerance_years: float = 1.0


class Caps(BaseModel):
    naukri_applies_per_day: int = 5
    ats_applies_per_day: int = 5
    emails_per_day: int = 5
    usd_per_day: float = 4.0


class Outreach(BaseModel):
    enabled: bool = False  # drafts are written either way; the outbox only sends when this is on
    send_hours: tuple[int, int] = (10, 18)  # IST, Monday-Friday
    follow_up_after_days: int = 6  # business days; one short follow-up, then stop
    company_cooldown_days: int = 30  # never email the same company twice within this window
    find_website: bool = True  # look up company websites (Claude web search) when no address is known


class Mode(BaseModel):
    dry_run: bool = True
    paused: bool = False


class Config(BaseModel):
    reviewed: bool = False
    search: Search
    blocklist: Blocklist = Blocklist()
    matching: Matching = Matching()
    caps: Caps = Caps()
    outreach: Outreach = Outreach()
    mode: Mode = Mode()


class Targets(BaseModel):
    greenhouse: list[str] = []
    lever: list[str] = []


def load_config() -> Config:
    return Config.model_validate(settings.load_yaml(settings.CONFIG_PATH))


def load_targets() -> Targets:
    path = settings.CONFIG_DIR / "targets.yaml"
    return Targets.model_validate(settings.load_yaml(path)) if path.exists() else Targets()


def load_answers() -> dict:
    path = settings.ANSWERS_PATH
    return settings.load_yaml(path) if path.exists() else {}


def answer(answers: dict, dotted_key: str) -> object | None:
    """Value at e.g. "employment.total_experience_years", or None if missing or still TODO."""
    value: object = answers
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, list):  # e.g. preferred_cities: [TODO]
        value = [item for item in value if item not in (settings.PLACEHOLDER, "", None)]
        return value or None
    return None if value in (settings.PLACEHOLDER, "", None) else value
