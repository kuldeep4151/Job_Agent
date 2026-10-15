"""What is still missing or a placeholder. `jobagent check` prints this; live mode needs it all green."""

from dataclasses import dataclass

from pydantic import ValidationError

from jobagent import llm, settings
from jobagent.profile import load_profile


@dataclass
class Item:
    name: str
    ok: bool
    detail: str
    required: bool = True  # optional items (Gmail) never block live applying


def _profile_item() -> tuple[Item, bool]:
    """Returns the profile item and whether a real (non-placeholder) profile exists."""
    path = settings.PROFILE_PATH
    if not path.exists():
        return Item("Profile", False, "missing → run `jobagent init --resume <file>`"), False
    try:
        profile = load_profile(path)
    except ValidationError as e:
        return Item("Profile", False, f"config/profile.json is invalid ({e.error_count()} errors)"), False
    if profile.is_placeholder():
        return Item("Profile", False, "sample placeholder → run `jobagent init --resume <file>`"), False
    detail = (
        f"{len(profile.experience)} roles, {len(profile.bullets())} bullets, "
        f"{len(profile.all_skills())} skills"
    )
    return Item("Profile", True, detail), True


def _master_item(real_profile: bool) -> Item:
    pdf, profile = settings.MASTER_RESUME_PDF, settings.PROFILE_PATH
    if not pdf.exists():
        return Item("Master resume PDF", False, "not rendered → run `jobagent render`")
    if profile.exists() and pdf.stat().st_mtime < profile.stat().st_mtime:
        return Item("Master resume PDF", False, "older than profile.json → run `jobagent render`")
    if not real_profile:
        return Item("Master resume PDF", False, "rendered from the sample profile")
    return Item("Master resume PDF", True, str(pdf.relative_to(settings.ROOT)))


def _answers_item() -> Item:
    if not settings.ANSWERS_PATH.exists():
        return Item("Application answers", False, "missing → copy config/answers.example.yaml to answers.yaml")
    todo = settings.unfilled(settings.load_yaml(settings.ANSWERS_PATH))
    if todo:
        preview = ", ".join(todo[:4]) + (", …" if len(todo) > 4 else "")
        return Item("Application answers", False, f"{len(todo)} fields still TODO ({preview})")
    return Item("Application answers", True, "all filled")


def _config_item() -> Item:
    reviewed = settings.load_yaml(settings.CONFIG_PATH).get("reviewed") is True
    if not reviewed:
        return Item("Search preferences", False, "placeholder roles/cities → edit config.yaml, set reviewed: true")
    return Item("Search preferences", True, "reviewed")


def setup_items() -> list[Item]:
    key_ok = llm.api_key_configured()
    profile_item, real_profile = _profile_item()
    return [
        Item("Anthropic API key", key_ok, "set" if key_ok else "still the placeholder in .env"),
        profile_item,
        _master_item(real_profile),
        _answers_item(),
        _config_item(),
        _gmail_item(),
    ]


def _gmail_item() -> Item:
    from jobagent.outreach import gmail

    creds = gmail.credentials()
    detail = f"set for {creds[0]}" if creds else "no App Password in .env (needed only to send outreach emails)"
    return Item("Gmail (outreach)", creds is not None, detail, required=False)


def ready_for_live() -> bool:
    return all(item.ok for item in setup_items() if item.required)
