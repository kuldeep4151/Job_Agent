"""macOS launchd LaunchAgents: `jobagent run` through the day, `jobagent outbox` every 15 minutes,
and the `jobagent digest` email at 20:45.

A LaunchAgent runs in the user's login session (the Naukri step opens a visible Chrome window).
launchd uses the Mac's local time zone (IST on this machine). Runs missed while the Mac slept
fire once on wake. The project must live outside ~/Documents, ~/Desktop and ~/Downloads: macOS
privacy protection blocks launchd jobs there ("Operation not permitted", exit 78).
"""

import os
import plistlib
import subprocess
from pathlib import Path

from jobagent import settings
from jobagent.config import load_config

RUN_LABEL = "local.jobagent.run"
OUTBOX_LABEL = "local.jobagent.outbox"
DIGEST_LABEL = "local.jobagent.digest"
RUN_HOURS = [8, 11, 14, 17, 20]
DIGEST_AT = (20, 45)
OUTBOX_EVERY_SECONDS = 15 * 60
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _agent(label: str, command: str, log_name: str, timing: dict) -> dict:
    return {
        "Label": label,
        "ProgramArguments": [str(settings.ROOT / ".venv" / "bin" / "jobagent"), command],
        "WorkingDirectory": str(settings.ROOT),
        "StandardOutPath": str(settings.LOG_DIR / log_name),
        "StandardErrorPath": str(settings.LOG_DIR / log_name),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},  # progress lands in the log as it happens
        "ProcessType": "Background",
        **timing,
    }


def agents() -> dict[str, dict]:
    """The pipeline at fixed hours, and the outbox (replies, follow-ups, one send) every 15 minutes."""
    return {
        RUN_LABEL: _agent(RUN_LABEL, "run", "run.log",
                          {"StartCalendarInterval": [{"Hour": h, "Minute": 0} for h in RUN_HOURS]}),
        OUTBOX_LABEL: _agent(OUTBOX_LABEL, "outbox", "outbox.log", {"StartInterval": OUTBOX_EVERY_SECONDS}),
        DIGEST_LABEL: _agent(DIGEST_LABEL, "digest", "digest.log",
                             {"StartCalendarInterval": {"Hour": DIGEST_AT[0], "Minute": DIGEST_AT[1]}}),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _path(label: str) -> Path:
    return AGENTS_DIR / f"{label}.plist"


def install() -> str:
    if not load_config().reviewed:
        raise RuntimeError(
            "config/config.yaml still has placeholder roles/cities (reviewed: false). "
            "Edit it and set reviewed: true first, so scheduled runs search for the right jobs."
        )
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for label, definition in agents().items():
        subprocess.run(["launchctl", "bootout", f"{_domain()}/{label}"], capture_output=True)  # replace if loaded
        with _path(label).open("wb") as f:
            plistlib.dump(definition, f)
        subprocess.run(["launchctl", "bootstrap", _domain(), str(_path(label))], check=True, capture_output=True)
    hours = ", ".join(f"{h:02d}:00" for h in RUN_HOURS)
    return (f"Installed: `jobagent run` daily at {hours}; `jobagent outbox` every {OUTBOX_EVERY_SECONDS // 60} min; "
            f"`jobagent digest` at {DIGEST_AT[0]:02d}:{DIGEST_AT[1]:02d}. Logs in data/logs/.")


def uninstall() -> str:
    removed = []
    for label in agents():
        subprocess.run(["launchctl", "bootout", f"{_domain()}/{label}"], capture_output=True)
        if _path(label).exists():
            _path(label).unlink()
            removed.append(label)
    return f"Removed {', '.join(removed)}; scheduled jobs stopped." if removed else "No schedule was installed."


def status() -> str:
    lines = []
    for label in agents():
        if not _path(label).exists():
            lines.append(f"{label}: not installed (`jobagent schedule install`)")
            continue
        result = subprocess.run(["launchctl", "print", f"{_domain()}/{label}"], capture_output=True, text=True)
        if result.returncode != 0:
            lines.append(f"{label}: installed but not loaded; run `jobagent schedule install` again")
            continue
        wanted = ("state =", "last exit code =", "runs =")
        details = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(wanted)]
        lines.append(f"{label}: " + "; ".join(details))
    return "\n".join(lines)
