"""The macOS launchd LaunchAgent that runs the main agent (`jobagent main`) every 15 minutes; it decides
which sub-agents work and sends the daily report itself.

A LaunchAgent runs in the user's login session (the Applier opens a visible Chrome window). While the
Mac sleeps nothing runs; launchd fires once on wake. The project must live outside ~/Documents,
~/Desktop and ~/Downloads: macOS privacy protection blocks launchd jobs there ("Operation not
permitted", exit 78).
"""

import os
import plistlib
import subprocess
from pathlib import Path

from jobagent import settings
from jobagent.config import load_config

MAIN_LABEL = "local.jobagent.main"
LEGACY_LABELS = ["local.jobagent.run", "local.jobagent.outbox", "local.jobagent.digest"]  # before the main agent
TICK_SECONDS = 15 * 60
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
    return {MAIN_LABEL: _agent(MAIN_LABEL, "main", "main.log", {"StartInterval": TICK_SECONDS, "RunAtLoad": True})}


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _path(label: str) -> Path:
    return AGENTS_DIR / f"{label}.plist"


def _remove(label: str) -> bool:
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{label}"], capture_output=True)
    if _path(label).exists():
        _path(label).unlink()
        return True
    return False


def install() -> str:
    if not load_config().reviewed:
        raise RuntimeError(
            "config/config.yaml still has placeholder roles/cities (reviewed: false). "
            "Edit it and set reviewed: true first, so scheduled runs search for the right jobs."
        )
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    replaced = [label for label in LEGACY_LABELS if _remove(label)]
    for label, definition in agents().items():
        subprocess.run(["launchctl", "bootout", f"{_domain()}/{label}"], capture_output=True)  # replace if loaded
        with _path(label).open("wb") as f:
            plistlib.dump(definition, f)
        subprocess.run(["launchctl", "bootstrap", _domain(), str(_path(label))], check=True, capture_output=True)
    old = f" Replaced the old schedule ({', '.join(replaced)})." if replaced else ""
    return (f"Installed: the main agent (`jobagent main`) every {TICK_SECONDS // 60} minutes; it sends the daily "
            f"report after 20:45. Log: data/logs/main.log.{old}")


def uninstall() -> str:
    removed = [label for label in [*agents(), *LEGACY_LABELS] if _remove(label)]
    return f"Removed {', '.join(removed)}; scheduled jobs stopped." if removed else "No schedule was installed."


def status() -> str:
    lines = [f"{label}: old schedule still installed; `jobagent schedule install` replaces it"
             for label in LEGACY_LABELS if _path(label).exists()]
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
