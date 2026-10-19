"""What a sub-agent is: a worker the main agent asks "anything to do now?" and then lets run for a while.

Sub-agents never call each other. They hand work on through the database (a job's state, an email's
status, an application's status), so any one of them can fail, wait or be switched off without
blocking the others.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func
from sqlmodel import Session, select

from jobagent.config import Config
from jobagent.match import MatchContext
from jobagent.pipeline import Tally


@dataclass
class Turn:
    """What a sub-agent gets for one turn."""

    session: Session
    match: MatchContext           # config, profile (None while it's the sample), answers
    now: datetime                 # IST
    until: float                  # time.monotonic() deadline: hand back once it passes (inf: no limit)
    force: bool = False           # `jobagent run` / `jobagent agent X`: a whole pass, cadence and hours ignored
    smoke: bool = False
    log: Callable[[str], None] = print

    @property
    def config(self) -> Config:
        return self.match.config


class SubAgent:
    name = ""
    title = ""
    claude_work: frozenset[str] = frozenset()  # queue entries that need Claude: they wait for its budget

    def settings(self, config: Config):
        return getattr(config.agents, self.name)

    def minutes(self, config: Config) -> float:
        return self.settings(config).minutes_per_tick

    def not_now(self, turn: Turn) -> str | None:
        """Why this agent shouldn't run now even with work waiting (outside its hours, …), or None."""
        return None

    def queue(self, turn: Turn) -> dict[str, int]:
        """What's waiting for this agent, e.g. {"naukri": 2}; empty when there's nothing to do."""
        raise NotImplementedError

    def idle_note(self, turn: Turn) -> str:
        return "nothing to do"

    def run(self, turn: Turn, tally: Tally) -> None:
        raise NotImplementedError


def count(session: Session, query) -> int:
    """Rows a select(Model).where(…) would return."""
    return session.exec(select(func.count()).select_from(query.subquery())).one()


def non_zero(queue: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in queue.items() if value > 0}
