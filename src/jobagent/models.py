"""Database tables. A Job moves through `State` one stage at a time; each stage is safe to re-run."""

from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class State(StrEnum):
    DISCOVERED = "discovered"  # stored, not yet filtered
    FILTERED = "filtered"      # passed hard filters, waiting for skill prefilter + Claude score
    SKIPPED = "skipped"        # dropped by a filter, the prefilter, or the score (see skip_reason)
    SCORED = "scored"          # score at or above threshold: eligible to apply
    TAILORED = "tailored"      # per-job resume rendered (Naukri-profile applies skip this stage)
    APPLIED = "applied"        # application submitted (or found already applied on the board)
    MANUAL = "manual"          # needs the user: unanswerable question, unknown page, unsupported site
    FAILED = "failed"          # an attempt went wrong; see apply_note and the screenshot


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dedupe_key: str = Field(index=True, unique=True)
    source: str                       # naukri / linkedin / indeed / glassdoor / greenhouse / lever
    external_id: str
    url: str
    apply_url: str | None = None      # company-site apply link when the board redirects there
    title: str
    company: str
    location: str = ""
    query_city: str | None = None     # city we searched for (boards sometimes drop the city)
    is_remote: bool = False
    experience_text: str | None = None
    exp_min: float | None = None
    exp_max: float | None = None
    salary_min_lpa: float | None = None
    salary_max_lpa: float | None = None
    description: str = ""
    emails: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # extracted by code only
    listed_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    posted_at: datetime | None = None
    first_seen_at: datetime = Field(default_factory=utcnow, index=True)
    also_seen_at: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # duplicate URLs

    state: str = Field(default=State.DISCOVERED, index=True)
    skip_reason: str | None = None
    flags: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # warnings, e.g. free-mail contact
    skill_overlap: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    score: int | None = None
    score_decision: str | None = None
    matched_must_haves: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    missing_must_haves: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    seniority_fit: str | None = None
    score_reason: str | None = None
    scored_at: datetime | None = None

    resume_path: str | None = None        # relative to the project root
    tailored_summary: str | None = None   # None when the original summary was kept
    tailor_rationale: str | None = None
    tailor_notes: str | None = None       # validator rejections / repairs, "; "-separated
    tailored_at: datetime | None = None

    applied_at: datetime | None = None    # set only when this agent submitted the application
    dry_run_at: datetime | None = None    # a dry run reached the point of submitting
    apply_note: str | None = None
    apply_screenshot: str | None = None   # relative to the project root
    apply_qa: str | None = None           # JSON list of {question, answer, source} for audit

    company_website: str | None = None    # from the board (Indeed/Glassdoor) when it gives one
    outreach_note: str | None = None      # set once contact-finding ran for this job (found / why not)


class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    found: int = 0
    new: int = 0
    duplicates: int = 0
    skipped: int = 0
    scored: int = 0
    matches: int = 0
    tailored: int | None = 0
    applied: int | None = 0
    drafted: int | None = 0
    cost_usd: float = 0.0
    source_counts: dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON))
    errors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    notes: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class EmailStatus(StrEnum):
    DRAFT = "draft"          # written; the outbox sends it once outreach is enabled
    SENT = "sent"
    REJECTED = "rejected"    # failed the anti-fabrication check: never sent
    CANCELLED = "cancelled"  # by the user, or the address/company became off-limits
    EXPIRED = "expired"      # sat unsent too long to still be timely
    FAILED = "failed"        # sending raised an error (see note)


class Email(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(index=True)
    company_key: str = Field(index=True)  # normalised company name: one company, one email per cooldown
    to_address: str
    contact_source: str                   # "job post" or the company page it was found on
    subject: str
    body: str
    attachment: str | None = None         # resume PDF, relative to the project root
    kind: str = "first"                   # first | follow_up
    status: str = Field(default=EmailStatus.DRAFT, index=True)
    note: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    sent_at: datetime | None = None
    message_id: str | None = None
    in_reply_to: str | None = None        # follow-ups thread under the first email
    replied_at: datetime | None = None
    reply_snippet: str | None = None
    bounced_at: datetime | None = None


class Suppression(SQLModel, table=True):
    """Addresses (or "@domain") never to email again: bounced, or asked not to be contacted."""

    address: str = Field(primary_key=True)
    reason: str
    created_at: datetime = Field(default_factory=utcnow)
