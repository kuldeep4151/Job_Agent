"""Outreach: contact finding (mock web), email drafting (mock Claude), outbox rules, replies and bounces."""

import json
from datetime import datetime, timedelta

import httpx
import pytest
from sqlmodel import select

from conftest import SAMPLE_PROFILE
from jobagent import db
from jobagent.config import Config, Mode, Outreach, Search
from jobagent.discover.normalize import IST
from jobagent.models import Email, EmailStatus, Job, State, Suppression, utcnow
from jobagent.outreach import compose, contacts, gmail, outbox
from jobagent.profile import load_profile

ANSWERS = {"personal": {"full_name": "Sample Candidate", "phone": "+91 90000 00000",
                        "email": "sample.candidate@example.com", "linkedin_url": "https://linkedin.com/in/sample"}}
WEEKDAY_11AM = datetime(2026, 10, 15, 11, 0, tzinfo=IST)  # a Thursday
GOOD_BODY = (
    "Dear Hiring Team,\n\nI'm writing about the Backend Engineer role at Nimbus. At Example Payments Pvt Ltd I "
    "built a refund-reconciliation service in Go processing 2M transactions/day, cutting manual finance "
    "escalations by 60%. I also led the migration of 14 Django endpoints to FastAPI with zero downtime.\n\n"
    "My resume is attached. Would you be open to considering me for the role?\n\nBest regards,\nSample"
)


def make_job(**fields) -> Job:
    defaults = dict(id=1, dedupe_key="k", source="linkedin", external_id="1", url="u", title="Backend Engineer",
                    company="Nimbus", location="Bengaluru", score=82, state=State.TAILORED)
    return Job(**(defaults | fields))


def site(pages: dict[str, str], robots: str = "", seen: list | None = None) -> httpx.Client:
    """A fake website: path → HTML (with a homepage unless given). Records every requested path in `seen`."""
    pages = {"": "<title>Home</title>"} | pages

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots)
        html = pages.get(request.url.path.strip("/"))
        return httpx.Response(200, html=html) if html is not None else httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture
def profile():
    return load_profile(SAMPLE_PROFILE)


# ---------- contacts ----------


@pytest.mark.parametrize(
    ("address", "from_posting", "rank"),
    [
        ("careers@acme.com", True, 0), ("talent@nimbus.com", True, 0), ("hr@acme.co.in", False, 1),
        ("sheetalsen@bestpeers.com", True, 1), ("sheetalsen@bestpeers.com", False, None),
        ("info@acme.com", False, 3), ("accessibility.support@tiaa.org", True, None),
        ("global.compliance@globalfoundries.com", True, None), ("applicationassistance@sailpoint.com", True, None),
        ("accomodations@powerschool.com", True, None), ("hr.acme@gmail.com", True, None), ("noreply@acme.com", True, None),
    ],
)
def test_classify(address, from_posting, rank):
    assert contacts.classify(address, from_posting=from_posting) == rank


def test_registrable_domain():
    assert contacts.registrable_domain("careers.examplelabs.com") == "examplelabs.com"
    assert contacts.registrable_domain("www.acme.co.in") == "acme.co.in"


def test_scrape_site_obeys_robots_and_stays_on_domain():
    seen: list[str] = []
    client = site(
        {"careers": '<a href="mailto:careers@acme.com?subject=Hi">Mail us</a> or partner@other.com',
         "contact": "hr@acme.com"},
        robots="User-agent: *\nDisallow: /contact\n", seen=seen,
    )
    found = contacts.scrape_site("acme.com", client)
    assert [(c.address, c.rank) for c in found] == [("careers@acme.com", 1)]
    assert "/contact" not in seen  # disallowed by robots.txt
    assert found[0].source == "https://acme.com/careers"


def test_posting_hiring_address_beats_the_website():
    seen: list[str] = []
    job = make_job(emails=["accessibility.support@nimbus.com", "talent@nimbus.com"], company_website="https://nimbus.com")
    contact, note = contacts.best_contact(job, allow_lookup=False, client=site({}, seen=seen))
    assert contact.address == "talent@nimbus.com" and contact.source == "job post"
    assert seen == []  # a rank-0 posting address means no need to crawl the site


def test_website_used_when_posting_has_only_boilerplate():
    job = make_job(emails=["accessibility.support@nimbus.com"])  # its domain still tells us the website
    contact, note = contacts.best_contact(job, allow_lookup=False, client=site({"careers": "careers@nimbus.com"}))
    assert contact.address == "careers@nimbus.com" and "via posting address" in note


def test_web_lookup_needs_verification(fake_api):
    fake_api.reply("The official website is nimbushq.io.")
    job = make_job()
    good = site({"": "<title>Nimbus | Cloud for clinics</title>", "careers": "jobs@nimbushq.io"})
    contact, note = contacts.best_contact(job, allow_lookup=True, client=good)
    assert contact.address == "jobs@nimbushq.io" and "web search (verified)" in note
    sent = fake_api.sent[0]["body"]
    assert sent["tools"][0]["type"] == "web_search_20260209"
    assert sent["messages"][0]["content"] == "Company: Nimbus\nLocation: Bengaluru"  # never the posting text

    fake_api.reply("acme-widgets.com")
    wrong = site({"": "<title>Acme Widgets</title>", "careers": "jobs@acme-widgets.com"})
    assert contacts.best_contact(job, allow_lookup=True, client=wrong) == (None, "no company website or hiring address found")


# ---------- drafting ----------


def write(fake_api, profile, body: str, **job_fields) -> Email:
    fake_api.reply_when("Write the email", json.dumps({"subject": "Backend Engineer - Sample Candidate", "body": body}))
    return compose.write(make_job(**job_fields), profile, ANSWERS, system=compose.system_prompt(profile),
                         to_address="talent@nimbus.com", source="job post", company_key="nimbus",
                         attachment="data/resumes/job-1/Sample_Candidate_Resume.pdf")


def test_good_draft_gets_signature_and_opt_out(fake_api, profile):
    email = write(fake_api, profile, GOOD_BODY)
    assert email.status == EmailStatus.DRAFT and email.note is None
    assert email.body.count("Best regards") == 1  # the model's own sign-off was stripped
    assert email.body.endswith(compose.OPT_OUT)
    assert "+91 90000 00000 · sample.candidate@example.com" in email.body
    content = fake_api.sent[0]["body"]["messages"][0]["content"]
    assert "has not applied yet" in content and fake_api.sent[0]["body"]["output_config"]["effort"] == "low"


def test_applied_jobs_say_so(fake_api, profile):
    write(fake_api, profile, GOOD_BODY, state=State.APPLIED, source="naukri")
    assert "already applied for this role on Naukri" in fake_api.sent[0]["body"]["messages"][0]["content"]


def test_fabricated_draft_is_rejected(fake_api, profile):
    email = write(fake_api, profile, "Dear Hiring Team,\n\nI have 5 years of experience with Terraform at Google.")
    assert email.status == EmailStatus.REJECTED
    assert "5 years" in email.note and "Terraform" in email.note and "Google" in email.note


# ---------- outbox ----------


@pytest.fixture
def outbox_world(monkeypatch, tmp_path):
    db.configure(tmp_path / "jobs.db")
    monkeypatch.setenv("GMAIL_ADDRESS", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    sent: list = []
    monkeypatch.setattr(gmail, "send", sent.append)
    config = Config(reviewed=True, search=Search(roles=["x"], cities=["Pune"]),
                    mode=Mode(dry_run=False), outreach=Outreach(enabled=True))
    return config, sent


def add_email(**fields) -> Email:
    defaults = dict(job_id=1, company_key="nimbus", to_address="talent@nimbus.com", contact_source="job post",
                    subject="Backend Engineer - Sample", body="Dear Hiring Team, …")
    with db.session() as session:
        email = Email(**(defaults | fields))
        session.add(email)
        session.commit()
        return email


def emails() -> list[Email]:
    with db.session() as session:
        return session.exec(select(Email).order_by(Email.id)).all()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda c: setattr(c.mode, "paused", True), "paused"),
        (lambda c: setattr(c.mode, "dry_run", True), "dry run"),
        (lambda c: setattr(c.outreach, "enabled", False), "outreach is off"),
    ],
)
def test_sending_blocked_by_config(outbox_world, change, reason):
    config, _ = outbox_world
    change(config)
    with db.session() as session:
        assert reason in outbox.sending_blocked(session, config, WEEKDAY_11AM)


def test_sending_window_and_credentials(outbox_world, monkeypatch):
    config, _ = outbox_world
    with db.session() as session:
        assert outbox.sending_blocked(session, config, WEEKDAY_11AM) is None
        assert "outside sending hours" in outbox.sending_blocked(session, config, WEEKDAY_11AM.replace(hour=9))
        assert "outside sending hours" in outbox.sending_blocked(session, config, datetime(2026, 10, 17, 11, tzinfo=IST))
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "REPLACE_ME")
        assert "no Gmail App Password" in outbox.sending_blocked(session, config, WEEKDAY_11AM)


def test_sends_one_draft_per_tick_with_attachment(outbox_world, profile, tmp_path, monkeypatch):
    config, sent = outbox_world
    resume = tmp_path / "Sample_Candidate_Resume.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    add_email(attachment=str(resume))
    add_email(to_address="careers@orbit.com", company_key="orbit")
    with db.session() as session:
        outbox.send_next(session, config, profile, now=WEEKDAY_11AM, log=lambda _: None)
    assert len(sent) == 1
    message = sent[0]
    assert message["To"] == "talent@nimbus.com" and message["From"] == "Sample Candidate <me@gmail.com>"
    assert [p.get_filename() for p in message.iter_attachments()] == ["Sample_Candidate_Resume.pdf"]
    first, second = emails()
    assert first.status == EmailStatus.SENT and first.message_id == message["Message-ID"]
    assert second.status == EmailStatus.DRAFT  # next tick


def test_daily_cap(outbox_world, profile):
    config, sent = outbox_world
    config.caps.emails_per_day = 1
    add_email(status=EmailStatus.SENT, sent_at=utcnow())
    add_email(to_address="careers@orbit.com", company_key="orbit")
    with db.session() as session:
        assert "daily email cap (1)" in outbox.sending_blocked(session, config, WEEKDAY_11AM)


def test_suppressed_expired_and_replied_drafts_are_not_sent(outbox_world, profile):
    config, sent = outbox_world
    with db.session() as session:
        session.add(Suppression(address="@bounced.com", reason="bounced"))
        session.commit()
    add_email(to_address="hr@bounced.com", company_key="bounced")
    add_email(to_address="hr@old.com", company_key="old", created_at=utcnow() - timedelta(days=8))
    add_email(to_address="hr@nimbus.com", status=EmailStatus.SENT, sent_at=utcnow(), replied_at=utcnow())
    add_email(to_address="talent@nimbus.com", kind="follow_up")
    with db.session() as session:
        outbox.send_next(session, config, profile, now=WEEKDAY_11AM, log=lambda _: None)
    assert sent == []
    statuses = [(e.to_address, e.status) for e in emails()]
    assert statuses == [("hr@bounced.com", "cancelled"), ("hr@old.com", "expired"),
                        ("hr@nimbus.com", "sent"), ("talent@nimbus.com", "cancelled")]


def test_follow_up_after_business_days(outbox_world, profile):
    config, _ = outbox_world
    add_email(status=EmailStatus.SENT, sent_at=utcnow() - timedelta(days=10), message_id="<first@gmail.com>")
    add_email(to_address="hr@orbit.com", company_key="orbit", status=EmailStatus.SENT, sent_at=utcnow() - timedelta(days=2))
    with db.session() as session:
        assert outbox.queue_follow_ups(session, config, ANSWERS, profile) == 1
        assert outbox.queue_follow_ups(session, config, ANSWERS, profile) == 0  # only one follow-up, ever
    follow = emails()[-1]
    assert follow.kind == "follow_up" and follow.in_reply_to == "<first@gmail.com>"
    assert follow.subject.startswith("Re: ") and follow.body.endswith(compose.OPT_OUT)


class FakeMailbox:
    def __init__(self, messages: dict[str, list[tuple[str, str]]]):
        self.messages = messages  # query substring → [(subject, text)]

    def search(self, query: str) -> list[bytes]:
        return [f"{key}|{i}".encode() for key, found in self.messages.items() if key in query for i in range(len(found))]

    def read(self, uid: bytes) -> tuple[str, str]:
        key, _, index = uid.decode().partition("|")
        return self.messages[key][int(index)]


def test_inbox_marks_replies_bounces_and_opt_outs(outbox_world):
    add_email(status=EmailStatus.SENT, sent_at=utcnow(), to_address="talent@nimbus.com")
    add_email(status=EmailStatus.SENT, sent_at=utcnow(), to_address="hr@orbit.com", company_key="orbit")
    add_email(status=EmailStatus.SENT, sent_at=utcnow(), to_address="jobs@gone.com", company_key="gone")
    mailbox = FakeMailbox({
        "from:nimbus.com": [("Re: Backend Engineer", "Thanks Sample, let's talk on Monday.")],
        "from:orbit.com": [("Re: hello", "No\n\nOn Thu, Sample wrote: …")],
        "mailer-daemon": [("Delivery Status Notification", "Your message wasn't delivered to jobs@gone.com")],
    })
    with db.session() as session:
        events = outbox.check_inbox(session, mailbox)
        suppressed = {s.address: s.reason for s in session.exec(select(Suppression)).all()}
    assert events == ["reply from nimbus.com", "reply from orbit.com", "bounced: jobs@gone.com"]
    nimbus, orbit, gone = emails()
    assert nimbus.replied_at and "let's talk on Monday" in nimbus.reply_snippet
    assert orbit.replied_at and gone.bounced_at
    assert suppressed == {"hr@orbit.com": "asked not to be contacted", "jobs@gone.com": "bounced"}


def test_site_reachable_only_on_www():
    """Live 2026-10-15: some company sites 404 or refuse connections unless you use www."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "acme.com":
            raise httpx.ConnectError("refused")
        if request.url.path.strip("/") == "careers":
            return httpx.Response(200, html='<a href="mailto:careers@acme.com">Careers</a>')
        return httpx.Response(200, text="" if request.url.path == "/robots.txt" else "<title>Acme</title>")
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    found = contacts.scrape_site("acme.com", client)
    assert [(c.address, c.source) for c in found] == [("careers@acme.com", "https://www.acme.com/careers")]


def test_imap_queries_with_quotes_are_escaped():
    """Live 2026-10-15: subject:"…" inside the X-GM-RAW string made Gmail reply "Could not parse command"."""
    assert gmail.imap_quote('subject:"Job agent test" newer_than:1d') == '"subject:\\"Job agent test\\" newer_than:1d"'
    assert gmail.imap_quote("from:acme.com after:2026/10/15") == '"from:acme.com after:2026/10/15"'


def test_concurrent_outbox_runs_never_double_send(outbox_world, profile, monkeypatch, tmp_path):
    import fcntl

    config, sent = outbox_world
    monkeypatch.setattr(outbox, "LOCK_PATH", tmp_path / "outbox.lock")
    add_email()
    logs: list[str] = []
    with (tmp_path / "outbox.lock").open("w") as held:  # another outbox run holds the lock
        fcntl.flock(held, fcntl.LOCK_EX)
        with db.session() as session:
            outbox.tick(session, config, ANSWERS, profile, log=logs.append)
    assert sent == [] and logs == ["outbox: another outbox run is in progress; skipping"]
