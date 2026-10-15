"""The whole run against a temporary database, with discovery and Claude mocked."""

import json
from pathlib import Path

import pytest
from sqlmodel import select

from conftest import SAMPLE_PROFILE
from jobagent import db, pipeline, schedule
from jobagent.config import Config, Search
from jobagent.discover import ats_boards, jobspy_src
from jobagent.models import Job, Run, State
from jobagent.profile import load_profile

GOOD_JD = "Backend role. Must have 3-5 years experience with Python, Django, PostgreSQL and Kafka."


def row(job_id: str, site: str, title: str, company: str, description: str, location="Bengaluru, India") -> dict:
    return {"id": job_id, "site": site, "job_url": f"https://{site}.example/{job_id}", "title": title,
            "company": company, "location": location, "description": description, "date_posted": None}


ROWS = [
    row("1", "naukri", "Backend Developer", "Acme Pvt Ltd", GOOD_JD + " Mail hr@acme.in"),
    row("2", "linkedin", "Backend Developer", "Acme", GOOD_JD),  # same opening as 1
    row("3", "indeed", "Frontend Developer", "Pixel", "React, CSS and Figma."),  # no skill overlap
    row("4", "glassdoor", "Backend Developer", "Faraway", GOOD_JD, location="Chennai, India"),  # location
    row("5", "naukri", "Data Engineer", "Scam Co", "Python, Kafka. Pay a registration fee to apply."),
]


@pytest.fixture
def world(monkeypatch, tmp_path, fake_api):
    db.configure(tmp_path / "jobs.db")
    monkeypatch.setattr(pipeline, "apply_naukri", lambda *a, **k: None)  # Naukri stage: see naukri_world
    monkeypatch.setattr(pipeline, "draft_outreach", lambda *a, **k: [])  # outreach stage: see test_outreach
    config = Config(reviewed=True, search=Search(roles=["Backend Developer"], cities=["Bengaluru"]))

    def fake_discover(search, batch, smoke=False):
        for r in ROWS:
            batch.jobs.append(jobspy_src.from_jobspy(r, query_city="Bengaluru"))
            batch.counts[r["site"]] += 1

    monkeypatch.setattr(jobspy_src, "discover", fake_discover)
    monkeypatch.setattr(ats_boards, "discover", lambda *a: None)
    monkeypatch.setattr(pipeline, "load_config", lambda: config)
    monkeypatch.setattr(schedule, "load_config", lambda: config)
    monkeypatch.setattr(pipeline, "load_answers", lambda: {"employment": {"total_experience_years": 4}})
    monkeypatch.setattr(pipeline, "_real_profile", lambda: load_profile(SAMPLE_PROFILE))
    fake_api.reply(json.dumps({"score": 84, "decision": "apply", "matched_must_haves": ["Python", "Kafka"],
                               "missing_must_haves": [], "seniority_fit": "fit", "reason": "Strong match."}))
    return config, fake_api


def jobs_by_company() -> dict[str, Job]:
    with db.session() as session:
        return {job.company: job for job in session.exec(select(Job)).all()}


def test_full_run(world):
    _, api = world
    record = pipeline.run(log=lambda _: None)

    assert (record.found, record.new, record.duplicates) == (5, 4, 1)
    assert record.scored == 1 and record.matches == 1
    jobs = jobs_by_company()
    acme = jobs["Acme Pvt Ltd"]
    assert acme.state == State.SCORED and acme.score == 84
    assert acme.also_seen_at == ["https://linkedin.example/2"]
    assert acme.emails == ["hr@acme.in"]
    assert jobs["Pixel"].skip_reason == "only 0 of your skills mentioned"
    assert jobs["Faraway"].skip_reason == "location: Chennai, India"
    assert jobs["Scam Co"].skip_reason == "possible scam: asks for money (registration fee)"

    # one Claude call, for Acme only; profile cached in system, JD in the user turn
    assert len(api.sent) == 1
    body = api.sent[0]["body"]
    assert "Example Payments Pvt Ltd" in body["system"][0]["text"]
    assert "Acme" in body["messages"][0]["content"] and "Acme" not in body["system"][0]["text"]


def test_second_run_never_rescores(world):
    _, api = world
    pipeline.run(log=lambda _: None)
    second = pipeline.run(log=lambda _: None)
    assert (second.new, second.duplicates, second.scored) == (0, 5, 0)
    assert len(api.sent) == 1
    with db.session() as session:
        assert len(session.exec(select(Run)).all()) == 2


def test_missing_description_is_skipped_honestly_then_recovered(world, monkeypatch):
    _, api = world
    batches = [
        [row("20", "linkedin", "ML Engineer", "Orbit", "")],  # LinkedIn withheld the text
        [row("21", "indeed", "ML Engineer", "Orbit", GOOD_JD)],  # same opening, with text
    ]

    def fake_discover(search, batch, smoke=False):
        for r in batches.pop(0):
            batch.jobs.append(jobspy_src.from_jobspy(r, query_city="Bengaluru"))

    monkeypatch.setattr(jobspy_src, "discover", fake_discover)
    pipeline.run(log=lambda _: None)
    assert jobs_by_company()["Orbit"].skip_reason == "linkedin returned no job description"
    assert not api.sent

    pipeline.run(log=lambda _: None)
    orbit = jobs_by_company()["Orbit"]
    assert orbit.score == 84 and orbit.url.startswith("https://linkedin")
    assert orbit.state == State.TAILORED  # a LinkedIn match gets its own resume
    assert len(api.sent) == 2  # score + tailor


def test_budget_cap_stops_scoring(world):
    config, api = world
    config.caps.usd_per_day = 0
    record = pipeline.run(log=lambda _: None)
    assert record.scored == 0 and not api.sent
    assert any("budget" in note for note in record.notes)
    assert jobs_by_company()["Acme Pvt Ltd"].state == State.FILTERED  # waits for tomorrow


def test_placeholder_profile_skips_scoring(world, monkeypatch):
    monkeypatch.setattr(pipeline, "_real_profile", lambda: None)
    record = pipeline.run(log=lambda _: None)
    assert record.scored == 0 and "sample placeholder" in record.notes[0]


def test_paused_does_nothing(world):
    config, _ = world
    config.mode.paused = True
    assert pipeline.run(log=lambda _: None) is None


def test_schedule_plist_and_review_guard(world):
    config, _ = world
    agents = schedule.agents()
    run, outbox = agents[schedule.RUN_LABEL], agents[schedule.OUTBOX_LABEL]
    assert run["ProgramArguments"][-1] == "run" and outbox["ProgramArguments"][-1] == "outbox"
    assert [entry["Hour"] for entry in run["StartCalendarInterval"]] == schedule.RUN_HOURS
    assert outbox["StartInterval"] == 900
    assert all("Documents" not in str(a["WorkingDirectory"]) for a in agents.values())  # macOS blocks launchd there
    config.reviewed = False
    with pytest.raises(RuntimeError, match="reviewed: false"):
        schedule.install()


TAILOR_PLAN = json.dumps({
    "summary": "Backend engineer with 4 years of experience in Python, Go and Kafka.",
    "bullet_ids": ["exp1.2", "exp1.1", "exp2.1", "exp2.3"],
    "skills": ["Kafka", "Python", "PostgreSQL"],
    "rationale": "Kafka and Python first: the posting's core stack.",
})


@pytest.fixture
def tailoring_world(world, monkeypatch, tmp_path):
    """Two equal matches: one only on Naukri (profile apply), one on LinkedIn (needs a tailored PDF)."""
    config, api = world
    rows = [
        row("10", "naukri", "Backend Developer", "Acme", GOOD_JD),
        row("11", "linkedin", "Backend Engineer", "Nimbus", GOOD_JD),
    ]

    def fake_discover(search, batch, smoke=False):
        for r in rows:
            batch.jobs.append(jobspy_src.from_jobspy(r, query_city="Bengaluru"))
            batch.counts[r["site"]] += 1

    monkeypatch.setattr(jobspy_src, "discover", fake_discover)
    api.reply_when("You screen", api.body["content"][0]["text"])  # score reply from `world`
    api.reply_when("You tailor", TAILOR_PLAN)
    return config, api


def test_run_tailors_matches_that_need_a_pdf(tailoring_world):
    record = pipeline.run(log=lambda _: None)
    jobs = jobs_by_company()

    assert record.matches == 2 and record.tailored == 1
    assert jobs["Acme"].state == State.SCORED and jobs["Acme"].resume_path is None  # Naukri profile apply
    nimbus = jobs["Nimbus"]
    assert nimbus.state == State.TAILORED
    assert nimbus.tailored_summary.startswith("Backend engineer with 4 years")
    assert nimbus.tailor_rationale.startswith("Kafka and Python first")
    pdf = Path(nimbus.resume_path)
    assert pdf.name == "Sample_Candidate_Resume.pdf" and pdf.read_bytes().startswith(b"%PDF")


def test_tailoring_failure_falls_back_to_original_resume(tailoring_world):
    _, api = tailoring_world
    api.reply_when("You tailor", '{"summary": "Backend', stop_reason="max_tokens")
    record = pipeline.run(log=lambda _: None)
    nimbus = jobs_by_company()["Nimbus"]
    assert nimbus.state == State.TAILORED and nimbus.tailored_summary is None
    assert "tailoring failed" in nimbus.tailor_notes and "using your original resume" in nimbus.tailor_notes
    assert Path(nimbus.resume_path).exists()
    assert any("max_tokens" in e for e in record.errors)


NAUKRI_FIXTURES = Path(__file__).parent / "fixtures" / "naukri"
REAL_APPLY_NAUKRI = pipeline.apply_naukri


@pytest.fixture
def naukri_world(world, monkeypatch, tmp_path):
    """Naukri jobs whose pages are local fixtures, opened in headless Chromium instead of the user's Chrome."""
    from contextlib import contextmanager

    from playwright.sync_api import sync_playwright

    config, api = world
    logged_in = {"value": True}

    @contextmanager
    def headless_context():
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            yield browser.new_context()
            browser.close()

    monkeypatch.setattr(pipeline, "apply_naukri", REAL_APPLY_NAUKRI)
    monkeypatch.setattr(pipeline.browser, "naukri_context", headless_context)
    monkeypatch.setattr(pipeline.naukri, "is_logged_in", lambda page: logged_in["value"])
    monkeypatch.setattr(pipeline.naukri, "SETTLE_MS", (0, 0))
    monkeypatch.setattr(pipeline, "PAUSE_SECONDS", (0, 0))

    def use_pages(*names: str) -> None:
        rows = []
        for i, name in enumerate(names):
            r = row(str(30 + i), "naukri", f"Backend Developer {i}", f"Naukri Co {i}", GOOD_JD)
            r["job_url"] = (NAUKRI_FIXTURES / f"{name}.html").as_uri()
            rows.append(r)

        def fake_discover(search, batch, smoke=False):
            batch.jobs.extend(jobspy_src.from_jobspy(r, query_city="Bengaluru") for r in rows)

        monkeypatch.setattr(jobspy_src, "discover", fake_discover)

    def go_live() -> None:
        config.mode.dry_run = False
        monkeypatch.setattr(pipeline.guards.setup_status, "ready_for_live", lambda: True)

    return config, use_pages, go_live, logged_in


def test_dry_run_applies_nothing(naukri_world):
    config, use_pages, _, _ = naukri_world
    use_pages("apply_instant")
    record = pipeline.run(log=lambda _: None)
    job = jobs_by_company()["Naukri Co 0"]
    assert record.applied == 0 and job.state == State.SCORED and job.applied_at is None
    assert job.dry_run_at is not None and job.apply_note.startswith("dry run")
    assert any("dry run" in note for note in record.notes)


def test_live_needs_setup_complete_even_if_dry_run_is_off(naukri_world):
    config, use_pages, _, _ = naukri_world
    use_pages("apply_instant")
    config.mode.dry_run = False  # but setup_status isn't all green in tests
    record = pipeline.run(log=lambda _: None)
    assert record.applied == 0 and any("setup incomplete" in note for note in record.notes)


def test_live_apply_records_outcomes(naukri_world):
    config, use_pages, go_live, _ = naukri_world
    use_pages("apply_instant", "company_site", "closed")
    go_live()
    record = pipeline.run(log=lambda _: None)
    jobs = jobs_by_company()

    assert record.applied == 1
    applied = jobs["Naukri Co 0"]
    assert applied.state == State.APPLIED and applied.applied_at is not None
    assert Path(applied.apply_screenshot).exists()
    company_site = jobs["Naukri Co 1"]
    assert company_site.state == State.SCORED and company_site.apply_url == company_site.url  # tailored next run
    assert jobs["Naukri Co 2"].skip_reason == "closed on Naukri"


def test_one_off_live_override(naukri_world, monkeypatch):
    config, use_pages, _, _ = naukri_world
    use_pages("apply_instant", "apply_instant")
    pipeline.run(log=lambda _: None)  # dry run by config: both jobs dry-run only
    jobs = jobs_by_company()
    first, second = jobs["Naukri Co 0"], jobs["Naukri Co 1"]
    assert first.state == second.state == State.SCORED

    with pytest.raises(ValueError):
        pipeline.apply_only(live=True)  # never a bulk live run
    assert pipeline.apply_only(job_id=first.id, live=True, log=lambda _: None).applied == 0  # setup not green
    monkeypatch.setattr(pipeline.guards.setup_status, "ready_for_live", lambda: True)
    assert pipeline.apply_only(job_id=first.id, live=True, log=lambda _: None).applied == 1

    jobs = jobs_by_company()
    assert jobs["Naukri Co 0"].state == State.APPLIED and jobs["Naukri Co 1"].state == State.SCORED
    assert config.mode.dry_run is True  # the user's config is untouched


def test_daily_cap(naukri_world):
    config, use_pages, go_live, _ = naukri_world
    use_pages("apply_instant", "apply_instant")
    go_live()
    config.caps.naukri_applies_per_day = 1
    first = pipeline.run(log=lambda _: None)
    second = pipeline.run(log=lambda _: None)
    assert (first.applied, second.applied) == (1, 0)
    assert any("daily cap (1) reached" in note for note in second.notes)


def test_logged_out_applies_nothing(naukri_world):
    config, use_pages, go_live, logged_in = naukri_world
    use_pages("apply_instant")
    go_live()
    logged_in["value"] = False
    record = pipeline.run(log=lambda _: None)
    assert record.applied == 0 and any("jobagent naukri login" in note for note in record.notes)
    assert jobs_by_company()["Naukri Co 0"].apply_note is None


def test_two_failures_in_a_row_stop_naukri(naukri_world):
    config, use_pages, go_live, _ = naukri_world
    use_pages("unknown", "unknown", "apply_instant")
    go_live()
    record = pipeline.run(log=lambda _: None)
    assert record.applied == 0  # stopped before reaching the third job
    assert any("2 failures in a row" in note for note in record.notes)


REAL_DRAFT_OUTREACH = pipeline.draft_outreach
ACME_EMAIL = json.dumps({"subject": "Backend Developer - Sample Candidate", "body": (
    "Dear Hiring Team,\n\nI'm writing about the Backend Developer role at Acme. I built a refund-reconciliation "
    "service in Go processing 2M transactions/day, cutting manual finance escalations by 60%.\n\n"
    "My resume is attached. Could we have a short call?")})


@pytest.fixture
def outreach_world(world, monkeypatch):
    config, api = world
    monkeypatch.setattr(pipeline, "draft_outreach", REAL_DRAFT_OUTREACH)
    rows = [
        row("40", "naukri", "Backend Developer", "Acme Pvt Ltd", GOOD_JD + " Share your CV at careers@acme.in"),
        row("41", "linkedin", "Backend Engineer", "Nimbus", GOOD_JD),  # no address, no website
    ]
    monkeypatch.setattr(jobspy_src, "discover", lambda search, batch, smoke=False: batch.jobs.extend(
        jobspy_src.from_jobspy(r, query_city="Bengaluru") for r in rows))
    api.reply_when("You screen", api.body["content"][0]["text"])
    api.reply_when("Write the email", ACME_EMAIL)
    api.reply_when("official website", "NONE")
    return config, api


def test_run_drafts_outreach_once_per_company(outreach_world):
    record = pipeline.run(log=lambda _: None)
    jobs = jobs_by_company()
    with db.session() as session:
        drafts = session.exec(select(pipeline.Email)).all()

    assert record.drafted == 1 and len(drafts) == 1
    email = drafts[0]
    assert (email.to_address, email.status, email.contact_source) == ("careers@acme.in", "draft", "job post")
    assert email.body.startswith("Dear Hiring Team") and email.attachment.endswith("Sample_Candidate_Resume.pdf")
    assert jobs["Acme Pvt Ltd"].outreach_note == "careers@acme.in from job post"
    assert jobs["Nimbus"].outreach_note == "no company website or hiring address found"

    again = pipeline.run(log=lambda _: None)
    assert again.drafted == 0  # contact-finding already ran for both jobs, and Acme is in its cooldown


def test_drafts_never_outnumber_a_days_sending(outreach_world):
    config, _ = outreach_world
    config.caps.emails_per_day = 0
    assert pipeline.run(log=lambda _: None).drafted == 0
    assert jobs_by_company()["Acme Pvt Ltd"].outreach_note is None  # left for a later run


def test_dashboard_outreach_and_cancel(outreach_world):
    pipeline.run(log=lambda _: None)
    app = open_dashboard(jobs_by_company()["Acme Pvt Ltd"].id)
    assert not app.exception, app.exception
    assert any("careers@acme.in from job post" in c.value for c in app.caption)
    cancel = next(b for b in app.button if b.label == "Cancel this email")
    cancel.click().run()
    assert not app.exception, app.exception
    with db.session() as session:
        assert session.exec(select(pipeline.Email)).one().status == "cancelled"


def open_dashboard(job_id: int | None = None):
    """Run the dashboard script headless. Streamlit's data cache is process-wide, so clear it per test."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    app = AppTest.from_file(str(db.settings.ROOT / "dashboard" / "app.py"), default_timeout=30)
    if job_id is not None:
        app.query_params["job"] = str(job_id)
    return app.run()


def test_dashboard_shows_tailoring(tailoring_world):
    pipeline.run(log=lambda _: None)
    app = open_dashboard(jobs_by_company()["Nimbus"].id)
    assert not app.exception, app.exception
    assert any("Tailored resume" in m.value for m in app.markdown)
    assert any("Backend engineer with 4 years" in m.value for m in app.markdown)
    assert len(app.get("download_button")) == 2


def test_dashboard_renders(world):
    pipeline.run(log=lambda _: None)
    app = open_dashboard()
    assert not app.exception, app.exception
    assert any("1 jobs at or above score" in c.value for c in app.caption)

    acme_id = jobs_by_company()["Acme Pvt Ltd"].id
    app.query_params["job"] = str(acme_id)
    app.run()
    assert not app.exception, app.exception
    assert app.subheader[0].value == "Backend Developer · Acme Pvt Ltd"
    assert app.info[0].value == "Strong match."
    assert any("hr@acme.in" in m.value for m in app.markdown)
    # missing values must not leak through as "nan" text or as links to nowhere
    assert [m.value for m in app.main.metric] == ["84", "fit", "–", "–"]
    assert not any("Company apply page" in m.value for m in app.markdown)
    assert not app.warning  # a scored job has no "Skipped: …" box

    pixel_id = jobs_by_company()["Pixel"].id  # skipped, never scored
    app.query_params["job"] = str(pixel_id)
    app.run()
    assert not app.exception, app.exception
    assert [m.value for m in app.main.metric][0] == "–"
    assert not any("Strong match" in i.value for i in app.info)  # no score reason for an unscored job
    assert app.warning[0].value == "Skipped: only 0 of your skills mentioned"
