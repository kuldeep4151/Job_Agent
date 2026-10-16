"""Answer mapping (no browser) and the Naukri flow against local fixture pages (headless Chromium)."""

import json
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from conftest import SAMPLE_PROFILE
from jobagent.apply import evidence, naukri
from jobagent.apply.answers import AnswerMapper, Question, Unanswerable, flatten, format_value
from jobagent.models import Job
from jobagent.profile import load_profile

FIXTURES = Path(__file__).parent / "fixtures" / "naukri"
ANSWERS = {
    "personal": {"full_name": "Sample Candidate", "linkedin_url": "TODO", "current_city": "Indore"},
    "employment": {"total_experience_years": 4, "current_designation": "Software Engineer II"},
    "compensation": {"current_ctc_lpa": 10.0, "expected_ctc_lpa": 15},
    "notice": {"notice_period_days": 30, "serving_notice": False},
    "preferences": {"willing_to_relocate": True, "preferred_cities": ["Pune", "TODO"]},
    "education": {"degree": "B.Tech, Mechanical Engineering", "degree_specialization": "Mechanical Engineering",
                  "degree_year": 2021, "postgrad_year": 2025},
    "skill_experience_years": {"Python": 4},
}


def mapping(key: str, unit: str = "as_is", option: str = "", free_text: str = "") -> str:
    return json.dumps({"key": key, "unit": unit, "option": option, "free_text": free_text})


def make_job(url: str = "https://www.naukri.com/job-listings-x", **fields) -> Job:
    return Job(id=7, dedupe_key="k", source="naukri", external_id="1", url=url,
               title="Backend Developer", company="Acme", **fields)


@pytest.fixture
def mapper():
    return AnswerMapper(ANSWERS, load_profile(SAMPLE_PROFILE))


# ---------- facts and conversions (pure code) ----------


def test_flatten_skips_placeholders():
    facts = flatten(ANSWERS)
    assert "personal.linkedin_url" not in facts
    assert facts["preferences.preferred_cities"] == ["Pune"]
    assert facts["skill_experience_years.Python"] == 4


@pytest.mark.parametrize(
    ("key", "value", "unit", "expected"),
    [
        ("compensation.current_ctc_lpa", 10.0, "lakhs", "10"),
        ("compensation.current_ctc_lpa", 10.0, "rupees_per_year", "1000000"),
        ("compensation.expected_ctc_lpa", 15, "rupees_per_month", "125000"),
        ("notice.notice_period_days", 30, "months", "1"),
        ("notice.notice_period_days", 0, "days", "0"),
        ("employment.total_experience_years", 1, "months", "12"),
        ("employment.total_experience_years", 0.6, "years", "0.6"),
        ("notice.notice_period_days", 30, "lakhs", "30"),  # nonsense unit for the key → value as stored
        ("preferences.willing_to_relocate", True, "as_is", "Yes"),
    ],
)
def test_format_value(key, value, unit, expected):
    assert format_value(key, value, unit) == expected


# ---------- Claude picks the key, code supplies the value ----------


def test_text_question_uses_the_fact_value(fake_api, mapper):
    fake_api.reply(mapping("compensation.current_ctc_lpa", unit="rupees_per_year"))
    answer = mapper.answer(Question("Current CTC (in Rs)?", "text", []), make_job())
    assert (answer.value, answer.source) == ("1000000", "compensation.current_ctc_lpa")

    body = fake_api.sent[0]["body"]
    keys = body["output_config"]["format"]["schema"]["properties"]["key"]["enum"]
    assert "compensation.current_ctc_lpa" in keys and "UNKNOWN" in keys
    assert "personal.linkedin_url" not in keys  # TODO facts are never offered
    assert "never answer it with total_experience_years" in body["system"][0]["text"]
    assert body["output_config"]["effort"] == "low"


def test_code_overrides_claude_on_yes_no(fake_api, mapper):
    fake_api.reply(mapping("preferences.willing_to_relocate", option="No"))  # wrong option from the model
    answer = mapper.answer(Question("Willing to relocate?", "choice", ["Yes", "No"]), make_job())
    assert answer.value == "Yes"  # from willing_to_relocate: true


def test_option_must_be_on_the_list(fake_api, mapper):
    fake_api.reply(mapping("notice.notice_period_days", option="Immediately"))
    with pytest.raises(Unanswerable):
        mapper.answer(Question("Notice period?", "choice", ["15 days or less", "1 month", "2 months"]), make_job())
    fake_api.reply(mapping("notice.notice_period_days", option="1 month"))
    assert mapper.answer(Question("Notice period?", "choice", ["15 days or less", "1 month"]), make_job()).value == "1 month"


def test_unknown_goes_to_a_human(fake_api, mapper):
    fake_api.reply(mapping("UNKNOWN"))
    with pytest.raises(Unanswerable, match="passport"):
        mapper.answer(Question("Do you hold a valid passport?", "choice", ["Yes", "No"]), make_job())


def test_free_text_is_checked_for_fabrication(fake_api, mapper):
    fake_api.reply(mapping("FREE_TEXT", free_text="I built Kafka consumers and FastAPI services for payments at Acme."))
    assert "Kafka" in mapper.answer(Question("Why this role?", "text", []), make_job()).value

    fake_api.reply(mapping("FREE_TEXT", free_text="I have led Terraform rollouts on GCP for 6 years."))
    with pytest.raises(Unanswerable, match="Terraform"):
        mapper.answer(Question("Why this role?", "text", []), make_job())


# ---------- Naukri pages (local fixtures, headless) ----------


@pytest.fixture
def page(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "SHOTS_DIR", tmp_path / "shots")
    monkeypatch.setattr(naukri, "SETTLE_MS", (0, 0))
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        yield browser.new_page()
        browser.close()


def fixture_url(name: str, variant: str = "") -> str:
    return (FIXTURES / f"{name}.html").as_uri() + (f"#{variant}" if variant else "")


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("logged_out", naukri.PageState.LOGGED_OUT),
        ("already_applied", naukri.PageState.ALREADY_APPLIED),
        ("closed", naukri.PageState.CLOSED),
        ("company_site", naukri.PageState.COMPANY_SITE),
        ("apply_instant", naukri.PageState.APPLY),
        ("unknown", naukri.PageState.UNKNOWN),
    ],
)
def test_detect_state(page, name, state):
    page.goto(fixture_url(name))
    assert naukri.detect_state(page) == state


def test_dry_run_never_clicks_apply(page, mapper):
    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_instant")), mapper, live=False)
    assert outcome.status == "dry_run" and outcome.screenshot.exists()
    assert page.evaluate("window.clicked") is False


def test_live_instant_apply(page, mapper):
    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_instant")), mapper, live=True)
    assert outcome.status == "applied" and page.evaluate("window.clicked") is True


def test_live_questionnaire_fills_facts(fake_api, page, mapper):
    # markers must be question text: words like "relocate" also appear in the facts in every system prompt
    fake_api.reply_when("current CTC (in lakhs)", mapping("compensation.current_ctc_lpa", unit="lakhs"))
    fake_api.reply_when("relocate to Bengaluru", mapping("preferences.willing_to_relocate", option="Yes"))

    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_questionnaire")), mapper, live=True)

    assert outcome.status == "applied", outcome.note
    assert page.evaluate("window.answers") == ["10", "Yes"]
    assert [(qa["answer"], qa["source"]) for qa in outcome.qa] == [
        ("10", "compensation.current_ctc_lpa"),
        ("Yes", "preferences.willing_to_relocate"),
    ]
    assert outcome.qa[1]["options"] == ["Yes", "No"]


def test_real_naukri_checkbox_question_after_greeting(fake_api, page, mapper, monkeypatch, tmp_path):
    """Regression: the first live run read the greeting/typing bubble as an empty question."""
    """Regressions from the first live runs: greeting/typing bubble read as an empty question; hidden
    checkbox inputs; a checkbox answer that sends itself, leaving Save greyed out for the next question."""
    fake_api.reply_when("Year of Graduation", mapping("education.postgrad_year", option="2025"))
    fake_api.reply_when("Current Location (City)", mapping("personal.current_city"))
    fake_api.reply_when("Full-Time Experience", mapping("UNKNOWN"))
    resume = tmp_path / "Sample_Candidate_Resume.pdf"
    resume.write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(naukri.settings, "MASTER_RESUME_PDF", resume)

    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_questionnaire", "graduation")), mapper, live=True)

    assert outcome.status == "applied", outcome.note
    # optional + unknown → skipped; "upload your resume" → the master resume, no Claude call
    assert page.evaluate("window.answers") == ["2025", "Indore", "(skipped)", "Sample_Candidate_Resume.pdf"]
    assert outcome.qa[1]["options"] == [] and outcome.qa[2]["source"] == "SKIPPED"
    assert outcome.qa[3] == {"question": "Please upload your resume.", "options": ["Upload Resume"],
                             "answer": "Sample_Candidate_Resume.pdf", "source": "RESUME_FILE"}
    assert outcome.qa[0] == {"question": "Year of Graduation/Post Graduation", "options": ["2024", "2025", "2026"],
                             "answer": "2025", "source": "education.postgrad_year"}
    asked = [json.dumps(sent["body"]["messages"]) for sent in fake_api.sent]
    assert len(asked) == 3 and not any("thank you for showing interest" in a for a in asked)


def test_profile_update_bot_with_searchable_options(fake_api, page, mapper):
    """Live 2026-10-15: Naukri's profile-update bot; the specialization list shows only a few chips."""
    fake_api.reply_when("undergraduate course", mapping("education.degree", option="B.Tech / B.E."))
    fake_api.reply_when("What specialization", mapping("education.degree_specialization"))
    fake_api.reply_when("date of birth", mapping("UNKNOWN"))
    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_questionnaire", "profile")), mapper, live=True)
    # the unanswerable follow-up would mean "manual", but the reloaded job page says Applied: it went through
    assert outcome.status == "applied", outcome.note
    assert outcome.note.startswith("applied on Naukri; then: Naukri asked 'Please share your date of birth'")
    assert [qa["answer"] for qa in outcome.qa] == ["B.Tech / B.E.", "Mechanical Engineering"]  # typed, then picked
    assert len(fake_api.sent) == 3  # the bot's intro lines are never treated as questions


def test_unanswerable_question_goes_to_manual(fake_api, page, mapper):
    fake_api.reply(mapping("UNKNOWN"))
    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_questionnaire", "passport")), mapper, live=True)
    assert outcome.status == "manual" and "valid passport" in outcome.note
    assert outcome.screenshot.with_suffix(".html").exists()  # page saved for checking selectors


def test_rejected_answer_goes_to_manual(fake_api, page, mapper, monkeypatch):
    monkeypatch.setattr(naukri, "ANSWER_ACCEPT_SECONDS", 1)
    fake_api.reply(mapping("compensation.current_ctc_lpa", unit="lakhs"))
    outcome = naukri.apply_to_job(page, make_job(fixture_url("apply_questionnaire", "reject")), mapper, live=True)
    assert outcome.status == "manual" and "didn't accept" in outcome.note


@pytest.mark.parametrize(
    ("name", "status"),
    [("logged_out", "logged_out"), ("already_applied", "already_applied"), ("closed", "closed"),
     ("company_site", "company_site"), ("unknown", "manual")],
)
def test_non_apply_pages_never_click(page, mapper, name, status):
    assert naukri.apply_to_job(page, make_job(fixture_url(name)), mapper, live=True).status == status


def test_units_in_free_text_boxes(fake_api, mapper):
    fake_api.reply(mapping("compensation.current_ctc_lpa"))
    q = Question("Current Compensation", "text", [])
    assert mapper.answer(q, make_job(), units_in_text=True).value == "10 LPA"
    assert mapper.answer(q, make_job()).value == "10"  # Naukri's chat boxes validate numbers: no unit
    fake_api.reply(mapping("notice.notice_period_days", unit="months"))
    assert mapper.answer(Question("Notice period (months)", "text", []), make_job(), units_in_text=True).value == "1 month"
