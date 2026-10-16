"""Greenhouse and Lever forms: planning answers (mock Claude) and filling local copies of the real forms."""

import json
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from conftest import SAMPLE_PROFILE
from jobagent.apply import evidence, greenhouse, lever
from jobagent.apply.answers import AnswerMapper, Unanswerable
from jobagent.models import Job
from jobagent.profile import load_profile

FIXTURES = Path(__file__).parent / "fixtures"
ANSWERS = {
    "personal": {"full_name": "Asha Rao", "email": "asha@example.com", "phone": "+91-9000000000",
                 "current_city": "Indore"},
    "employment": {"current_company": "Example Payments Pvt Ltd"},
    "compensation": {"current_ctc_lpa": 10.0},
    "preferences": {"ok_with_work_from_office": True},
    "other": {"relatives_at_hiring_company": False},
}
GH_QUESTIONS = [
    {"label": "First Name", "required": True, "fields": [{"name": "first_name", "type": "input_text"}]},
    {"label": "Last Name", "required": True, "fields": [{"name": "last_name", "type": "input_text"}]},
    {"label": "Email", "required": True, "fields": [{"name": "email", "type": "input_text"}]},
    {"label": "Phone", "required": True, "fields": [{"name": "phone", "type": "input_text"}]},
    {"label": "Resume/CV", "required": True, "fields": [{"name": "resume", "type": "input_file"}]},
    {"label": "Cover Letter", "required": False, "fields": [{"name": "cover_letter", "type": "input_file"}]},
    {"label": "Do you have relatives working at Acme?", "required": True,
     "fields": [{"name": "question_1", "type": "multi_value_single_select", "values": [{"label": "Yes"}, {"label": "No"}]}]},
    {"label": "Current Compensation", "required": False, "fields": [{"name": "question_2", "type": "input_text"}]},
]


def mapping(key: str, unit: str = "as_is", option: str = "") -> str:
    return json.dumps({"key": key, "unit": unit, "option": option, "free_text": ""})


@pytest.fixture
def mapper():
    return AnswerMapper(ANSWERS, load_profile(SAMPLE_PROFILE))


@pytest.fixture
def resume(tmp_path):
    path = tmp_path / "Asha_Rao_Resume.pdf"
    path.write_bytes(b"%PDF-1.4 test")
    return path


@pytest.fixture
def page(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "SHOTS_DIR", tmp_path / "shots")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        yield browser.new_page()
        browser.close()


def make_job(ats: str, url: str) -> Job:
    return Job(id=5, dedupe_key="k", source="linkedin", external_id="1", url=url, title="Backend Engineer",
               company="Acme", ats=ats, apply_target=url)


# ---------- Greenhouse ----------


@pytest.mark.parametrize(
    ("url", "parsed"),
    [
        ("https://job-boards.eu.greenhouse.io/acme/jobs/4001?gh_jid=4001", ("acme", "4001")),
        ("https://boards.greenhouse.io/acme/jobs/42", ("acme", "42")),
        ("https://boards.greenhouse.io/embed/job_app?for=acme&token=42", ("acme", "42")),
        ("https://careers.acme.com/jobs/42", None),
    ],
)
def test_greenhouse_urls(url, parsed):
    assert greenhouse.parse_url(url) == parsed


def test_greenhouse_plan_uses_facts_and_stops_on_required_gaps(fake_api, mapper, resume):
    fake_api.reply_when("relatives working", mapping("other.relatives_at_hiring_company", option="No"))
    fake_api.reply_when("Current Compensation", mapping("compensation.current_ctc_lpa"))
    fills = greenhouse.plan(GH_QUESTIONS, mapper, ANSWERS, make_job("greenhouse", "u"), resume)
    assert [(f.field, f.value) for f in fills] == [
        ("first_name", "Asha"), ("last_name", "Rao"), ("email", "asha@example.com"), ("phone", "9000000000"),
        ("resume", str(resume)), ("question_1", "No"), ("question_2", "10 LPA"),
    ]  # no cover letter: optional and no file for it

    fake_api.reply_when("relatives working", mapping("UNKNOWN"))
    with pytest.raises(Unanswerable, match="required question 'Do you have relatives"):
        greenhouse.plan(GH_QUESTIONS, mapper, ANSWERS, make_job("greenhouse", "u"), resume)


@pytest.mark.parametrize("live", [False, True])
def test_greenhouse_fills_and_only_submits_when_live(fake_api, mapper, resume, page, monkeypatch, live):
    monkeypatch.setattr(greenhouse, "fetch_questions", lambda board, job_id, client: GH_QUESTIONS)
    monkeypatch.setattr(greenhouse, "parse_url", lambda url: ("acme", "1"))
    fake_api.reply_when("relatives working", mapping("other.relatives_at_hiring_company", option="No"))
    fake_api.reply_when("Current Compensation", mapping("compensation.current_ctc_lpa"))
    job = make_job("greenhouse", (FIXTURES / "greenhouse" / "form.html").as_uri())

    outcome = greenhouse.apply(page, job, mapper, ANSWERS, resume, live=live, client=None)

    assert outcome.status == ("applied" if live else "dry_run"), outcome.note
    assert page.evaluate("window.submitted") is live
    if not live:  # the filled form is still on screen
        assert page.evaluate("window.picked") == {"country": "India +91", "question_1": "No"}  # not "British Indian…"
        assert page.input_value("#phone") == "9000000000" and page.input_value("#question_2") == "10 LPA"
        assert page.eval_on_selector("#resume", "e => e.files[0].name") == "Asha_Rao_Resume.pdf"


def test_greenhouse_required_gap_fills_nothing(fake_api, mapper, resume, page, monkeypatch):
    monkeypatch.setattr(greenhouse, "fetch_questions", lambda board, job_id, client: GH_QUESTIONS)
    monkeypatch.setattr(greenhouse, "parse_url", lambda url: ("acme", "1"))
    fake_api.reply(mapping("UNKNOWN"))
    job = make_job("greenhouse", (FIXTURES / "greenhouse" / "form.html").as_uri())
    outcome = greenhouse.apply(page, job, mapper, ANSWERS, resume, live=True, client=None)
    assert outcome.status == "manual" and "relatives" in outcome.note
    assert page.url == "about:blank"  # decided from the API alone: the form was never opened


# ---------- Lever ----------


@pytest.mark.parametrize("live", [False, True])
def test_lever_fills_and_only_submits_when_live(fake_api, mapper, resume, page, monkeypatch, live):
    fake_api.reply_when("Pune office", mapping("preferences.ok_with_work_from_office", option="Yes"))
    fake_api.reply_when("current salary in LPA", mapping("compensation.current_ctc_lpa", unit="lakhs", option="10"))
    fake_api.reply_when("Anything else", mapping("UNKNOWN"))  # optional: skipped, not a blocker
    monkeypatch.setattr(lever, "apply_url", lambda url: url)  # a local file, not a jobs.lever.co address
    job = make_job("lever", (FIXTURES / "lever" / "form.html").as_uri())

    outcome = lever.apply(page, job, mapper, ANSWERS, resume, live=live)

    assert outcome.status == ("applied" if live else "dry_run"), outcome.note
    assert page.evaluate("window.submitted") is live
    if not live:
        assert page.input_value("input[name=location]") == "Indore, Madhya Pradesh, IND"  # not "Indore, WV, USA"
        assert page.input_value("input[name=org]") == "Example Payments Pvt Ltd"
        assert page.eval_on_selector("input[value=Yes]", "e => e.checked")  # hidden input, chosen via its label
        assert page.eval_on_selector("select", "e => e.selectedOptions[0].text") == "10"
    sources = {qa["question"]: qa["source"] for qa in outcome.qa}
    assert sources["Current location"] == "personal.current_city" and "Anything else? (optional)" not in sources


def test_lever_apply_url():
    assert lever.apply_url("https://jobs.lever.co/acme/123?lever-source=x") == "https://jobs.lever.co/acme/123/apply"
    assert lever.apply_url("https://jobs.lever.co/acme/123/apply") == "https://jobs.lever.co/acme/123/apply"
