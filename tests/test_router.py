"""Routing jobs to auto-apply (Greenhouse/Lever) or the "Needs you" queue, and the copy-ready answers."""

import httpx
import pytest

from jobagent.apply import router
from jobagent.apply.answers import copy_ready_answers
from jobagent.models import Job


def job(**fields) -> Job:
    return Job(**({"dedupe_key": "k", "source": "linkedin", "external_id": "1", "url": "https://www.linkedin.com/jobs/view/1",
                   "title": "t", "company": "c"} | fields))


@pytest.mark.parametrize(
    ("url", "ats"),
    [
        ("https://job-boards.eu.greenhouse.io/acme/jobs/4001", "greenhouse"),
        ("https://boards.greenhouse.io/acme/jobs/1", "greenhouse"),
        ("https://jobs.lever.co/acme/5ef858da", "lever"),
        ("https://examplecorp.wd1.myworkdayjobs.com/en-US/careers/job/1", "workday"),
        ("https://exampleco.taleo.net/careersection/1", "taleo"),
        ("https://examplestartup.keka.com/careers/jobdetails/1", "keka"),
        ("https://www.linkedin.com/jobs/view/1", "linkedin"),
        ("https://careers.example.com/global/en/job/1", "company site"),
    ],
)
def test_classify(url, ats):
    assert router.classify(url) == ats


def test_route_prefers_company_link_and_resolves_short_links(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    monkeypatch.setattr(router, "resolve", lambda url, client: "https://job-boards.eu.greenhouse.io/acme/jobs/5")
    assert router.route(job(apply_url="https://grnh.se/abc"), client) == (
        "greenhouse", "https://job-boards.eu.greenhouse.io/acme/jobs/5")
    assert router.route(job(), client) == ("linkedin", "https://www.linkedin.com/jobs/view/1")
    lever = job(source="lever", url="https://jobs.lever.co/acme/1", apply_url="https://jobs.lever.co/acme/1/apply")
    assert router.route(lever, client) == ("lever", "https://jobs.lever.co/acme/1/apply")


def test_how_to_apply_never_suggests_linkedin_automation():
    assert "never automated" in router.how_to_apply("linkedin")
    assert router.how_to_apply("company site").startswith("Company careers site")
    assert router.how_to_apply("smartrecruiters").startswith("Smartrecruiters")


def test_copy_ready_answers_skip_blanks_and_format():
    answers = {"personal": {"full_name": "Asha Rao", "linkedin_url": "TODO"},
               "compensation": {"current_ctc_lpa": 10.0}, "notice": {"immediate_joiner": True},
               "preferences": {"preferred_cities": ["Pune", "Indore"]}}
    assert copy_ready_answers(answers) == [
        ("Full name", "Asha Rao"), ("Current CTC (LPA)", "10"), ("Immediate joiner", "Yes"),
        ("Preferred cities", "Pune, Indore"),
    ]
