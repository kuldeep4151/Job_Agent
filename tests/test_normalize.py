import math
from datetime import date

import pytest

from jobagent.discover.normalize import (
    dedupe_key,
    extract_emails,
    from_greenhouse,
    from_jobspy,
    from_lever,
    html_to_text,
    mentions_city,
    norm_company,
    parse_experience,
    salary_lpa,
)


def jobspy_row(**overrides) -> dict:
    row = {
        "id": "in-123",
        "site": "indeed",
        "job_url": "https://in.indeed.com/viewjob?jk=123",
        "job_url_direct": math.nan,
        "title": "Java/J2EE Developer",
        "company": "TQuanta Technologies Pvt. Ltd.",
        "location": "KA, India",
        "date_posted": date(2026, 10, 15),
        "is_remote": False,
        "min_amount": 2_000_000.0,
        "max_amount": math.nan,
        "interval": "yearly",
        "currency": "INR",
        "emails": math.nan,
        "description": "We need 3-5 years of experience in Java. Mail hr@tquanta.in",
        "skills": math.nan,
        "experience_range": math.nan,
    }
    return row | overrides


@pytest.mark.parametrize(
    ("text", "keyword", "expected"),
    [
        ("3-5 Yrs", False, (3, 5)),
        ("2 to 4 years", False, (2, 4)),
        ("5+ years", False, (5, None)),
        ("Fresher", False, (0, 1)),
        ("Minimum 3 years of experience with Go", True, (3, None)),
        ("4 - 6 years experience in backend", True, (4, 6)),
        ("A company with 20 years of history in retail.", True, (None, None)),
        (None, False, (None, None)),
    ],
)
def test_parse_experience(text, keyword, expected):
    assert parse_experience(text, require_keyword=keyword) == expected


def test_company_and_dedupe_key_ignore_legal_suffixes():
    assert norm_company("TQuanta Technologies Pvt. Ltd.") == norm_company("TQuanta Technologies") == "tquanta technologies"
    indeed = from_jobspy(jobspy_row(), query_city="Bengaluru")
    glassdoor = from_jobspy(
        jobspy_row(site="glassdoor", id="gd-9", company="TQuanta Technologies", location="Bengaluru, India"),
        query_city="Bengaluru",
    )
    assert dedupe_key(indeed) == dedupe_key(glassdoor) == "tquanta technologies|java j2ee developer"


def test_from_jobspy_handles_nan_and_derives_fields():
    job = from_jobspy(jobspy_row(), query_city="Bengaluru")
    assert job.apply_url is None
    assert job.salary_min_lpa == 20.0 and job.salary_max_lpa is None
    assert (job.exp_min, job.exp_max) == (3, 5)  # from the description
    assert job.emails == ["hr@tquanta.in"]
    assert job.posted_at.date() == date(2026, 10, 15)


def test_naukri_experience_and_emails_columns():
    job = from_jobspy(
        jobspy_row(site="naukri", experience_range="6-10 Yrs", emails="A.B@Quess.com", skills="Java, Spring"),
        query_city="Bengaluru",
    )
    assert (job.exp_min, job.exp_max) == (6, 10)
    assert job.emails[0] == "a.b@quess.com"
    assert job.listed_skills == ["Java", "Spring"]


def test_salary_lpa():
    assert salary_lpa(1_200_000, "yearly", "INR") == 12.0
    assert salary_lpa(50_000, "monthly", "INR") == 6.0
    assert salary_lpa(100_000, "yearly", "USD") is None
    assert salary_lpa(math.nan, "yearly", "INR") is None


def test_city_aliases():
    assert mentions_city("Bangalore Urban, Karnataka", "Bengaluru")
    assert mentions_city("Gurgaon, Haryana", "Gurugram")
    assert not mentions_city("Punjab", "Pune")


def test_extract_emails_skips_images_and_dupes():
    text = "Send CV to HR@acme.co.in or hr@acme.co.in. Logo: logo@2x.png"
    assert extract_emails(text) == ["hr@acme.co.in"]


def test_greenhouse_and_lever_conversion():
    gh = from_greenhouse(
        {
            "id": 1,
            "title": "Backend Engineer",
            "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
            "company_name": "Acme",
            "location": {"name": "Remote, India"},
            "content": "&lt;p&gt;Need 2+ years experience. Python.&lt;/p&gt;",
            "first_published": "2026-10-15T09:00:00+05:30",
        },
        "acme",
    )
    assert gh.description == "Need 2+ years experience. Python."
    assert gh.is_remote and gh.exp_min == 2

    lv = from_lever(
        {
            "id": "abc",
            "text": "Backend Engineer",
            "hostedUrl": "https://jobs.lever.co/acme/abc",
            "applyUrl": "https://jobs.lever.co/acme/abc/apply",
            "categories": {"location": "Pune"},
            "workplaceType": "onsite",
            "descriptionPlain": "Intro",
            "lists": [{"text": "Requirements", "content": "<li>Go</li><li>SQL</li>"}],
            "additionalPlain": "",
            "createdAt": 1786469891368,
        },
        "acme",
    )
    assert "Requirements\nGo\nSQL" in lv.description
    assert lv.apply_url.endswith("/apply") and not lv.is_remote


def test_html_to_text_collapses_blank_lines():
    assert html_to_text("<p>a</p><br><br><br><p>b</p>") in ("a\nb", "a\n\nb")
