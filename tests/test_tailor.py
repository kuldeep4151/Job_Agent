import json
import sqlite3

import pytest

from conftest import SAMPLE_PROFILE
from jobagent import db
from jobagent.models import Job
from jobagent.profile import load_profile
from jobagent.tailor import tailor
from jobagent.tailor.validate import check_summary

GOOD_SUMMARY = (
    "Backend engineer with 4 years of experience in Python, Go and Kafka. "
    "Led migration of Django endpoints to FastAPI and added Redis caching to high-traffic services."
)


@pytest.fixture
def profile():
    return load_profile(SAMPLE_PROFILE)


def plan(**overrides) -> tailor.TailorPlan:
    fields = dict(
        summary=GOOD_SUMMARY,
        bullet_ids=["exp1.3", "exp1.1", "exp2.2", "exp9.9", "exp1.3"],
        skills=["Kafka", "Django", "Terraform"],
        rationale="Led with the Kafka and Django work.",
    )
    return tailor.TailorPlan(**(fields | overrides))


# ---------- summary validator ----------


def test_faithful_summary_passes(profile):
    assert check_summary(GOOD_SUMMARY, profile) == []
    assert check_summary("Ships REST APIs and Kafka's consumers in Python/Go.", profile) == []


@pytest.mark.parametrize(
    ("summary", "problem"),
    [
        ("Backend engineer with 6 years of experience in Python.", "claims experience not in your profile: 6 years"),
        ("Backend engineer with 5+ years in Go.", "5+ years"),
        ("Backend engineer skilled in Python and Terraform.", "not in your profile: Terraform"),
        ("Backend engineer who scaled services at Google.", "not in your profile: Google"),
        ("Built pipelines in Python/Rust.", "Python/Rust"),
        ("Cut costs by 75% across payment services.", "75"),
        ("I build backend services in Python.", "first person"),
        ("Python " * 100, "words (max 90)"),
    ],
)
def test_fabrications_are_rejected(profile, summary, problem):
    problems = check_summary(summary, profile)
    assert any(problem in p for p in problems), problems


# ---------- assembling the tailored profile ----------


def test_bullets_are_picked_by_id_in_plan_order(profile):
    result = tailor.build_tailored(profile, plan())
    first, second = result.profile.experience
    assert first.company == "Example Payments Pvt Ltd"  # roles keep chronological order
    assert first.bullets == [profile.experience[0].bullets[2], profile.experience[0].bullets[0]]
    # only exp2.2 chosen → topped up from the original order to the minimum of 2
    assert second.bullets == [profile.experience[1].bullets[1], profile.experience[1].bullets[0]]
    assert result.profile.projects[0].bullets == profile.projects[0].bullets
    assert "ignored unknown bullet IDs: exp9.9" in result.notes
    # every rendered bullet is verbatim from the profile
    assert set(sum((r.bullets for r in result.profile.experience), [])) <= set(profile.bullets().values())


def test_skills_reordered_never_dropped_or_added(profile):
    result = tailor.build_tailored(profile, plan())
    groups = result.profile.skills
    assert [g.name for g in groups] == ["Data & Infra", "Backend", "Languages"]
    assert groups[0].skills[0] == "Kafka" and groups[1].skills[0] == "Django"
    assert sorted(result.profile.all_skills()) == sorted(profile.all_skills())
    assert "ignored skills not in your profile: Terraform" in result.notes


def test_bad_summary_keeps_original_but_keeps_reordering(profile):
    result = tailor.build_tailored(profile, plan(summary="Senior engineer with 8 years at Google."))
    assert result.summary_kept_original
    assert result.profile.summary == profile.summary
    assert result.profile.experience[0].bullets[0] == profile.experience[0].bullets[2]
    assert any(note.startswith("kept your original summary") for note in result.notes)


def test_good_summary_is_used(profile):
    result = tailor.build_tailored(profile, plan())
    assert not result.summary_kept_original and result.profile.summary == GOOD_SUMMARY


# ---------- Claude request ----------


def test_plan_request(fake_api, profile):
    system = tailor.tailoring_system_prompt(profile)
    assert system == tailor.tailoring_system_prompt(profile)  # byte-stable → cached
    fake_api.reply_when("You tailor", plan().model_dump_json())
    job = Job(dedupe_key="k", source="linkedin", external_id="1", url="u", title="Backend Engineer",
              company="Nimbus", description="Kafka and Django.")

    result = tailor.plan_for(job, system)

    assert result.bullet_ids[0] == "exp1.3"
    body = fake_api.sent[0]["body"]
    assert body["output_config"]["effort"] == "medium"
    assert "exp1.1: Built a refund-reconciliation service" in body["system"][0]["text"]
    content = body["messages"][0]["content"]
    assert content.startswith("<job_posting>") and "Nimbus" in content
    assert "Nimbus" not in body["system"][0]["text"]


def test_naukri_routing():
    job = Job(dedupe_key="k", source="naukri", external_id="1", url="u", title="t", company="c")
    assert tailor.uses_naukri_profile(job)
    job.apply_url = "https://careers.acme.com/apply/1"  # "Apply on company site"
    assert not tailor.uses_naukri_profile(job)


# ---------- database upgrade ----------


def test_new_columns_are_added_to_an_existing_database(tmp_path):
    path = tmp_path / "old.db"
    db.configure(path)
    with sqlite3.connect(path) as connection:  # simulate a database created before Phase 3
        connection.execute("ALTER TABLE job DROP COLUMN tailor_notes")
        connection.execute("ALTER TABLE run DROP COLUMN tailored")
    engine = db.configure(path)
    with engine.connect() as connection:
        columns = [row[1] for row in connection.exec_driver_sql("PRAGMA table_info(job)")]
    assert "tailor_notes" in columns
    assert json.dumps(db.add_missing_columns(engine)) == "[]"  # idempotent


def test_dash_styles_are_the_same_number(profile):
    """Live 2026-10-15: the resume says "40–55%" (en dash); Claude wrote "40-55%" and was wrongly rejected."""
    from jobagent.tailor.validate import unsupported_terms

    profile = profile.model_copy(deep=True)
    profile.experience[0].bullets.append("Cut LLM token usage by 40–55% while improving accuracy.")
    assert unsupported_terms("Reduced token usage by 40-55% with Kafka.", profile) == []
    assert unsupported_terms("Reduced token usage by 40-65% with Kafka.", profile) == ["40-65"]
