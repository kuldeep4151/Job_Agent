from datetime import timedelta

from conftest import SAMPLE_PROFILE
from jobagent.config import Config, Search
from jobagent.match import (
    JobScore,
    MatchContext,
    apply_score,
    experience_ok,
    hard_filter,
    is_stale,
    job_prompt,
    location_ok,
    matching_skills,
    scam_signals,
    scoring_system_prompt,
)
from jobagent.models import Job, State, utcnow
from jobagent.profile import load_profile


def make_job(**fields) -> Job:
    defaults = dict(dedupe_key="k", source="naukri", external_id="1", url="u", title="Backend Developer",
                    company="Acme", location="Bengaluru, India", description="")
    return Job(**(defaults | fields))


def make_config(**search) -> Config:
    return Config(search=Search(roles=["Backend Developer"], cities=["Bengaluru", "Pune"], **search))


def test_location():
    config = make_config(include_remote=True)
    assert location_ok(make_job(location="Bangalore, Karnataka"), config)
    assert not location_ok(make_job(location="Chennai, India"), config)
    assert location_ok(make_job(location="Chennai", is_remote=True), config)
    assert location_ok(make_job(location="KA, India", query_city="Bengaluru"), config)  # board dropped the city
    assert not location_ok(make_job(location="Chennai", query_city="Bengaluru"), config)
    assert not location_ok(make_job(location="Chennai", is_remote=True), make_config(include_remote=False))


def test_experience_window():
    assert experience_ok(make_job(exp_min=3, exp_max=5), my_years=4, tolerance=1)
    assert experience_ok(make_job(exp_min=5, exp_max=None), my_years=4, tolerance=1)
    assert not experience_ok(make_job(exp_min=8, exp_max=12), my_years=4, tolerance=1)
    assert not experience_ok(make_job(exp_min=0, exp_max=1), my_years=4, tolerance=1)
    assert experience_ok(make_job(exp_min=8, exp_max=12), my_years=None, tolerance=1)


def test_scam_signals():
    strong, _ = scam_signals(make_job(description="Pay a registration fee of Rs 2000 to confirm."))
    assert strong == ["asks for money (registration fee)"]
    _, weak = scam_signals(make_job(description="WhatsApp your CV", emails=["jobs.acme@gmail.com"]))
    assert weak == ["free-mail contact only", "WhatsApp contact"]
    assert scam_signals(make_job(description="Mail hr@acme.com", emails=["hr@acme.com"])) == ([], [])


def test_matching_skills_boundaries_and_case():
    job = make_job(description="Strong C++ and Node.js; .NET a plus. Let's go build it with PostgreSQL.")
    found = matching_skills(job, ["C++", "C", "Node.js", ".NET", "Go", "postgresql", "Java"])
    assert found == ["C++", "Node.js", ".NET", "postgresql"]  # lowercase "go" is not the language Go
    assert matching_skills(make_job(description="Golang or Go services"), ["Go"]) == ["Go"]


def test_matching_skills_accepts_short_forms_and_plurals():
    job = make_job(description="Experience with an LLM, RAG, embedding models and GCP. Vector DBs a plus.")
    skills = ["Large Language Models (LLMs)", "Retrieval-Augmented Generation (RAG)", "Embeddings",
              "Google Cloud Platform (GCP)", "Redis", "Transformers"]
    assert matching_skills(job, skills) == skills[:4]


def test_answer_ignores_placeholders_inside_lists():
    from jobagent.config import answer

    assert answer({"p": {"cities": ["TODO"]}}, "p.cities") is None
    assert answer({"p": {"cities": ["Indore", "TODO"]}}, "p.cities") == ["Indore"]


def test_scoring_prompt_leaves_location_to_code():
    prompt = scoring_system_prompt(load_profile(SAMPLE_PROFILE), {"preferences": {"preferred_cities": ["TODO"]}})
    assert "TODO" not in prompt and "preferred_cities" not in prompt
    assert "Location has already been checked" in prompt


def test_hard_filter_order_and_flags():
    ctx = MatchContext(config=make_config(), profile=None, answers={"employment": {"total_experience_years": 4}})
    ctx.config.blocklist.companies = ["Acme Pvt Ltd"]
    assert hard_filter(make_job(company="ACME Private Limited"), ctx) == "blocklisted company (Acme Pvt Ltd)"
    ctx.config.blocklist.companies = []
    assert hard_filter(make_job(location="Delhi"), ctx) == "location: Delhi"
    assert hard_filter(make_job(exp_min=9, exp_max=12, experience_text="9-12 Yrs"), ctx).startswith("experience 9-12 Yrs")
    job = make_job(description="whatsapp us")
    assert hard_filter(job, ctx) is None and job.flags == ["WhatsApp contact"]


def test_answers_todo_disables_experience_filter():
    ctx = MatchContext(config=make_config(), profile=None, answers={"employment": {"total_experience_years": "TODO"}})
    assert ctx.my_years is None
    assert hard_filter(make_job(exp_min=15, exp_max=20), ctx) is None


def test_system_prompt_is_stable_and_skips_todo_answers():
    profile = load_profile(SAMPLE_PROFILE)
    answers = {"employment": {"total_experience_years": 4.5, "current_designation": "TODO"}}
    first, second = scoring_system_prompt(profile, answers), scoring_system_prompt(profile, answers)
    assert first == second  # byte-stable → prompt cache hits across jobs
    assert '"total_experience_years": 4.5' in first and "current_designation" not in first
    assert "Example Payments Pvt Ltd" in first


def test_job_prompt_truncates_and_wraps():
    prompt = job_prompt(make_job(description="x" * 20000, salary_min_lpa=12.0))
    assert prompt.startswith("<job_posting>") and "12.0–? LPA" in prompt
    assert prompt.count("x") < 12100


def test_apply_score_thresholds():
    result = JobScore(score=82, decision="apply", matched_must_haves=["Go"], missing_must_haves=[],
                      seniority_fit="fit", reason="Good fit.")
    job = make_job()
    apply_score(job, result, min_score=70)
    assert job.state == State.SCORED

    job = make_job()
    apply_score(job, result.model_copy(update={"score": 60}), min_score=70)
    assert (job.state, job.skip_reason) == (State.SKIPPED, "score 60 < 70")

    job = make_job()
    apply_score(job, result.model_copy(update={"decision": "skip", "reason": "Needs a PhD."}), min_score=70)
    assert job.skip_reason == "Claude: Needs a PhD."


def test_stale():
    assert is_stale(make_job(first_seen_at=utcnow() - timedelta(hours=73)))
    assert not is_stale(make_job(first_seen_at=utcnow() - timedelta(hours=2)))
