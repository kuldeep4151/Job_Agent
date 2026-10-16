"""Greenhouse application forms (job-boards.greenhouse.io / boards.greenhouse.io).

Greenhouse publishes every form's questions (label, required, type, options) through its public job
board API, and the form's inputs carry the same ids (first_name, resume, question_123…). So answers are
planned from the API, all before touching the page, and a required question none of your facts
answers sends the job to "Needs you" without filling anything. Dry run fills the form and stops.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from playwright.sync_api import Page

from jobagent import llm
from jobagent.apply.answers import Answer, AnswerMapper, Question, Unanswerable
from jobagent.apply.evidence import Outcome, save_evidence
from jobagent.config import answer
from jobagent.models import Job

API = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}?questions=true"
_URL = re.compile(r"greenhouse\.io/(?:embed/job_app\?for=)?([\w-]+)/jobs/(\d+)|greenhouse\.io/embed/job_app\?for=([\w-]+)&token=(\d+)")
_KINDS = {"input_text": "text", "textarea": "text", "multi_value_single_select": "choice",
          "multi_value_multi_select": "multi"}
SUCCESS = re.compile(r"thank you for applying|application (has been )?(submitted|received)|we.ve received your application", re.I)
SECURITY_CODE = re.compile(r"security code|verification code", re.I)


@dataclass
class Fill:
    field: str          # input id on the page
    label: str
    kind: str           # text | choice | multi | file
    value: str
    source: str


def code_from_gmail(job: Job, wait_seconds: int = 60) -> str | None:
    """Greenhouse's "security code for your application" email → the code (None if no Gmail or it never came)."""
    from jobagent.outreach import gmail

    if gmail.credentials() is None:
        return None
    for _ in range(wait_seconds // 5):
        with gmail.Mailbox() as mailbox:
            uids = mailbox.search('from:greenhouse newer_than:1h "security code"')
            if uids:
                _subject, text = mailbox.read(uids[-1])
                if match := re.search(r"code[^:]*:\s*([A-Za-z0-9]{6,10})\b", text):
                    return match[1]
        time.sleep(5)
    return None


def parse_url(url: str) -> tuple[str, str] | None:
    match = _URL.search(url)
    if not match:
        return None
    board, job_id = (match[1], match[2]) if match[1] else (match[3], match[4])
    return board, job_id


def fetch_questions(board: str, job_id: str, client: httpx.Client) -> list[dict]:
    response = client.get(API.format(board=board, job_id=job_id))
    response.raise_for_status()
    return response.json().get("questions", [])


def _standard(name: str, answers: dict) -> str | None:
    """Contact fields filled straight from answers.yaml."""
    full = str(answer(answers, "personal.full_name") or "")
    first, _, last = full.partition(" ")
    phone = re.sub(r"^\+?91[-\s]?", "", str(answer(answers, "personal.phone") or ""))  # India picked in the country box
    return {"first_name": first, "last_name": last, "email": answer(answers, "personal.email"),
            "phone": re.sub(r"[^\d]", "", phone)}.get(name)


def plan(questions: list[dict], mapper: AnswerMapper, answers: dict, job: Job, resume: Path) -> list[Fill]:
    """Every field's value, decided before the page is touched. Raises Unanswerable for a required gap."""
    fills: list[Fill] = []
    for question in questions:
        field = question["fields"][0]
        name, ftype, label, required = field["name"], field["type"], question["label"].strip(), question["required"]
        if name in ("first_name", "last_name", "email", "phone"):
            value = _standard(name, answers)
            if not value:
                raise Unanswerable(f"no {name.replace('_', ' ')} in answers.yaml")
            fills.append(Fill(name, label, "text", str(value), f"personal ({name})"))
        elif ftype == "input_file":
            if name == "resume":
                fills.append(Fill(name, label, "file", str(resume), "RESUME_FILE"))
            elif required:
                raise Unanswerable(f"the form requires a file for {label!r}")
        elif ftype in _KINDS:
            options = [v["label"] for v in field.get("values", [])]
            try:
                result: Answer = mapper.answer(Question(label, _KINDS[ftype], options), job, units_in_text=True)
            except (Unanswerable, llm.LLMError) as e:
                if required:
                    raise Unanswerable(f"required question {label!r}: {e}") from e
                continue
            fills.append(Fill(name, label, _KINDS[ftype], result.value, result.source))
        elif required:
            raise Unanswerable(f"unsupported required field type {ftype!r} for {label!r}")
    return fills


def _pick(page: Page, field_id: str, value: str) -> None:
    """Greenhouse's dropdowns are comboboxes: open, type, choose the exact option."""
    box = page.locator(f"#{field_id}")
    box.click()
    box.press_sequentially(value, delay=40)
    page.wait_for_timeout(500)
    exact = page.get_by_role("option", name=value, exact=True)
    # live: the country list says "India +91", and typing "India" also offers "British Indian Ocean Territory"
    starts = page.get_by_role("option", name=re.compile(rf"^{re.escape(value)}(\s|$)"))
    option = exact if exact.count() else starts
    option.first.wait_for(timeout=5000)
    option.first.click()


def fill(page: Page, fills: list[Fill]) -> None:
    if page.locator("#country").count():  # phone's country box (new Greenhouse forms require it)
        _pick(page, "country", "India")
    for item in fills:
        if item.kind == "file":
            page.set_input_files(f"#{item.field}", item.value)
        elif item.kind == "text":
            page.locator(f"#{item.field}").fill(item.value)
        else:
            _pick(page, item.field, item.value)


def apply(page: Page, job: Job, mapper: AnswerMapper, answers: dict, resume: Path, *, live: bool,
          client: httpx.Client, security_code=None) -> Outcome:
    parsed = parse_url(job.apply_target or job.url)
    if parsed is None:
        return Outcome("manual", f"not a Greenhouse form URL: {job.apply_target}")
    try:
        fills = plan(fetch_questions(*parsed, client), mapper, answers, job, resume)
    except Unanswerable as e:
        return Outcome("manual", f"Greenhouse form: {e}")
    except httpx.HTTPError as e:
        return Outcome("failed", f"couldn't read the Greenhouse form: {e}")
    qa = [{"question": f.label, "answer": Path(f.value).name if f.kind == "file" else f.value, "source": f.source}
          for f in fills]

    page.goto(job.apply_target or job.url, wait_until="networkidle", timeout=60_000)
    fill(page, fills)
    if not live:
        return Outcome("dry_run", "dry run: Greenhouse form filled, not submitted",
                       save_evidence(page, job.id, "greenhouse-dry-run"), qa)
    page.get_by_role("button", name=re.compile(r"submit application", re.I)).click()
    for _ in range(30):
        page.wait_for_timeout(1000)
        text = page.inner_text("body")
        if SUCCESS.search(text) or "/confirmation" in page.url:
            return Outcome("applied", "applied on Greenhouse", save_evidence(page, job.id, "greenhouse-applied"), qa)
        if SECURITY_CODE.search(text) and security_code is not None:
            code = security_code(job)
            if not code:
                return Outcome("manual", "Greenhouse asked for an emailed security code that didn't arrive",
                               save_evidence(page, job.id, "greenhouse-code", html=True), qa)
            page.locator("input[id*='security'], input[name*='security']").first.fill(code)
            page.get_by_role("button", name=re.compile(r"submit", re.I)).click()
            security_code = None  # only once
    return Outcome("failed", "no Greenhouse confirmation within 30s of submitting",
                   save_evidence(page, job.id, "greenhouse-timeout", html=True), qa)
