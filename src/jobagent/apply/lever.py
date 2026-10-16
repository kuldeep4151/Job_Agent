"""Lever application forms (jobs.lever.co/<company>/<id>/apply).

Lever has no public list of a form's custom questions, so they're read off the page: each
li.application-question has a label (✱ = required) and its controls (text, radio group, checkboxes,
select, file). Answers are planned for every question before anything is typed; a required question
none of your facts answers sends the job to "Needs you". Lever forms carry hCaptcha: if a challenge
appears on submit, that's a "Needs you" too.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from jobagent import llm
from jobagent.apply.answers import AnswerMapper, Question, Unanswerable
from jobagent.apply.evidence import Outcome, save_evidence
from jobagent.config import answer
from jobagent.models import Job

_STANDARD = {
    "name": "personal.full_name", "email": "personal.email", "phone": "personal.phone",
    "org": "employment.current_company", "location": "personal.current_city",
    "urls[LinkedIn]": "personal.linkedin_url", "urls[GitHub]": "personal.github_url",
    "urls[Portfolio]": "personal.portfolio_url",
}
SUCCESS = re.compile(r"application submitted|thanks for applying|thank you for applying", re.I)
_PLACEHOLDER_OPTION = re.compile(r"^\s*(select\.*|choose\.*|--+)?\s*$", re.I)


@dataclass
class LeverQuestion:
    label: str
    required: bool
    name: str           # the control's name attribute
    kind: str           # text | choice | multi | select | file | standard
    options: list[str]


@dataclass
class Fill:
    name: str
    label: str
    kind: str
    value: str
    source: str


def apply_url(url: str) -> str:
    url = url.split("?")[0].rstrip("/")
    return url if url.endswith("/apply") else f"{url}/apply"


def _option_labels(block: Locator, selector: str) -> list[str]:
    return [t.strip() for t in block.locator(selector).evaluate_all(
        "els => els.map(e => (e.closest('label') || e.parentElement).innerText)")]


def read_questions(page: Page) -> list[LeverQuestion]:
    questions: list[LeverQuestion] = []
    for block in page.locator("li.application-question").all():
        label_el = block.locator(".application-label").first
        raw = (label_el.inner_text() if label_el.count() else block.inner_text().split("\n")[0]).strip()
        required = "✱" in raw or block.locator("[required]").count() > 0
        label = " ".join(raw.replace("✱", "").split())
        controls = block.locator("input:not([type=hidden]), textarea, select")
        if not controls.count():
            continue
        first = controls.first
        name = first.get_attribute("name") or ""
        ctype = (first.get_attribute("type") or first.evaluate("e => e.tagName")).lower()
        if name in _STANDARD:
            kind, options = "standard", []
        elif ctype == "file":
            kind, options = "file", []
        elif ctype == "radio":
            kind, options = "choice", _option_labels(block, "input[type=radio]")
        elif ctype == "checkbox":
            kind, options = "multi", _option_labels(block, "input[type=checkbox]")
        elif ctype == "select":
            texts = block.locator("select option").evaluate_all("els => els.map(e => e.innerText.trim())")
            kind, options = "select", [t for t in texts if not _PLACEHOLDER_OPTION.match(t)]
        else:
            kind, options = "text", []
        questions.append(LeverQuestion(label, required, name, kind, options))
    return questions


def plan(questions: list[LeverQuestion], mapper: AnswerMapper, answers: dict, job: Job, resume: Path) -> list[Fill]:
    fills: list[Fill] = []
    for q in questions:
        if q.kind == "standard":
            value = answer(answers, _STANDARD[q.name])
            if value:
                fills.append(Fill(q.name, q.label, "text", str(value), _STANDARD[q.name]))
            elif q.required:
                raise Unanswerable(f"no {_STANDARD[q.name]} in answers.yaml for {q.label!r}")
        elif q.kind == "file":
            if q.name == "resume":
                fills.append(Fill(q.name, q.label, "file", str(resume), "RESUME_FILE"))
            elif q.required:
                raise Unanswerable(f"the form requires a file for {q.label!r}")
        elif q.name == "opportunityLocationId" and not q.required:
            continue  # "which location did you find the job in?": optional bookkeeping
        else:
            kind = "choice" if q.kind == "select" else q.kind
            try:
                result = mapper.answer(Question(q.label, kind, q.options), job, units_in_text=True)
            except (Unanswerable, llm.LLMError) as e:
                if q.required:
                    raise Unanswerable(f"required question {q.label[:80]!r}: {e}") from e
                continue
            fills.append(Fill(q.name, q.label, q.kind, result.value, result.source))
    return fills


def _choose(page: Page, name: str, kind: str, value: str) -> None:
    controls = page.locator(f"input[name='{name}']")
    for i in range(controls.count()):
        control = controls.nth(i)
        text = control.evaluate("e => (e.closest('label') || e.parentElement).innerText").strip()
        if text == value:
            # live: Lever styles a <label> over the real input, so clicking the input itself can miss
            label = control.locator("xpath=ancestor::label[1]")
            (label if label.count() else control).click()
            if not control.is_checked():
                control.evaluate("e => { e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true})); }")
            return
    raise Unanswerable(f"option {value!r} not on the page")


def _pick_location(page: Page, city: str) -> None:
    """Lever only keeps a location picked from its suggestions ("Indore, WV, USA" is offered too)."""
    box = page.locator("input[name='location']").first
    box.click()
    box.press_sequentially(city, delay=60)
    suggestions = page.locator(".dropdown-location").filter(visible=True)
    try:
        suggestions.first.wait_for(timeout=6000)
    except PlaywrightTimeout:
        return  # no suggestions: the typed city stays
    texts = suggestions.all_inner_texts()
    starts = [i for i, t in enumerate(texts) if t.strip().lower().startswith(city.lower())]
    indian = [i for i in starts if _is_india(texts[i])]
    suggestions.nth((indian or starts or [0])[0]).click()


def _is_india(text: str) -> bool:
    return bool(re.search(r",\s*(IND|India)\s*$", text.strip()))


def fill(page: Page, fills: list[Fill]) -> None:
    for item in fills:
        selector = f"[name='{item.name}']"
        if item.name == "location":
            _pick_location(page, item.value)
        elif item.kind == "file":
            page.set_input_files(selector, item.value)
        elif item.kind in ("text", "standard"):
            page.locator(selector).first.fill(item.value)
        elif item.kind == "select":
            page.select_option(selector, label=item.value)
        else:
            _choose(page, item.name, item.kind, item.value)


def apply(page: Page, job: Job, mapper: AnswerMapper, answers: dict, resume: Path, *, live: bool) -> Outcome:
    page.goto(apply_url(job.apply_target or job.url), wait_until="networkidle", timeout=60_000)
    if not page.locator("li.application-question").count():
        return Outcome("manual", "couldn't read this Lever form (screenshot + HTML saved)",
                       save_evidence(page, job.id, "lever-unknown", html=True))
    try:
        fills = plan(read_questions(page), mapper, answers, job, resume)
    except Unanswerable as e:
        return Outcome("manual", f"Lever form: {e}", save_evidence(page, job.id, "lever-question"))
    qa = [{"question": f.label, "answer": Path(f.value).name if f.kind == "file" else f.value, "source": f.source}
          for f in fills]
    fill(page, fills)
    if not live:
        return Outcome("dry_run", "dry run: Lever form filled, not submitted", save_evidence(page, job.id, "lever-dry-run"), qa)
    page.locator("#btn-submit").click()
    for _ in range(30):
        page.wait_for_timeout(1000)
        if SUCCESS.search(page.inner_text("body")) or page.url.rstrip("/").endswith("/thanks"):
            return Outcome("applied", "applied on Lever", save_evidence(page, job.id, "lever-applied"), qa)
        # hCaptcha may preload its challenge frame hidden: only a visible one means a challenge is showing
        if page.locator("iframe[src*='hcaptcha'][src*='challenge']").filter(visible=True).count():
            return Outcome("manual", "Lever showed a captcha: finish this one by hand",
                           save_evidence(page, job.id, "lever-captcha", html=True), qa)
    return Outcome("failed", "no Lever confirmation within 30s of submitting",
                   save_evidence(page, job.id, "lever-timeout", html=True), qa)
