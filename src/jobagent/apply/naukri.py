"""Naukri apply flow, driven on the user's logged-in session in a visible Chrome window.

Naukri has no applicant API and logged-out visitors don't even see the Apply button. Every page
selector lives in SELECTORS, so when Naukri changes its markup they're updated in one place;
`jobagent naukri inspect` saves a real page (screenshot + HTML) to check them against.

Dry run never clicks Apply: on Naukri a single click often submits the application.
"""

import json
import random
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from playwright.sync_api import Locator, Page
from selectolax.lexbor import LexborHTMLParser

from jobagent import llm, settings
from jobagent.apply.answers import Answer, AnswerMapper, Question, Unanswerable, clean_question
from jobagent.apply.evidence import Outcome, save_evidence
from jobagent.models import Job
from jobagent.profile import Profile
from jobagent.tailor import tailor

HOME_URL = "https://www.naukri.com/mnjuser/homepage"
APPLIES_URL = "https://www.naukri.com/myapply/historypage"  # "Job application status"
LOGIN_URL = "https://www.naukri.com/nlogin/login"
MAX_QUESTIONS = 15
RESULT_TIMEOUT_SECONDS = 90
SETTLE_MS = (2500, 4500)  # wait after loading a job page: lets it render, reads like a person
ANSWER_ACCEPT_SECONDS = 20  # same question still showing this long after answering → not accepted
SAVE_WAIT_SECONDS = 5  # after answering: how long to wait for Save to enable (or the bot to move on)

# First visible match wins. Update these when Naukri changes its pages (see `jobagent naukri inspect`).
SELECTORS: dict[str, list[str]] = {
    "logged_out": ["#login_Layer", "text=/continue with google/i", "text=/(login|register) to apply/i"],
    "already_applied": ["#already-applied", "button:text-is('Applied')", "span:text-is('Applied')"],
    "closed": ["text=/no longer accepting applications|job has expired|this job is closed/i"],
    "company_site": ["#company-site-button", "button:has-text('Apply on company site')"],
    "apply": ["#apply-button", "button:text-is('Apply')", "button:text-is('Apply now')"],
    # live 2026-10-15: after the questionnaire Naukri shows 'Applied to "<job title>"' on a confirmation page
    "success": ["text=/successfully applied|applied successfully|you have applied|application (has been )?sent/i",
                "text=/^\\s*applied to\\s*[\"“]/i", ".job-title-text:has-text('Applied to')"],
    # the recruiter questionnaire ("chatbot" drawer) that can open after clicking Apply
    "drawer": ["[class*='chatbot_DrawerContentWrapper']", "[class*='chatbot_Drawer']", "[class*='chatbot-drawer']"],
    "bot_message": ["[class*='botMsg']", "[class*='bot-msg']"],
    "text_input": ["[contenteditable='true']", "input[type='text']", "textarea"],
    "radio": ["input[type='radio']"],
    "checkbox": ["input[type='checkbox']"],
    # exact class: [class*='chip'] also hits div.chipMsg (the logo on every message) and the
    # div.chatbot_Chips container. "Skip this question" is a chip too, but it's handled as `skip`.
    "chip": [".chatbot_Chip"],
    "skip": [".chatbot_Chip:has-text('Skip this question')", ".chatbot_Chip:has-text(\"I'll do it later\")"],
    "file_input": ["input[type='file']"],
    # the clickable Save is div.sendMsg inside div.sendMsgbtn_container: match the exact class first
    "send": [".sendMsg", "button:has-text('Save')", "button:has-text('Submit')", "button:has-text('Send')"],
}


class PageState(StrEnum):
    LOGGED_OUT = "logged_out"
    ALREADY_APPLIED = "already_applied"
    CLOSED = "closed"
    COMPANY_SITE = "company_site"
    APPLY = "apply"
    UNKNOWN = "unknown"


def find_all(scope: Page | Locator, name: str, visible: bool = True) -> Locator | None:
    """All matches of the first selector in SELECTORS[name] that matches anything."""
    for selector in SELECTORS[name]:
        match = scope.locator(selector)
        if visible:
            match = match.filter(visible=True)
        if match.count():
            return match
    return None


def find(scope: Page | Locator, name: str) -> Locator | None:
    match = find_all(scope, name)
    return match.first if match else None


def detect_state(page: Page) -> PageState:
    for name in ("logged_out", "already_applied", "closed", "company_site", "apply"):
        if find(page, name):
            return PageState(name)
    return PageState.UNKNOWN


def is_logged_in(page: Page) -> bool:
    page.goto(HOME_URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(3000)
    return "login" not in page.url.lower() and find(page, "logged_out") is None


def wait_for_login(page: Page, timeout_seconds: int = 600) -> bool:
    """Open the login page and wait (passively, so typing isn't disturbed) for Naukri's post-login redirect."""
    page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if any("/mnjuser/" in tab.url for tab in page.context.pages):
            return True
        page.wait_for_timeout(3000)
    return False


def inspect(page: Page, url: str, tag: str) -> dict:
    """Read-only look at a job page: detected state, apply-related buttons, screenshot + HTML. Never clicks."""
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(4000)
    buttons = page.eval_on_selector_all(
        "button, a, [role=button]",
        """els => els.filter(e => e.offsetParent && /apply|applied|interest|company site/i.test(e.innerText))
                    .map(e => ({tag: e.tagName, id: e.id, cls: e.className.toString().slice(0, 80),
                                text: e.innerText.trim().slice(0, 50)}))""",
    )
    return {"state": detect_state(page), "buttons": buttons, "screenshot": save_evidence(page, tag, "inspect", html=True)}


# ---------- your applications' status (read-only) ----------


@dataclass
class AppliedJob:
    job_id: str   # Naukri's id: a Job's external_id is "nk-" + this
    title: str
    company: str
    status: str   # the card's chip, e.g. "Application sent 2 days ago"


def parse_applies(html: str) -> list[AppliedJob]:
    """The cards on the "Applies on Naukri" list. Each card's links carry the job id."""
    found = []
    for card in LexborHTMLParser(html).css(".applyHistoryListContainer .jdTupleContainer"):
        title, company, status = (card.css_first(sel) for sel in (".jdTitle", ".company", ".apply-status"))
        ids = re.findall(r"(?:jobId|flowId)=(\d+)", card.html or "")
        if title is not None and ids:
            found.append(AppliedJob(ids[0], title.text(strip=True), company.text(strip=True) if company else "",
                                    status.text(strip=True) if status else ""))
    return found


def read_applies(page: Page, max_scrolls: int = 15) -> list[AppliedJob]:
    """Open the application-status page and read every card (the list loads more as it scrolls). Never clicks."""
    page.goto(APPLIES_URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(4000)
    cards = page.locator(".applyHistoryListContainer .jdTupleContainer")
    for _ in range(max_scrolls):
        before = cards.count()
        if before:
            cards.last.scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        if cards.count() == before:
            break
    return parse_applies(page.content())


# ---------- questionnaire ----------


def _label(drawer: Locator, control: Locator) -> str:
    control_id = control.get_attribute("id")
    if control_id:
        label = drawer.locator(f"label[for='{control_id}']")
        if label.count():
            return label.first.inner_text().strip()
    return (control.get_attribute("value") or "").strip()


_SKIP_LABELS = {"skip this question", "i'll do it later"}
_RESUME_REQUEST = re.compile(r"upload (your )?(updated )?(resume|cv)", re.IGNORECASE)

# Bot messages that aren't questions: the greeting ("Hi <name>, thank you for showing interest. Kindly
# answer…"), error notes ("File upload was unsuccessful. Please try again.", then it re-asks), and the
# sign-off ("Thank you for your responses.", then the 'Applied to …' confirmation page).
_GREETING = re.compile(
    r"thank you for showing interest|kindly answer|unsuccessful|please try again|thank you for your responses"
    r"|update your profile|helps us to recommend",  # the profile-update bot's intro (live 2026-10-15)
    re.IGNORECASE,
)


def read_question(drawer: Locator) -> Question | None:
    """The question currently awaiting an answer, or None while the bot is still greeting/typing."""
    messages = find_all(drawer, "bot_message")
    if messages is None:
        return None
    text = clean_question(messages.last.inner_text())
    if not text or _GREETING.search(text):  # empty = typing bubble still animating
        return None
    skippable = find(drawer, "skip") is not None
    # Naukri hides the real radio/checkbox inputs behind styled labels, so don't require visibility.
    # Only options with a visible text count: anything else is page furniture, not a choice.
    for name, kind in (("radio", "choice"), ("checkbox", "multi")):
        controls = find_all(drawer, name, visible=False)
        options = [_label(drawer, controls.nth(i)) for i in range(controls.count())] if controls else []
        if any(options):
            return Question(text, kind, [o for o in options if o], skippable)
    chips = find_all(drawer, "chip")
    options = [chips.nth(i).inner_text().strip() for i in range(chips.count())] if chips else []
    options = [o for o in options if o and o.lower() not in _SKIP_LABELS]
    if options:
        # chips plus a search box ("Type or click on more to see all the courses") = a partial list
        typeahead = find(drawer, "text_input") is not None
        return Question(text, "choice", options, skippable, typeahead)
    return Question(text, "text", [], skippable)


def upload_resume(drawer: Locator, question: Question, job: Job, profile: Profile) -> Answer:
    """"Please upload your resume." → click "Upload Resume" and pick the file, as a person would.

    Verified live 2026-10-15: pushing the file into the hidden input without clicking the chip gets
    "File upload was unsuccessful".
    """
    resume = tailor.resume_for_job(job, profile)
    chip = drawer.locator(".chatbot_Chip").filter(has_text=re.compile(r"upload (resume|cv)", re.IGNORECASE))
    if chip.count():
        with drawer.page.expect_file_chooser(timeout=10_000) as chooser:
            chip.first.click()
        chooser.value.set_files(str(resume))
    else:
        uploader = find_all(drawer, "file_input", visible=False)
        if uploader is None:
            raise Unanswerable("no upload control in the questionnaire")
        uploader.first.set_input_files(str(resume))
    _submit(drawer, question)
    return Answer(value=resume.name, source="RESUME_FILE")


def fill_answer(drawer: Locator, question: Question, answer: Answer) -> None:
    if question.kind == "text":
        _type(drawer, question, answer.value)
    else:
        controls = find_all(drawer, "radio" if question.kind == "choice" else "checkbox", visible=False)
        if controls:
            _check_option(drawer, controls, answer.value)
        elif not _click_chip(drawer, answer.value):
            # not among the chips on screen: search for it, then pick the matching suggestion
            _type(drawer, question, answer.value)
            drawer.page.wait_for_timeout(1200)
            _click_chip(drawer, answer.value)
    _submit(drawer, question)


def _check_option(drawer: Locator, controls: Locator, value: str) -> None:
    picked = next((controls.nth(i) for i in range(controls.count()) if _label(drawer, controls.nth(i)) == value), None)
    if picked is None:
        raise Unanswerable(f"option {value!r} not on the page")
    if not picked.is_checked():
        # Naukri hides the real input (display: none) and styles its <label>: click what a person clicks
        label = drawer.locator(f"label[for='{picked.get_attribute('id')}']")
        (label.first if label.count() else picked).click()


def _type(drawer: Locator, question: Question, value: str) -> None:
    box = find(drawer, "text_input")
    if box is None:
        raise Unanswerable(f"no input box for {question.text!r}")
    # Keystrokes only: Naukri's chat box ignores programmatic edits (fill), and enables Save on keyup.
    # Verified live 2026-10-15: click + keyboard.type → text shows and Save enables.
    box.click()
    box.press("ControlOrMeta+a")
    box.press("Backspace")
    drawer.page.keyboard.type(value, delay=60)


def _click_chip(drawer: Locator, value: str) -> bool:
    """Click the chip whose text is exactly `value` (case-insensitive). False if none is on screen."""
    chips = find_all(drawer, "chip")
    if chips is None:
        return False
    match = chips.filter(has_text=re.compile(rf"^\s*{re.escape(value)}\s*$", re.IGNORECASE))
    if not match.count():
        return False
    match.first.click()
    return True


_SAVE_ENABLED = "e => !e.closest('.disabled') && getComputedStyle(e).pointerEvents !== 'none'"


def _submit(drawer: Locator, question: Question) -> None:
    """Click Save once it's enabled, unless the bot already took the answer (ticking a box can send it)."""
    deadline = time.monotonic() + SAVE_WAIT_SECONDS
    while time.monotonic() < deadline:
        current = read_question(drawer)
        if current is None or current.text != question.text:
            return  # moved on: typing the next question, or already showing it
        send = find(drawer, "send")
        if send is not None and send.evaluate(_SAVE_ENABLED):
            send.click(timeout=5000)
            return
        drawer.page.wait_for_timeout(300)
    raise Unanswerable(f"Save stayed disabled after answering {question.text!r}")


def _wait_for_result(page: Page, job: Job, mapper: AnswerMapper) -> Outcome:
    qa: list[dict] = []
    last_question, answered_at, seen_once = None, 0.0, None
    deadline = time.monotonic() + RESULT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if find(page, "success") or find(page, "already_applied"):
            return Outcome("applied", "applied on Naukri", save_evidence(page, job.id, "applied"), qa)
        drawer = find(page, "drawer")
        question = read_question(drawer) if drawer else None
        if question and question.text != last_question and question.text != seen_once:
            seen_once = question.text  # answer only once the same question reads identically twice
            page.wait_for_timeout(700)
            continue
        if question and question.text == last_question and time.monotonic() - answered_at > ANSWER_ACCEPT_SECONDS:
            note = f"Naukri didn't accept the answer to {question.text!r}"
            return Outcome("manual", note, save_evidence(page, job.id, "not-accepted", True), qa)
        if question and question.text != last_question:
            if len(qa) >= MAX_QUESTIONS:
                return Outcome("manual", "questionnaire longer than expected", save_evidence(page, job.id, "q", True), qa)
            try:
                if _RESUME_REQUEST.search(question.text):
                    answer = upload_resume(drawer, question, job, mapper.profile)
                else:
                    answer = mapper.answer(question, job)
                    fill_answer(drawer, question, answer)
            except (Unanswerable, llm.LLMError) as e:
                if not question.skippable:
                    note = f"Naukri asked {question.text!r}: {e}"
                    return Outcome("manual", note, save_evidence(page, job.id, "question", True), qa)
                # optional question none of your facts answers: skipping claims nothing, guessing might
                find(drawer, "skip").click()
                answer = Answer(value="(skipped)", source="SKIPPED")
            qa.append({"question": question.text, "options": question.options,
                       "answer": answer.value, "source": answer.source})
            last_question, answered_at = question.text, time.monotonic()
        page.wait_for_timeout(1000)
    return Outcome("failed", "no confirmation within 90s of clicking Apply", save_evidence(page, job.id, "timeout", True), qa)


def apply_to_job(page: Page, job: Job, mapper: AnswerMapper, *, live: bool) -> Outcome:
    page.goto(job.url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(random.randint(*SETTLE_MS))
    state = detect_state(page)
    if state == PageState.LOGGED_OUT:
        return Outcome("logged_out", "Naukri session expired", save_evidence(page, job.id, "logged-out"))
    if state == PageState.ALREADY_APPLIED:
        return Outcome("already_applied", "already applied on Naukri", save_evidence(page, job.id, "already"))
    if state == PageState.CLOSED:
        return Outcome("closed", "job closed on Naukri", save_evidence(page, job.id, "closed"))
    if state == PageState.COMPANY_SITE:
        return Outcome("company_site", "Naukri sends this one to the company's own site",
                       save_evidence(page, job.id, "company-site"))
    if state == PageState.UNKNOWN:
        return Outcome("manual", "couldn't recognise this Naukri page (screenshot + HTML saved)",
                       save_evidence(page, job.id, "unknown", html=True))
    if not live:
        return Outcome("dry_run", "dry run: Apply button found, not clicked", save_evidence(page, job.id, "dry-run"))
    find(page, "apply").click()
    outcome = _wait_for_result(page, job, mapper)
    if outcome.status in ("manual", "failed"):
        # Naukri can accept the application first and ask follow-ups afterwards (its profile-update bot,
        # live 2026-10-15), while the button only turns into "Applied" on reload. Check before reporting.
        page.reload(wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(random.randint(*SETTLE_MS))
        if detect_state(page) == PageState.ALREADY_APPLIED:
            note = f"applied on Naukri; then: {outcome.note} (left unanswered)"
            return Outcome("applied", note, save_evidence(page, job.id, "applied"), outcome.qa)
    return outcome

