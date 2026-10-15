"""Job Agent dashboard. Start with `uv run jobagent dashboard` (serves on localhost only)."""

import json
from datetime import timezone

import pandas as pd
import streamlit as st
from sqlmodel import select

from jobagent import db, guards, settings, setup_status
from jobagent.config import load_config
from jobagent.discover.normalize import IST
from jobagent.models import Email, EmailStatus, Job, Run, State
from jobagent.profile import load_profile

st.set_page_config(page_title="Job Agent", page_icon="💼", layout="wide")
config = load_config()


def _ist(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return value.astimezone(IST).strftime("%d %b %H:%M")


@st.cache_data(ttl=30)
def load_jobs() -> pd.DataFrame:
    with db.session() as session:
        rows = [job.model_dump() for job in session.exec(select(Job)).all()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["seen"] = frame["first_seen_at"].map(_ist)
    return frame


@st.cache_data(ttl=30)
def load_emails() -> pd.DataFrame:
    with db.session() as session:
        rows = [e.model_dump() for e in session.exec(select(Email).order_by(Email.created_at.desc())).all()]
    return pd.DataFrame(rows)


def cancel_email(email_id: int) -> None:
    with db.session() as session:
        email = session.get(Email, email_id)
        if email is not None and email.status == EmailStatus.DRAFT:
            email.status, email.note = EmailStatus.CANCELLED, "cancelled in the dashboard"
            session.commit()


def show_email(email: pd.Series, key: str) -> None:
    email = email.astype(object).where(email.notna(), None)
    when = f"sent {_ist(email['sent_at'])}" if email["sent_at"] is not None else f"drafted {_ist(email['created_at'])}"
    st.markdown(f"**{email['status']}** · {email['kind'].replace('_', '-')} · to `{email['to_address']}` · {when}")
    st.caption(f"Contact found in: {email['contact_source']}")
    if email["note"]:
        st.warning(email["note"])
    if email["replied_at"] is not None:
        st.success(f"Replied {_ist(email['replied_at'])}: {email['reply_snippet'] or ''}")
    if email["bounced_at"] is not None:
        st.error(f"Bounced {_ist(email['bounced_at'])}: this address is now suppressed")
    st.text(f"Subject: {email['subject']}\n\n{email['body']}")
    if email["attachment"]:
        st.caption(f"Attachment: {email['attachment']}")
    if email["status"] == EmailStatus.DRAFT and st.button("Cancel this email", key=f"cancel-{key}-{email['id']}"):
        cancel_email(int(email["id"]))
        st.rerun()


def load_runs() -> pd.DataFrame:
    with db.session() as session:
        rows = [run.model_dump() for run in session.exec(select(Run).order_by(Run.id.desc()).limit(50)).all()]
    return pd.DataFrame(rows)


# ---------- sidebar: mode, spend, last run, setup ----------
with st.sidebar:
    st.title("Job Agent")
    if config.mode.paused:
        st.error("Paused (kill switch on)")
    st.caption(("🧪 Dry run" if config.mode.dry_run else "🟢 Live") + f" · min score {config.matching.min_score}")

    spent, cap = guards.spent_today_usd(), config.caps.usd_per_day
    st.metric("Claude spend today", f"${spent:.2f}", help=f"Daily cap ${cap:.2f} (config.yaml → caps.usd_per_day)")
    st.progress(min(spent / cap, 1.0) if cap else 0.0)

    runs = load_runs()
    if not runs.empty:
        last = runs.iloc[0]
        st.caption(
            f"Last run {_ist(last['started_at'])}: {last['found']} found, {last['new']} new, "
            f"{last['matches']} matches"
        )

    st.subheader("Setup")
    for item in setup_status.setup_items():
        st.markdown(f"{'✅' if item.ok else '❌'} **{item.name}**  \n<small>{item.detail}</small>", unsafe_allow_html=True)

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


# ---------- job detail ----------
def show_job(job: pd.Series) -> None:
    job = job.astype(object).where(job.notna(), None)  # pandas turns missing values into NaN, which is truthy
    st.divider()
    st.subheader(f"{job['title']} · {job['company']}")
    links = [f"[Open on {job['source']}]({job['url']})"]
    if job["apply_url"]:
        links.append(f"[Company apply page]({job['apply_url']})")
    links += [f"[Also listed here]({url})" for url in job["also_seen_at"]]
    st.markdown(" · ".join(links))

    cols = st.columns(4)
    cols[0].metric("Score", "–" if job["score"] is None else int(job["score"]))
    cols[1].metric("Seniority", job["seniority_fit"] or "–")
    cols[2].metric("Experience asked", job["experience_text"] or "–")
    low, high = job["salary_min_lpa"], job["salary_max_lpa"]
    salary = f"{low or '?'}–{high or '?'} LPA" if low is not None or high is not None else "–"
    cols[3].metric("Salary", salary)

    if job["score_reason"]:
        st.info(job["score_reason"])
    if job["skip_reason"]:
        st.warning(f"Skipped: {job['skip_reason']}")
    for flag in job["flags"]:
        st.error(f"⚠️ {flag}")
    left, right = st.columns(2)
    left.markdown("**Matched must-haves**\n" + "".join(f"\n- {m}" for m in job["matched_must_haves"]))
    right.markdown("**Missing must-haves**\n" + "".join(f"\n- {m}" for m in job["missing_must_haves"]))
    st.markdown(f"**Your skills in this posting:** {', '.join(job['skill_overlap']) or '–'}")
    if job["apply_note"]:
        show_application(job)
    if job["outreach_note"]:
        st.markdown("#### Outreach")
        st.caption(f"Contact: {job['outreach_note']}")
        job_emails = emails[emails["job_id"] == job["id"]] if not emails.empty else emails
        for _, email in job_emails.iterrows():
            show_email(email, key="job")
    if job["resume_path"]:
        show_tailoring(job)
    if job["emails"]:
        st.markdown(f"**Contacts found in posting:** {', '.join(job['emails'])}")
    with st.expander("Job description"):
        st.text(job["description"])


def _pdf_button(column, label: str, path, key: str) -> None:
    if path.exists():
        column.download_button(label, path.read_bytes(), file_name=path.name, mime="application/pdf", key=key)
    else:
        column.caption(f"{label}: file missing ({path.name})")


def show_application(job: pd.Series) -> None:
    st.markdown("#### Application")
    when = f" on {_ist(job['applied_at'])}" if job["applied_at"] is not None else ""
    st.markdown(f"**{job['state']}**{when}: {job['apply_note']}")
    if job["apply_qa"]:
        st.dataframe(pd.DataFrame(json.loads(job["apply_qa"]))[["question", "answer", "source"]], hide_index=True)
    if job["apply_screenshot"]:
        shot = settings.ROOT / job["apply_screenshot"]
        if shot.exists():
            st.image(str(shot), caption="What the agent saw")


def show_tailoring(job: pd.Series) -> None:
    st.markdown("#### Tailored resume")
    if job["tailor_rationale"]:
        st.caption(job["tailor_rationale"])
    if job["tailor_notes"]:
        st.warning(f"Tailoring notes: {job['tailor_notes']}")
    original = load_profile(settings.PROFILE_PATH).summary if settings.PROFILE_PATH.exists() else ""
    left, right = st.columns(2)
    left.markdown("**Your original summary**")
    left.write(original)
    right.markdown("**Summary for this job**")
    right.write(job["tailored_summary"] or "_Original summary kept._")
    _pdf_button(left, "Download original resume", settings.MASTER_RESUME_PDF, f"master-{job['id']}")
    _pdf_button(right, "Download tailored resume", settings.ROOT / job["resume_path"], f"tailored-{job['id']}")


def job_table(frame: pd.DataFrame, key: str) -> None:
    if frame.empty:
        st.info("Nothing here yet.")
        return
    columns = ["score", "title", "company", "location", "source", "state", "seen", "url"]
    event = st.dataframe(
        frame[columns],
        key=key,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "score": st.column_config.NumberColumn("Score", format="%d"),
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        },
    )
    if event.selection.rows:
        show_job(frame.iloc[event.selection.rows[0]])


# ---------- main ----------
jobs = load_jobs()
emails = load_emails()

# Deep link (e.g. from the daily digest): ?job=<id> shows that job above the tabs.
linked_id = st.query_params.get("job")
if linked_id and not jobs.empty:
    linked = jobs[jobs["id"].astype(str) == linked_id]
    if linked.empty:
        st.warning(f"No job with id {linked_id}.")
    else:
        show_job(linked.iloc[0])

matches_tab, applications_tab, outreach_tab, all_tab, runs_tab = st.tabs(
    ["Matches", "Applications", "Outreach", "All jobs", "Runs"])

with matches_tab:
    if jobs.empty:
        st.info("No jobs yet. Run `uv run jobagent run`.")
    else:
        matches = jobs[jobs["state"].isin([State.SCORED, State.TAILORED])].sort_values("score", ascending=False)
        st.caption(f"{len(matches)} jobs at or above score {config.matching.min_score}. Select a row for details.")
        job_table(matches.reset_index(drop=True), "matches")

with all_tab:
    if not jobs.empty:
        c1, c2, c3 = st.columns([2, 2, 3])
        states = c1.multiselect("State", sorted(jobs["state"].unique()), placeholder="all")
        sources = c2.multiselect("Source", sorted(jobs["source"].unique()), placeholder="all")
        text = c3.text_input("Search title/company")
        view = jobs
        if states:
            view = view[view["state"].isin(states)]
        if sources:
            view = view[view["source"].isin(sources)]
        if text:
            needle = text.lower()
            view = view[(view["title"] + " " + view["company"]).str.lower().str.contains(needle, regex=False)]
        st.caption(f"{len(view)} of {len(jobs)} jobs")
        job_table(view.sort_values("first_seen_at", ascending=False).reset_index(drop=True), "all")

with applications_tab:
    if not jobs.empty:
        acted = jobs[jobs["state"].isin([State.APPLIED, State.MANUAL, State.FAILED]) | jobs["dry_run_at"].notna()]
        st.caption(
            f"Applied {int((acted['state'] == State.APPLIED).sum())} · needs you {int((acted['state'] == State.MANUAL).sum())}"
            f" · failed {int((acted['state'] == State.FAILED).sum())} · dry runs {int(acted['dry_run_at'].notna().sum())}"
        )
        job_table(acted.sort_values("first_seen_at", ascending=False).reset_index(drop=True), "applications")
    else:
        st.info("No applications yet.")

with outreach_tab:
    if emails.empty:
        st.info("No emails yet. Drafts appear here after a run finds a hiring contact for a match.")
    else:
        if not config.outreach.enabled:
            st.info("Outreach is off: drafts wait here and nothing is sent. Set `outreach.enabled: true` in config.yaml to send.")
        counts = emails["status"].value_counts().to_dict()
        replied = int(emails["replied_at"].notna().sum())
        st.caption(" · ".join(f"{k} {v}" for k, v in counts.items()) + f" · replies {replied}")
        companies = jobs.set_index("id")["company"].to_dict() if not jobs.empty else {}
        table = emails.assign(company=emails["job_id"].map(companies), created=emails["created_at"].map(_ist))
        event = st.dataframe(
            table[["status", "company", "to_address", "subject", "created"]], key="emails", hide_index=True,
            width="stretch", on_select="rerun", selection_mode="single-row",
        )
        if event.selection.rows:
            show_email(emails.iloc[event.selection.rows[0]], key="tab")

with runs_tab:
    if runs.empty:
        st.info("No runs yet.")
    else:
        table = runs.assign(
            started=runs["started_at"].map(_ist),
            sources=runs["source_counts"].map(lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items()))),
        )
        st.dataframe(
            table[["id", "started", "found", "new", "duplicates", "skipped", "scored", "matches", "cost_usd", "sources"]],
            hide_index=True,
            width="stretch",
        )
        latest = runs.iloc[0]
        for note in latest["notes"]:
            st.caption(f"Note (run {latest['id']}): {note}")
        for error in latest["errors"]:
            st.error(f"Run {latest['id']}: {error}")
