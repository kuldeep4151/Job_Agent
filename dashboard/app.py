"""Job Agent dashboard. Start with `uv run jobagent dashboard` (serves on localhost only)."""

import json
from datetime import timezone

import pandas as pd
import streamlit as st
from sqlmodel import select

from jobagent import db, guards, pipeline, settings, setup_status
from jobagent.agents import main as main_agent
from jobagent.agents import tracker
from jobagent.apply.answers import copy_ready_answers
from jobagent.config import load_answers, load_config
from jobagent.discover.normalize import IST
from jobagent.models import AgentRun, Application, AppStatus, Email, EmailStatus, Job, Run, State, utcnow
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


@st.cache_data(ttl=30)
def load_applications() -> pd.DataFrame:
    with db.session() as session:
        rows = [app.model_dump() for app in session.exec(select(Application)).all()]
    return pd.DataFrame(rows)


def load_sweeps() -> pd.DataFrame:
    with db.session() as session:
        rows = [run.model_dump() for run in session.exec(select(Run).order_by(Run.id.desc()).limit(30)).all()]
    return pd.DataFrame(rows)


def load_turns() -> pd.DataFrame:
    with db.session() as session:
        rows = [t.model_dump() for t in session.exec(select(AgentRun).order_by(AgentRun.id.desc()).limit(100)).all()]
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

    turns = load_turns()
    ticks = turns[turns["agent"] == "main"] if not turns.empty else turns
    if not ticks.empty:
        st.caption(f"Main agent, last tick {_ist(ticks.iloc[0]['started_at'])}: {ticks.iloc[0]['summary']}")

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
    tracked = applications[applications["job_id"] == job["id"]] if not applications.empty else applications
    if not tracked.empty:
        show_tracking(tracked.iloc[0])
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


def set_outcome(job_id: int, applied: bool) -> None:
    with db.session() as session:
        job = session.get(Job, job_id)
        if applied:
            job.state, job.applied_at, job.apply_note = State.APPLIED, utcnow(), "applied by you (marked in the dashboard)"
        else:
            job.state, job.skip_reason = State.SKIPPED, "not interested (dismissed in the dashboard)"
        session.commit()
        if applied:  # on the Tracker's list right away, not at its next turn
            tracker.sync_applications(session)
            tracker.export_csv(session)
    st.cache_data.clear()


def save_status(app_id: int, status: str, note: str) -> None:
    with db.session() as session:
        tracker.record_your_edit(session, session.get(Application, app_id), status, note)
    st.cache_data.clear()


def show_tracking(app: pd.Series) -> None:
    """Where this application stands, its history, and a way to set the status yourself."""
    app = app.astype(object).where(app.notna(), None)
    st.markdown("#### Tracker")
    st.markdown(f"**{app['status']}** since {_ist(app['status_at'])}" + (f": {app['status_note']}" if app["status_note"] else ""))
    history = pd.DataFrame(app["history"])
    if not history.empty:
        history["at"] = pd.to_datetime(history["at"]).map(_ist)
        st.dataframe(history[["at", "status", "by", "note"]], hide_index=True)
    statuses = [s.value for s in AppStatus]
    left, mid, right = st.columns([2, 4, 1])
    status = left.selectbox("Status", statuses, index=statuses.index(app["status"]), key=f"status-{app['id']}")
    note = mid.text_input("Note (optional)", key=f"status-note-{app['id']}")
    right.write("")
    if right.button("Save", key=f"status-save-{app['id']}"):
        save_status(int(app["id"]), status, note)
        st.rerun()


def show_needs_you(job: pd.Series) -> None:
    """Everything to apply by hand in a couple of minutes: where, the resume, and copy-ready answers."""
    job = job.astype(object).where(job.notna(), None)
    st.divider()
    st.subheader(f"{job['title']} · {job['company']}")
    st.info(job["apply_note"] or "Open the link and apply.")
    left, right = st.columns(2)
    left.link_button("Open application page", job["apply_target"] or job["url"])
    resume = settings.ROOT / job["resume_path"] if job["resume_path"] else settings.MASTER_RESUME_PDF
    _pdf_button(right, "Download resume for this job", resume, f"needs-{job['id']}")

    answers = copy_ready_answers(load_answers())
    if answers:
        st.markdown("**Your answers** (copy with the icon at the top right)")
        st.code("\n".join(f"{label}: {value}" for label, value in answers), language=None)
    about = job["tailored_summary"] or (load_profile(settings.PROFILE_PATH).summary if settings.PROFILE_PATH.exists() else "")
    if about:
        st.markdown("**About me** (for a summary / cover-note box)")
        st.code(about, language=None, wrap_lines=True)

    done, dismiss = st.columns(2)
    if done.button("Mark applied", key=f"applied-{job['id']}", type="primary"):
        set_outcome(int(job["id"]), applied=True)
        st.rerun()
    if dismiss.button("Not interested", key=f"dismiss-{job['id']}"):
        set_outcome(int(job["id"]), applied=False)
        st.rerun()


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
applications = load_applications()

# Deep link (e.g. from the daily digest): ?job=<id> shows that job above the tabs.
linked_id = st.query_params.get("job")
if linked_id and not jobs.empty:
    linked = jobs[jobs["id"].astype(str) == linked_id]
    if linked.empty:
        st.warning(f"No job with id {linked_id}.")
    else:
        if linked.iloc[0]["state"] == State.MANUAL:
            show_needs_you(linked.iloc[0])
        show_job(linked.iloc[0])

matches_tab, needs_tab, applications_tab, outreach_tab, agents_tab, all_tab = st.tabs(
    ["Matches", "Needs you", "Applications", "Outreach", "Agents", "All jobs"])

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

with needs_tab:
    waiting = jobs[jobs["state"] == State.MANUAL].sort_values("score", ascending=False) if not jobs.empty else jobs
    if waiting.empty:
        st.info("Nothing waiting for you. Jobs the agent can't apply to by itself (Workday, LinkedIn, company sites…) appear here.")
    else:
        st.caption(f"{len(waiting)} jobs to apply to by hand, best first. Select one for its link, resume and answers.")
        table = waiting.assign(where=waiting["ats"].fillna("?"))
        picked = st.dataframe(
            table[["score", "title", "company", "where", "seen"]], key="needs", hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row",
            column_config={"score": st.column_config.NumberColumn("Score", format="%d")},
        )
        if picked.selection.rows:
            show_needs_you(table.iloc[picked.selection.rows[0]])
            show_job(table.iloc[picked.selection.rows[0]])

with applications_tab:
    if applications.empty:
        st.info("No applications yet. The Tracker lists every job the agent applies to, every one you mark "
                "applied, and every company the Mailer emails.")
    else:
        counts = applications["status"].value_counts()
        st.caption(" · ".join(f"{status} {n}" for status, n in counts.items())
                   + f" · also in {settings.stored_path(tracker.CSV_PATH)}")
        picked_statuses = st.multiselect("Status", list(counts.index), placeholder="all")
        view = applications[applications["status"].isin(picked_statuses)] if picked_statuses else applications
        view = view.assign(applied=view["applied_at"].map(_ist).replace("", "earlier"), since=view["status_at"].map(_ist))
        view = view.sort_values("applied_at", ascending=False, na_position="last").reset_index(drop=True)
        picked = st.dataframe(
            view[["company", "title", "via", "applied", "status", "since", "status_note", "emailed_to"]],
            key="applications", hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            column_config={"status_note": "note", "emailed_to": "emailed"},
        )
        if tracker.CSV_PATH.exists():
            st.download_button("Download as CSV", tracker.CSV_PATH.read_bytes(), file_name="applications.csv",
                               mime="text/csv")
        if picked.selection.rows:
            job_row = jobs[jobs["id"] == view.iloc[picked.selection.rows[0]]["job_id"]]
            if not job_row.empty:
                show_job(job_row.iloc[0])
    if not jobs.empty:
        tried = jobs[(jobs["state"] == State.FAILED) | (jobs["dry_run_at"].notna() & (jobs["state"] != State.APPLIED))]
        if not tried.empty:
            with st.expander(f"Dry runs and failed attempts ({len(tried)})"):
                job_table(tried.sort_values("first_seen_at", ascending=False).reset_index(drop=True), "attempts")

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

with agents_tab:
    with db.session() as session:
        agent_rows = main_agent.status(session, pipeline.context())
        today = main_agent.ticks_today(session)
    offline = sum(t.outcome == "offline" for t in today)
    st.caption(f"The main agent ticks every 15 minutes: {len(today) - offline} ticks today"
               + (f" ({offline} more found no internet: Mac asleep)" if offline else "")
               + ". Each sub-agent with work gets a turn of a few minutes.")
    st.dataframe(
        pd.DataFrame([{
            "agent": row.agent.title,
            "on": "on" if row.enabled else "off",
            "last turn": _ist(row.last.started_at) if row.last else "never",
            "result": row.last.summary if row.last else "",
            "waiting": ", ".join(f"{name} {n}" for name, n in row.queue.items()) or "nothing",
            "next tick": row.waiting or "runs",
        } for row in agent_rows]),
        hide_index=True, width="stretch",
    )
    for row in agent_rows:
        if row.last and row.last.outcome == "error":
            st.error(f"{row.agent.title}, {_ist(row.last.started_at)}: {row.last.errors[0] if row.last.errors else ''}")
    recent = turns[turns["agent"] != "main"] if not turns.empty else turns
    if not recent.empty:
        st.markdown("#### Recent turns")
        st.dataframe(
            recent.assign(at=recent["started_at"].map(_ist))[["at", "agent", "outcome", "summary", "cost_usd"]],
            hide_index=True, width="stretch", column_config={"cost_usd": st.column_config.NumberColumn("Claude $", format="%.2f")},
        )
    sweeps = load_sweeps()
    if not sweeps.empty:
        with st.expander("Search sweeps"):
            table = sweeps.assign(
                started=sweeps["started_at"].map(_ist),
                progress=sweeps.apply(lambda r: "searching" if pd.notna(r["next_unit"]) else _ist(r["finished_at"]), axis=1),
                sources=sweeps["source_counts"].map(lambda c: ", ".join(f"{k} {v}" for k, v in sorted(c.items()))),
            )
            st.dataframe(table[["id", "started", "progress", "found", "new", "duplicates", "sources"]],
                         hide_index=True, width="stretch")
