"""Command-line entry point: `uv run jobagent <command>`."""

import argparse
import sys
from pathlib import Path

from jobagent import llm, settings, setup_status
from jobagent.profile import load_profile, save_profile


def cmd_hello(_args: argparse.Namespace) -> int:
    text, cost = llm.ask(
        task="hello",
        system="You are the assistant inside a personal job-search agent for the Indian job market.",
        content="Reply with one short sentence confirming you're ready.",
        effort="low",
        max_tokens=1000,
    )
    print(text.strip())
    print(f"(model {llm.MODEL}, cost ≈ ${cost:.4f})")
    return 0


def cmd_check(_args: argparse.Namespace) -> int:
    items = setup_status.setup_items()
    width = max(len(item.name) for item in items)
    for item in items:
        mark = "✓" if item.ok else ("✗" if item.required else "–")
        print(f"  [{mark}] {item.name:<{width}}  {item.detail}{'' if item.required else '  (optional)'}")
    ready = all(item.ok for item in items if item.required)
    print("\nReady for live runs." if ready else "\nNot ready for live runs: --dry-run only.")
    return 0 if ready else 1


def cmd_init(args: argparse.Namespace) -> int:
    from jobagent import onboarding
    from jobagent.tailor.render import render_pdf

    resume = Path(args.resume).expanduser()
    if not resume.exists():
        print(f"error: {resume} not found", file=sys.stderr)
        return 1
    existing = settings.PROFILE_PATH
    if existing.exists() and not load_profile(existing).is_placeholder() and not args.force:
        print(f"error: {existing} already holds your profile; pass --force to overwrite it", file=sys.stderr)
        return 1

    print(f"Reading {resume.name} with Claude…")
    profile = onboarding.extract_profile(resume)
    save_profile(profile, existing)
    render_pdf(profile, settings.MASTER_RESUME_PDF)
    print(
        f"Saved {existing.relative_to(settings.ROOT)}: {len(profile.experience)} roles, "
        f"{len(profile.bullets())} bullets, {len(profile.all_skills())} skills.\n"
        f"Rendered {settings.MASTER_RESUME_PDF.relative_to(settings.ROOT)}.\n"
        "Compare it with your original; fix anything in profile.json, then run `jobagent render`."
    )
    return 0


def cmd_render(_args: argparse.Namespace) -> int:
    from jobagent.tailor.render import render_pdf

    profile = load_profile(settings.PROFILE_PATH)
    out = render_pdf(profile, settings.MASTER_RESUME_PDF)
    print(f"Rendered {out.relative_to(settings.ROOT)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from datetime import datetime

    from jobagent import pipeline
    from jobagent.discover.normalize import IST

    settings.rotate_logs()
    print(f"=== jobagent run · {datetime.now(IST):%d %b %Y %H:%M} IST ===")
    try:
        record = pipeline.run(smoke=args.smoke)
    except Exception as e:  # last line of defence: record it, alert, exit non-zero
        import traceback

        from jobagent import alerts, db
        from jobagent.models import Run

        traceback.print_exc()
        with db.session() as session:
            crash = Run(errors=[f"crashed: {type(e).__name__}: {e}"[:400]])
            session.add(crash)
            session.commit()
            alerts.from_run(session, crash)
        return 1
    if record is None:
        return 0
    counts = ", ".join(f"{site} {n}" for site, n in sorted(record.source_counts.items())) or "none"
    print(
        f"\nRun #{record.id}: found {record.found} ({counts}); {record.new} new, {record.duplicates} duplicates\n"
        f"  skipped {record.skipped} · scored {record.scored} · matches {record.matches} · "
        f"tailored {record.tailored} · applied {record.applied} · emails drafted {record.drafted} · "
        f"Claude cost ≈ ${record.cost_usd:.2f}"
    )
    for note in record.notes:
        print(f"  note: {note}")
    for error in record.errors:
        print(f"  error: {error}")
    return 1 if record.found == 0 and record.errors else 0


def cmd_tailor(args: argparse.Namespace) -> int:
    from jobagent import db
    from jobagent.models import Job
    from jobagent.tailor import tailor

    profile = load_profile(settings.PROFILE_PATH)
    if profile.is_placeholder():
        print("error: config/profile.json is still the sample; run `jobagent init` first", file=sys.stderr)
        return 1
    with db.session() as session:
        job = session.get(Job, args.job_id)
        if job is None:
            print(f"error: no job {args.job_id}", file=sys.stderr)
            return 1
        print(f"Tailoring for {job.title} at {job.company}…")
        result = tailor.build_tailored(profile, tailor.plan_for(job, tailor.tailoring_system_prompt(profile)))
        path = tailor.save(job, profile, result)
        session.commit()
    print(f"Summary: {'original kept' if result.summary_kept_original else result.profile.summary}")
    print(f"Why: {result.plan.rationale}")
    for note in result.notes:
        print(f"  note: {note}")
    print(f"Saved {path.relative_to(settings.ROOT)}")
    return 0


def cmd_naukri(args: argparse.Namespace) -> int:
    from jobagent.apply import browser, naukri

    if args.action == "inspect" and not args.target:
        print("error: give a job id or a Naukri job URL to inspect", file=sys.stderr)
        return 1
    with browser.naukri_context() as context:
        page = browser.first_page(context)
        if args.action == "login":
            print("Log in to Naukri in the Chrome window that just opened (enter the OTP if asked).")
            print("Waiting up to 10 minutes…")
            if not naukri.wait_for_login(page):
                print("error: didn't see a successful login; run `jobagent naukri login` again", file=sys.stderr)
                return 1
            print("Logged in. The session is saved in data/browser/naukri for future runs.")
            return 0
        if not naukri.is_logged_in(page):
            print("Naukri: logged out. Run `jobagent naukri login`.")
            return 1
        if args.action == "status":
            print("Naukri: logged in.")
            return 0
        url, tag = args.target, "url"
        if args.target.isdigit():
            from jobagent import db
            from jobagent.models import Job

            with db.session() as session:
                job = session.get(Job, int(args.target))
            if job is None:
                print(f"error: no job {args.target}", file=sys.stderr)
                return 1
            url, tag = job.url, args.target
        info = naukri.inspect(page, url, tag)
    print(f"Detected: {info['state']}")
    for button in info["buttons"]:
        print(f"  <{button['tag'].lower()} id={button['id']!r} class={button['cls']!r}> {button['text']}")
    print(f"Saved {settings.stored_path(info['screenshot'])} (+ .html)")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from jobagent import pipeline

    if args.live and args.job is None:
        print("error: --live needs --job (it's for submitting one chosen job while you watch)", file=sys.stderr)
        return 1
    tally = pipeline.apply_only(job_id=args.job, live=args.live)
    print(f"Applied: {tally.applied}")
    for note in tally.notes:
        print(f"  note: {note}")
    return 0


def _outreach_context():
    from jobagent.config import load_answers, load_config

    profile = load_profile(settings.PROFILE_PATH)
    if profile.is_placeholder():
        raise SystemExit("error: config/profile.json is still the sample; run `jobagent init` first")
    return load_config(), load_answers(), profile


def cmd_outbox(_args: argparse.Namespace) -> int:
    from jobagent import db
    from jobagent.outreach import outbox

    from datetime import datetime

    from jobagent.discover.normalize import IST

    settings.rotate_logs()
    config, answers, profile = _outreach_context()
    with db.session() as session:
        outbox.tick(session, config, answers, profile, log=lambda line: print(f"{datetime.now(IST):%d %b %H:%M} {line}"))
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    from jobagent import db, digest
    from jobagent.config import load_config

    settings.rotate_logs()
    config = load_config()
    with db.session() as session:
        if args.print:
            subject, body = digest.build(session, config)
            print(f"Subject: {subject}\n\n{body}")
            return 0
        print(digest.send(session, config))
        if removed := digest.prune_screenshots(session):
            print(f"housekeeping: removed {removed} old screenshot(s)")
    return 0


def cmd_outreach(args: argparse.Namespace) -> int:
    from jobagent import db, pipeline
    from jobagent.match import MatchContext
    from jobagent.models import Job

    config, answers, profile = _outreach_context()
    ctx = MatchContext(config=config, profile=profile, answers=answers)
    with db.session() as session:
        drafted = pipeline.draft_outreach(session, ctx, pipeline.Tally(), job_id=args.job)
        job = session.get(Job, args.job)
        print(f"Contact: {job.outreach_note if job else 'no such job'}")
        for email in drafted:
            print(f"\n[{email.status}] To: {email.to_address}  ({email.contact_source})\nSubject: {email.subject}\n")
            print(email.body)
            if email.note:
                print(f"\nnote: {email.note}")
    return 0


def cmd_gmail(args: argparse.Namespace) -> int:
    import imaplib
    import smtplib

    from jobagent.outreach import gmail

    creds = gmail.credentials()
    if creds is None:
        print("error: set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env "
              "(App Password: https://myaccount.google.com/apppasswords)", file=sys.stderr)
        return 1
    try:
        gmail.check_login()
    except (smtplib.SMTPException, imaplib.IMAP4.error, OSError) as e:
        print(f"error: Gmail login failed for {creds[0]}: {e}", file=sys.stderr)
        return 1
    if args.action == "check":
        print(f"Gmail: login OK for {creds[0]} (SMTP + IMAP). Nothing was sent.")
        return 0
    message = gmail.build_message(sender=creds[0], sender_name="Job Agent", to=creds[0],
                                  subject="Job agent test email",
                                  body="This is a test from your job agent. If you can read this, sending works.")
    gmail.send(message)
    print(f"Sent a test email to yourself ({creds[0]}). Check your inbox (and spam).")
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    from jobagent import schedule

    action = {"install": schedule.install, "uninstall": schedule.uninstall, "status": schedule.status}
    try:
        print(action[args.action]())
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_dashboard(_args: argparse.Namespace) -> int:
    import subprocess

    app = settings.ROOT / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app)]
    # localhost only and no "Deploy" button: the dashboard shows personal data
    command += ["--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"]
    command += ["--client.toolbarMode", "viewer"]
    return subprocess.call(command)


def main() -> int:
    parser = argparse.ArgumentParser(prog="jobagent", description="Personal job-hunt agent (India).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check the Claude API connection").set_defaults(func=cmd_hello)
    sub.add_parser("check", help="list setup items still missing or placeholder").set_defaults(func=cmd_check)
    init = sub.add_parser("init", help="build config/profile.json from your resume, render the master PDF")
    init.add_argument("--resume", required=True, help="resume file (.pdf, .docx, .txt, .md)")
    init.add_argument("--force", action="store_true", help="overwrite an existing (non-sample) profile")
    init.set_defaults(func=cmd_init)
    sub.add_parser("render", help="re-render the master resume PDF from profile.json").set_defaults(func=cmd_render)
    run = sub.add_parser("run", help="discover fresh jobs, filter them, and score them with Claude")
    run.add_argument("--smoke", action="store_true", help="tiny run: first role, first city, 5 jobs per site")
    run.set_defaults(func=cmd_run)
    tailor = sub.add_parser("tailor", help="(re)build the tailored resume for one job")
    tailor.add_argument("job_id", type=int, help="job id, as shown in the dashboard")
    tailor.set_defaults(func=cmd_tailor)
    naukri_cmd = sub.add_parser("naukri", help="Naukri session: login, status, or inspect a job page (read-only)")
    naukri_cmd.add_argument("action", choices=["login", "status", "inspect"])
    naukri_cmd.add_argument("target", nargs="?", default="", help="for inspect: a job id or a Naukri job URL")
    naukri_cmd.set_defaults(func=cmd_naukri)
    apply_cmd = sub.add_parser("apply", help="run only the apply steps: Naukri, Greenhouse, Lever (dry run unless live)")
    apply_cmd.add_argument("--job", type=int, help="just this job id")
    apply_cmd.add_argument("--live", action="store_true", help="with --job: submit for real even if dry_run is on")
    apply_cmd.set_defaults(func=cmd_apply)
    sub.add_parser("outbox", help="check replies, queue follow-ups, send one email (run every 15 min)").set_defaults(func=cmd_outbox)
    outreach_cmd = sub.add_parser("outreach", help="find a contact and draft an email for one job")
    outreach_cmd.add_argument("--job", type=int, required=True, help="job id, as shown in the dashboard")
    outreach_cmd.set_defaults(func=cmd_outreach)
    digest_cmd = sub.add_parser("digest", help="email yourself today's summary (scheduled at 20:45)")
    digest_cmd.add_argument("--print", action="store_true", help="show it here instead of emailing it")
    digest_cmd.set_defaults(func=cmd_digest)
    gmail_cmd = sub.add_parser("gmail", help="check the Gmail App Password, or send yourself a test email")
    gmail_cmd.add_argument("action", choices=["check", "test"])
    gmail_cmd.set_defaults(func=cmd_gmail)
    schedule = sub.add_parser("schedule", help="run `jobagent run` automatically through the day (launchd)")
    schedule.add_argument("action", choices=["install", "uninstall", "status"])
    schedule.set_defaults(func=cmd_schedule)
    sub.add_parser("dashboard", help="open the web dashboard (localhost)").set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    try:
        return args.func(args)
    except llm.LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
