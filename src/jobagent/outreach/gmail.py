"""Gmail over SMTP (send) and IMAP (replies, bounces) with an App Password.

App Passwords need 2-Step Verification on the Google account; create one at
https://myaccount.google.com/apppasswords and put it in .env as GMAIL_APP_PASSWORD.
"""

import email
import imaplib
import os
import smtplib
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
IMAP_HOST = "imap.gmail.com"
IMAP_TIMEOUT_SECONDS = 60
PLACEHOLDER = "REPLACE_ME"


def credentials() -> tuple[str, str] | None:
    address = os.environ.get("GMAIL_ADDRESS", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")  # Google shows it in groups of four
    if not address or "@" not in address or not password or PLACEHOLDER in password:
        return None
    return address, password


def build_message(*, sender: str, sender_name: str, to: str, subject: str, body: str,
                  attachment: Path | None = None, in_reply_to: str | None = None) -> EmailMessage:
    message = EmailMessage()
    message["From"] = formataddr((sender_name, sender))
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2])
    if in_reply_to:  # follow-up: same thread
        message["In-Reply-To"] = in_reply_to
        message["References"] = in_reply_to
    message.set_content(body)
    if attachment is not None:
        message.add_attachment(attachment.read_bytes(), maintype="application", subtype="pdf", filename=attachment.name)
    return message


def send(message: EmailMessage) -> None:
    address, password = credentials() or ("", "")
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.login(address, password)
        smtp.send_message(message)


def check_login() -> None:
    """Raises if Gmail rejects the App Password (IMAP and SMTP both checked; nothing is sent)."""
    address, password = credentials() or ("", "")
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.login(address, password)
    imap = imaplib.IMAP4_SSL(IMAP_HOST, timeout=IMAP_TIMEOUT_SECONDS)
    imap.login(address, password)
    imap.logout()


def imap_quote(text: str) -> str:
    """An IMAP quoted string: inner quotes and backslashes escaped (else Gmail answers "Could not parse")."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Mailbox:
    """Read-only INBOX search with Gmail's own query syntax (X-GM-RAW)."""

    def __enter__(self) -> "Mailbox":
        address, password = credentials() or ("", "")
        # a timeout, so a connection that dies (Mac asleep mid-check) fails this tick instead of hanging every later one
        self.imap = imaplib.IMAP4_SSL(IMAP_HOST, timeout=IMAP_TIMEOUT_SECONDS)
        self.imap.login(address, password)
        self.imap.select("INBOX", readonly=True)
        return self

    def __exit__(self, *_exc) -> None:
        try:
            self.imap.logout()
        except imaplib.IMAP4.error:
            pass

    def search(self, gmail_query: str) -> list[bytes]:
        status, data = self.imap.uid("SEARCH", "X-GM-RAW", imap_quote(gmail_query))
        return data[0].split() if status == "OK" and data and data[0] else []

    def read(self, uid: bytes) -> tuple[str, str]:
        """(subject, plain-text body) of one message."""
        status, data = self.imap.uid("FETCH", uid, "(BODY.PEEK[])")
        if status != "OK" or not data or not isinstance(data[0], tuple):
            return "", ""
        message = email.message_from_bytes(data[0][1])
        parts = message.walk() if message.is_multipart() else [message]
        text = ""
        for part in parts:
            if part.get_content_type() in ("text/plain", "message/delivery-status"):
                payload = part.get_payload(decode=True) or b""
                text += payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        return str(message.get("Subject", "")), text
