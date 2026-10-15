"""Profile → ATS-friendly HTML → A4 PDF (rendered by Playwright's Chromium).

Tailoring produces a modified copy of the Profile and renders it through the same template.
"""

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape
from playwright.sync_api import sync_playwright

from jobagent.profile import Profile

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _display_url(url: str) -> str:
    return url.removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")


_HYPHENATED = re.compile(r"[^\s<>]*\w-\w[^\s<>]*")


def _nobreak(text: str) -> Markup:
    """Escape, then keep hyphenated words ("react-native-reanimated") on one line."""
    escaped = str(escape(text))
    return Markup(_HYPHENATED.sub(lambda m: f'<span class="nw">{m.group(0)}</span>', escaped))


_env.filters["display_url"] = _display_url
_env.filters["nobreak"] = _nobreak


def render_html(profile: Profile) -> str:
    return _env.get_template("resume.html.j2").render(p=profile)


def render_pdf(profile: Profile, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(profile)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(
                path=str(out_path),
                format="A4",
                print_background=True,
                margin={"top": "12mm", "bottom": "12mm", "left": "14mm", "right": "14mm"},
            )
        finally:
            browser.close()
    return out_path
