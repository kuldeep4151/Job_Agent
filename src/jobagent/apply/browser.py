"""The agent's own Chrome profile for Naukri (data/browser/naukri): you log in once, every run reuses it.

It is separate from your everyday Chrome profile. Visible by default: Naukri's bot protection answers
headless browsers with "Access Denied".
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import BrowserContext, Error as PlaywrightError, Page, sync_playwright

from jobagent import settings

NAUKRI_PROFILE_DIR = settings.DATA_DIR / "browser" / "naukri"


@contextmanager
def naukri_context(headless: bool = False, profile_dir: Path = NAUKRI_PROFILE_DIR) -> Iterator[BrowserContext]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = {"headless": headless, "viewport": {"width": 1366, "height": 900}}
    with sync_playwright() as pw:
        try:
            context = pw.chromium.launch_persistent_context(str(profile_dir), channel="chrome", **options)
        except PlaywrightError as e:
            if "is not found" not in str(e):  # e.g. the profile is already open in another run
                raise
            context = pw.chromium.launch_persistent_context(str(profile_dir), **options)  # no Google Chrome
        try:
            yield context
        finally:
            context.close()


def first_page(context: BrowserContext) -> Page:
    return context.pages[0] if context.pages else context.new_page()
