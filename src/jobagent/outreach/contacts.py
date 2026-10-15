"""Who to email about a job: an address found by code in the posting or on the company's own website.

Claude never supplies an address. When neither the board nor the posting gives the company's website,
Claude may name its domain (web search, given only the company name and city), and code then checks
that the site really is that company's before scraping its careers/contact pages.
"""

import re
import urllib.robotparser
from dataclasses import dataclass

import dns.exception
import dns.resolver
import httpx
from selectolax.lexbor import LexborHTMLParser

from jobagent import llm
from jobagent.discover.normalize import FREE_MAIL, extract_emails, norm_company
from jobagent.models import Job

USER_AGENT = "Mozilla/5.0 (personal job search; contact page lookup)"
SITE_PAGES = ["careers", "career", "jobs", "join-us", "contact", "contact-us", "about", "about-us", ""]
MAX_PAGE_BYTES = 1_500_000

# Addresses in postings that aren't hiring contacts (accommodation, compliance, privacy boilerplate…)
_NOT_HIRING = re.compile(
    r"^(no-?reply|do-?not-?reply|accessib|accom+o?dat|complian|privacy|legal|security|abuse|support|help"
    r"|sales|billing|finance|accounts|press|media|investor|ir\b|webmaster|postmaster|admin|marketing|partner"
    r"|feedback|grievance|ethics|whistle|gdpr|dpo|unsubscribe|applicationassistance|bounce)",
    re.IGNORECASE,
)
_HIRING = re.compile(r"career|job|hr\b|^hr|talent|recruit|hiring|people|resume|\bcv|hire|opportunit", re.IGNORECASE)
_GENERIC = ("info", "contact", "hello", "enquir", "inquir", "office", "mail")
_GENERIC_NAME_WORDS = {
    "technologies", "technology", "tech", "solutions", "solution", "software", "systems", "services", "labs",
    "lab", "global", "group", "consulting", "infotech", "digital", "company", "enterprises", "international",
    "india", "private", "limited", "pvt", "ltd", "the", "and", "of",
}


@dataclass
class Contact:
    address: str
    source: str   # "job post" or the page URL it was found on
    rank: int     # lower is better


def registrable_domain(host: str) -> str:
    """careers.examplelabs.com → examplelabs.com; www.acme.co.in → acme.co.in."""
    labels = host.lower().strip(".").removeprefix("www.").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in {"co", "com", "org", "net", "ac", "gov", "edu"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _domain_of(url_or_host: str) -> str:
    host = re.sub(r"^[a-z]+://", "", url_or_host.strip().lower()).split("/")[0].split(":")[0]
    return registrable_domain(host)


def classify(address: str, *, from_posting: bool) -> int | None:
    """Rank an address (lower = better), or None if it must not be emailed about a job."""
    local, _, domain = address.lower().partition("@")
    # check every part: "global.compliance@" and "accessibility.support@" are boilerplate, not recruiters
    if domain in FREE_MAIL or any(_NOT_HIRING.match(part) for part in re.split(r"[._-]", local) if part):
        return None
    if _HIRING.search(local):
        return 0 if from_posting else 1
    if local.startswith(_GENERIC):
        return 2 if from_posting else 3
    # a named person: fine when the posting itself gives it for applications, not when scraped off a site
    return 1 if from_posting else None


def has_mx(domain: str) -> bool:
    try:
        return bool(dns.resolver.resolve(domain, "MX", lifetime=8))
    except (dns.exception.DNSException, OSError):
        return False


def _distinctive_words(company: str) -> list[str]:
    return [w for w in norm_company(company).split() if len(w) >= 3 and w not in _GENERIC_NAME_WORDS]


def base_url(domain: str, client: httpx.Client) -> str | None:
    """The site's working origin: many only answer on www. (seen live: some 404 or refuse connections on the bare domain)."""
    for origin in (f"https://{domain}", f"https://www.{domain}"):
        try:
            response = client.get(origin + "/")
        except httpx.HTTPError:
            continue
        if response.status_code < 400:
            return f"{response.url.scheme}://{response.url.host}"
    return None


def verify_site(domain: str, company: str, client: httpx.Client) -> bool:
    """The site plausibly belongs to `company`: a distinctive word of its name is in the domain or page title."""
    words = _distinctive_words(company)
    if not words:
        return False
    if any(word in domain.replace("-", "") for word in words):
        return True
    origin = base_url(domain, client)
    if origin is None:
        return False
    try:
        page = client.get(origin + "/")
    except httpx.HTTPError:
        return False
    title = LexborHTMLParser(page.text[:200_000]).css_first("title")
    return title is not None and any(word in title.text().lower() for word in words)


_LOOKUP_RULES = """You find a company's official website. Search the web for the company described, then \
reply with only its main website domain (for example: example.com), or NONE if you can't identify it with \
confidence. Never guess a domain, and never give a job board, LinkedIn, Naukri or other directory site."""


def lookup_website(company: str, location: str) -> str | None:
    text, _cost = llm.ask(
        task="website_lookup",
        system=_LOOKUP_RULES,
        content=f"Company: {company}\nLocation: {location or 'India'}",
        effort="low",
        max_tokens=2000,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 2}],
    )
    found = re.findall(r"\b((?:[a-z0-9-]+\.)+[a-z]{2,})\b", text.lower())
    if not found or "none" in text.lower().split():
        return None
    return registrable_domain(found[-1])


def company_domain(job: Job, client: httpx.Client, *, allow_lookup: bool) -> tuple[str | None, str]:
    """(domain, how it was found). Board-given site first, then a company address in the posting, then search."""
    if job.company_website:
        return _domain_of(job.company_website), "board"
    for address in job.emails:
        domain = address.rpartition("@")[2]
        if domain not in FREE_MAIL:
            return registrable_domain(domain), "posting address"
    if allow_lookup:
        domain = lookup_website(job.company, job.location or job.query_city or "")
        if domain and verify_site(domain, job.company, client):
            return domain, "web search (verified)"
    return None, "not found"


def scrape_site(domain: str, client: httpx.Client) -> list[Contact]:
    """Hiring/generic addresses on the company's own careers/contact pages (same domain only, robots.txt obeyed)."""
    origin = base_url(domain, client)
    if origin is None:
        return []
    robots = urllib.robotparser.RobotFileParser()
    try:
        robots.parse(client.get(f"{origin}/robots.txt").text.splitlines())
    except httpx.HTTPError:
        robots.parse([])
    found: list[Contact] = []
    for path in SITE_PAGES:
        url = f"{origin}/{path}"
        if not robots.can_fetch(USER_AGENT, url):
            continue
        try:
            response = client.get(url)
        except httpx.HTTPError:
            continue
        if response.status_code != 200 or "html" not in response.headers.get("content-type", "html"):
            continue
        html = response.text[:MAX_PAGE_BYTES]
        tree = LexborHTMLParser(html)
        mailtos = [a.attributes.get("href", "")[7:].split("?")[0] for a in tree.css("a[href^='mailto:']")]
        for address in dict.fromkeys(extract_emails(" ".join(mailtos) + " " + tree.text(separator=" "))):
            if registrable_domain(address.rpartition("@")[2]) != domain:
                continue
            rank = classify(address, from_posting=False)
            if rank is not None and all(c.address != address for c in found):
                found.append(Contact(address, str(response.url), rank))
        if any(c.rank <= 1 for c in found):
            break  # a careers/HR address: good enough, stop fetching
    return found


def best_contact(job: Job, *, allow_lookup: bool, client: httpx.Client | None = None) -> tuple[Contact | None, str]:
    """The address to email for this job, and a short note on how (or why not) it was found."""
    candidates = [
        Contact(address, "job post", rank)
        for address in job.emails
        if (rank := classify(address, from_posting=True)) is not None
    ]
    site_note = ""
    own_client = client is None
    client = client or httpx.Client(timeout=12, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    try:
        if not any(c.rank == 0 for c in candidates):
            domain, how = company_domain(job, client, allow_lookup=allow_lookup)
            if domain:
                candidates += scrape_site(domain, client)
                site_note = f" (site {domain} via {how})"
            elif not candidates:
                return None, "no company website or hiring address found"
    finally:
        if own_client:
            client.close()
    for contact in sorted(candidates, key=lambda c: c.rank):
        if has_mx(contact.address.rpartition("@")[2]):
            return contact, f"{contact.address} from {contact.source}{site_note}"
    return None, f"no hiring address with a working mail domain{site_note}"
