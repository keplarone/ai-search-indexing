#!/usr/bin/env python3
"""Stealth headless miner for publicly listed digital-agency contacts.

Discovers agency sites from public search indexes, reads company names,
canonical websites, and a fixed set of leadership titles, and appends each
profile to ``leads.json`` as soon as it is captured. When a page exposes a
contact form and no CAPTCHA, the script types the Forensic Citation Leak
note into the visible text fields and submits it.

Run from the workspace:

    python3 -m pip install -r requirements-lead-miner.txt
    python3 -m playwright install chromium
    python3 lead_miner.py

``LEAD_LOCATION``, ``LEAD_MAX``, and ``LEAD_HEADLESS`` override the defaults.
``LEAD_MINER_SENDER_NAME`` and ``LEAD_MINER_SENDER_EMAIL`` are what the
contact form receives. The sender email is left blank when it is unset,
rather than inventing an address.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import random
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import playwright.async_api
from playwright.async_api import (
    Locator,
    Page,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Playwright also injects --enable-automation, which re-enables the signal
# this flag is meant to remove. The launch call drops that default.
CHROMIUM_ARGS: tuple[str, ...] = (
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
)

AGENCY_QUERIES: tuple[str, ...] = (
    "Performance Marketing Agency",
    "Enterprise SEO Firm",
)

TITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bChief Technology Officer\b", re.IGNORECASE), "CTO"),
    (re.compile(r"\bCTO\b"), "CTO"),
    (re.compile(r"\bVice President of SEO\b", re.IGNORECASE), "VP of SEO"),
    (re.compile(r"\bVP of SEO\b", re.IGNORECASE), "VP of SEO"),
    (re.compile(r"\bHead of Performance\b", re.IGNORECASE), "Head of Performance"),
    (re.compile(r"\bManaging Director\b", re.IGNORECASE), "Managing Director"),
)

FORENSIC_CITATION_LEAK_TEMPLATE = (
    "Hello, this is {sender} at IndexGrid. We review whether AI answer "
    "engines cite {company} ({website}) for commercial searches in {location}. "
    "When the brand is absent, those engines name other firms instead. That "
    "gap is a citation leak. I can run the IndexGrid audit and send the "
    "keyword-level report."
)

DIRECTORY_HOSTS: frozenset[str] = frozenset(
    {
        "clutch.co",
        "www.clutch.co",
        "sortlist.com",
        "www.sortlist.com",
        "goodfirms.co",
        "www.goodfirms.co",
        "upcity.com",
        "www.upcity.com",
        "designrush.com",
        "www.designrush.com",
    }
)

BLOCKED_HOSTS: frozenset[str] = frozenset(
    {
        "duckduckgo.com",
        "www.duckduckgo.com",
        "bing.com",
        "www.bing.com",
        "google.com",
        "www.google.com",
        "facebook.com",
        "www.facebook.com",
        "instagram.com",
        "www.instagram.com",
        "linkedin.com",
        "www.linkedin.com",
        "twitter.com",
        "x.com",
        "youtube.com",
        "www.youtube.com",
        "wikipedia.org",
        "en.wikipedia.org",
        "yelp.com",
        "www.yelp.com",
    }
)

TEAM_LINK = re.compile(r"\b(team|leadership|our people|about)\b", re.IGNORECASE)
CONTACT_LINK = re.compile(r"\b(contact|get in touch|work with us)\b", re.IGNORECASE)
VISIT_WEBSITE = re.compile(
    r"^(visit( our)? website|visit site|company website|website)$",
    re.IGNORECASE,
)
CAPTCHA_SELECTORS: tuple[str, ...] = (
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[src*='challenges.cloudflare.com']",
    ".g-recaptcha",
    ".h-captcha",
)
SKIPPED_INPUT_TYPES: frozenset[str] = frozenset(
    {"hidden", "password", "file", "checkbox", "radio", "submit", "button", "image"}
)

LOGGER = logging.getLogger("lead_miner")
_CONFIG: contextvars.ContextVar[ScraperConfig | None] = contextvars.ContextVar(
    "scraper_config",
    default=None,
)
_STORE: contextvars.ContextVar[LeadStore | None] = contextvars.ContextVar(
    "lead_store",
    default=None,
)


class ScraperConfig:
    """Operational knobs for one mining run."""

    target_location: str
    max_leads: int
    headless_mode: bool

    def __init__(
        self,
        target_location: str = "New York",
        max_leads: int = 50,
        headless_mode: bool = True,
    ) -> None:
        location = target_location.strip()
        if not location:
            raise ValueError("target_location cannot be blank")
        if isinstance(max_leads, bool) or not isinstance(max_leads, int):
            raise TypeError("max_leads must be an integer")
        if max_leads < 1:
            raise ValueError("max_leads must be at least 1")
        if not isinstance(headless_mode, bool):
            raise TypeError("headless_mode must be a bool")
        self.target_location = location
        self.max_leads = max_leads
        self.headless_mode = headless_mode


def active_config() -> ScraperConfig:
    config = _CONFIG.get()
    if config is None:
        config = ScraperConfig()
        _CONFIG.set(config)
    return config


def leads_path() -> Path:
    override = os.getenv("LEADS_PATH", "").strip()
    if override:
        return Path(override)
    return Path.cwd() / "leads.json"


# ---------------------------------------------------------------------------
# Lead file
# ---------------------------------------------------------------------------


class LeadStore:
    """JSON array on disk, rewritten atomically after every new profile."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._run_new = 0

    @property
    def run_count(self) -> int:
        return self._run_new

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            backup = self.path.with_suffix(".json.bak")
            os.replace(self.path, backup)
            LOGGER.error("leads file was not valid JSON; moved it to %s", backup)
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def contains(self, website_url: str) -> bool:
        key = canonical_website(website_url)
        if not key:
            return False
        with self._lock:
            return any(canonical_website(str(item.get("website_url", ""))) == key for item in self._read())

    def append(self, lead: dict[str, Any]) -> None:
        with self._lock:
            leads = self._read()
            leads.append(lead)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(leads, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self._run_new += 1
        LOGGER.info(
            "captured %s %s",
            lead.get("company_name", ""),
            lead.get("website_url", ""),
        )


def active_store() -> LeadStore:
    store = _STORE.get()
    if store is None:
        store = LeadStore(leads_path())
        _STORE.set(store)
    return store


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


def canonical_website(url: str) -> str:
    """Reduce a URL to scheme plus host so deep links collapse to one lead."""

    if not url:
        return ""
    candidate = url.strip()
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname.lower()
    host = host.removeprefix("www.")
    return f"https://{host}"


def canonical_title(value: str) -> str:
    """Return the public title label, or an empty string when it is not one we track."""

    text = " ".join(value.split())
    if not text:
        return ""
    for pattern, label in TITLE_RULES:
        if pattern.search(text):
            return label
    return ""


def _looks_like_name(value: str) -> bool:
    parts = value.replace(",", " ").split()
    if not 2 <= len(parts) <= 4:
        return False
    return all(re.fullmatch(r"[A-Z][A-Za-z'.-]+", part) for part in parts)


def people_from_text(text: str) -> list[tuple[str, str]]:
    """Pull (name, title) pairs out of visible page text. Missing names are empty."""

    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    previous = ""
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        name = ""
        title = ""
        for separator in (" - ", " – ", " — ", " | ", ", "):
            if separator not in line:
                continue
            left, right = line.split(separator, 1)
            right_title = canonical_title(right)
            left_title = canonical_title(left)
            if right_title and not left_title:
                name = left.strip()
                title = right_title
                break
            if left_title and not right_title and _looks_like_name(right):
                name = right.strip()
                title = left_title
                break
        if not title:
            title = canonical_title(line)
            if title and _looks_like_name(previous):
                name = previous
        if title:
            if name and not _looks_like_name(name):
                name = ""
            pair = (name, title)
            if pair not in seen:
                seen.add(pair)
                found.append(pair)
        previous = line
    return found


def people_from_json_ld(payload: Any) -> list[tuple[str, str]]:
    """Read Person and Employee nodes that publish one of the tracked titles."""

    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        raw_type = node.get("@type", node.get("type", ""))
        if isinstance(raw_type, list):
            type_names = [str(item) for item in raw_type]
        else:
            type_names = [str(raw_type)]
        if any(name.lower() in {"person", "employee"} for name in type_names):
            title = canonical_title(str(node.get("jobTitle") or ""))
            name = " ".join(str(node.get("name") or "").split())
            if title:
                if name and not _looks_like_name(name):
                    name = ""
                pair = (name, title)
                if pair not in seen:
                    seen.add(pair)
                    found.append(pair)
        for value in node.values():
            if isinstance(value, (dict, list)):
                walk(value)

    walk(payload)
    return found


def classify_field(attrs: dict[str, str]) -> str:
    """Map a form control to a delivery field. Unknown controls return an empty string."""

    tag = attrs.get("tag", "")
    field_type = attrs.get("type", "")
    blob = " ".join(
        attrs.get(key, "")
        for key in ("type", "name", "id", "placeholder", "aria", "autocomplete")
    ).lower()
    if tag == "textarea" or any(
        token in blob for token in ("message", "comment", "inquiry", "enquiry", "details")
    ):
        return "message"
    if field_type == "email" or "email" in blob:
        return "email"
    if field_type == "tel" or any(token in blob for token in ("phone", "mobile", "tel")):
        return "phone"
    if any(token in blob for token in ("company", "organization", "organisation", "business")):
        return "company"
    if any(token in blob for token in ("subject", "topic")):
        return "subject"
    if any(token in blob for token in ("name", "firstname", "first_name", "lastname", "last_name")):
        return "name"
    return ""


def forensic_message(lead: dict[str, Any]) -> str:
    company = str(lead.get("company_name") or "").strip() or "your firm"
    website = str(lead.get("website_url") or "").strip() or "your site"
    location = str(lead.get("location") or "").strip() or "your market"
    sender = os.getenv("LEAD_MINER_SENDER_NAME", "IndexGrid").strip() or "IndexGrid"
    return FORENSIC_CITATION_LEAK_TEMPLATE.format(
        sender=sender,
        company=company,
        website=website,
        location=location,
    )


def delivery_values(lead: dict[str, Any]) -> dict[str, str]:
    company = str(lead.get("company_name") or "").strip()
    sender = os.getenv("LEAD_MINER_SENDER_NAME", "IndexGrid").strip() or "IndexGrid"
    return {
        "name": sender,
        "email": os.getenv("LEAD_MINER_SENDER_EMAIL", "").strip(),
        "company": "IndexGrid",
        "subject": f"Forensic Citation Leak: {company}".strip() if company else "Forensic Citation Leak",
        "message": forensic_message(lead),
        "phone": "",
    }


def _unwrap_search_href(href: str) -> str:
    if not href:
        return ""
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or ""):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    return href


def _is_agency_candidate(url: str) -> bool:
    canonical = canonical_website(url)
    if not canonical:
        return False
    host = urlparse(canonical).hostname or ""
    return host not in BLOCKED_HOSTS and host not in {item.removeprefix("www.") for item in BLOCKED_HOSTS}


def _is_directory_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    bare = host.removeprefix("www.")
    return host in DIRECTORY_HOSTS or bare in {item.removeprefix("www.") for item in DIRECTORY_HOSTS}


def _join_people(people: list[tuple[str, str]]) -> tuple[str, str]:
    if not people:
        return "", ""
    titles: list[str] = []
    first_name = ""
    seen: set[str] = set()
    for name, title in people:
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
        if name and not first_name:
            first_name = name
    return first_name, "; ".join(titles)


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------


async def _human_pause() -> None:
    """Wait between clicks and focus events so the session is not a tight loop."""

    await asyncio.sleep(random.uniform(1.5, 4.2))


@asynccontextmanager
async def stealth_browser(config: ScraperConfig) -> AsyncIterator[Page]:
    """Launch one Linux Chromium context and close it when the run ends."""

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=config.headless_mode,
            args=list(CHROMIUM_ARGS),
            ignore_default_args=["--enable-automation"],
        )
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = await context.new_page()
        page.set_default_timeout(10_000)
        page.set_default_navigation_timeout(10_000)
        try:
            yield page
        finally:
            await context.close()
            await browser.close()


async def _goto(page: Page, url: str) -> bool:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
    except PlaywrightTimeoutError:
        LOGGER.warning("navigation timed out for %s", url)
        return False
    except playwright.async_api.Error as exc:
        LOGGER.warning("navigation failed for %s (%s)", url, exc.__class__.__name__)
        return False
    return True


async def _selector_ready(page: Page, selector: str) -> bool:
    try:
        await page.wait_for_selector(selector, timeout=10_000)
    except PlaywrightTimeoutError:
        LOGGER.warning("selector timed out: %s", selector)
        return False
    return True


async def _safe_text(locator: Locator) -> str:
    try:
        if await locator.count() == 0:
            return ""
        return " ".join((await locator.first.inner_text()).split())
    except playwright.async_api.Error:
        return ""


async def _safe_attribute(locator: Locator, name: str) -> str:
    try:
        if await locator.count() == 0:
            return ""
        value = await locator.first.get_attribute(name)
    except playwright.async_api.Error:
        return ""
    return (value or "").strip()


# ---------------------------------------------------------------------------
# Discovery and extraction
# ---------------------------------------------------------------------------


async def _collect_result_hrefs(page: Page, selector: str) -> list[str]:
    anchors = page.locator(selector)
    count = await anchors.count()
    found: list[str] = []
    seen: set[str] = set()
    for index in range(count):
        href = await anchors.nth(index).get_attribute("href") or ""
        absolute = urljoin(page.url, _unwrap_search_href(href))
        canonical = canonical_website(absolute)
        if not _is_agency_candidate(canonical) or canonical in seen:
            continue
        seen.add(canonical)
        found.append(canonical)
    return found


async def _discover_from_duckduckgo(page: Page, search_text: str) -> list[str]:
    target = "https://html.duckduckgo.com/html/?q=" + quote_plus(search_text)
    LOGGER.info("searching duckduckgo for %s", search_text)
    if not await _goto(page, target):
        return []
    if not await _selector_ready(page, ".result"):
        return []
    return await _collect_result_hrefs(page, ".result__a")


async def _discover_from_bing(page: Page, search_text: str) -> list[str]:
    target = "https://www.bing.com/search?q=" + quote_plus(search_text)
    LOGGER.info("searching bing for %s", search_text)
    if not await _goto(page, target):
        return []
    if not await _selector_ready(page, "li.b_algo h2 a"):
        return []
    return await _collect_result_hrefs(page, "li.b_algo h2 a")


async def _directory_outbound_link(page: Page) -> str:
    """On a B2B directory profile, follow the agency's own website when it is labeled."""

    anchors = page.locator("a[href]")
    count = min(await anchors.count(), 100)
    current = canonical_website(page.url)
    for index in range(count):
        anchor = anchors.nth(index)
        text = await _safe_text(anchor)
        if not text or len(text) > 40 or VISIT_WEBSITE.fullmatch(text) is None:
            continue
        href = await anchor.get_attribute("href") or ""
        absolute = urljoin(page.url, href)
        canonical = canonical_website(absolute)
        if not _is_agency_candidate(canonical) or canonical == current:
            continue
        if _is_directory_host(canonical):
            continue
        return canonical
    return ""


async def _company_name(page: Page) -> str:
    site_name = await _safe_attribute(page.locator("meta[property='og:site_name']"), "content")
    if site_name:
        return site_name[:200]
    title = ""
    try:
        title = " ".join((await page.title()).split())
    except playwright.async_api.Error:
        title = ""
    if title:
        for separator in (" | ", " - ", " – ", " — "):
            if separator in title:
                title = title.split(separator, 1)[0].strip()
                break
        if title:
            return title[:200]
    heading = await _safe_text(page.locator("h1"))
    return heading[:200]


async def _json_ld_people(page: Page) -> list[tuple[str, str]]:
    scripts = page.locator("script[type='application/ld+json']")
    try:
        count = min(await scripts.count(), 20)
    except playwright.async_api.Error:
        return []
    people: list[tuple[str, str]] = []
    for index in range(count):
        raw = ""
        try:
            raw = await scripts.nth(index).inner_text()
        except playwright.async_api.Error:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        people.extend(people_from_json_ld(payload))
    return people


async def _visible_people(page: Page) -> list[tuple[str, str]]:
    try:
        text = await page.locator("body").inner_text()
    except playwright.async_api.Error:
        return []
    return people_from_text(text[:50_000])


def _merge_people(*groups: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for pair in group:
            if pair in seen:
                continue
            seen.add(pair)
            merged.append(pair)
    return merged


async def _internal_links(page: Page, kind: re.Pattern[str]) -> list[str]:
    origin = canonical_website(page.url)
    anchors = page.locator("a[href]")
    try:
        count = min(await anchors.count(), 150)
    except playwright.async_api.Error:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for index in range(count):
        anchor = anchors.nth(index)
        href = await anchor.get_attribute("href") or ""
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        text = await _safe_text(anchor)
        absolute = urljoin(page.url, href)
        if canonical_website(absolute) != origin:
            continue
        path = urlparse(absolute).path.lower()
        if kind.search(text) is None and kind.search(path.replace("-", " ").replace("/", " ")) is None:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        found.append(absolute)
        if len(found) >= 2:
            break
    return found


async def _form_present(page: Page) -> bool:
    form = page.locator("form").filter(has=page.locator("textarea, input[type='email']"))
    try:
        return await form.count() > 0
    except playwright.async_api.Error:
        return False


async def _inspect_agency(page: Page, website: str, query: str, config: ScraperConfig) -> dict[str, Any] | None:
    await _human_pause()
    if not await _goto(page, website):
        return None
    try:
        await page.wait_for_selector("body", timeout=10_000)
    except PlaywrightTimeoutError:
        return None

    if _is_directory_host(page.url):
        external = await _directory_outbound_link(page)
        if external:
            await _human_pause()
            if not await _goto(page, external):
                return None

    website_url = canonical_website(page.url) or website
    company_name = await _company_name(page)
    people = _merge_people(await _json_ld_people(page), await _visible_people(page))
    team_urls = await _internal_links(page, TEAM_LINK)
    contact_urls = await _internal_links(page, CONTACT_LINK)
    form_found = await _form_present(page)

    for team_url in team_urls:
        await _human_pause()
        if not await _goto(page, team_url):
            continue
        people = _merge_people(people, await _json_ld_people(page), await _visible_people(page))

    for contact_url in contact_urls:
        await _human_pause()
        if not await _goto(page, contact_url):
            continue
        if await _form_present(page):
            form_found = True
            break

    contact_name, professional_title = _join_people(people)
    return {
        "company_name": company_name or "",
        "website_url": website_url or "",
        "professional_title": professional_title or "",
        "contact_name": contact_name or "",
        "query": query or "",
        "location": config.target_location,
        "contact_form_detected": form_found,
        "delivery_status": "not_attempted",
        "captured_at": datetime.now(UTC).isoformat(),
    }


async def _captcha_present(page: Page) -> bool:
    for selector in CAPTCHA_SELECTORS:
        try:
            if await page.locator(selector).count() > 0:
                return True
        except playwright.async_api.Error:
            continue
    return False


async def _field_attributes(field: Locator) -> dict[str, str]:
    try:
        raw = await field.evaluate(
            """(element) => ({
                tag: (element.tagName || '').toLowerCase(),
                type: (element.getAttribute('type') || '').toLowerCase(),
                name: element.getAttribute('name') || '',
                id: element.getAttribute('id') || '',
                placeholder: element.getAttribute('placeholder') || '',
                aria: element.getAttribute('aria-label') || '',
                autocomplete: element.getAttribute('autocomplete') || ''
            })"""
        )
    except playwright.async_api.Error:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value or "") for key, value in raw.items()}


async def execute_outbound_delivery(page: Page, lead_data: dict[str, Any]) -> dict[str, Any]:
    """Type the Forensic Citation Leak note into a detected contact form and submit it.

    Password, file, and consent controls are never touched. A CAPTCHA stops
    delivery; this function does not try to solve one. Each click and each
    focus is separated by a 1.5 to 4.2 second pause.
    """

    if await _captcha_present(page):
        LOGGER.info("skipping delivery for %s because a captcha is present", lead_data.get("website_url", ""))
        lead_data["delivery_status"] = "captcha"
        return lead_data

    form = page.locator("form").filter(has=page.locator("textarea, input[type='email']")).first
    try:
        if await form.count() == 0:
            lead_data["delivery_status"] = "no_form"
            return lead_data
    except playwright.async_api.Error:
        lead_data["delivery_status"] = "no_form"
        return lead_data

    values = delivery_values(lead_data)
    if not values["email"]:
        LOGGER.warning("LEAD_MINER_SENDER_EMAIL is unset; the email field will be left blank")

    fields = form.locator("input:visible, textarea:visible")
    try:
        count = min(await fields.count(), 12)
    except playwright.async_api.Error:
        lead_data["delivery_status"] = "failed"
        return lead_data

    populated = 0
    for index in range(count):
        field = fields.nth(index)
        attributes = await _field_attributes(field)
        if attributes.get("type", "") in SKIPPED_INPUT_TYPES:
            continue
        kind = classify_field(attributes)
        text = values.get(kind, "")
        if not kind or not text:
            continue
        try:
            if await field.is_disabled():
                continue
            await field.click()
            await _human_pause()
            await field.fill("")
            await field.fill(text)
            populated += 1
            await _human_pause()
        except playwright.async_api.Error:
            LOGGER.warning("could not fill %s on %s", kind, lead_data.get("website_url", ""))

    if populated == 0:
        lead_data["delivery_status"] = "no_fields"
        return lead_data

    submit = form.locator("button[type='submit'], input[type='submit']").first
    try:
        if await submit.count() == 0:
            submit = form.get_by_role("button", name=re.compile(r"send|submit|contact", re.IGNORECASE)).first
        if await submit.count() == 0:
            lead_data["delivery_status"] = "populated"
            return lead_data
        await submit.click()
        await _human_pause()
    except playwright.async_api.Error:
        lead_data["delivery_status"] = "populated"
        return lead_data

    lead_data["delivery_status"] = "submitted"
    LOGGER.info("delivered citation note to %s", lead_data.get("website_url", ""))
    return lead_data


async def extract_agency_leads(page: Page, query: str) -> list[dict[str, Any]]:
    """Crawl public indexes for ``query`` and persist each new agency profile.

    DuckDuckGo's HTML index is tried first. Bing is the fallback when that
    page has no result nodes within 10 seconds. A missing company name, URL,
    or title becomes an empty string and the loop continues.
    """

    config = active_config()
    store = active_store()
    search_text = f'{query} "{config.target_location}"'
    discovered: list[str] = []
    for opener in (_discover_from_duckduckgo, _discover_from_bing):
        try:
            discovered = await opener(page, search_text)
        except PlaywrightTimeoutError:
            LOGGER.warning("index timed out for %s", query)
            discovered = []
        if discovered:
            break

    captured: list[dict[str, Any]] = []
    for website in discovered:
        if store.run_count >= config.max_leads:
            break
        if store.contains(website):
            LOGGER.info("skipping duplicate %s", website)
            continue
        try:
            lead = await _inspect_agency(page, website, query, config)
        except PlaywrightTimeoutError:
            LOGGER.warning("timed out while reading %s", website)
            continue
        except playwright.async_api.Error as exc:
            LOGGER.warning("browser error on %s (%s)", website, exc.__class__.__name__)
            continue
        if lead is None:
            continue
        lead["company_name"] = str(lead.get("company_name") or "")
        lead["website_url"] = str(lead.get("website_url") or "")
        lead["professional_title"] = str(lead.get("professional_title") or "")
        lead["contact_name"] = str(lead.get("contact_name") or "")
        if not lead["website_url"] or store.contains(lead["website_url"]):
            continue
        if lead.get("contact_form_detected"):
            try:
                lead = await execute_outbound_delivery(page, lead)
            except PlaywrightTimeoutError:
                lead["delivery_status"] = "failed"
            except playwright.async_api.Error:
                lead["delivery_status"] = "failed"
        store.append(lead)
        captured.append(lead)
    return captured


def _config_from_environment() -> ScraperConfig:
    raw_max = os.getenv("LEAD_MAX", "50").strip() or "50"
    try:
        max_leads = int(raw_max)
    except ValueError as exc:
        raise SystemExit(f"LEAD_MAX must be an integer, received {raw_max!r}") from exc
    headless_raw = os.getenv("LEAD_HEADLESS", "true").strip().lower()
    return ScraperConfig(
        target_location=os.getenv("LEAD_LOCATION", "New York"),
        max_leads=max_leads,
        headless_mode=headless_raw not in {"0", "false", "no", "off"},
    )


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _config_from_environment()
    store = LeadStore(leads_path())
    _CONFIG.set(config)
    _STORE.set(store)
    LOGGER.info(
        "mining up to %s agencies in %s",
        config.max_leads,
        config.target_location,
    )
    try:
        async with stealth_browser(config) as page:
            for query in AGENCY_QUERIES:
                if store.run_count >= config.max_leads:
                    break
                await extract_agency_leads(page, query)
    except playwright.async_api.Error as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "browserType.launch" in message:
            raise SystemExit(
                "Chromium is not installed for Playwright. "
                "Run: python3 -m playwright install chromium"
            ) from exc
        raise
    LOGGER.info("finished with %s new leads in %s", store.run_count, store.path)


if __name__ == "__main__":
    asyncio.run(main())
