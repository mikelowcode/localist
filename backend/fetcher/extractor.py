"""
LORA Fetcher — content extraction (readability-lxml + lxml.html)

Single responsibility: given raw HTML bytes, extract clean article text,
title, author, and publish date.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import lxml.html
from readability import Document

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    title:          str
    author:         str
    date_published: str
    cleaned_text:   str
    word_count:     int


def extract(html: bytes, url: str = "") -> ExtractedContent:
    """
    Extract clean article content from raw HTML bytes.

    Uses readability-lxml for main content detection, then lxml.html
    to strip residual tags from the readability output.

    Parameters
    ----------
    html :
        Raw HTML bytes from the HTTP response.
    url :
        Source URL — passed to readability for relative link resolution.

    Returns
    -------
    ExtractedContent

    Raises
    ------
    ValueError
        If readability produces empty content (e.g. login walls, paywalls).
    """
    # --- readability pass ---------------------------------------------------
    # readability-lxml 0.8.x expects a decoded string, not raw bytes.
    html_str = html.decode("utf-8", errors="replace")
    doc    = Document(html_str, url=url)
    title  = (doc.title() or "").strip()
    # doc.summary() returns an HTML fragment — strip tags next
    summary_html = doc.summary(html_partial=True)

    if not summary_html:
        raise ValueError("readability returned empty content — possible paywall or login wall")

    # --- lxml tag stripping -------------------------------------------------
    root        = lxml.html.fromstring(summary_html)
    cleaned_text = _clean_text(root.text_content())

    if not cleaned_text:
        raise ValueError("extraction produced empty text after tag stripping")

    # --- metadata extraction ------------------------------------------------
    # Parse the original HTML (not summary) for meta tags
    try:
        full_doc = lxml.html.fromstring(html)
        author   = _extract_meta(full_doc, ["author", "article:author"])
        date     = _extract_meta(full_doc, [
            "article:published_time",
            "datePublished",
            "pubdate",
            "date",
        ])
    except Exception:
        author = ""
        date   = ""

    word_count = len(cleaned_text.split())

    logger.debug(
        "extract() ← title=%r  words=%d  author=%r  date=%r",
        title, word_count, author, date,
    )

    return ExtractedContent(
        title          = title,
        author         = author,
        date_published = date,
        cleaned_text   = cleaned_text,
        word_count     = word_count,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip control characters."""
    # Collapse runs of whitespace (including \n, \t) to single spaces
    text = re.sub(r"\s+", " ", raw)
    return text.strip()


def _extract_meta(doc: lxml.html.HtmlElement, names: list[str]) -> str:
    """
    Extract the first matching meta tag content from an lxml document.
    Checks both name= and property= attributes.
    """
    for name in names:
        # <meta name="..."> or <meta property="...">
        for attr in ("name", "property"):
            nodes = doc.xpath(
                f'//meta[@{attr}="{name}"]/@content'
            )
            if nodes:
                return str(nodes[0]).strip()
    return ""
