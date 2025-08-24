import time
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_text(elem: Optional[ET.Element]) -> str:
    return _clean(elem.text if elem is not None else "")


def _get_link(entry: ET.Element, rel_value: str) -> str:
    for link in entry.findall("atom:link", NS):
        rel = link.attrib.get("rel", "")
        if rel == rel_value:
            href = link.attrib.get("href", "")
            if href:
                return href
    return ""


def _get_pdf_link(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", NS):
        title = link.attrib.get("title", "")
        typ = link.attrib.get("type", "")
        href = link.attrib.get("href", "")
        if (title and title.lower() == "pdf") or (typ == "application/pdf"):
            if href:
                return href
    return ""


def _retry_get(
    url: str,
    params: dict,
    timeout: int = 30,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> requests.Response:
    last_exc = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff**i)
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed without exception")


def search_arxiv(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start = 0
    max_results = max(1, min(200, int(max_results)))
    rows_per_page = min(50, max_results)

    while len(results) < max_results:
        size = min(rows_per_page, max_results - len(results))
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": size,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = _retry_get(ARXIV_API, params=params)
        root = ET.fromstring(resp.text)

        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for entry in entries:
            eid = _extract_text(entry.find("atom:id", NS))
            title = _extract_text(entry.find("atom:title", NS))
            summary = _extract_text(entry.find("atom:summary", NS))
            published = _extract_text(entry.find("atom:published", NS))
            year = ""
            if published:
                m = re.match(r"(\d{4})", published)
                year = m.group(1) if m else ""

            authors_list = []
            for a in entry.findall("atom:author", NS):
                name = _extract_text(a.find("atom:name", NS))
                if name:
                    authors_list.append(name)
            authors = ", ".join(authors_list)

            link_abs = _get_link(entry, "alternate")
            link_pdf = _get_pdf_link(entry)

            cats = []
            for c in entry.findall("atom:category", NS):
                term = c.attrib.get("term", "")
                if term:
                    cats.append(term)
            categories = ", ".join(cats)

            results.append(
                {
                    "id": eid,
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "published": published,
                    "year": year,
                    "link_abs": link_abs,
                    "link_pdf": link_pdf,
                    "categories": categories,
                }
            )

        start += size
        time.sleep(0.2)

    return results
