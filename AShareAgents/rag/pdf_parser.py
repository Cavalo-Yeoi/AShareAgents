"""Offline page extraction with explicit failures and unmodified extracted text.

No OCR or table reconstruction is performed. Layout text preserves available
spacing; quality flags make this limitation visible to downstream annotation.
"""
import hashlib
import re
from pathlib import Path


PARSER_VERSION = "page-extraction-v1"


def text_quality_flags(content):
    """Flag mixed-script font mapping corruption in Chinese financial documents.

    A flag is a quarantine signal, not an attempted OCR correction. Latin words,
    financial symbols and a few foreign names alone must not trigger it.
    """
    visible = [c for c in content if not c.isspace()]
    unusual = sum('\u0400' <= c <= '\u1fff' for c in visible)
    chinese = sum('\u4e00' <= c <= '\u9fff' for c in visible)
    flags = []
    if unusual >= 8 and unusual / max(1, len(visible)) >= .02 and unusual > chinese / 3:
        flags.append('suspicious_unicode_mapping')
    if '\ufffd' in content:
        flags.append('replacement_characters')
    return flags


def _result(path):
    path = Path(path)
    result = {"pages": [], "errors": [], "file_sha256": "", "page_count": 0,
              "parser_version": PARSER_VERSION}
    try:
        data = path.read_bytes()
        result["file_sha256"] = hashlib.sha256(data).hexdigest()
    except OSError as exc:
        result["errors"].append({"page_number": None, "error": str(exc)})
        return result, None
    return result, data


def _page(content, number, metadata, result, path, flags):
    evidence = {**metadata, "page_number": number,
                "file_sha256": result["file_sha256"], "raw_path": str(path),
                "parser_version": PARSER_VERSION,
                "quality_flags": list(dict.fromkeys([*metadata.get("quality_flags", []), *flags]))}
    return {"page_number": number, "content": content,
            "quality_flags": evidence["quality_flags"], "metadata": evidence}


def parse_pdf(path: Path, metadata: dict) -> dict:
    """Return all pages, including empty/failed pages, and structured errors.

    ``metadata['source']`` is preserved across pages for stable document identity.
    An unreadable file has no pages and a document-level error (page_number=None).
    """
    result, data = _result(path)
    if data is None:
        return result
    try:
        from io import BytesIO
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        result["page_count"] = len(reader.pages)
    except Exception as exc:
        result["errors"].append({"page_number": None, "error": str(exc)})
        return result
    for index in range(result["page_count"]):
        flags, content = [], ""
        try:
            page = reader.pages[index]
            try:
                content = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                content = page.extract_text() or ""
                flags.append("layout_mode_unavailable")
            if not content.strip():
                flags.append("empty_text")
                try:
                    if len(page.images):
                        flags.append("scan_suspected_no_ocr")
                except Exception as exc:
                    flags.append("image_inspection_error")
                    result["errors"].append({"page_number": index + 1, "error": str(exc)})
            if re.search(r"\S[ \t]{2,}\S|\|.+\|", content):
                flags.append("table_layout_unverified")
            flags.extend(text_quality_flags(content))
        except Exception as exc:
            flags.extend(["extraction_error", "empty_text"])
            result["errors"].append({"page_number": index + 1, "error": str(exc)})
        result["pages"].append(_page(content, index + 1, metadata, result, path, flags))
    if not result["page_count"]:
        result["errors"].append({"page_number": None, "error": "PDF has no pages"})
    return result


def parse_html(path: Path, metadata: dict) -> dict:
    """Extract an official HTML article as one logical page; retain its heading."""
    result, data = _result(path)
    if data is None:
        return result
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, "html.parser")
        heading = soup.find("h1") or soup.find("title")
        title = heading.get_text(" ", strip=True) if heading else ""
        for node in soup.select("script, style, nav, footer, header, noscript"):
            node.decompose()
        article = soup.select_one("article, #UCAP-CONTENT, .TRS_Editor, #zoom, main, [role=main]")
        flags = []
        if article is None:
            article = soup.body or soup
            flags.append("article_container_fallback")
        content = article.get_text("\n", strip=True)
        if title and title not in content:
            content = title + "\n" + content
        if article.find("table") is not None:
            flags.append("table_layout_unverified")
        if not content.strip():
            flags.append("empty_text")
        result["page_count"] = 1
        result["pages"].append(_page(content, 1, metadata, result, path, flags))
    except Exception as exc:
        result["errors"].append({"page_number": None, "error": str(exc)})
    return result
