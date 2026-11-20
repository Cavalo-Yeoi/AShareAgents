"""Typed provenance fields shared by page ingestion and retrieval filters."""
from datetime import date


UNVERIFIED_DATE_STATUSES = frozenset({"unknown", "unverified", "date_unverified"})
VERIFIED_DATE_STATUSES = frozenset({"verified", "official_verified", "verified_official"})


def normalize_evidence_metadata(metadata):
    """Keep allowlisted, JSON-safe evidence without accepting arbitrary objects."""
    result = {}
    for key in ("issuer", "source_url", "file_sha256", "raw_path", "doc_id", "date_status",
                "expire_date", "revision_date", "section_title", "parser_version"):
        if key in metadata:
            value = metadata[key]
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            result[key] = value.strip()
    if "date_status" in result:
        result["date_status"] = result["date_status"].lower()
    for key in ("page_number", "page_start", "page_end"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            result[key] = value
    if result.get("page_start", 1) > result.get("page_end", result.get("page_start", 1)):
        raise ValueError("page_end must not precede page_start")
    for key in ("expire_date", "revision_date"):
        if result.get(key):
            date.fromisoformat(result[key])
    if "quality_flags" in metadata:
        flags = metadata["quality_flags"]
        if not isinstance(flags, (list, tuple)) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError("quality_flags must be a list of strings")
        result["quality_flags"] = list(dict.fromkeys(flag.strip() for flag in flags if flag.strip()))
    return result
