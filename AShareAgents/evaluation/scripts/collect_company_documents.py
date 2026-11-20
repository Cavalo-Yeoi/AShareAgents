"""Collect original company disclosures and independently preserved CNINFO metadata.

Run with the project Python and -B. Uses only public official HTTP endpoints.
No credentials, project database, or dependencies are modified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from collections import Counter
import hashlib
import io
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/rag_eval_v1"
RAW = DATA / "raw/company"
MANIFESTS = DATA / "manifests"
METADATA = MANIFESTS / "company_metadata"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
COMPANIES = [
    ("600519", "贵州茅台", "白酒"), ("002594", "比亚迪", "新能源汽车"),
    ("300750", "宁德时代", "动力电池"), ("688981", "中芯国际", "半导体"),
    ("600036", "招商银行", "银行"), ("601398", "工商银行", "银行"),
    ("000333", "美的集团", "家电"), ("600276", "恒瑞医药", "医药"),
    ("601899", "紫金矿业", "有色金属"), ("601012", "隆基绿能", "光伏"),
]
FIELDS = ["document_id", "title", "knowledge_type", "document_type", "ticker",
          "company_name", "industry", "issuer", "source_url", "source_level",
          "publish_date", "report_period", "effective_date", "expire_date",
          "downloaded_at", "original_filename", "file_sha256", "raw_path",
          "date_status", "date_evidence_url", "date_evidence_text", "collection_status",
          "source_record_id", "date_evidence_path", "pdf_pages", "pdf_metadata_title",
          "first_page_excerpt"]
LOCK = threading.Lock()
LOG = MANIFESTS / "company_collection_log.jsonl"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def log(event, **details):
    entry = {"at": utcnow(), "event": event, **details}
    with LOCK:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps(entry, ensure_ascii=True), flush=True)


def session():
    client = requests.Session()
    client.trust_env = False
    client.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    return client


def request(client, method, url, **kwargs):
    for attempt in range(3):
        try:
            result = client.request(method, url, timeout=(15, 25), **kwargs)
            result.raise_for_status()
            return result
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                raise
            log("request_retry", url=url, error=str(exc), attempt=attempt + 1)
            time.sleep(attempt + 1)


def clean_title(value):
    return re.sub("<[^>]+>", "", value).strip()


def fetch_metadata(client, ticker, orgid, cutoff):
    records = {}
    # Specific searches preserve older annual reports even for frequent disclosers.
    for keyword in ["年度报告", "半年度报告", "季度报告", "投资者关系", ""]:
        payload = {"pageNum": 1, "pageSize": 100, "column": "szse", "tabName": "fulltext",
                   "stock": f"{ticker},{orgid}", "searchkey": keyword, "secid": "",
                   "plate": "", "category": "", "trade": "", "seDate": f"2025-01-01~{cutoff}",
                   "sortName": "time", "sortType": "desc", "isHLtitle": "true"}
        label = {"年度报告": "annual", "半年度报告": "interim", "季度报告": "quarterly",
                 "投资者关系": "investor_relations", "": "recent"}[keyword]
        evidence_path = METADATA / f"{ticker}_{label}.json"
        if evidence_path.exists():
            data = json.loads(evidence_path.read_text(encoding="utf-8"))["response"]
        else:
            response = request(client, "POST", QUERY_URL, data=payload)
            data = response.json()
            evidence = {"endpoint": QUERY_URL, "request": payload, "retrieved_at": utcnow(), "response": data}
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        announcements = data.get("announcements") or []
        log("metadata_query", ticker=ticker, keyword=keyword, returned=len(announcements),
            total=data.get("totalAnnouncement"), evidence_path=evidence_path.relative_to(ROOT).as_posix())
        for record in announcements:
            if record.get("secCode") == ticker and record.get("adjunctType", "").upper() == "PDF":
                item = dict(record)
                item["title"] = clean_title(record["announcementTitle"])
                item["_evidence_path"] = evidence_path.relative_to(ROOT).as_posix()
                records.setdefault(record["announcementId"], item)
        if not keyword:
            # CNINFO currently caps pages at 30 records regardless of pageSize.
            pages = min(20, (int(data.get("totalAnnouncement") or 0) + 29) // 30)
            for page in range(2, pages + 1):
                next_payload = {**payload, "pageNum": page, "pageSize": 30}
                next_path = METADATA / f"{ticker}_recent_page{page}.json"
                try:
                    if next_path.exists():
                        next_data = json.loads(next_path.read_text(encoding="utf-8"))["response"]
                    else:
                        time.sleep(0.5)
                        next_data = request(client, "POST", QUERY_URL, data=next_payload).json()
                        next_path.write_text(json.dumps({"endpoint": QUERY_URL, "request": next_payload,
                            "retrieved_at": utcnow(), "response": next_data}, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as exc:
                    log("metadata_pagination_failed", ticker=ticker, page=page, error=repr(exc))
                    break
                for record in next_data.get("announcements") or []:
                    if record.get("secCode") == ticker and record.get("adjunctType", "").upper() == "PDF":
                        item = dict(record)
                        item["title"] = clean_title(record["announcementTitle"])
                        item["_evidence_path"] = next_path.relative_to(ROOT).as_posix()
                        records.setdefault(record["announcementId"], item)
            log("metadata_pagination", ticker=ticker, pages=pages, distinct_records=len(records))
    return list(records.values())


def classify(record):
    title = record["title"]
    if any(term in title for term in ["摘要", "英文", "H股", "审计", "监事", "董事", "取消", "补充", "更正"]):
        return None
    year = re.search(r"(202[456])年?", title)
    if not year:
        return None
    y = year.group(1)
    if re.search(r"(?:年?年度报告|年报报告)(?:全文)?(?:[（(].*?[）)])?$", title) and "半年度" not in title:
        return ("annual_report", y)
    if re.search(r"半年度报告(?:全文)?(?:[（(].*?[）)])?$", title):
        return ("interim_report", y + "H1")
    if re.search(r"(?:第三季度|三季度|三季)报告(?:全文)?(?:[（(].*?[）)])?$", title):
        return ("quarterly_report", y + "Q3")
    if re.search(r"(?:第一季度|一季度|一季)报告(?:全文)?(?:[（(].*?[）)])?$", title):
        return ("quarterly_report", y + "Q1")
    return None


def candidates(records):
    records.sort(key=lambda r: (r.get("announcementTime") or 0, r["announcementId"]), reverse=True)
    selected, used = [], set()
    targets = [("annual_report", "2024"), ("annual_report", "2025"),
               ("interim_report", "2025H1"), ("interim_report", "2026H1"),
               ("quarterly_report", "2025Q3"), ("quarterly_report", "2026Q3")]
    for target in targets:
        choices = [r for r in records if classify(r) == target]
        if choices:
            item = choices[0]
            selected.append((item, target[0], target[1]))
            used.add(item["announcementId"])
    substantive = []
    excluded = ["H股公告", "摘要", "法律意见", "股东大会", "章程", "董事会", "监事会", "募集说明书",
                "受托管理", "会议通知", "登记日", "回购", "股份变动", "债券付息", "取消", "更正", "召开"]
    for item in records:
        title = item["title"]
        if item["announcementId"] in used or any(word in title for word in excluded):
            continue
        if any(word in title for word in ["投资者关系活动记录", "业绩说明会会议记录", "业绩说明会纪要", "投资者交流记录"]):
            substantive.append((item, "investor_relations", ""))
        elif any(word in title for word in ["产销快报", "经营数据", "生产经营", "临床试验", "药品注册",
                    "药品上市", "药品批准", "项目投资", "对外投资", "收购", "资源量", "储量", "战略合作",
                    "项目进展", "获得", "经营业绩", "业绩快报", "业绩预告", "投资建设", "权益分派实施",
                    "发行金融债券", "二级资本债券发行", "无固定期限资本债券发行", "购买资产"]):
            substantive.append((item, "material_announcement", ""))
    # Prefer different substantive titles; downloads take the first two valid files.
    selected.extend(substantive[:12])
    # Additional periodic reports are fallback only, with annual/interim diversity preserved above.
    for item in records:
        kind = classify(item)
        if kind and item["announcementId"] not in used:
            selected.append((item, *kind))
    return selected


def download(client, record, dtype, period, ticker, name, industry, seen):
    source = "https://static.cninfo.com.cn/" + record["adjunctUrl"].lstrip("/")
    original = record["adjunctUrl"].split("/")[-1]
    target = RAW / ticker / original
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        content = target.read_bytes()
    else:
        log("download_started", ticker=ticker, source_url=source, title=record["title"])
        with request(client, "GET", source, stream=True) as response:
            content = b"".join(response.iter_content(128 * 1024))
    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded response does not have a PDF header")
    digest = hashlib.sha256(content).hexdigest()
    if digest in seen:
        log("duplicate_hash", ticker=ticker, title=record["title"], file_sha256=digest)
        return None
    reader = PdfReader(io.BytesIO(content), strict=False)
    if not reader.pages:
        raise ValueError("PDF has no pages")
    first = (reader.pages[0].extract_text() or "").strip()
    # Preserve the original byte stream only after validation.
    if not target.exists():
        target.write_bytes(content)
    timestamp = record.get("announcementTime")
    publish = datetime.fromtimestamp(timestamp / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat() if timestamp else ""
    evidence = {"announcementId": record["announcementId"], "announcementTime": timestamp,
                "announcementTitle": record["announcementTitle"], "secCode": record["secCode"],
                "orgId": record["orgId"], "adjunctUrl": record["adjunctUrl"],
                "interpretation": "announcementTime epoch milliseconds converted in Asia/Shanghai"}
    identity = f"cninfo:{record['announcementId']}:{digest}"
    manifest = dict(document_id="company_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        title=record["title"], knowledge_type="company_official", document_type=dtype,
        ticker=ticker, company_name=name, industry=industry, issuer=name, source_url=source,
        source_level="official", publish_date=publish, report_period=period, effective_date="",
        expire_date="", downloaded_at=datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat(), original_filename=original, file_sha256=digest,
        raw_path=target.relative_to(ROOT).as_posix(), date_status="verified" if publish else "date_unverified",
        date_evidence_url=QUERY_URL, date_evidence_text=json.dumps(evidence, ensure_ascii=False),
        collection_status="downloaded_validated", source_record_id=record["announcementId"],
        date_evidence_path=record["_evidence_path"], pdf_pages=len(reader.pages),
        pdf_metadata_title=str((reader.metadata or {}).get("/Title", "")), first_page_excerpt=first[:1600])
    seen.add(digest)
    log("download_validated", ticker=ticker, document_id=manifest["document_id"],
        title=record["title"], pages=len(reader.pages), bytes=len(content), publish_date=publish)
    return manifest


def write_manifests(stem, rows):
    path = MANIFESTS / stem
    path.with_suffix(".jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_existing():
    """Independently recheck the final files and dates without making network calls."""
    rows = [json.loads(line) for line in (MANIFESTS / "company_documents.jsonl").read_text(encoding="utf-8").splitlines() if line]
    errors = []
    ids, hashes, periods = set(), set(), set()
    for row in rows:
        file_path = ROOT / row["raw_path"]
        contents = file_path.read_bytes()
        identity = row["document_id"]
        if identity in ids or row["file_sha256"] in hashes:
            errors.append({"document_id": identity, "error": "duplicate identity or file hash"})
        ids.add(identity)
        hashes.add(row["file_sha256"])
        period_key = (row["ticker"], row["document_type"], row["report_period"])
        if row["report_period"] and period_key in periods:
            errors.append({"document_id": identity, "error": "duplicate report period"})
        periods.add(period_key)
        if not contents.startswith(b"%PDF-") or hashlib.sha256(contents).hexdigest() != row["file_sha256"]:
            errors.append({"document_id": identity, "error": "invalid header or hash mismatch"})
        if len(PdfReader(io.BytesIO(contents), strict=False).pages) != row["pdf_pages"]:
            errors.append({"document_id": identity, "error": "page count mismatch"})
        evidence = json.loads((ROOT / row["date_evidence_path"]).read_text(encoding="utf-8"))
        source_record = next(record for record in evidence["response"]["announcements"] if record["announcementId"] == row["source_record_id"])
        official_date = datetime.fromtimestamp(source_record["announcementTime"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()
        if official_date != row["publish_date"] or source_record["secCode"] != row["ticker"]:
            errors.append({"document_id": identity, "error": "official date/ticker mismatch"})
        if clean_title(source_record["announcementTitle"]) != row["title"]:
            errors.append({"document_id": identity, "error": "official title mismatch"})
        if row["effective_date"]:
            errors.append({"document_id": identity, "error": "unexpected inferred effective date"})
        row["downloaded_at"] = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat()
    # Keep original download times when originals were reused across validation runs.
    for ticker in sorted({row["ticker"] for row in rows}):
        write_manifests(f"company_{ticker}", [row for row in rows if row["ticker"] == ticker])
    write_manifests("company_documents", rows)
    result = {"validated_at": utcnow(), "documents": len(rows), "companies": len({row["ticker"] for row in rows}),
              "counts_by_company": dict(Counter(row["ticker"] for row in rows)),
              "counts_by_document_type": dict(Counter(row["document_type"] for row in rows)),
              "total_pages": sum(row["pdf_pages"] for row in rows),
              "total_bytes": sum((ROOT / row["raw_path"]).stat().st_size for row in rows),
              "date_status_counts": dict(Counter(row["date_status"] for row in rows)),
              "empty_first_page_excerpts": sum(not row["first_page_excerpt"].strip() for row in rows),
              "unique_file_hashes": len(hashes), "errors": errors,
              "limitations": ["CNINFO pagination returned HTTP 403; saved targeted searches and latest-30 records were used.",
                  "No separate IR activity records were counted; several companies use additional first-quarter reports.",
                  "Uncounted preliminary/H-share/meeting-notice originals are preserved on disk but excluded from the final manifests."]}
    (MANIFESTS / "company_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log("independent_validation_completed", **result)
    if errors:
        raise SystemExit(1)


def collect_company(company, orgids, cutoff, max_docs):
    ticker, name, industry = company
    client = session()
    rows, seen = [], set()
    try:
        records = fetch_metadata(client, ticker, orgids[ticker], cutoff)
        options = candidates(records)
        for record, dtype, period in options:
            if len(rows) >= max_docs:
                break
            if period and any(row["document_type"] == dtype and row["report_period"] == period for row in rows):
                continue
            # Once two substantive disclosures succeeded, continue only report fallbacks.
            if dtype in {"investor_relations", "material_announcement"} and sum(r["document_type"] in {"investor_relations", "material_announcement"} for r in rows) >= 2:
                continue
            try:
                item = download(client, record, dtype, period, ticker, name, industry, seen)
                if item:
                    rows.append(item)
                    write_manifests(f"company_{ticker}", rows)
            except Exception as exc:
                log("download_failed", ticker=ticker, title=record["title"],
                    source_record_id=record["announcementId"], error=repr(exc))
        log("company_complete", ticker=ticker, documents=len(rows), distinct_metadata_records=len(records))
    except Exception as exc:
        log("company_failed", ticker=ticker, error=repr(exc))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2026-11-16")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-per-company", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for directory in [RAW, MANIFESTS, METADATA]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        validate_existing()
        return
    log("collection_started", cutoff=args.cutoff, max_per_company=args.max_per_company)
    stock_path = METADATA / "official_stock_list.json"
    if stock_path.exists():
        stocks = json.loads(stock_path.read_text(encoding="utf-8"))
    else:
        response = request(session(), "GET", "https://www.cninfo.com.cn/new/data/szse_stock.json")
        stocks = response.json()
        stock_path.write_text(json.dumps(stocks, ensure_ascii=False), encoding="utf-8")
    orgids = {item["code"]: item["orgId"] for item in stocks["stockList"]}
    selected = [c for c in COMPANIES if not args.tickers or c[0] in args.tickers]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_company, c, orgids, args.cutoff, args.max_per_company) for c in selected]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    rows = []
    for ticker, _, _ in COMPANIES:
        path = MANIFESTS / f"company_{ticker}.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    unique = {row["file_sha256"]: row for row in rows}
    rows = sorted(unique.values(), key=lambda r: (r["ticker"], r["publish_date"], r["document_id"]))
    write_manifests("company_documents", rows)
    log("collection_finished", documents=len(rows), companies=len({r["ticker"] for r in rows}),
        bytes=sum((ROOT / r["raw_path"]).stat().st_size for r in rows))


if __name__ == "__main__":
    main()
