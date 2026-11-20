"""Stage the complete real corpus, estimate cost, then optionally embed into an isolated DB."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, CORPUS, read_jsonl, write_json, write_jsonl, write_csv, write_text, fingerprint,
)
from AShareAgents.evaluation.ingestion import stage_document, CHUNK_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=CORPUS / 'manifests/corpus_manifest.jsonl')
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1, choices=(1, 2))
    parser.add_argument('--cloud-budget-cny', type=float, default=17,
                        help='Shared persistent embedding+rerank conservative reservation limit')
    args = parser.parse_args()
    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[:args.limit]
    report, staged = [], []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    pending = {pool.submit(stage_document, r, CORPUS / 'processed'): r for r in records} if pool else {}
    jobs = ((pending[f], f) for f in as_completed(pending)) if pool else ((r, None) for r in records)
    for record, future in jobs:
        try:
            result = future.result() if future is not None else stage_document(record, CORPUS / 'processed')
            staged.append(result)
            report.append({'manifest_document_id': record['document_id'], 'document_id': result['document_id'],
                           'title': record['title'], 'knowledge_type': record['knowledge_type'],
                           'document_type': record['document_type'], 'file_sha256': record['file_sha256'],
                           'pages': result['page_count'], 'parent_chunks': result['parent_count'],
                           'child_chunks': result['child_count'], 'parse_errors': len(result['errors']),
                           'empty_pages': sum(not p['content'].strip() for p in result['pages']),
                           'quarantined_pages': len(result.get('quarantined_pages', [])),
                           'processed_path': result['processed_path'], 'parse_status': 'parsed' if result['chunks'] else 'failed',
                           'ingest_status': 'awaiting_cloud_embedding', 'cache_hit': result['cached']})
            print(f"PARSED {record['document_id']} pages={result['page_count']} children={result['child_count']} cached={result['cached']}", flush=True)
        except Exception as exc:
            report.append({'manifest_document_id': record['document_id'], 'title': record['title'],
                           'parse_status': 'failed', 'error_type': type(exc).__name__,
                           'ingest_status': 'not_ingested'})
            print(f"PARSE_FAILED {record['document_id']} {type(exc).__name__}", flush=True)
        write_csv(CORPUS / 'manifests/ingestion_status.csv', report)
    if pool:
        pool.shutdown()
    report.sort(key=lambda r: r['manifest_document_id'])
    staged.sort(key=lambda r: r['manifest_document_id'])
    chunks = [chunk for result in staged for chunk in result['chunks']]
    pages = [page for result in staged for page in result['pages']]
    if len({x['metadata']['chunk_id'] for x in chunks}) != len(chunks):
        raise ValueError('Duplicate chunk IDs across document pages')
    write_jsonl(CORPUS / 'processed/chunks.jsonl', chunks)
    write_jsonl(CORPUS / 'processed/pages.jsonl', pages)
    snapshot = fingerprint([(r['document_id'], r['file_sha256']) for r in sorted(records, key=lambda r: r['document_id'])])
    characters = sum(len(x['page_content']) for x in chunks)
    summary = {'corpus_hash': snapshot, 'documents_planned': len(records), 'documents_parsed': len(staged),
               'pages': len(pages), 'parent_chunks': len({x['metadata']['parent_id'] for x in chunks}),
               'child_chunks': len(chunks), 'characters_to_embed': characters,
               'token_estimate_lower': int(characters * .4), 'token_estimate_upper': characters * 2,
               'embedding_http_requests_upper': (len(chunks) + 9) // 10 + len(staged),
               'estimated_embedding_cny_upper_at_official_rate': characters * 2 / 1_000_000 * .5,
               'document_types': dict(Counter(r['document_type'] for r in records)),
               'metadata_complete_fraction': sum(all(x['metadata'].get(k) for k in ['document_id','parent_id','chunk_id','source','publish_date','file_sha256','page_number']) for x in chunks) / len(chunks) if chunks else 0,
               'chunk_config': CHUNK_CONFIG, 'embedding_status': 'not_started_no_paid_flag'}
    write_json(CORPUS / 'manifests/ingestion_summary.json', summary)
    if args.allow_paid:
        # Keep one paid implementation: it validates parse_key as well as file
        # identity and uses bounded document memory with durable cloud caching.
        write_csv(CORPUS / 'manifests/ingestion_status.csv', report)
        from AShareAgents.evaluation.scripts.embed_staged_corpus import main as embed_staged
        return embed_staged(['--allow-paid', '--cloud-budget-cny', str(args.cloud_budget_cny)])
    write_csv(CORPUS / 'manifests/ingestion_status.csv', report)
    write_text(EVALUATION_DIR / 'reports/ingestion_report.md', '# 解析与入库报告\n\n'
               + '本报告由实际执行结果生成。原业务数据库不作为写入目标。\n\n'
               + '```json\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n```\n\n'
               + '逐文档状态见 data/rag_eval_v1/manifests/ingestion_status.csv。'
               + '表格为layout文本提取，未做OCR、图形理解或财务表格语义重建；quality_flags必须随证据审查。\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
