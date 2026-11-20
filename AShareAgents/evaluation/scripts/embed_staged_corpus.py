"""Embed one staged document at a time; resume using both parse identity and hash."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import EVALUATION_DIR, ROOT, CORPUS, write_csv, write_json, write_text
from AShareAgents.evaluation.ingestion import CHUNK_CONFIG
from AShareAgents.evaluation.snapshot import load_staged_snapshot, attest_database_snapshot


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--cloud-budget-cny',type=float,default=17)
    args=parser.parse_args(argv)
    status_path=CORPUS/'manifests/ingestion_status.csv'
    with status_path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    summary_path=CORPUS/'manifests/ingestion_summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    total_bytes=0
    for row in rows:
        if row['parse_status']!='parsed': continue
        result=json.loads((ROOT/row['processed_path']).read_text(encoding='utf-8'))
        total_bytes+=sum(len(c['page_content'].encode('utf-8'))+16 for c in result['chunks'])
    print(json.dumps({'utf8_upper_proxy_tokens':total_bytes,
                      'reserved_embedding_upper_cny':total_bytes*.5/1e6},ensure_ascii=False),flush=True)
    if not args.allow_paid: return
    from dotenv import load_dotenv
    from langchain_core.documents import Document
    from AShareAgents.rag.cloud_adapters.bailian_embeddings import BailianEmbeddings
    from AShareAgents.storage.sqlite_knowledge_store import SQLiteKnowledgeStore
    from AShareAgents.evaluation.cloud_cache import ResponseCache,CachedEmbeddings
    load_dotenv(ROOT/'.env',override=False)
    cache=ResponseCache(CORPUS/'knowledge_base/cloud_cache.sqlite',max_requests=20000,
                        max_cost_cny=args.cloud_budget_cny)
    embedding=CachedEmbeddings(BailianEmbeddings(),cache,workers=8,requests_per_second=8)
    db=CORPUS/'knowledge_base/knowledge_v1.sqlite'
    store=SQLiteKnowledgeStore(str(db),embedding.adapter.dimension)
    identity=json.dumps(embedding.identity,sort_keys=True)
    with sqlite3.connect(db) as conn:
        prior=conn.execute("SELECT value FROM rag_settings WHERE key='embedding_identity'").fetchone()
        if prior and prior[0]!=identity: raise ValueError('Embedding identity changed; use new database')
        conn.execute("INSERT OR REPLACE INTO rag_settings VALUES ('embedding_identity',?)",(identity,))
        conn.execute('CREATE TABLE IF NOT EXISTS eval_ingestion (document_id TEXT PRIMARY KEY,parse_key TEXT,file_sha256 TEXT,child_count INTEGER)')
    for number,row in enumerate(rows,1):
        if row['parse_status']!='parsed': continue
        result=json.loads((ROOT/row['processed_path']).read_text(encoding='utf-8'))
        expected=(result['parse_key'],result['file_sha256'],result['child_count'])
        with sqlite3.connect(db) as conn:
            prior=conn.execute('SELECT parse_key,file_sha256,child_count FROM eval_ingestion WHERE document_id=?',
                               (result['document_id'],)).fetchone()
            count=conn.execute('SELECT count(*) FROM rag_chunks WHERE document_id=?',(result['document_id'],)).fetchone()[0]
        if prior==expected and count==result['child_count']:
            row['ingest_status']='already_ingested'
        else:
            try:
                docs=[Document(**chunk) for chunk in result['chunks']]
                vectors=embedding.embed_documents([d.page_content for d in docs])
                count=store.upsert_documents(docs,vectors,replace_parents=True)
                if count!=result['child_count']: raise ValueError('Stored child count mismatch')
                with sqlite3.connect(db) as conn:
                    conn.execute('INSERT OR REPLACE INTO eval_ingestion VALUES (?,?,?,?)',(result['document_id'],*expected))
                row.update(ingest_status='ingested',inserted_chunks=count)
            except Exception as exc:
                row.update(ingest_status='failed',ingest_error_type=type(exc).__name__)
                print(f'INGEST_FAILED {row["manifest_document_id"]} {type(exc).__name__}: {str(exc)[:120]}',flush=True)
                write_csv(status_path,rows)
                if isinstance(exc,RuntimeError) and 'budget' in str(exc).lower(): raise
                response=getattr(exc,'response',None)
                if response is not None and response.status_code in (401,403):
                    summary.update(embedding_status='blocked_service_permission',
                                   blocking_http_status=response.status_code,
                                   documents_ingested=sum(r.get('ingest_status') in ('ingested','already_ingested') for r in rows))
                    write_json(summary_path,summary)
                    raise RuntimeError('Embedding access denied; existing progress preserved, enable provider access before resuming') from None
                continue
        write_csv(status_path,rows)
        print(f'INGEST {number}/{len(rows)} {row["manifest_document_id"]} children={count} {row["ingest_status"]} http={cache.requests}',flush=True)
    with sqlite3.connect(db) as conn:
        counts=conn.execute("SELECT count(*),count(DISTINCT document_id),count(DISTINCT json_extract(metadata,'$.parent_id')) FROM rag_chunks").fetchone()
    with sqlite3.connect(cache.path) as conn:
        attempts,reserved=conn.execute('SELECT count(*),COALESCE(sum(estimated_cny),0) FROM paid_reservations').fetchone()
    succeeded=sum(r.get('ingest_status') in ('ingested','already_ingested') for r in rows)
    summary.update(embedding_status='complete' if succeeded==len(rows) else 'partial',embedding_identity=embedding.identity,
                   embedding_requests_this_process=cache.requests,cloud_reserved_upper_cny=reserved,
                   cloud_http_attempts=attempts,
                   documents_ingested=succeeded,failed_or_pending_documents=len(rows)-succeeded,
                   database_documents=counts[1],database_child_chunks=counts[0],database_parent_chunks=counts[2])
    if summary['embedding_status']=='complete':
        snapshot = attest_database_snapshot(db, load_staged_snapshot(CORPUS / 'processed', rows),
                                            CHUNK_CONFIG, embedding.identity)
        summary['knowledge_snapshot_hash'] = snapshot['knowledge_snapshot_hash']
        summary['raw_corpus_hash'] = summary['corpus_hash']
        summary['snapshot_version'] = snapshot['snapshot_version']
        summary.pop('blocking_http_status',None)
        summary.pop('blocking_code',None)
        summary['historical_service_interruption']='403 AccessDenied.Unpurchased resolved after user enabled provider access'
    write_json(summary_path,summary)
    write_text(EVALUATION_DIR / 'reports/ingestion_report.md','# 解析与入库报告\n\n以下为实际执行记录，部分页质量见逐文档状态和 quality_flags。\n\n```json\n'
               +json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n原业务数据库未用作写入目标。扫描/图形未OCR；layout表格未作语义重建。\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
