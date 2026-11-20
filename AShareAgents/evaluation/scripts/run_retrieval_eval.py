"""Run full-query, same-snapshot IR comparisons using the production algorithms."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import sqlite3

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import yaml
from langchain_core.documents import Document
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, RETRIEVAL_CONFIG, ROOT, CORPUS, read_jsonl, write_json, write_jsonl, write_csv,
    write_text, fingerprint,
)
from AShareAgents.evaluation.metrics import ranking_metrics
from AShareAgents.evaluation.retrieval import (
    STRATEGIES, FilteredIndex, eligible_for_question, retrieve_all, BM25PoolCache,
)
from AShareAgents.evaluation.ingestion import CHUNK_CONFIG
from AShareAgents.evaluation.snapshot import load_staged_snapshot, attest_database_snapshot


def serialize(docs):
    return [{'page_content': d.page_content,
             'metadata': {k: v for k, v in d.metadata.items() if k != 'parent_content'}} for d in docs]


def merge_checkpoint(previous, updates):
    """Replace visited query/strategy rows while preserving the rest on resume."""
    indexed = {(r['query_id'], r['strategy']): r for r in previous}
    indexed.update({(r['query_id'], r['strategy']): r for r in updates})
    return list(indexed.values())


def matching_completed(previous, config_hash, corpus_hash, dataset_hash, knowledge_snapshot_hash):
    return {(r['query_id'], r['strategy']): r for r in previous
            if r.get('status') == 'ok' and r.get('config_hash') == config_hash
            and r.get('corpus_hash') == corpus_hash and r.get('dataset_hash') == dataset_hash
            and bool(knowledge_snapshot_hash) and r.get('knowledge_snapshot_hash') == knowledge_snapshot_hash}


def validate_snapshot_completion(summary, children, documents):
    if (summary.get('embedding_status') != 'complete' or children != summary['child_chunks']
            or documents != summary['documents_planned']):
        raise RuntimeError('Complete the independent knowledge snapshot before cloud comparison')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES)
    parser.add_argument('--config', type=Path, default=RETRIEVAL_CONFIG)
    parser.add_argument('--dataset', type=Path, default=EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cloud-budget-cny', type=float, default=17)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config_hash = fingerprint(cfg)
    summary = json.loads((CORPUS/'manifests/ingestion_summary.json').read_text(encoding='utf-8'))
    corpus_hash = summary['corpus_hash']
    db = ROOT / cfg['knowledge_database']
    with (CORPUS / 'manifests/ingestion_status.csv').open(encoding='utf-8-sig', newline='') as stream:
        staged_inventory = load_staged_snapshot(CORPUS / 'processed', list(csv.DictReader(stream)))
    snapshot = attest_database_snapshot(db, staged_inventory, CHUNK_CONFIG, summary.get('embedding_identity'))
    knowledge_snapshot_hash = snapshot['knowledge_snapshot_hash']
    questions = read_jsonl(args.dataset)
    dataset_hash = fingerprint(questions)
    if args.limit:
        questions = questions[:args.limit]
    # Stream the 290MB JSONL to avoid keeping a second complete dict list.
    with (CORPUS/'processed/chunks.jsonl').open(encoding='utf-8') as stream:
        documents=[Document(**json.loads(line)) for line in stream if line.strip()]
    bm25_cache=BM25PoolCache(max_pools=2)
    index = reranker = None
    cache = None
    if args.allow_paid:
        from dotenv import load_dotenv
        from AShareAgents.rag.index_construction import IndexConstructionModule
        from AShareAgents.storage.sqlite_knowledge_store import SQLiteKnowledgeStore
        from AShareAgents.rag.cloud_adapters.bailian_embeddings import BailianEmbeddings
        from AShareAgents.rag.cloud_adapters.bailian_reranker import BailianReranker
        from AShareAgents.evaluation.cloud_cache import ResponseCache, CachedEmbeddings, CachedReranker
        load_dotenv(ROOT/'.env', override=False)
        db = ROOT/cfg['knowledge_database']
        if not db.is_file():
            raise RuntimeError('Complete independent cloud ingestion first')
        with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as conn:
            children,document_count=conn.execute('SELECT count(*),count(DISTINCT document_id) FROM rag_chunks').fetchone()
        validate_snapshot_completion(summary,children,document_count)
        store = SQLiteKnowledgeStore(str(db), cfg['embedding_dimension'])
        cache = ResponseCache(ROOT/cfg['cloud_cache'], max_requests=1000,
                              max_cost_cny=args.cloud_budget_cny)
        embedding = CachedEmbeddings(BailianEmbeddings(), cache)
        index = FilteredIndex(IndexConstructionModule(uri='', collection_name='eval_v1',
                    model_name=cfg['embedding_model'], dimension=cfg['embedding_dimension'],
                    embeddings=embedding, store=store))
        reranker = CachedReranker(BailianReranker(), cache)
    strategies = args.strategies or STRATEGIES
    output = EVALUATION_DIR / 'results'
    output.mkdir(parents=True, exist_ok=True)
    old_runs = read_jsonl(output/'retrieval_runs.jsonl') if (output/'retrieval_runs.jsonl').exists() else []
    completed = matching_completed(old_runs, config_hash, corpus_hash, dataset_hash, knowledge_snapshot_hash)
    runs = []
    raw_runs = []
    for number,q in enumerate(questions,1):
        pending = [s for s in strategies if (q['query_id'],s) not in completed]
        if pending:
            filters = {'knowledge_type':q['knowledge_type'],
                       'publish_before_ts':int(datetime.fromisoformat(q['as_of_date']).replace(
                           hour=23, minute=59, second=59, tzinfo=timezone.utc).timestamp()),
                       'date_verified_only':True}
            for key in ('ticker','document_type','report_period'):
                if q.get(key):
                    filters[key]=q[key]
            if q.get('knowledge_type')=='policy_industry' and q.get('industry'):
                filters['industry']=q['industry']
            eligible = [d for d in documents if eligible_for_question(d.metadata,q)]
            start = time.perf_counter()
            pool_key=fingerprint([knowledge_snapshot_hash,filters])
            result = retrieve_all(q['query'],eligible,index,reranker,filters,cfg,strategies=pending,
                                  bm25_search=lambda query,docs,k:bm25_cache.search(query,docs,k,pool_key))
            raw_runs.append({'query_id':q['query_id'],'corpus_hash':corpus_hash,'config_hash':config_hash,
                             'raw_corpus_hash':corpus_hash,'knowledge_snapshot_hash':knowledge_snapshot_hash,
                             'dataset_hash':dataset_hash,'question_hash':fingerprint(q),
                             'eligible_pool_count':len(eligible),'filters':filters,
                             'raw_candidates':{k:serialize(v) for k,v in result['raw_candidates'].items()}})
            for strategy in pending:
                data = result['strategies'].get(strategy)
                row = {'query_id':q['query_id'],'strategy':strategy,'as_of_date':q['as_of_date'],
                       'filters':filters,'config_hash':config_hash,'corpus_hash':corpus_hash,
                       'raw_corpus_hash':corpus_hash,'knowledge_snapshot_hash':knowledge_snapshot_hash,
                       'dataset_hash':dataset_hash,'question_hash':fingerprint(q),
                       'annotation_basis':q['annotation_status'],'retrieved_at':datetime.now(timezone.utc).isoformat(),
                       'status':'ok' if data else 'blocked','elapsed_total':time.perf_counter()-start}
                if data:
                    row.update(children=serialize(data['children']),parents=serialize(data['parents']),
                               latency=data['latency'])
                else:
                    row.update(error=result['errors'].get(strategy,result['errors'].get('Dense','unavailable')),
                               children=[],parents=[])
                completed[(q['query_id'],strategy)] = row
        for strategy in strategies:
            runs.append(completed[(q['query_id'],strategy)])
        old_runs = merge_checkpoint(old_runs, runs)
        write_jsonl(output/'retrieval_runs.jsonl',old_runs)
        # Preserve previous raw candidates when a resumed query needed no new calls.
        if raw_runs:
            previous = read_jsonl(output/'retrieval_raw_candidates.jsonl') if (output/'retrieval_raw_candidates.jsonl').exists() else []
            indexed={r['query_id']:r for r in previous if r.get('config_hash')==config_hash
                     and r.get('corpus_hash')==corpus_hash and r.get('dataset_hash')==dataset_hash
                     and r.get('knowledge_snapshot_hash')==knowledge_snapshot_hash}
            indexed.update({r['query_id']:r for r in raw_runs})
            write_jsonl(output/'retrieval_raw_candidates.jsonl',indexed.values())
            raw_runs=[]
        print(f'RETRIEVAL {number}/{len(questions)} {q["query_id"]}',flush=True)
    qmap={q['query_id']:q for q in questions}
    rows=[]
    ranked_rows=[]
    for run in runs:
        if run['status']!='ok':
            continue
        q=qmap[run['query_id']]
        for grain,id_key in [('child','chunk_id'),('parent','parent_id')]:
            docs=run['children' if grain=='child' else 'parents']
            qrels={r[id_key]:r['relevance'] for r in q['qrels_'+grain]}
            scores=ranking_metrics([d['metadata'][id_key] for d in docs],qrels)
            rows.append({'query_id':q['query_id'],'strategy':run['strategy'],'grain':grain,
                         'category':q['category'],'split':q['split'],'answerability':q['answerability'],
                         'latency':run['latency'],'annotation_status':q['annotation_status'],**scores})
            for rank,d in enumerate(docs,1):
                m=d['metadata']
                ranked_rows.append({'query_id':q['query_id'],'strategy':run['strategy'],'grain':grain,
                    'document_id':m['document_id'],'parent_id':m['parent_id'],'chunk_id':m['chunk_id'],
                    'retrieval_rank':rank,'retrieval_score':m.get('rrf_score',m.get('bm25_score',m.get('score'))),
                    'rerank_score':m.get('rerank_score'),'publish_date':m.get('publish_date'),
                    'as_of_date':q['as_of_date'],'retrieval_latency':run['latency'],
                    'filter_conditions':json.dumps(run['filters'],ensure_ascii=False)})
    write_csv(output/'retrieval_per_query_metrics.csv',rows)
    write_csv(output/'retrieval_rankings.csv',ranked_rows)
    aggregates=[]
    for strategy in strategies:
        for grain in ('child','parent'):
            for split in ('dev','test','all'):
                for category in ['all',*sorted({q['category'] for q in questions})]:
                    selected=[r for r in rows if r['strategy']==strategy and r['grain']==grain
                              and (split=='all' or r['split']==split) and(category=='all' or r['category']==category)]
                    if not selected: continue
                    metrics={k:sum(r[k] for r in selected if r.get(k) is not None)/sum(r.get(k)is not None for r in selected)
                             for k in [f'{m}@{k}'for m in ('recall','mrr','ndcg')for k in (5,10,20)]
                             if any(r.get(k)is not None for r in selected)}
                    aggregates.append({'strategy':strategy,'grain':grain,'split':split,'category':category,
                        'queries':len(selected),'answerable_queries':sum(r['answerability']=='answerable' for r in selected),
                        'mean_latency':sum(r['latency']for r in selected)/len(selected),**metrics})
    write_csv(output/'retrieval_metrics.csv',aggregates)
    candidates=[r for r in aggregates if r['grain']=='parent' and r['split']=='dev' and r['category']=='all']
    if candidates:
        choice=max(candidates,key=lambda x:(x.get('ndcg@5',-1),STRATEGIES.index(x['strategy'])))
        write_json(output/'selected_strategy.json',{'strategy':choice['strategy'],'selection_basis':'dev parent nDCG@5 only',
                    'annotation_basis':'pending_review; provisional','config_hash':config_hash,
                    'corpus_hash':corpus_hash,'dataset_hash':dataset_hash,
                    'raw_corpus_hash':corpus_hash,'knowledge_snapshot_hash':knowledge_snapshot_hash})
    write_json(output/'retrieval_summary.json',{'queries':len(questions),'planned_strategies':strategies,
        'completed_queries_per_strategy':{s:sum(r['strategy']==s and r['status']=='ok'for r in runs)for s in strategies},
        'cloud_attempts_this_process':cache.requests if cache else 0,'corpus_hash':corpus_hash,
        'config_hash':config_hash,'dataset_hash':dataset_hash,
        'raw_corpus_hash':corpus_hash,'knowledge_snapshot_hash':knowledge_snapshot_hash})
    table='|策略|粒度|Recall@5|Recall@10|Recall@20|MRR@5|nDCG@5|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in aggregates:
        if r['split']=='test' and r['category']=='all':
            table+='|'+r['strategy']+'|'+r['grain']+'|'+ '|'.join(f'{r.get(k,0):.4f}'for k in ['recall@5','recall@10','recall@20','mrr@5','ndcg@5'])+'|\n'
    write_text(EVALUATION_DIR / 'reports/retrieval_evaluation_report.md','# 独立检索评测\n\n'
        '四种策略共享语料、问题、时间过滤及qrels；BM25使用全部合格块，未按ID截断。所有指标使用pending_review机器辅助标注，仅为初步结果，不能宣传为人工Golden Truth。\n\n'
        +table+'\n原始子块候选、父块提升结果、逐查询分数和过滤条件见results。最终生成仅取前5父块；@10/@20用于检索诊断，未增加原有召回预算。'
        '无答案问题不计Recall/MRR，另分析非空召回和生成拒答。Test未用于参数调优；策略选择只使用Dev。\n')


if __name__=='__main__':
    main()
