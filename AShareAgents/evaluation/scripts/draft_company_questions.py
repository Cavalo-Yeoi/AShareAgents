"""Draft 150 source-grounded company questions; all remain pending human review."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AShareAgents.evaluation.annotation import check_seeds
from AShareAgents.evaluation.cloud_cache import ResponseCache
from AShareAgents.evaluation.io import EVALUATION_DIR, ROOT, CORPUS, read_jsonl, write_json, write_jsonl
from AShareAgents.rag.pdf_parser import text_quality_flags


def initialize_annotation_budget(budget,historical_reserved_cny,historical_attempts):
    """Import the pre-ledger task history once; future reruns share the same cap."""
    from AShareAgents.evaluation.cloud_cache import sqlite_connection
    with sqlite_connection(budget.ledger_path) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE IF NOT EXISTS annotation_history_import (id TEXT PRIMARY KEY,cost REAL,attempts INTEGER)')
        if not db.execute("SELECT 1 FROM annotation_history_import WHERE id='20261116-preledger'").fetchone():
            db.execute("INSERT INTO annotation_history_import VALUES ('20261116-preledger',?,?)",(historical_reserved_cny,historical_attempts))
            db.execute("UPDATE generation_budget SET reserved_microcny=reserved_microcny+?,attempts=attempts+? WHERE scope='rag-eval-v1'",
                       (round(historical_reserved_cny*1e6),historical_attempts))
        budget.reserved_cny=db.execute("SELECT reserved_microcny FROM generation_budget WHERE scope='rag-eval-v1'").fetchone()[0]/1e6


def source_packets(ticker, manifests):
    all_pages = read_jsonl(CORPUS / 'manifests/company_seed_firstpages.jsonl')
    path = CORPUS / f'manifests/company_seed_risk_{ticker}.jsonl'
    if path.exists():
        all_pages += read_jsonl(path)
    pages = {}
    for p in all_pages:
        if p['ticker'] != ticker or len(p['content'].strip()) < 100:
            continue
        if 'suspicious_unicode_mapping' in text_quality_flags(p['content']):
            continue
        pages[(p['manifest_document_id'], p['page_number'])] = p['content']
    # Keep several early accounting/business pages in each period plus actual risk sections.
    chosen, chars = [], 0
    by_doc = {}
    for (doc_id, page), text in pages.items():
        by_doc.setdefault(doc_id, []).append((page, text))
    for doc_id, items in by_doc.items():
        ordered = sorted(items)
        picked = dict(ordered[:3] + ordered[7:12] + ordered[-5:])
        doc_chars = 0
        for page, text in sorted(picked.items()):
            if doc_chars + len(text) > 23000:
                continue
            if chars + len(text) > 48000:
                continue
            r = manifests[doc_id]
            chosen.append({'manifest_document_id': doc_id, 'title': r['title'],
                           'publish_date': r['publish_date'], 'report_period': r['report_period'],
                           'page_number': page, 'content': text})
            chars += len(text)
            doc_chars += len(text)
    return chosen, pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--tickers', nargs='*')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--budget-cny', type=float, default=5)
    args = parser.parse_args()
    if not 0<args.budget_cny<=5:
        parser.error('Annotation lifetime allocation is 5 CNY')
    manifests = {r['document_id']: r for r in read_jsonl(CORPUS / 'manifests/company_documents.jsonl')}
    all_tickers = sorted({r['ticker'] for r in manifests.values()})
    requested = args.tickers or all_tickers
    if not args.allow_paid:
        print('Use --allow-paid only within explicitly approved task budget; planned 10–20 bounded calls, at most5CNY.')
        return
    from dotenv import load_dotenv
    import httpx
    from AShareAgents.llm.factory import create_llm_client
    from AShareAgents.llm.guardrails import LLMPolicy, TaskBudget
    from AShareAgents.evaluation.scripts.run_generation_eval import CallBudget
    load_dotenv(ROOT / '.env', override=False)
    budget = CallBudget(20, budget_cny=args.budget_cny, max_output_tokens=14000, max_input_tokens=190000,
                        ledger_path=CORPUS/'cloud_annotation_budget.sqlite')
    initialize_annotation_budget(budget,4.408048,19)
    transport = httpx.Client(timeout=120, follow_redirects=False, trust_env=False,
                              event_hooks={'request': [budget.consume]})
    runtime = TaskBudget(LLMPolicy(max_attempts=20, max_retries=0, read_timeout=120,
                                  call_timeout=150, task_timeout=3600))
    runtime.own_http_client(transport)
    llm = create_llm_client(provider='deepseek', model=args.model, runtime=runtime,
                            base_url='https://api.deepseek.com', http_client=transport,
                            max_retries=0).get_llm()
    cache = ResponseCache(CORPUS / 'knowledge_base/annotation_cache.sqlite')
    directory = EVALUATION_DIR / 'annotation/company_drafts'
    directory.mkdir(parents=True, exist_ok=True)
    statuses = []
    try:
        for ticker in requested:
            index = all_tickers.index(ticker)
            counts = {'single_document_fact': 4 if index < 5 else 3,
                      'multi_evidence_synthesis': 4 if index < 5 else 3,
                      'cross_period_comparison': 3, 'business_risk_analysis': 3,
                      'temporal_no_answer_conflict': 2}
            packets, original_pages = source_packets(ticker, manifests)
            company = next(r for r in manifests.values() if r['ticker'] == ticker)
            request = {'model': args.model, 'packets': packets, 'counts': counts, 'version': 2}
            completed = directory / f'{ticker}.jsonl'
            if completed.exists():
                prior = read_jsonl(completed)
                if not check_seeds(prior, original_pages, counts):
                    statuses.append({'ticker': ticker, 'questions': len(prior), 'errors': [], 'cached': True})
                    continue
            system = ('你正在编制正式金融RAG评测题的机器辅助草稿。只用给定真实文件页面和元数据，不得根据常识编造事实或数值。'
                      '返回JSON对象{"questions":[...]}。所有问题annotation_status必须pending_review。每题字段：query,category,'
                      'answerability(answerable或unanswerable),expected_behavior(answer或refuse),as_of_date,reference_answer,'
                      'atomic_claims(字符串数组),sources(数组，每项manifest_document_id,page_number,quote),negative_sources(同结构数组)。'
                      '证据quote必须逐字摘自给定页面，建议20～80个字符，最多100字，不改数字和标点；空白换行可归一化。'
                      '每个标准答案的重要事实都要有证据，不要把报告期当作披露日，不要混用银行集团/本行、年初至今/季度、元/万元。'
                      '跨报告期题必须引用两份实际文档，并核对可比口径；综合题至少引用两个不同PDF物理页或不同文档，'
                      '真正组合经营变化、业务构成、现金流、海外布局等关联证据，禁止把同一句的营收和利润简单并列冒充多段综合；'
                      '经营风险必须具体贴合该公司实际披露。'
                      '问题要多样、真实业务导向，避免简单换年份/股票代码。不要写不能从页面支持的原因判断或投资预测。'
                      '时间/无答案/冲突类共2题：1题将分析日设在某报告披露日前一天，明确询问当时是否能引用该报告的特定事实，'
                      '标unanswerable，sources=[]，negative_sources写该未来报告；另1题辨别所给原文中的口径或时间误读，能回答时提供证据。'
                      '其他可回答题as_of_date=2026-11-16。严格满足各类别数量。')
            prompt = {'company': company['company_name'], 'ticker': ticker, 'required_category_counts': counts,
                      'source_pages': packets}
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)}]
            response = cache.get('annotation', request)
            errors = []
            for attempt in range(2):
                if response is None:
                    actual_messages = messages if not errors else messages + [{'role': 'user', 'content':
                        '上次草稿不满足校验，请重新输出完整questions。错误：' + json.dumps(errors,ensure_ascii=False)}]
                    out = llm.invoke(actual_messages, max_tokens=14000, temperature=0,
                                     response_format={'type': 'json_object'},
                                     extra_body={'thinking': {'type': 'disabled'}})
                    response = {'raw': out.content, 'usage': out.usage_metadata or {}}
                    cache.put('annotation' if not attempt else 'annotation_repair', request, response)
                raw = response['raw'].strip()
                if raw.startswith('```'):
                    raw = raw[raw.find('\n') + 1:raw.rfind('```')]
                try:
                    seeds = json.loads(raw)['questions']
                    errors = check_seeds(seeds, original_pages, counts)
                except (ValueError, KeyError, TypeError):
                    seeds, errors = [], ['invalid JSON/questions shape']
                write_json(directory / f'{ticker}_draft.json', {'request': request, 'response': response,
                                                                 'validation_errors': errors})
                if not errors:
                    for i, seed in enumerate(seeds, 1):
                        seed.update(seed_id=f'{ticker}-{i:02}', ticker=ticker,
                                    company_name=company['company_name'], industry=company['industry'],
                                    knowledge_type='company_official', annotation_status='pending_review',
                                    annotation_method='cloud_assisted_source_grounded', annotation_model=args.model)
                    write_jsonl(directory / f'{ticker}.jsonl', seeds)
                    break
                response = None
            statuses.append({'ticker': ticker, 'questions': len(seeds) if not errors else 0, 'errors': errors})
            write_json(CORPUS / 'manifests/annotation_drafting_status.json', {'companies': statuses,
                'http_attempts': budget.used, 'reserved_cost_upper_proxy_cny': budget.reserved_cny,
                'budget_cny': args.budget_cny})
            print(f'DRAFT {ticker} questions={statuses[-1]["questions"]} errors={len(errors)} calls={budget.used}', flush=True)
    finally:
        runtime.close()
    combined = []
    for path in sorted(directory.glob('*.jsonl')):
        combined += read_jsonl(path)
    write_jsonl(EVALUATION_DIR / 'annotation/company_question_seeds.jsonl', combined)


if __name__ == '__main__':
    main()
