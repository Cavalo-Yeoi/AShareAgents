"""Summarize actual retrieval and judge artifacts, keeping provisional labels explicit."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, RETRIEVAL_CONFIG, CORPUS, read_jsonl, write_json, write_text, fingerprint,
)
from AShareAgents.evaluation.metrics import ranking_metrics


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize_retrieval():
    dataset=read_jsonl(EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl')
    questions={q['query_id']:q for q in dataset}
    all_runs=read_jsonl(EVALUATION_DIR / 'results/retrieval_runs.jsonl')
    corpus_hash=json.loads((CORPUS/'manifests/ingestion_summary.json').read_text(encoding='utf-8'))['corpus_hash']
    import yaml
    config_hash=fingerprint(yaml.safe_load(RETRIEVAL_CONFIG.read_text(encoding='utf-8')))
    dataset_hash=fingerprint(dataset)
    runs=[r for r in all_runs if r['query_id'] in questions and r.get('corpus_hash')==corpus_hash
          and r.get('config_hash')==config_hash and r.get('dataset_hash')==dataset_hash]
    rows=[]
    for run in runs:
        if run['status']!='ok': continue
        q=questions[run['query_id']]
        for grain,idkey in [('parent','parent_id'),('child','chunk_id')]:
            rel={r[idkey]:r['relevance'] for r in q[f'qrels_{grain}']}
            ranking=[r['metadata'][idkey] for r in run[f'{grain}ren' if grain=='child' else 'parents']]
            metric=ranking_metrics(ranking,rel)
            rows.append({'query_id':q['query_id'],'strategy':run['strategy'],'grain':grain,
                         'category':q['category'],'split':q['split'],'company':q.get('company_name','政策'),
                         'answerability':q['answerability'],'latency':run['latency'],**metric})
    groups=defaultdict(list)
    for row in rows: groups[(row['strategy'],row['grain'],row['split'])].append(row)
    aggregate=[]
    metric_keys=[k for k in rows[0] if k.startswith(('recall','mrr','ndcg'))] if rows else []
    for (strategy,grain,split),values in sorted(groups.items()):
        aggregate.append({'strategy':strategy,'grain':grain,'split':split,'queries':len(values),
                          **{k:mean([r[k] for r in values]) for k in metric_keys},
                          'mean_latency_seconds':mean([r['latency'] for r in values])})
    leaks=[]
    for run in runs:
        for doc in run.get('children',[]):
            if doc['metadata'].get('publish_date','9999')>run['as_of_date']:
                leaks.append([run['query_id'],run['strategy'],doc['metadata']['chunk_id']])
    noanswers=[]
    for strategy in sorted({r['strategy'] for r in runs}):
        selected=[r for r in runs if r['strategy']==strategy and questions[r['query_id']]['answerability']!='answerable']
        noanswers.append({'strategy':strategy,'queries':len(selected),
                          'nonempty_retrievals':sum(bool(r.get('parents')) for r in selected)})
    failures=[]
    runmap={(r['query_id'],r['strategy']):r for r in runs}
    for strategy in sorted({r['strategy'] for r in runs}):
        ranked=sorted([r for r in rows if r['grain']=='parent' and r['strategy']==strategy and r['answerability']=='answerable'],
                     key=lambda r:(r.get('recall@5',1),r.get('ndcg@5',1),r['query_id']))
        worst=[]
        seen_categories=set()
        for r in ranked:
            if r['category'] not in seen_categories:
                worst.append(r)
                seen_categories.add(r['category'])
            if len(worst)==3:
                break
        for row in worst:
            q=questions[row['query_id']]; run=runmap[(q['query_id'],strategy)]
            failures.append({'query_id':q['query_id'],'query':q['query'],'strategy':strategy,
                             'category':q['category'],'recall@5':row.get('recall@5'),'ndcg@5':row.get('ndcg@5'),
                             'required_documents':q.get('source_doc_ids',[]),
                             'expected_evidence':q.get('evidence',[])[:3],
                             'top_results':[{'document_id':d['metadata'].get('document_id'),'page':d['metadata'].get('page_number'),
                                             'title':d['metadata'].get('title'),'text':d['page_content'][:180]}
                                            for d in run.get('parents',[])[:3]]})
    differences=[]
    byq=defaultdict(dict)
    for row in rows:
        if row['grain']=='parent' and row['answerability']=='answerable': byq[row['query_id']][row['strategy']]=row
    for qid,values in byq.items():
        if 'BM25' in values and 'Dense' in values:
            delta=values['Dense']['ndcg@5']-values['BM25']['ndcg@5']
            differences.append({'query_id':qid,'query':questions[qid]['query'],'category':questions[qid]['category'],
                                'dense_minus_bm25_ndcg5':delta})
    result={'annotation_basis':'pending_review; incomplete machine qrels; provisional scores',
            'excluded_stale_runs':len(all_runs)-len(runs),'dataset_hash':dataset_hash,
            'statuses':dict(Counter(r['status'] for r in runs)),'aggregates':aggregate,
            'time_filter_violations':leaks,'no_answer_retrieval':noanswers,'failure_cases':failures,
            'strategy_difference_cases':sorted(differences,key=lambda r:r['dense_minus_bm25_ndcg5'])[:5]
                +sorted(differences,key=lambda r:r['dense_minus_bm25_ndcg5'],reverse=True)[:5]}
    write_json(EVALUATION_DIR / 'results/retrieval_analysis.json',result)
    text='# 独立检索评测报告\n\n仅根据实际结果生成。所有标准答案和qrels尚待人工复核，以下数值为暂定结果。未穷尽标注的相关块可能被误计为不相关；不能将指标解释为已人工认证的系统能力。\n\n'
    text+='同一101份语料快照，同一问题、时间与公司过滤；复用生产BM25、Dense、RRF、Reranker；BM25遍历全部符合过滤的子块。参数未依据Test调优，生成策略只按Dev父块nDCG@5选择。\n\n'
    text+='|划分|粒度|策略|Recall@5|Recall@10|Recall@20|MRR@5|nDCG@5|nDCG@10|nDCG@20|\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in aggregate:
        vals=[row.get(k) for k in ('recall@5','recall@10','recall@20','mrr@5','ndcg@5','ndcg@10','ndcg@20')]
        text+='|'+ '|'.join([row['split'],row['grain'],row['strategy']]+[f'{v:.4f}' if v is not None else 'N/A' for v in vals])+'|\n'
    test_parent={r['strategy']:r for r in aggregate if r['grain']=='parent' and r['split']=='test'}
    text+='\n相同Test父块口径下的实际差值（正数表示后者高，不能外推到未覆盖语料）：\n\n'
    for before,after in [('BM25','Dense'),('Dense','RRF'),('RRF','RRF+Reranker')]:
        if before in test_parent and after in test_parent:
            left,right=test_parent[before],test_parent[after]
            text+=f'- {before} → {after}：Recall@5变化 {right["recall@5"]-left["recall@5"]:+.4f}，nDCG@5变化 {right["ndcg@5"]-left["ndcg@5"]:+.4f}。\n'
    text+='\n|Test类别|策略|父块Recall@5|父块nDCG@5|\n|---|---|---:|---:|\n'
    categories=defaultdict(list)
    for r in rows:
        if r['grain']=='parent' and r['split']=='test' and r['answerability']=='answerable':
            categories[(r['category'],r['strategy'])].append(r)
    for (category,strategy),rs in sorted(categories.items()):
        text+=f'|{category}|{strategy}|{mean([r["recall@5"] for r in rs]):.4f}|{mean([r["ndcg@5"] for r in rs]):.4f}|\n'
    text+='\n逐问题及类别指标见 retrieval_metrics.csv；详细父子排名、分数、日期、过滤和原始候选见 retrieval_runs.jsonl、retrieval_rankings.csv、retrieval_raw_candidates.jsonl。候选Dense20/BM2520限制可发现父块数，Reranker只调整前30候选；它不能找回候选池外的证据。\n\n'
    text+=f'时间过滤违规：{len(leaks)}。无答案问题不计算Recall/MRR/nDCG，非空召回数量如下；召回非空本身不等于回答错误，也不表示系统具有拒答能力。\n\n```json\n'+json.dumps(noanswers,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='以下是真实失败案例，包含预期证据与实际Top3，供复核词面匹配、跨期混淆、分块及多证据覆盖不足。具体原因仍需结合原始页人工确认，不能从低分直接断言模型机制。\n\n'
    for case in failures:
        text+=f'### {case["query_id"]} / {case["strategy"]}\n\n{case["query"]}\n\n```json\n'+json.dumps(case,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='BM25/Dense逐题差异最大的正反案例保存在 retrieval_analysis.json。多段与跨期问题需要全部关键证据；MRR较高而Recall低通常只说明早期命中部分证据。RRF是否改善召回及精排是否改善排序，应分别比较相同粒度、划分和K，不能用单一平均数判定普遍优越。\n'
    write_text(EVALUATION_DIR / 'reports/retrieval_evaluation_report.md',text)


def summarize_generation():
    path=EVALUATION_DIR / 'results/generated_answers.jsonl'
    if not path.exists(): return
    rows=read_jsonl(path)
    from AShareAgents.evaluation.generation import active_generation_rows
    selected=json.loads((EVALUATION_DIR / 'results/selected_strategy.json').read_text(encoding='utf-8'))
    preflight=json.loads((EVALUATION_DIR / 'results/preflight.json').read_text(encoding='utf-8'))
    rows,excluded=active_generation_rows(rows,read_jsonl(EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl'),selected,preflight)
    answerable=[r for r in rows if r['answerability']=='answerable']
    negatives=[r for r in rows if r['answerability']!='answerable']
    judged=[r for r in negatives if r.get('no_answer_assessment')]
    metrics={k:{'mean':mean([r.get('metrics',{}).get(k) for r in answerable]),
                'scored':sum(k in r.get('metrics',{}) for r in answerable),
                'missing':sum(k not in r.get('metrics',{}) for r in answerable)}
             for k in ('context_precision','context_recall','faithfulness')}
    qmap={q['query_id']:q for q in read_jsonl(EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl')}
    by_split={s:{'answerable':sum(qmap[r['query_id']]['split']==s for r in answerable),
        **{k:mean([r.get('metrics',{}).get(k) for r in answerable if qmap[r['query_id']]['split']==s])
           for k in ('context_precision','context_recall','faithfulness')}} for s in ('dev','test')}
    summary={'queries':len(rows),'answerable':len(answerable),'unanswerable':len(negatives),
             'excluded_stale_rows':excluded,
             'statuses':dict(Counter(r['status'] for r in rows)),'metrics':metrics,
             'metrics_by_split':by_split,
             'heuristic_refusals':sum(r.get('refusal_heuristic',False) for r in negatives),
             'cloud_judged_no_answers':len(judged),
             'provisional_judge_refusal_accuracy':mean([int(r['no_answer_assessment']['refusal_correct']) for r in judged]),
             'provisional_judge_unsupported_answer_rate':mean([int(r['no_answer_assessment']['unsupported_answer']) for r in judged]),
             'human_verified_refusal_accuracy':None,
             'caveat':'References, qrels and heuristic refusal labels are pending human verification.'}
    write_json(EVALUATION_DIR / 'results/generation_analysis.json',summary)
    text='# RAGAS生成质量报告\n\n实际使用隔离环境中的RAGAS 0.4.4 Collections API，云端DeepSeek生成与判断。生成Prompt只含问题、日期及实际Top5父块；标准答案仅用于评测。机器参考答案尚未人工核验，结果暂定。失败评分缺失，不填0。\n\n```json\n'
    text+=json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='无答案问题单独保留回答、启发式标志和独立云模型判断。拒答正确率/无依据回答率为云评测员暂定判断，分母为成功判断数；未将这些数值冒充人工确认结果。全部20题仍需人工逐项核对。\n\n'
    failures=[r for r in answerable if r.get('metric_errors') or r.get('metrics',{}).get('faithfulness',1)<.7][:8]
    for r in failures:
        text+=f'### {r["query_id"]}\n\n{r["query"]}\n\n生成回答：{r.get("answer","生成失败")}\n\n参考答案：{r["reference_answer"]}\n\n评分：{json.dumps(r.get("metrics",{}),ensure_ascii=False)}；错误：{json.dumps(r.get("metric_errors",{}),ensure_ascii=False)}\n\n'
    text+='全部真实上下文、原始回答、参考答案、评分、模型、时间和来源ID保存在 generated_answers.jsonl；用量/持久预算见 run_summary.json及cloud_generation_budget.sqlite。标准答案与生成上下文未混合。\n'
    write_text(EVALUATION_DIR / 'reports/ragas_evaluation_report.md',text)


if __name__=='__main__':
    summarize_retrieval()
    summarize_generation()
