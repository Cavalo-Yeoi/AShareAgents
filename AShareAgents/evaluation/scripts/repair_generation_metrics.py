"""Retry only missing actual judge metrics; never regenerate or change answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
import math
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AShareAgents.evaluation.io import EVALUATION_DIR, CORPUS, read_jsonl, write_jsonl, write_json, write_csv
from AShareAgents.evaluation.cloud_cache import ResponseCache
from AShareAgents.evaluation.scripts.run_generation_eval import (
    build_tasks, judge_request, paid_backends, CallBudget, verify_total_budget, METRICS,
)


def repair_row(row,task,judges,cache,output_limit):
    repaired=deepcopy(row)
    config={**row['config'],'repair_judge_output_limit':output_limit,'repair_policy':'missing_metric_only_v1'}
    for metric in list(row.get('metric_errors',{})):
        if metric not in METRICS or metric not in judges:
            continue
        request=judge_request(task,row['answer'],row['judge_model'],metric,config)
        try:
            result=cache.get('metric_repair',request)
            if result is None:
                result=judges[metric](task,row['answer'])
                value=float(result['value'])
                if not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError('Metric undefined/nonfinite; not replaced with zero')
                cache.put('metric_repair',request,result)
            repaired['metrics'][metric]=result['value']
            repaired['usage'][metric]=result.get('usage',{})
            repaired['metric_errors'].pop(metric,None)
            repaired.setdefault('metric_repairs',{})[metric]={
                'judge_max_output_tokens':output_limit,'reason':row['metric_errors'][metric],
                'repaired_at':datetime.now(timezone.utc).isoformat(),'model':row['judge_model']}
        except Exception as exc:
            repaired['metric_errors'][metric]=type(exc).__name__
    repaired['status']='partial_metric_error' if repaired['metric_errors'] else 'ok'
    return repaired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--budget-cny',type=float,default=28)
    parser.add_argument('--max-calls',type=int,default=600)
    parser.add_argument('--judge-output-limit',type=int,default=8192)
    parser.add_argument('--workers',type=int,default=8,choices=(1,2,3,4,6,8))
    args=parser.parse_args()
    rows=read_jsonl(EVALUATION_DIR / 'results/generated_answers.jsonl')
    strategy=rows[0]['strategy']
    tasks=build_tasks(read_jsonl(EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl'),
                      read_jsonl(EVALUATION_DIR / 'results/retrieval_runs.jsonl'),strategy)
    taskmap={t['query_id']:t for t in tasks}
    pending=[r for r in rows if r.get('answer') and r.get('metric_errors')]
    print(f'PENDING_METRIC_REPAIR {len(pending)} judge_limit={args.judge_output_limit}',flush=True)
    if not args.allow_paid or not pending:
        return
    verify_total_budget(CORPUS/'knowledge_base/cloud_cache.sqlite',args.budget_cny)
    args.model=rows[0]['model'];args.judge_model=rows[0]['judge_model']
    args.base_url='https://api.deepseek.com';args.max_output_tokens=rows[0]['config']['max_output_tokens']
    args.judge_max_output_tokens=args.judge_output_limit
    budget=CallBudget(args.max_calls,args.budget_cny,max(args.max_output_tokens,args.judge_output_limit),
                       64000,ledger_path=CORPUS/'cloud_generation_budget.sqlite')
    cache=ResponseCache(CORPUS/'generation_cache.sqlite')
    def worker(row):
        task=taskmap[row['query_id']]
        if any(task[k]!=row[k] for k in ('context','reference_answer','query','as_of_date')):
            raise ValueError('Metric repair input changed; freeze the original run')
        _,judges,close=paid_backends(args,budget)
        try:
            return repair_row(row,task,judges,cache,args.judge_output_limit)
        finally:
            close()
    index={r['query_id']:i for i,r in enumerate(rows)}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(worker,r) for r in pending]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result();rows[index[row['query_id']]]=row
            write_jsonl(EVALUATION_DIR / 'results/generated_answers.jsonl',rows)
            print(f'METRIC_REPAIR {i}/{len(pending)} {row["query_id"]} {row["status"]} calls={budget.used} reserved={budget.reserved_cny:.6f}',flush=True)
    write_csv(EVALUATION_DIR / 'results/ragas_metrics.csv',[{'query_id':r['query_id'],'strategy':r['strategy'],
        'model':r['model'],'judge_model':r['judge_model'],'status':r['status'],**r['metrics'],
        'judge_refusal_correct':r.get('no_answer_assessment',{}).get('refusal_correct'),
        'judge_unsupported_answer':r.get('no_answer_assessment',{}).get('unsupported_answer'),
        'annotation_status':r['annotation_status'],'evaluation_time':r['evaluation_time']} for r in rows])
    write_json(EVALUATION_DIR / 'results/metric_repair_summary.json',{'attempted_queries':len(pending),
        'remaining_partial_queries':sum(bool(r.get('metric_errors')) for r in rows),
        'http_attempts_this_process':budget.used,'judge_output_limit':args.judge_output_limit,
        'generation_answers_unchanged':True,'net_reserved_cny':budget.reserved_cny})


if __name__=='__main__':main()
