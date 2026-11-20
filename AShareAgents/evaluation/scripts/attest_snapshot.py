"""Attest a completed frozen experiment without API calls or DB mutations.

This is explicitly post-run provenance, never a claim that the new semantic hash
was captured during the original HTTP calls. Generation is opt-in after writers
have finished and must equal tasks rebuilt from the frozen dataset/retrieval.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import yaml
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, RETRIEVAL_CONFIG, ROOT, CORPUS, fingerprint, read_jsonl, write_json,
    write_jsonl,
)
from AShareAgents.evaluation.ingestion import CHUNK_CONFIG
from AShareAgents.evaluation.retrieval import STRATEGIES
from AShareAgents.evaluation.snapshot import (
    load_staged_snapshot, attest_database_snapshot, validate_retrieval_identity,
)


def validate_generation_identity(rows, tasks):
    by_id = {task['query_id']: task for task in tasks}
    if len(rows) != len(by_id) or len({row['query_id'] for row in rows}) != len(rows):
        raise ValueError('Generation inventory does not match complete frozen task inventory')
    for row in rows:
        task = by_id.get(row['query_id'])
        if task is None:
            raise ValueError('Unknown generation query identity')
        for field in ('query', 'as_of_date', 'reference_answer', 'context', 'contexts',
                      'strategy', 'corpus_hash', 'retrieval_config_hash', 'prompt_version'):
            if row.get(field) != task.get(field):
                raise ValueError(f'Generation {field} mismatch: {row["query_id"]}')
        for field in ('dataset_hash', 'question_hash', 'knowledge_snapshot_hash'):
            if row.get(field) not in (None, task.get(field)):
                raise ValueError(f'Generation {field} mismatch: {row["query_id"]}')
    return {'rows': len(rows), 'verified_fields': ['query', 'as_of_date', 'reference_answer',
            'context', 'contexts', 'strategy', 'corpus_hash', 'retrieval_config_hash', 'prompt_version']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=RETRIEVAL_CONFIG)
    parser.add_argument('--dataset', type=Path, default=EVALUATION_DIR / 'datasets/golden_dataset_v1.jsonl')
    parser.add_argument('--results', type=Path, default=EVALUATION_DIR / 'results')
    parser.add_argument('--include-generation', action='store_true', help='Only after the generation/metric repair writer has finished')
    parser.add_argument('--expected-documents', type=int, default=101)
    parser.add_argument('--expected-queries', type=int, default=190)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    summary_path = CORPUS / 'manifests/ingestion_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('chunk_config') != CHUNK_CONFIG or summary.get('embedding_status') != 'complete':
        raise ValueError('Ingestion is incomplete or configured chunking differs from frozen staging')
    with (CORPUS / 'manifests/ingestion_status.csv').open(encoding='utf-8-sig', newline='') as stream:
        staged = load_staged_snapshot(CORPUS / 'processed', list(csv.DictReader(stream)))
    if len(staged) != args.expected_documents:
        raise ValueError('Frozen staged document count differs from expected experiment')
    attestation = attest_database_snapshot(ROOT / config['knowledge_database'], staged, CHUNK_CONFIG,
                                          summary.get('embedding_identity'))
    if summary.get('child_chunks') != attestation['children']:
        raise ValueError('Frozen summary child count differs from exact DB inventory')
    knowledge_hash = attestation['knowledge_snapshot_hash']
    raw_hash, config_hash = summary['corpus_hash'], fingerprint(config)
    questions = read_jsonl(args.dataset)
    if len(questions) != args.expected_queries:
        raise ValueError('Frozen dataset count differs from expected experiment')
    dataset_hash = fingerprint(questions)
    runs_path = args.results / 'retrieval_runs.jsonl'
    runs = read_jsonl(runs_path)
    verified = validate_retrieval_identity(runs, questions, config_hash, raw_hash, knowledge_hash, STRATEGIES)
    question_map = {q['query_id']: q for q in questions}
    raw_path = args.results / 'retrieval_raw_candidates.jsonl'
    raw = read_jsonl(raw_path)
    synthetic = []
    for row in raw:
        q = question_map.get(row['query_id'])
        if q is None:
            raise ValueError('Raw candidate record has unknown question identity')
        synthetic.append({**row, 'status': 'ok', 'strategy': 'raw_candidates', 'as_of_date': q['as_of_date'],
                          'children': [doc for docs in row['raw_candidates'].values() for doc in docs], 'parents': []})
    validate_retrieval_identity(synthetic, questions, config_hash, raw_hash, knowledge_hash, ['raw_candidates'])
    metadata_paths = [args.results / 'retrieval_summary.json', args.results / 'selected_strategy.json']
    metadata = [json.loads(path.read_text(encoding='utf-8')) for path in metadata_paths]
    for value in metadata:
        for field, expected in [('corpus_hash', raw_hash), ('config_hash', config_hash), ('dataset_hash', dataset_hash)]:
            if value.get(field) != expected:
                raise ValueError(f'Frozen retrieval metadata {field} mismatch')
        if value.get('knowledge_snapshot_hash') not in (None, knowledge_hash):
            raise ValueError('Frozen retrieval metadata semantic hash mismatch')
    stamp = {'knowledge_snapshot_hash': knowledge_hash, 'raw_corpus_hash': raw_hash,
             'identity_attestation': 'post_run_verified_frozen_snapshot',
             'identity_attested_at': datetime.now(timezone.utc).isoformat(),
             'knowledge_snapshot_hash_recorded_at_execution': False}
    tagged_runs = [{**r, **stamp} for r in runs]
    generation_rows = None
    generation_report = None
    if args.include_generation:
        from AShareAgents.evaluation.scripts.run_generation_eval import build_tasks
        selected_strategy = metadata[1]['strategy']
        tasks = build_tasks(questions, tagged_runs, selected_strategy, max_queries=None)
        generation_path = args.results / 'generated_answers.jsonl'
        generation_rows = read_jsonl(generation_path)
        generation_report = validate_generation_identity(generation_rows, tasks)
        task_map = {task['query_id']: task for task in tasks}
        generation_rows = [{**r, **stamp, 'dataset_hash': dataset_hash,
                            'question_hash': task_map[r['query_id']]['question_hash']}
                           for r in generation_rows]
    # No artifacts are changed until all applicable checks above have passed.
    write_jsonl(runs_path, tagged_runs)
    write_jsonl(raw_path, [{**r, **stamp} for r in raw])
    for path, value in zip(metadata_paths, metadata):
        write_json(path, {**value, **stamp})
    write_json(summary_path, {**summary, 'knowledge_snapshot_hash': knowledge_hash,
               'raw_corpus_hash': raw_hash, 'snapshot_version': attestation['snapshot_version'],
               'knowledge_identity_attestation': stamp['identity_attestation']})
    if generation_rows is not None:
        write_jsonl(args.results / 'generated_answers.jsonl', generation_rows)
    report = {**attestation, **stamp, 'raw_corpus_hash': raw_hash, 'config_hash': config_hash,
              'dataset_hash': dataset_hash, 'retrieval_validation': verified,
              'raw_candidate_records': len(raw), 'generation_validation': generation_report,
              'database_access': 'read_only', 'paid_api_calls': 0,
              'provenance_caveat': 'Semantic identity attached only after frozen post-run verification; not recorded during original paid calls.'}
    write_json(EVALUATION_DIR / 'reports/snapshot_attestation.json', report)
    print(json.dumps({'knowledge_snapshot_hash': knowledge_hash, 'documents': attestation['documents'],
          'children': attestation['children'], 'retrieval_runs': len(runs),
          'generation_rows_attested': len(generation_rows) if generation_rows is not None else 0,
          'paid_api_calls': 0}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
