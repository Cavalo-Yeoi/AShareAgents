"""Build reviewable gold drafts from real page-grounded seeds, entirely offline."""
import argparse
from collections import Counter
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from AShareAgents.evaluation.dataset import CATEGORY_COUNTS, stratified_split, validate_dataset
from AShareAgents.evaluation.evidence_mapping import build_record
from AShareAgents.evaluation.io import (
    EVALUATION_DIR, ROOT, CORPUS, fingerprint, read_jsonl, write_json, write_jsonl, write_text,
    write_csv,
)
import json


def load_staged_documents(directory):
    """Read isolated parse artifacts, preserving original and staged identities."""
    documents = {}
    for path in sorted(Path(directory).glob('*/*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        manifest_id = item.get('manifest_document_id')
        if not manifest_id or 'pages' not in item or 'chunks' not in item:
            continue
        if manifest_id in documents:
            raise ValueError(f'multiple staged artifacts for manifest document {manifest_id}')
        # Keep the content needed for proof without repeated parent text copies.
        chunks, parents = [], {}
        for child in item['chunks']:
            meta = child['metadata']
            chunks.append({'page_content': child['page_content'], 'metadata': {
                key: meta[key] for key in ('chunk_id', 'parent_id', 'document_id',
                    'page_number', 'publish_date', 'date_status', 'doc_type',
                    'file_sha256', 'title', 'knowledge_type', 'ticker', 'industry') if key in meta}})
            if meta.get('parent_content') and meta['parent_id'] not in parents:
                parents[meta['parent_id']] = {'page_content': meta['parent_content'],
                    'metadata': {key: meta[key] for key in ('parent_id', 'document_id',
                        'page_number', 'publish_date', 'date_status') if key in meta}}
        documents[manifest_id] = {key: item.get(key) for key in (
            'manifest_document_id', 'document_id', 'file_sha256', 'parse_key', 'parser_version')}
        documents[manifest_id].update(pages=item['pages'], chunks=chunks, parents=list(parents.values()),
                                      processed_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    return documents


def build_artifacts(seeds, documents, allow_partial=False, split_seed=20261116):
    records, mapping_gaps = [], []
    for seed in seeds:
        record, issues = build_record(seed, documents)
        records.append(record)
        if issues:
            mapping_gaps.append({'seed_id': seed['seed_id'], 'issues': issues})
    records = stratified_split(records, seed=split_seed)
    all_chunks = [c for d in documents.values() for c in d['chunks']]
    quality = validate_dataset(records, all_chunks)
    actual_counts = dict(Counter(x['category'] for x in records))
    quality.update(expected_categories=CATEGORY_COUNTS,
                   expected_total=190, expected_splits={'dev': 38, 'test': 152},
                   full_dataset_quota_met=actual_counts == CATEGORY_COUNTS,
                   mapping_gaps=mapping_gaps,
                   exact_mapping_complete=not mapping_gaps,
                   annotation_status='pending_review',
                   human_verified_questions=0,
                   split_seed=split_seed,
                   seed_fingerprint=fingerprint(seeds),
                   staged_documents=len(documents),
                   provisional_metrics_only=True,
                   source_identity_notes='manifest_document_id is original manifest identity; source_doc_ids are staged document hashes')
    if not allow_partial and actual_counts != CATEGORY_COUNTS:
        quality['errors'].append('full dataset category counts do not meet requested 190-question quota')
    if not allow_partial and quality['splits'] != {'dev': 38, 'test': 152}:
        quality['errors'].append('full dataset split does not meet 38 dev / 152 test quota')
    quality['errors'].extend(f"{x['seed_id']}: {issue}" for x in mapping_gaps for issue in x['issues'])
    quality['structural_evidence_completeness'] = quality['evidence_completeness']
    for field in ('company_name', 'industry', 'category'):
        coverage = []
        labels = sorted({q.get(field) or '政策/未指定' for q in records})
        for label in labels:
            group = [q for q in records if (q.get(field) or '政策/未指定') == label]
            positives = [q for q in group if q['answerability'] == 'answerable']
            complete = sum(bool(q['evidence']) and not any(issue.startswith(('positive:', 'answerable'))
                            for issue in q['mapping_issues']) for q in positives)
            coverage.append({'group': label, 'questions': len(group), 'answerable': len(positives),
                             'fully_mapped_answerable': complete,
                             'evidence_completeness': complete / len(positives) if positives else None})
        quality['coverage_by_' + field] = coverage
    total_positive = [q for q in records if q['answerability'] == 'answerable']
    complete_count = sum(bool(q['evidence']) and not any(issue.startswith(('positive:', 'answerable'))
                         for issue in q['mapping_issues']) for q in total_positive)
    quality['evidence_completeness'] = complete_count / len(total_positive) if total_positive else None
    by_id = {q['query_id']: q for q in records}
    reviews = []
    for first, second in quality['similar_pairs']:
        left, right = by_id[first], by_id[second]
        decision, reason = 'pending_semantic_review', '相似度规则触发，保留原题供人工判断，未自动去重。'
        if left.get('ticker') != right.get('ticker') and left['answerability'] == right['answerability'] == 'unanswerable':
            decision = 'retain_distinct_company_time_boundaries'
            reason = '不同公司、披露日期和截止日的独立时间边界用例；模板相似，不能据此宣称跨语义族泛化。'
        elif ('仿制药' in left['query'] and '创新药' in right['query']) or ('创新药' in left['query'] and '仿制药' in right['query']):
            decision = 'retain_distinct_business_risk_objects'
            reason = '仿制药集采降价及销售减少，与创新药竞争、降价和医院准入障碍属于不同业务风险对象；标准答案和证据不同。'
        reviews.append({'query_ids': [first, second], 'queries': [left['query'], right['query']],
                        'splits': [left['split'], right['split']], 'decision': decision,
                        'reason': reason, 'review_type': 'assistant_semantic_review',
                        'annotation_status': 'pending_review'})
    quality['similar_pair_reviews'] = reviews
    return records, quality


GUIDELINES = """# Human review guidelines

All questions, reference answers, atomic claims and relevance labels are machine
assisted drafts with `annotation_status=pending_review`. No human verification is
claimed. Retrieval metrics against these qrels are provisional.

Review the original source page, not only the retrieved fragment. Confirm company,
report period, units, gross/net/consolidated scope, exact numbers, publication
evidence and the cutoff date. A report period is not its public disclosure date.
Official HTML uses logical page 1; PDF numbers are physical one-based pages.

For each atomic claim, check that the cited source actually supports the claim.
Multiple fragments with one `source_quote_sha256` are a boundary-spanning source
clause and must be considered together. The `quote` is literal child text;
`source_quote` preserves the original seed wording, aligned with whitespace-only
normalization. No approximate wording, number replacement or invented evidence is
permitted. Unmapped clauses remain visible in `mapping_issues` and the quality
report, and are not valid completed gold labels.

Parent and child qrels are separate. Relevance 2 marks selected necessary evidence
fragments; relevance 1 marks alternate overlapping child evidence/context and
requires review; relevance 0 comes from explicit negative-source seeds, including
future policy material excluded by the cutoff. Absence of a qrel is unjudged, not
an automatic zero. A parent label does not imply every sibling child is relevant.

Unanswerable questions have no positive evidence or positive qrels. They state
insufficient evidence within the current closed corpus and time boundary, not a
claim that the information is unknowable worldwide. Verify that a negative case
is not answered by another eligible document before accepting it.

Use the review queue to record corrections, reviewer identity and review date.
Only a real human review may change annotation_status to verified. Keep the fixed
stratified split after review; do not tune against the test answers.
"""


def write_review_exports(records, output_dir, annotation_dir):
    output_dir, annotation_dir = Path(output_dir), Path(annotation_dir)
    for grain, key in [('child', 'chunk_id'), ('parent', 'parent_id')]:
        text = ''.join(f"{q['query_id']}\t0\t{r[key]}\t{r['relevance']}\n"
                       for q in records for r in q['qrels_' + grain])
        write_text(output_dir / f'qrels_{grain}.tsv', text)
    queue = [{'query_id': q['query_id'], 'seed_id': q['seed_id'], 'category': q['category'],
              'split': q['split'], 'query': q['query'], 'annotation_status': 'pending_review',
              'reviewer': '', 'reviewed_at': '', 'review_decision': '',
              'mapping_issues': json.dumps(q['mapping_issues'], ensure_ascii=False)} for q in records]
    write_csv(annotation_dir / 'human_review_queue.csv', queue,
              fields=['query_id', 'seed_id', 'category', 'split', 'query', 'annotation_status',
                      'reviewer', 'reviewed_at', 'review_decision', 'mapping_issues'])
    write_text(annotation_dir / 'annotation_guidelines.md', GUIDELINES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='*', type=Path)
    parser.add_argument('--processed', type=Path, default=CORPUS / 'processed')
    parser.add_argument('--output-dir', type=Path, default=EVALUATION_DIR / 'datasets')
    parser.add_argument('--report-dir', type=Path, default=EVALUATION_DIR / 'reports')
    parser.add_argument('--allow-partial', action='store_true', help='Allow a clearly reported subset; never suppress mapping errors')
    parser.add_argument('--split-seed', type=int, default=20261116)
    args = parser.parse_args()
    seed_paths = args.seeds or [EVALUATION_DIR / 'annotation/company_question_seeds.jsonl',
                               EVALUATION_DIR / 'annotation/policy_question_seeds.jsonl']
    seeds = []
    for path in seed_paths:
        seeds.extend(read_jsonl(path))
    documents = load_staged_documents(args.processed)
    records, quality = build_artifacts(seeds, documents, args.allow_partial, args.split_seed)
    quality['seed_paths'] = [str(x) for x in seed_paths]
    quality['dataset_fingerprint'] = fingerprint(records)
    out = args.output_dir
    write_jsonl(out / 'golden_dataset_v1.jsonl', records)
    for split in ('dev', 'test'):
        write_jsonl(out / f'{split}.jsonl', [x for x in records if x['split'] == split])
    child_qrels, parent_qrels = [], []
    for q in records:
        child_qrels.extend({'query_id': q['query_id'], **r} for r in q['qrels_child'])
        parent_qrels.extend({'query_id': q['query_id'], **r} for r in q['qrels_parent'])
    write_jsonl(out / 'qrels_child.jsonl', child_qrels)
    write_jsonl(out / 'qrels_parent.jsonl', parent_qrels)
    queue = [{'query_id': q['query_id'], 'seed_id': q['seed_id'], 'category': q['category'],
              'split': q['split'], 'annotation_status': 'pending_review',
              'reviewer': None, 'reviewed_at': None, 'review_decision': None,
              'mapping_issues': q['mapping_issues'], 'query': q['query'],
              'checks': ['original page and quote', 'all atomic claims', 'date and units',
                         'parent/child relevance', 'answerability and closed-corpus coverage']}
             for q in records]
    write_jsonl(out / 'human_review_queue.jsonl', queue)
    write_text(out / 'annotation_guidelines.md', GUIDELINES)
    write_review_exports(records, out, EVALUATION_DIR / 'annotation' if not args.allow_partial else out / 'annotation')
    write_json(args.report_dir / 'dataset_quality_report.json', quality)
    report = ['# Golden dataset quality report', '',
              f"Questions: {len(records)}; categories: {quality['categories']}.",
              f"Splits: {quality['splits']}; requested full split: 38 dev / 152 test.",
              f"Answerable: {quality['answerable']}; unanswerable: {quality['unanswerable']}.",
              f"Pending human review: {quality['pending_review']}; human verified: 0.",
              f"Exact evidence mapping gaps: {len(quality['mapping_gaps'])} seeds.",
              f"Validation errors: {len(quality['errors'])}; similar query pairs: {len(quality['similar_pairs'])}.", '',
              'Labels and any resulting metrics remain provisional until human review.', '',
              '## Mapping and validation gaps', '']
    report.extend('- ' + error for error in quality['errors'])
    if not quality['errors']:
        report.append('No mapping or structural errors detected. This does not establish human-verified correctness.')
    report.extend(['', '## 公司、行业及证据完整率', '',
                   f"完整映射全部必要正证据的可回答题比例：{quality['evidence_completeness']}; 部分证据命中不算完整。", ''])
    for field, title in [('company_name', '公司'), ('industry', '行业'), ('category', '题型')]:
        report.extend([f'### {title}', '', '|分组|题数|可回答|全部证据映射完整|证据完整率|', '|---|---:|---:|---:|---:|'])
        for row in quality['coverage_by_' + field]:
            completeness = f"{row['evidence_completeness']:.2%}" if row['evidence_completeness'] is not None else 'N/A'
            report.append(f"|{row['group']}|{row['questions']}|{row['answerable']}|{row['fully_mapped_answerable']}|{completeness}|")
        report.append('')
    report.extend(['## 相似题去重审查', '', '以下为助手语义审查，仍需人工确认；保留决定不等于人工认证。', ''])
    for review in quality['similar_pair_reviews']:
        report.extend([f"### {review['query_ids'][0]} / {review['query_ids'][1]}", '',
                       '- ' + review['queries'][0], '- ' + review['queries'][1], '',
                       f"划分：{review['splits']}；处理：{review['decision']}。{review['reason']}", ''])
    write_text(args.report_dir / 'dataset_quality_report.md', '\n'.join(report) + '\n')
    print(json.dumps({key: quality[key] for key in ('questions', 'categories', 'splits',
          'answerable', 'unanswerable', 'pending_review', 'full_dataset_quota_met', 'exact_mapping_complete')}, ensure_ascii=True))
    print(f"mapping_gap_seeds={len(quality['mapping_gaps'])} validation_errors={len(quality['errors'])}")
    return 1 if quality['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
