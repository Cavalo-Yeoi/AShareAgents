"""Golden dataset validation and deterministic label-independent splitting."""
from collections import Counter, defaultdict
from datetime import date
import random
import re
from difflib import SequenceMatcher


CATEGORY_COUNTS = {'single_document_fact': 35, 'multi_evidence_synthesis': 35,
                   'cross_period_comparison': 30, 'business_risk_analysis': 30,
                   'policy_industry': 30, 'temporal_no_answer_conflict': 30}


def stratified_split(records, seed=20261116):
    groups = defaultdict(list)
    for record in records:
        groups[record['category']].append(dict(record))
    output = []
    for category in sorted(groups):
        ordered = sorted(groups[category], key=lambda x: x['query_id'])
        random.Random(f'{seed}:{category}').shuffle(ordered)
        # Greedy company/industry coverage within each category's exact quota.
        selected = []
        company, industry = Counter(), Counter()
        while len(selected) < round(len(ordered) * .2):
            candidates = [x for x in ordered if x not in selected]
            chosen = min(candidates, key=lambda x: (company[x.get('ticker', '')],
                         industry[x.get('industry', '')], ordered.index(x)))
            selected.append(chosen)
            company[chosen.get('ticker', '')] += 1
            industry[chosen.get('industry', '')] += 1
        ids = {x['query_id'] for x in selected}
        for record in ordered:
            record['split'] = 'dev' if record['query_id'] in ids else 'test'
            output.append(record)
    return sorted(output, key=lambda x: x['query_id'])


def normalize_text(text):
    return re.sub(r'\s+', '', str(text))


def validate_dataset(records, chunks):
    children = {c['metadata']['chunk_id']: c for c in chunks}
    parents = {c['metadata']['parent_id']: c for c in chunks}
    documents = {c['metadata']['document_id'] for c in chunks}
    errors, warnings, duplicates = [], [], []
    ids, questions = set(), {}
    answerable, complete = 0, 0
    for q in records:
        qid = q['query_id']
        if qid in ids:
            errors.append(f'{qid}: duplicate query_id')
        ids.add(qid)
        normalized = normalize_text(q.get('query', ''))
        if not normalized:
            errors.append(f'{qid}: empty query')
        if normalized in questions:
            errors.append(f'{qid}: duplicate question {questions[normalized]}')
        questions[normalized] = qid
        try:
            cutoff = date.fromisoformat(q['as_of_date'])
        except (ValueError, KeyError):
            errors.append(f'{qid}: invalid as_of_date')
            continue
        positive_child = [r for r in q.get('qrels_child', []) if r['relevance'] == 2]
        positive_parent = [r for r in q.get('qrels_parent', []) if r['relevance'] == 2]
        is_answerable = q.get('answerability') == 'answerable'
        if is_answerable:
            answerable += 1
            if not q.get('reference_answer') or not positive_child or not positive_parent or not q.get('evidence'):
                errors.append(f'{qid}: answerable lacks answer/positive evidence')
            else:
                complete += 1
        elif positive_child or positive_parent:
            errors.append(f'{qid}: unanswerable has positive qrels')
        for kind, table in [('child', children), ('parent', parents)]:
            for rel in q.get(f'qrels_{kind}', []):
                key = rel.get(f'{kind}_id') if kind == 'parent' else rel.get('chunk_id')
                if rel.get('relevance') not in (0, 1, 2):
                    errors.append(f'{qid}: invalid relevance')
                if key not in table:
                    errors.append(f'{qid}: unknown {kind} {key}')
                    continue
                meta = table[key]['metadata']
                if rel['relevance'] > 0:
                    published = meta.get('publish_date')
                    if not published:
                        errors.append(f'{qid}: unknown publication date')
                    elif date.fromisoformat(published) > cutoff:
                        errors.append(f'{qid}: future positive {key}')
                    if meta.get('date_status') in ('date_unverified', 'unknown', 'unverified'):
                        errors.append(f'{qid}: unverified date positive {key}')
        for doc in q.get('source_doc_ids', []):
            if doc not in documents:
                errors.append(f'{qid}: unknown source document {doc}')
        for evidence in q.get('evidence', []):
            chunk = children.get(evidence.get('chunk_id'))
            if chunk is None:
                errors.append(f'{qid}: evidence chunk not found')
                continue
            meta = chunk['metadata']
            if evidence.get('page_number') != meta.get('page_number'):
                errors.append(f'{qid}: page mismatch')
            if not evidence.get('quote') or normalize_text(evidence['quote']) not in normalize_text(chunk['page_content']):
                errors.append(f'{qid}: evidence quote not in child')
        if q.get('annotation_status') != 'verified':
            warnings.append(f'{qid}: pending human review')
    values = list(questions)
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if SequenceMatcher(None, a, b).ratio() >= .9:
                duplicates.append([questions[a], questions[b]])
    return {'questions': len(records), 'answerable': answerable,
            'unanswerable': len(records) - answerable,
            'categories': dict(Counter(q['category'] for q in records)),
            'companies': dict(Counter(q.get('company_name') or '政策/跨公司' for q in records)),
            'industries': dict(Counter(q.get('industry') or '未指定' for q in records)),
            'splits': dict(Counter(q.get('split', 'unset') for q in records)),
            'evidence_completeness': complete / answerable if answerable else None,
            'pending_review': sum(q.get('annotation_status') != 'verified' for q in records),
            'similar_pairs': duplicates, 'errors': errors, 'warnings': warnings}
