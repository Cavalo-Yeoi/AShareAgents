"""Checks for machine-assisted question drafts; never grants human verification."""
from collections import Counter
from .dataset import normalize_text


def check_seeds(seeds, pages, expected_counts):
    errors = []
    if dict(Counter(s.get('category') for s in seeds)) != expected_counts:
        errors.append('category counts differ from requested quota')
    queries = set()
    for i, seed in enumerate(seeds):
        query = normalize_text(seed.get('query', ''))
        if not query or query in queries:
            errors.append(f'{i}: empty/duplicate question')
        queries.add(query)
        if not seed.get('reference_answer'):
            errors.append(f'{i}: missing reference answer')
        sources = seed.get('sources', [])
        if seed.get('answerability') == 'answerable' and not sources:
            errors.append(f'{i}: answer lacks evidence')
        if seed.get('answerability') == 'unanswerable' and sources:
            errors.append(f'{i}: unanswerable has positive sources')
        if seed.get('category') == 'cross_period_comparison' and len({s['manifest_document_id'] for s in sources}) < 2:
            errors.append(f'{i}: comparison must cite two documents')
        if seed.get('category') == 'multi_evidence_synthesis' and len({(s.get('manifest_document_id'), s.get('page_number'), normalize_text(s.get('quote', ''))) for s in sources}) < 2:
            errors.append(f'{i}: synthesis needs multiple distinct evidence clauses')
        for source in sources:
            key = (source.get('manifest_document_id'), source.get('page_number'))
            quote = normalize_text(source.get('quote', ''))
            if key not in pages or not quote or quote not in normalize_text(pages[key]):
                errors.append(f'{i}: quote is not on stated original page {key}')
            elif len(quote) > 130:
                errors.append(f'{i}: quote exceeds precise evidence-span limit')
    return errors
