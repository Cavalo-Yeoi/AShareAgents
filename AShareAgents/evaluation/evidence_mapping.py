"""Exact page/child evidence alignment with explicit gaps and temporal gating."""
from collections import defaultdict
from datetime import date
import hashlib

from .dataset import normalize_text


def _normalized_offsets(text):
    # MarkdownHeaderTextSplitter drops nonprintable font/control glyphs before
    # splitting. Mirror that deterministic formatting filter for alignment only;
    # the original source quotation is proved unchanged against the raw page.
    offsets = [i for i, character in enumerate(text) if not character.isspace() and character.isprintable()]
    return ''.join(text[i] for i in offsets), offsets


def map_quote(source, document):
    """Map a literal source quote to a complete, minimal child interval cover.

    Whitespace may differ between extraction modes; no fuzzy wording/number
    substitution is allowed. Quotes crossing a child boundary produce multiple
    exact child fragments, each retaining the complete original source quote.
    """
    failure = {'status': 'unmapped', 'coverage_complete': False, 'evidence': [],
               'overlap_children': [], 'reason': ''}
    page_number = source.get('page_number')
    page = next((p for p in document.get('pages', []) if p['page_number'] == page_number), None)
    if page is None:
        return dict(failure, reason='source physical/logical page absent from staged original')
    needle = normalize_text(source.get('quote', ''))
    original_page_text = normalize_text(page['content'])
    original_start = original_page_text.find(needle) if needle else -1
    if original_start < 0:
        return dict(failure, reason='exact quote not found on declared staged page')
    page_text = ''.join(c for c in original_page_text if c.isprintable())
    start = sum(c.isprintable() for c in original_page_text[:original_start])
    aligned_needle = ''.join(c for c in needle if c.isprintable())
    end = start + len(aligned_needle)
    if not aligned_needle:
        return dict(failure, reason='quote contains no printable evidence characters')
    intervals = []
    for child in document.get('chunks', []):
        meta = child['metadata']
        if meta.get('doc_type', 'child') != 'child' or meta.get('page_number') != page_number:
            continue
        text, offsets = _normalized_offsets(child['page_content'])
        if not text:
            continue
        position = page_text.find(text)
        while position >= 0:
            stop = position + len(text)
            if position < end and stop > start:
                intervals.append((position, stop, child, offsets))
            position = page_text.find(text, position + 1)
    cursor, chosen = start, []
    while cursor < end:
        choices = [x for x in intervals if x[0] <= cursor < x[1]]
        if not choices:
            return dict(failure, reason=f'exact quote has uncovered child interval at normalized page offset {cursor}')
        chosen_interval = max(choices, key=lambda x: (x[1], -x[0], x[2]['metadata']['chunk_id']))
        lo, hi, child, offsets = chosen_interval
        covered_end = min(end, hi)
        local_lo, local_hi = cursor - lo, covered_end - lo
        fragment = child['page_content'][offsets[local_lo]:offsets[local_hi - 1] + 1]
        meta = child['metadata']
        chosen.append({'manifest_document_id': document['manifest_document_id'],
                       'document_id': document['document_id'], 'page_number': page_number,
                       'chunk_id': meta['chunk_id'], 'parent_id': meta['parent_id'],
                       'quote': fragment, 'source_quote': source['quote'],
                       'source_quote_sha256': hashlib.sha256(source['quote'].encode()).hexdigest(),
                       'source_page_span_normalized': [start, end],
                       'alignment_filter': 'whitespace_and_chunker_nonprintable_glyph_filter',
                       'fragment_page_span_normalized': [cursor, covered_end],
                       'mapping_method': 'exact_whitespace_normalized_interval_cover'})
        cursor = covered_end
    selected = {x['chunk_id'] for x in chosen}
    overlap = {x[2]['metadata']['chunk_id']: x[2] for x in intervals
               if x[2]['metadata']['chunk_id'] not in selected}
    return {'status': 'mapped', 'coverage_complete': True, 'evidence': chosen,
            'overlap_children': list(overlap.values()), 'reason': '',
            'source_quote_occurrences_on_page': original_page_text.count(needle)}


def build_record(seed, documents):
    """Build provisional parent/child labels without promoting future evidence."""
    issues, evidence, source_documents, negative_documents, mappings = [], [], set(), set(), []
    child_rels, parent_rels = {}, {}
    cutoff = date.fromisoformat(seed['as_of_date'])
    answerable = seed.get('answerability') == 'answerable'
    positive_sources = seed.get('sources', [])
    if not answerable and positive_sources:
        issues.append('unanswerable seed declares positive sources; positives excluded')
        positive_sources = []
    if answerable and not positive_sources:
        issues.append('answerable seed lacks positive sources')

    def add_relevance(chunk, relevance, reason):
        meta = chunk['metadata']
        cid, pid = meta['chunk_id'], meta['parent_id']
        if cid not in child_rels or relevance > child_rels[cid]['relevance']:
            child_rels[cid] = {'chunk_id': cid, 'relevance': relevance, 'reason': reason}
        if pid not in parent_rels or relevance > parent_rels[pid]['relevance']:
            parent_rels[pid] = {'parent_id': pid, 'relevance': relevance, 'reason': reason}

    for polarity, sources in [('positive', positive_sources), ('negative', seed.get('negative_sources', []))]:
        for source in sources:
            manifest_id = source.get('manifest_document_id')
            document = documents.get(manifest_id)
            if document is None:
                issues.append(f'{polarity}: staged document missing {manifest_id}')
                continue
            result = map_quote(source, document)
            mappings.append({'polarity': polarity, 'manifest_document_id': manifest_id,
                             'page_number': source.get('page_number'), 'quote': source.get('quote'),
                             'status': result['status'], 'reason': result['reason'],
                             'coverage_complete': result['coverage_complete']})
            if result['status'] != 'mapped':
                issues.append(f"{polarity}: {manifest_id} page {source.get('page_number')}: {result['reason']}")
                continue
            children = {x['metadata']['chunk_id']: x for x in document['chunks']}
            selected = [children[x['chunk_id']] for x in result['evidence']]
            if polarity == 'positive':
                valid = True
                for child in selected + result['overlap_children']:
                    meta = child['metadata']
                    published = meta.get('publish_date')
                    if not published or meta.get('date_status') != 'verified':
                        issues.append(f'positive: unknown/unverified publication date {manifest_id}')
                        valid = False
                    elif date.fromisoformat(published) > cutoff:
                        issues.append(f'positive: future publication excluded {manifest_id}')
                        valid = False
                if not valid:
                    continue
                source_documents.add(document['document_id'])
                evidence.extend(result['evidence'])
                for child in selected:
                    add_relevance(child, 2, 'necessary_exact_evidence_fragment')
                # An alternate overlapping child contains a literal part of the
                # quoted evidence; this is provisional partial support, not an
                # automatically relevant neighboring paragraph.
                for child in result['overlap_children']:
                    add_relevance(child, 1, 'alternate_child_overlapping_exact_evidence')
            else:
                negative_documents.add(document['document_id'])
                for child in selected + result['overlap_children']:
                    add_relevance(child, 0, source.get('reason', 'explicit_seed_negative'))
            # Judge parent text independently. A second overlapping parent can
            # contain the complete clause even when none of its children was
            # needed by the minimal child interval cover.
            for parent in document.get('parents', []):
                meta = parent['metadata']
                if meta.get('page_number') != source['page_number']:
                    continue
                if normalize_text(source['quote']) not in normalize_text(parent['page_content']):
                    continue
                relevance = 0
                if polarity == 'positive':
                    published = meta.get('publish_date')
                    if meta.get('date_status') != 'verified' or not published or date.fromisoformat(published) > cutoff:
                        continue
                    relevance = 2
                pid = meta['parent_id']
                if pid not in parent_rels or relevance > parent_rels[pid]['relevance']:
                    parent_rels[pid] = {'parent_id': pid, 'relevance': relevance,
                        'reason': 'parent_contains_complete_exact_source_clause' if relevance else source.get('reason', 'explicit_seed_negative')}

    record = {k: v for k, v in seed.items() if k not in ('sources', 'negative_sources', 'annotation_status')}
    record.update(query_id='ragv1_' + seed['seed_id'], annotation_status='pending_review',
                  source_doc_ids=sorted(source_documents), negative_source_doc_ids=sorted(negative_documents),
                  evidence=evidence, qrels_child=sorted(child_rels.values(), key=lambda x: x['chunk_id']),
                  qrels_parent=sorted(parent_rels.values(), key=lambda x: x['parent_id']),
                  source_mappings=mappings,
                  annotation_method=seed.get('annotation_method', 'machine_assisted_seed_to_exact_page_mapping'),
                  qrels_status='provisional_pending_human_review',
                  expected_behavior=seed.get('expected_behavior', 'answer' if answerable else 'refuse'))
    if answerable and not evidence:
        issues.append('answerable evidence coverage incomplete; cannot be considered validated gold')
    record['mapping_issues'] = issues
    return record, issues
