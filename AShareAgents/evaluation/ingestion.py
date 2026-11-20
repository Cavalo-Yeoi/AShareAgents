"""Hash-aware page staging; cloud ingestion is an explicit separate operation."""
import hashlib
import json
from pathlib import Path

from AShareAgents.rag.data_preparation import DataPreparationModule
from AShareAgents.rag.pdf_parser import PARSER_VERSION, parse_html, parse_pdf, text_quality_flags
from .io import ROOT, fingerprint, write_json


CHUNK_CONFIG = {'parent_chunk_size': 800, 'parent_chunk_overlap': 100,
                'child_chunk_size': 200, 'child_chunk_overlap': 40}


def stage_document(record, processed_root):
    raw = Path(record['raw_path'])
    raw = raw if raw.is_absolute() else ROOT / raw
    actual_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    if actual_hash != record['file_sha256']:
        raise ValueError(f'SHA256 mismatch: {raw.name}')
    identity_fields = ['title', 'knowledge_type', 'document_type', 'ticker', 'company_name',
                       'industry', 'issuer', 'source_url', 'publish_date', 'date_status',
                       'report_period', 'effective_date', 'expire_date', 'revision_date', 'document_id']
    key = fingerprint({'file_sha256': actual_hash, 'parser': PARSER_VERSION, 'chunker': CHUNK_CONFIG,
                       'metadata': {k: record.get(k) for k in identity_fields}})
    target = Path(processed_root) / ('company' if record['knowledge_type'] == 'company_official' else 'policy') / f'{key}.json'
    if target.exists():
        result = json.loads(target.read_text(encoding='utf-8'))
        bad_pages = set()
        for page in result['pages']:
            flags = text_quality_flags(page['content'])
            if flags:
                page['quality_flags'] = sorted(set(page['quality_flags'] + flags))
                page['metadata']['quality_flags'] = page['quality_flags']
            if 'suspicious_unicode_mapping' in page['quality_flags']:
                bad_pages.add(page['page_number'])
        if bad_pages:
            result['chunks'] = [c for c in result['chunks'] if c['metadata']['page_number'] not in bad_pages]
            result['child_count'] = len(result['chunks'])
            result['parent_count'] = len({c['metadata']['parent_id'] for c in result['chunks']})
            result['quarantined_pages'] = sorted(bad_pages)
            write_json(target, result)
        result['cached'] = True
        return result
    metadata = {**record, 'doc_id': record.get('document_id', ''), 'source': record['source_url'],
                'source_url': record['source_url'], 'source_level': record.get('source_level', 'official'),
                'publish_date': record.get('publish_date') or '',
                'effective_date': record.get('effective_date') or ''}
    parsed = parse_pdf(raw, metadata) if raw.suffix.lower() == '.pdf' else parse_html(raw, metadata)
    # Official HTML body selector from collection is preserved beside its raw original.
    if raw.suffix.lower() != '.pdf' and record.get('body_text_path') and parsed['pages']:
        body_path = ROOT / record['body_text_path']
        if body_path.is_file():
            parsed['pages'][0]['content'] = body_path.read_text(encoding='utf-8-sig')
            parsed['pages'][0]['metadata']['quality_flags'] = ['official_html_article_extract', 'logical_page_not_pdf']
            parsed['pages'][0]['quality_flags'] = ['official_html_article_extract', 'logical_page_not_pdf']
    pages, children = [], []
    for page in parsed['pages']:
        pages.append({**page, 'manifest_document_id': record.get('document_id'),
                      'document_id': hashlib.sha256(record['source_url'].encode()).hexdigest()})
        if not page['content'].strip() or 'suspicious_unicode_mapping' in page['quality_flags']:
            continue
        preparation = DataPreparationModule(knowledge_type=record['knowledge_type'], **CHUNK_CONFIG)
        docs = preparation.prepare_records([{'content': page['content'], 'metadata': page['metadata']}])
        for chunk in preparation.chunk_documents(docs):
            children.append({'page_content': chunk.page_content, 'metadata': chunk.metadata})
    result = {'manifest_document_id': record.get('document_id'), 'document_id': hashlib.sha256(record['source_url'].encode()).hexdigest(),
              'file_sha256': actual_hash, 'parse_key': key, 'parser_version': PARSER_VERSION,
              'page_count': parsed['page_count'], 'pages': pages, 'errors': parsed['errors'],
              'chunks': children, 'parent_count': len({c['metadata']['parent_id'] for c in children}),
              'child_count': len(children), 'cached': False, 'processed_path': str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)}
    result['quarantined_pages'] = [p['page_number'] for p in pages if 'suspicious_unicode_mapping' in p['quality_flags']]
    write_json(target, result)
    return result
