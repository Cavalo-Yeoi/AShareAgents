"""Semantic knowledge identity and read-only attestation of frozen artifacts.

Raw corpus bytes alone do not identify a knowledge base: corrected publication
metadata, parsing/chunking settings and embedding identity also change its meaning.
No provider setup, environment reads or database mutations occur in this module.
"""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sqlite3

from .io import ROOT, fingerprint


SNAPSHOT_VERSION = 'rag-eval-knowledge-snapshot-v1'
POST_RUN_ATTESTATION = 'post_run_verified_frozen_snapshot'


def _canonical_rows(rows):
    canonical, seen = [], set()
    for row in rows:
        item = {key: row[key] for key in ('document_id', 'parse_key', 'file_sha256', 'child_count')}
        if not all(isinstance(item[key], str) and item[key] for key in ('document_id', 'parse_key', 'file_sha256')):
            raise ValueError('Incomplete document/parse/hash snapshot identity')
        if type(item['child_count']) is not int or item['child_count'] < 0:
            raise ValueError('Invalid child_count snapshot identity')
        if item['document_id'] in seen:
            raise ValueError('Duplicate document_id in snapshot identity')
        seen.add(item['document_id'])
        canonical.append(item)
    return sorted(canonical, key=lambda x: x['document_id'])


def compute_knowledge_snapshot_hash(ingestion_rows, chunk_config, embedding_identity):
    if not isinstance(chunk_config, dict) or not chunk_config:
        raise ValueError('Chunk configuration is required for semantic snapshot identity')
    if not isinstance(embedding_identity, dict) or not embedding_identity:
        raise ValueError('Embedding identity is required for semantic snapshot identity')
    return fingerprint({'snapshot_version': SNAPSHOT_VERSION,
                        'documents': _canonical_rows(ingestion_rows),
                        'chunk_config': chunk_config, 'embedding_identity': embedding_identity})


def load_staged_snapshot(processed_root, status_rows=None):
    """Read current parse artifacts; an inventory avoids accidentally using old caches."""
    root = Path(processed_root).resolve()
    paths = []
    if status_rows is None:
        paths = sorted(root.glob('*/*.json'))
    else:
        for row in status_rows:
            if row.get('parse_status') != 'parsed':
                raise ValueError('Snapshot inventory contains unparsed documents')
            path = Path(row['processed_path'])
            path = path if path.is_absolute() else ROOT / path
            if not path.resolve().is_relative_to(root):
                raise ValueError('Staged artifact is outside the isolated processed directory')
            paths.append(path)
    output = []
    for path in paths:
        item = json.loads(path.read_text(encoding='utf-8'))
        chunk_ids = [c['metadata']['chunk_id'] for c in item['chunks']]
        if len(chunk_ids) != item['child_count'] or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f'Staged child identity/count mismatch: {path.name}')
        output.append({key: item[key] for key in ('manifest_document_id', 'document_id',
                      'parse_key', 'file_sha256', 'child_count')} | {'chunk_ids': chunk_ids})
    _canonical_rows(output)
    return output


def attest_database_snapshot(database_path, staged_documents, chunk_config, expected_embedding_identity=None):
    """Check every stage/ledger tuple and exact stored child IDs, opening DB read-only."""
    staged = list(staged_documents.values()) if isinstance(staged_documents, dict) else list(staged_documents)
    expected = {row['document_id']: row for row in _canonical_rows(staged)}
    uri = Path(database_path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        ledger = [dict(zip(('document_id', 'parse_key', 'file_sha256', 'child_count'), row))
                  for row in db.execute('SELECT document_id,parse_key,file_sha256,child_count FROM eval_ingestion')]
        actual = {row['document_id']: row for row in _canonical_rows(ledger)}
        if set(actual) != set(expected):
            raise ValueError('Stage/DB document inventory mismatch')
        for doc_id in expected:
            if actual[doc_id] != expected[doc_id]:
                raise ValueError(f'Stage/DB parse key, file hash or child count mismatch: {doc_id}')
        stored = {}
        for chunk_id, document_id in db.execute('SELECT chunk_id,document_id FROM rag_chunks'):
            stored.setdefault(document_id, set()).add(chunk_id)
        if set(stored) != set(expected):
            raise ValueError('Actual DB child document inventory differs from ingestion ledger')
        stage_lookup = {x['document_id']: x for x in staged}
        for doc_id, row in expected.items():
            actual_ids = stored[doc_id]
            if len(actual_ids) != row['child_count']:
                raise ValueError(f'Actual DB child count mismatch: {doc_id}')
            stage_ids = stage_lookup[doc_id].get('chunk_ids')
            if stage_ids is not None and actual_ids != set(stage_ids):
                raise ValueError(f'Actual DB child IDs differ from staged child IDs: {doc_id}')
        value = db.execute("SELECT value FROM rag_settings WHERE key='embedding_identity'").fetchone()
        if value is None:
            raise ValueError('DB embedding identity absent')
        identity = json.loads(value[0])
    if expected_embedding_identity is not None and identity != expected_embedding_identity:
        raise ValueError('DB embedding identity differs from expected identity')
    snapshot_hash = compute_knowledge_snapshot_hash(ledger, chunk_config, identity)
    return {'knowledge_snapshot_hash': snapshot_hash, 'snapshot_version': SNAPSHOT_VERSION,
            'documents': len(ledger), 'children': sum(x['child_count'] for x in ledger),
            'embedding_identity': identity, 'chunk_config': dict(chunk_config),
            'verified_ingestion_rows': _canonical_rows(ledger),
            'database_attestation': 'all_staged_parse_keys_raw_hashes_counts_and_child_ids_verified'}


def validate_retrieval_identity(runs, questions, config_hash, raw_corpus_hash,
                                knowledge_snapshot_hash, strategies):
    """Fail closed before attaching a semantic identity to old successful results."""
    by_id = {q['query_id']: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError('Duplicate current question identity')
    dataset_hash = fingerprint(questions)
    expected = {(qid, strategy) for qid in by_id for strategy in strategies}
    observed = []
    returned = 0
    for row in runs:
        key = (row.get('query_id'), row.get('strategy'))
        if key not in expected or row.get('status') != 'ok':
            raise ValueError('Retrieval run inventory/status is not the complete current experiment')
        observed.append(key)
        question = by_id[key[0]]
        for field, wanted in [('config_hash', config_hash), ('corpus_hash', raw_corpus_hash),
                              ('dataset_hash', dataset_hash), ('question_hash', fingerprint(question)),
                              ('as_of_date', question['as_of_date'])]:
            if row.get(field) != wanted:
                raise ValueError(f'Retrieval {field} mismatch for question {key[0]}')
        if row.get('knowledge_snapshot_hash') not in (None, knowledge_snapshot_hash):
            raise ValueError('Retrieval semantic knowledge snapshot mismatch')
        cutoff = date.fromisoformat(question['as_of_date'])
        for document in row.get('children', []) + row.get('parents', []):
            meta = document['metadata']
            published = meta.get('publish_date')
            if not published or meta.get('date_status') != 'verified':
                raise ValueError('Retrieved document has unknown/unverified publication metadata')
            if date.fromisoformat(published) > cutoff:
                raise ValueError(f'future retrieved evidence for question {key[0]}')
            for field in ('knowledge_type', 'ticker', 'document_type', 'report_period'):
                if question.get(field) and meta.get(field) != question[field]:
                    raise ValueError(f'Retrieved {field} scope mismatch for question {key[0]}')
            if question.get('knowledge_type') == 'policy_industry' and question.get('industry') and meta.get('industry') != question['industry']:
                raise ValueError('Retrieved policy industry scope mismatch')
            returned += 1
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Duplicate or missing current retrieval question/strategy runs')
    return {'runs': len(runs), 'questions': len(questions), 'strategies': list(strategies),
            'returned_documents_date_scope_verified': returned, 'dataset_hash': dataset_hash}
