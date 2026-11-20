"""Content-addressed, resumable cloud responses; credentials never enter keys/logs."""
import json
import sqlite3
import time
import threading
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .io import fingerprint, CORPUS


@contextmanager
def sqlite_connection(path, **options):
    """Commit/rollback and close: sqlite's own context manager does not close."""
    options.setdefault('timeout',60)
    conn=sqlite3.connect(path,**options)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


class ResponseCache:
    def __init__(self, path, max_requests=None, max_cost_cny=None):
        if max_cost_cny is not None and (not isinstance(max_cost_cny,(int,float))
                or not math.isfinite(max_cost_cny) or max_cost_cny<=0):
            raise ValueError('Cloud cost limit must be finite and positive')
        self.path = Path(path)
        if self.path.resolve()==(CORPUS/'knowledge_base/cloud_cache.sqlite').resolve():
            if max_cost_cny is not None and max_cost_cny>17:
                raise ValueError('Cloud lifetime allocation cannot exceed 17 CNY')
            max_cost_cny=17 if max_cost_cny is None else max_cost_cny
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite_connection(self.path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS responses '
                         '(key TEXT PRIMARY KEY, kind TEXT, request TEXT, response TEXT, created_at TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS paid_reservations '
                         '(id INTEGER PRIMARY KEY AUTOINCREMENT, estimated_cny REAL, created_at TEXT)')
        self.hits = self.misses = self.requests = 0
        self.max_requests = max_requests
        self.max_cost_cny = max_cost_cny
        self._request_lock = threading.Lock()

    def invoke(self, fn, estimated_cny=0.0):
        if not math.isfinite(estimated_cny) or estimated_cny<0:
            raise ValueError('Cloud estimated cost must be finite and nonnegative')
        def counted():
            with self._request_lock, sqlite_connection(self.path, timeout=60) as conn:
                if self.max_requests is not None and self.requests >= self.max_requests:
                    raise RuntimeError('Cloud request budget exhausted before send')
                conn.execute('BEGIN IMMEDIATE')
                total = conn.execute('SELECT COALESCE(sum(estimated_cny),0) FROM paid_reservations').fetchone()[0]
                if self.max_cost_cny is not None and total + estimated_cny > self.max_cost_cny:
                    raise RuntimeError('Cloud money budget exhausted before send')
                conn.execute('INSERT INTO paid_reservations (estimated_cny,created_at) VALUES (?,?)',
                             (estimated_cny,datetime.now(timezone.utc).isoformat()))
                self.requests += 1
            return fn()
        return bounded_call(counted)

    def get(self, kind, request):
        with sqlite_connection(self.path) as conn:
            row = conn.execute('SELECT response FROM responses WHERE key=?',
                               (fingerprint([kind, request]),)).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put(self, kind, request, response):
        serialized = json.dumps(response, ensure_ascii=False, allow_nan=False)
        with sqlite_connection(self.path) as conn:
            conn.execute('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)',
                         (fingerprint([kind, request]), kind,
                          json.dumps(request, ensure_ascii=False, sort_keys=True), serialized,
                          datetime.now(timezone.utc).isoformat()))

    def get_many(self, kind, requests):
        keys = [fingerprint([kind, r]) for r in requests]
        found = {}
        with sqlite_connection(self.path, timeout=60) as conn:
            for start in range(0, len(keys), 500):
                part = keys[start:start+500]
                for key, response in conn.execute('SELECT key,response FROM responses WHERE key IN ('
                        + ','.join('?' for _ in part) + ')', part):
                    found[key] = json.loads(response)
        self.hits += sum(key in found for key in keys)
        self.misses += sum(key not in found for key in keys)
        return [found.get(key) for key in keys]

    def put_many(self, kind, requests, responses):
        now = datetime.now(timezone.utc).isoformat()
        rows = [(fingerprint([kind,r]), kind, json.dumps(r,ensure_ascii=False,sort_keys=True),
                 json.dumps(s,ensure_ascii=False,allow_nan=False), now) for r,s in zip(requests,responses)]
        with sqlite_connection(self.path, timeout=60) as conn:
            conn.executemany('INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)',rows)


def bounded_call(fn, attempts=3):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if isinstance(exc, RuntimeError) and 'budget' in str(exc).lower():
                raise
            response = getattr(exc, 'response', None)
            if response is not None and getattr(response, 'status_code', None) in (400, 401, 403, 404):
                raise  # Permanent request/access errors cannot be repaired by retrying.
            if attempt + 1 == attempts:
                raise
            time.sleep(.5 * 2 ** attempt)


class CachedEmbeddings:
    def __init__(self, adapter, cache, workers=8, requests_per_second=8):
        self.adapter, self.cache = adapter, cache
        self.workers = workers
        self.interval = 1 / requests_per_second
        self._rate_lock = threading.Lock()
        self._last_send = 0.0
        self.identity = {'model': adapter.model, 'dimension': adapter.dimension,
                         'endpoint_host': urlsplit(adapter.url).hostname,
                         'model_revision': 'provider_alias_unpinned', 'cache_version': 1}

    def embed_documents(self, texts):
        unique = list(dict.fromkeys(texts))
        requests = [{**self.identity, 'text': t} for t in unique]
        vectors = self.cache.get_many('embedding', requests)
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        batches = [missing[start:start+self.adapter.batch_size]
                   for start in range(0,len(missing),self.adapter.batch_size)]
        def send(positions):
            batch = [unique[i] for i in positions]
            upper_tokens = sum(len(text.encode('utf-8')) + 16 for text in batch)
            def call():
                with self._rate_lock:
                    delay = self.interval - (time.monotonic()-self._last_send)
                    if delay > 0:
                        time.sleep(delay)
                    self._last_send = time.monotonic()
                return self.adapter.embed_documents(batch)
            output = self.cache.invoke(call,
                                       estimated_cny=upper_tokens * .5 / 1_000_000)
            if len(output) != len(positions):
                raise ValueError('Embedding count mismatch')
            self.cache.put_many('embedding', [requests[i] for i in positions],output)
            return positions,output
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for positions,output in pool.map(send,batches):
                for i, vector in zip(positions, output):
                    vectors[i] = vector
        by_text = dict(zip(unique,vectors))
        return [by_text[text] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class CachedReranker:
    def __init__(self, adapter, cache):
        self.adapter, self.cache = adapter, cache
        self.identity = {'model': adapter.model, 'endpoint_host': urlsplit(adapter.url).hostname,
                         'model_revision': 'provider_alias_unpinned', 'cache_version': 1}

    def score(self, query, documents):
        request = {**self.identity, 'query': query,
                   'documents': [d.page_content for d in documents]}
        scores = self.cache.get('rerank', request)
        if scores is None:
            upper_tokens = sum(len((query + doc.page_content).encode('utf-8')) + 32 for doc in documents)
            scores = self.cache.invoke(lambda: self.adapter.score(query, documents),
                                       estimated_cny=upper_tokens * .5 / 1_000_000)
            if len(scores) != len(documents):
                raise ValueError('Rerank count mismatch')
            self.cache.put('rerank', request, scores)
        return scores
