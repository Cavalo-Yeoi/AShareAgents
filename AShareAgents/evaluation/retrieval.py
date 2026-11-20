"""Independent experiment orchestration around existing retrieval algorithms."""
from copy import deepcopy
from collections import OrderedDict
import time

from langchain_core.documents import Document
from AShareAgents.rag.retrieval_optimization import RetrievalOptimizationModule


STRATEGIES = ['BM25', 'Dense', 'RRF', 'RRF+Reranker']


class BM25PoolCache:
    """Bounded index reuse; identical production tokenizer, BM25 and stable ties."""
    def __init__(self, max_pools=2):
        self.pools=OrderedDict()
        self.max_pools=max_pools
        self.builds=0

    def search(self, query, candidates, top_k, pool_key):
        if not candidates:
            return []
        if pool_key not in self.pools:
            from rank_bm25 import BM25Okapi
            tokenized=[RetrievalOptimizationModule._tokenize(d.page_content) for d in candidates]
            self.pools[pool_key]=(BM25Okapi(tokenized),candidates)
            self.builds+=1
            while len(self.pools)>self.max_pools:
                self.pools.popitem(last=False)
        self.pools.move_to_end(pool_key)
        bm25,documents=self.pools[pool_key]
        scores=bm25.get_scores(RetrievalOptimizationModule._tokenize(query))
        ranked=sorted(zip(documents,scores),key=lambda x:float(x[1]),reverse=True)
        return [Document(page_content=d.page_content,metadata={**deepcopy(d.metadata),'bm25_score':float(s)})
                for d,s in ranked[:top_k] if float(s)>0]


def eligible_for_question(metadata, question):
    """Equivalent text-staging eligibility before vectors exist; fail closed on dates."""
    if metadata.get('date_status') not in ('verified', 'official_verified', 'verified_official'):
        return False
    published = metadata.get('publish_date')
    if not published or published > question['as_of_date']:
        return False
    for field in ('knowledge_type', 'ticker', 'document_type', 'report_period'):
        if question.get(field) and metadata.get(field) != question[field]:
            return False
    if question.get('company_name') and not question.get('ticker') and metadata.get('company_name') != question['company_name']:
        return False
    if question.get('industry') and question.get('knowledge_type') == 'policy_industry' and metadata.get('industry') != question['industry']:
        return False
    return True


def clone(documents):
    return [Document(page_content=d.page_content, metadata=deepcopy(d.metadata)) for d in documents]


def retrieve_all(query, eligible_documents, index, reranker, filters, config, strategies=None, bm25_search=None):
    strategies = strategies or STRATEGIES
    optimizer = RetrievalOptimizationModule(index, rerank_enabled=False,
                                            rrf_k=config.get('rrf_k', 60))
    started = time.perf_counter()
    bm25 = (bm25_search(query, eligible_documents, config.get('bm25_top_k', 20)) if bm25_search
            else optimizer._bm25_search(query, clone(eligible_documents), config.get('bm25_top_k', 20)))
    bm25_seconds = time.perf_counter() - started
    output = {'eligible_pool_count': len(eligible_documents), 'strategies': {},
              'raw_candidates': {'BM25': clone(bm25)}, 'errors': {}}
    dense, dense_seconds = [], 0.0
    if any(s != 'BM25' for s in strategies):
        if index is None:
            output['errors']['Dense'] = 'Document/query embeddings unavailable; no synthetic vectors used'
        else:
            started = time.perf_counter()
            try:
                dense = index.similarity_search(query=query, k=config.get('dense_top_k', 20), **filters)
            except Exception as exc:
                output['errors']['Dense'] = type(exc).__name__
            dense_seconds = time.perf_counter() - started
    output['raw_candidates']['Dense'] = clone(dense)
    if 'BM25' in strategies:
        output['strategies']['BM25'] = {'children': clone(bm25), 'latency': bm25_seconds}
    if 'Dense' in strategies and 'Dense' not in output['errors']:
        output['strategies']['Dense'] = {'children': clone(dense), 'latency': dense_seconds}
    if any(s in strategies for s in ['RRF', 'RRF+Reranker']) and 'Dense' not in output['errors']:
        started = time.perf_counter()
        fused = optimizer._rrf_fuse(clone(dense), clone(bm25))
        fused_seconds = time.perf_counter() - started + bm25_seconds + dense_seconds
        output['raw_candidates']['RRF'] = clone(fused)
        if 'RRF' in strategies:
            output['strategies']['RRF'] = {'children': clone(fused), 'latency': fused_seconds}
        if 'RRF+Reranker' in strategies:
            if reranker is None:
                output['errors']['RRF+Reranker'] = 'Cloud reranker unavailable'
            else:
                started = time.perf_counter()
                optimizer.reranker = reranker
                try:
                    reranked = optimizer._cross_encoder_rerank(query, clone(fused[:config.get('rerank_top_n', 30)]))
                    output['strategies']['RRF+Reranker'] = {'children': reranked,
                        'latency': fused_seconds + time.perf_counter() - started}
                except Exception as exc:
                    output['errors']['RRF+Reranker'] = type(exc).__name__
    for result in output['strategies'].values():
        result['parents'] = optimizer._promote_parent_chunks(clone(result['children']), config.get('max_rank', 20))
        result['children'] = result['children'][:config.get('max_rank', 20)]
    return output


class FilteredIndex:
    """Apply identical extended SQLite filters for BM25 and dense without a new algorithm."""
    def __init__(self, index):
        self.index = index

    def similarity_search(self, query, k, **filters):
        vector = self.index.embeddings.embed_query(query)
        return self.index.store.search(query_vector=vector, limit=k, **filters)
