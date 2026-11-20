"""AShareAgents 的默认配置。"""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("ASHAREAGENTS_RESULTS_DIR", os.path.join(_CACHE_DIR, "results")),
    "data_cache_dir": os.getenv("ASHAREAGENTS_CACHE_DIR", os.path.join(_CACHE_DIR, "ohlcv")),
    "memory_log_path": os.getenv(
        "ASHAREAGENTS_MEMORY_LOG_PATH",
        os.path.join(_CACHE_DIR, "memory", "trading_memory.md"),
    ),
    "memory_db_path": os.getenv(
        "ASHAREAGENTS_MEMORY_DB_PATH",
        os.path.join(_CACHE_DIR, "memory", "trading_memory.sqlite"),
    ),
    "memory_vector_backend": os.getenv("ASHAREAGENTS_MEMORY_VECTOR_BACKEND", "milvus"),
    "memory_milvus_collection": os.getenv(
        "ASHAREAGENTS_MEMORY_MILVUS_COLLECTION",
        "ashareagents_trading_memory_bge",
    ),
    "memory_vector_path": os.getenv(
        "ASHAREAGENTS_MEMORY_VECTOR_PATH",
        os.path.join(_CACHE_DIR, "memory", "trading_memory_vectors.sqlite"),
    ),
    "memory_log_max_entries": None,
    "rag_enabled": _env_bool("RAG_ENABLED", True),
    "rag_vector_backend": os.getenv("RAG_VECTOR_BACKEND", "milvus").strip().lower(),
    "rag_sqlite_path": os.getenv( "RAG_SQLITE_PATH",os.path.join(_CACHE_DIR, "rag", "knowledge.db")),
    "rag_embedding_backend": os.getenv("RAG_EMBEDDING_BACKEND", "local").lower(),
    "rag_reranker_backend": os.getenv("RAG_RERANK_BACKEND", "local").lower(),
    "rag_local_embedding_model": os.getenv("RAG_LOCAL_EMBEDDING_MODEL","BAAI/bge-small-zh-v1.5",),
    "rag_embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
    "rag_embedding_dimension": int(os.getenv("RAG_EMBEDDING_DIMENSION", "512")),
    "rag_model_device": os.getenv("RAG_MODEL_DEVICE", "cpu"),
    "rag_reranker_model": os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
    "rag_rerank_enabled": _env_bool("RAG_RERANK_ENABLED", True),
    "rag_parent_chunk_size": int(os.getenv("RAG_PARENT_CHUNK_SIZE", "800")),
    "rag_parent_chunk_overlap": int(os.getenv("RAG_PARENT_CHUNK_OVERLAP", "100")),
    "rag_child_chunk_size": int(os.getenv("RAG_CHILD_CHUNK_SIZE", "200")),
    "rag_child_chunk_overlap": int(os.getenv("RAG_CHILD_CHUNK_OVERLAP", "40")),
    "rag_vector_top_k": int(os.getenv("RAG_VECTOR_TOP_K", "20")),
    "rag_bm25_top_k": int(os.getenv("RAG_BM25_TOP_K", "20")),
    "rag_bm25_candidate_pool": int(os.getenv("RAG_BM25_CANDIDATE_POOL", "1000")),
    "rag_rerank_top_n": int(os.getenv("RAG_RERANK_TOP_N", "30")),
    "rag_rrf_k": int(os.getenv("RAG_RRF_K", "60")),
    "rag_retrieval_top_k": int(os.getenv("RAG_RETRIEVAL_TOP_K", "5")),
    "rag_max_context_chars": int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000")),
    "milvus_uri": os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
    "milvus_token": os.getenv("MILVUS_TOKEN", ""),
    "milvus_database": os.getenv("MILVUS_DATABASE", "default"),
    "milvus_collection": os.getenv("MILVUS_COLLECTION", "ashareagents_knowledge_bge"),
    "llm_provider": os.getenv("ASHAREAGENTS_LLM_PROVIDER", "openai"),
    "deep_think_llm": os.getenv("ASHAREAGENTS_DEEP_THINK_LLM", "gpt-5.4"),
    "quick_think_llm": os.getenv("ASHAREAGENTS_QUICK_THINK_LLM", "gpt-5.4-mini"),
    "backend_url": os.getenv("BACKEND_URL") or None,
    # Server-owned limits; not exposed as per-request configuration fields.
    "llm_max_attempts": int(os.getenv("ASHAREAGENTS_LLM_MAX_ATTEMPTS", "64")),
    "llm_max_retries": int(os.getenv("ASHAREAGENTS_LLM_MAX_RETRIES", "1")),
    "llm_connect_timeout": float(os.getenv("ASHAREAGENTS_LLM_CONNECT_TIMEOUT", "5")),
    "llm_read_timeout": float(os.getenv("ASHAREAGENTS_LLM_READ_TIMEOUT", "60")),
    "llm_write_timeout": float(os.getenv("ASHAREAGENTS_LLM_WRITE_TIMEOUT", "30")),
    "llm_pool_timeout": float(os.getenv("ASHAREAGENTS_LLM_POOL_TIMEOUT", "5")),
    "llm_call_timeout": float(os.getenv("ASHAREAGENTS_LLM_CALL_TIMEOUT", "90")),
    "llm_task_timeout": float(os.getenv("ASHAREAGENTS_LLM_TASK_TIMEOUT", "600")),
    "llm_retry_backoff": float(os.getenv("ASHAREAGENTS_LLM_RETRY_BACKOFF", "0.5")),
    "llm_retry_backoff_max": float(os.getenv("ASHAREAGENTS_LLM_RETRY_BACKOFF_MAX", "4")),
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    "checkpoint_enabled": False,
    "output_language": "Chinese",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "data_vendors": {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    },
    "tool_vendors": {},
}
