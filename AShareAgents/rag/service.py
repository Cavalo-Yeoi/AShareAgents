"""组合 RAG 各模块，并提供进程内延迟初始化服务。"""

import logging
from threading import Lock
from typing import Optional

from AShareAgents.datasource.config import get_config

from .generation_integration import GenerationIntegrationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule

logger = logging.getLogger(__name__)


class RAGService:
    """提供面向 Agent 的公司资料与政策知识检索接口。"""

    def __init__(
        self, index: IndexConstructionModule, max_context_chars: int,
        rerank_enabled: bool = True,
    ):
        self.retrieval = RetrievalOptimizationModule(index, rerank_enabled=rerank_enabled)
        self.integration = GenerationIntegrationModule(max_context_chars)

    # @classmethod
    # def from_config(cls) -> "RAGService":
    #     config = get_config()
    #     index = IndexConstructionModule(
    #         uri=config["milvus_uri"],
    #         token=config.get("milvus_token") or None,
    #         database=config.get("milvus_database", "default"),
    #         collection_name=config["milvus_collection"],
    #         model_name=config["rag_embedding_model"],
    #         dimension=int(config["rag_embedding_dimension"]),
    #         device=config.get("rag_model_device", "cpu"),
    #     )
    #     service = cls(
    #         index, int(config.get("rag_max_context_chars", 12000)),
    #         rerank_enabled=config.get("rag_rerank_enabled", True),
    #     )
    #     service.retrieval = RetrievalOptimizationModule(
    #         index=index,
    #         reranker_model=config["rag_reranker_model"],
    #         device=config.get("rag_model_device", "cpu"),
    #         vector_top_k=int(config.get("rag_vector_top_k", 20)),
    #         bm25_top_k=int(config.get("rag_bm25_top_k", 20)),
    #         bm25_candidate_pool=int(config.get("rag_bm25_candidate_pool", 1000)),
    #         rerank_top_n=int(config.get("rag_rerank_top_n", 30)),
    #         rrf_k=int(config.get("rag_rrf_k", 60)),
    #         rerank_enabled=config.get("rag_rerank_enabled", True),
    #     )
    #     return service
    @classmethod
    def from_config(cls) -> "RAGService":
        config = get_config()

        # 1. 选择 Embedding 运行模式
        embedding_backend = config.get(
            "rag_embedding_backend", "local"
        ).lower()

        if embedding_backend == "cloud":
            from .cloud_adapters.bailian_embeddings import (
                BailianEmbeddings,
            )
            embeddings = BailianEmbeddings()
            embedding_model = config["rag_embedding_model"]

        elif embedding_backend == "local":
            embeddings = None
            embedding_model = config.get("rag_local_embedding_model") or config["rag_embedding_model"]

        else:
            raise ValueError(
                f"不支持的 Embedding 模式: {embedding_backend}"
            )

        # 2. 选择 Reranker 运行模式
        reranker_backend = config.get(
            "rag_reranker_backend", "local"
        ).lower()
        rerank_enabled = config.get("rag_rerank_enabled", True)

        if not rerank_enabled:
            reranker = None
            reranker_model = config.get("rag_local_reranker_model") or config.get(
                "rag_reranker_model", "BAAI/bge-reranker-base"
            )
        elif reranker_backend == "cloud":
            from .cloud_adapters.bailian_reranker import (
                BailianReranker,
            )
            reranker = BailianReranker()
            reranker_model = config["rag_reranker_model"]

        elif reranker_backend == "local":
            reranker = None
            reranker_model = config.get("rag_local_reranker_model") or config.get(
                "rag_reranker_model", "BAAI/bge-reranker-base"
            )

        else:
            raise ValueError(
                f"不支持的 Reranker 模式: {reranker_backend}"
            )

        # 3. 选择向量存储后端
        vector_backend = config.get(
            "rag_vector_backend", "milvus"
        ).lower()

        if vector_backend == "sqlite":
            from AShareAgents.storage.sqlite_knowledge_store import (
                SQLiteKnowledgeStore,
            )

            store = SQLiteKnowledgeStore(
                path=config["rag_sqlite_path"],
                dimension=int(config["rag_embedding_dimension"]),
            )

        elif vector_backend == "milvus":
            store = None

        else:
            raise ValueError(
                f"不支持的 RAG 向量存储后端: {vector_backend}"
            )
        # 3. 初始化索引模块
        # 当前仍使用原 Milvus 存储适配器。
        # SQLite 将在下一阶段接入。
        index = IndexConstructionModule(
            uri=config["milvus_uri"],
            token=config.get("milvus_token") or None,
            database=config.get("milvus_database", "default"),
            collection_name=config["milvus_collection"],
            model_name=embedding_model,
            dimension=int(config["rag_embedding_dimension"]),
            device=config.get("rag_model_device", "cpu"),
            embeddings=embeddings,
            store=store,
        )

        # 4. 初始化 RAG Service
        service = cls(
            index,
            int(config.get("rag_max_context_chars", 12000)),
            rerank_enabled=rerank_enabled,
        )

        # 5. 初始化混合检索模块
        service.retrieval = RetrievalOptimizationModule(
            index=index,
            reranker_model=reranker_model,
            device=config.get("rag_model_device", "cpu"),
            vector_top_k=int(config.get("rag_vector_top_k", 20)),
            bm25_top_k=int(config.get("rag_bm25_top_k", 20)),
            bm25_candidate_pool=int(
                config.get("rag_bm25_candidate_pool", 1000)
            ),
            rerank_top_n=int(config.get("rag_rerank_top_n", 30)),
            rrf_k=int(config.get("rag_rrf_k", 60)),
            reranker=reranker,
            rerank_enabled=rerank_enabled,
        )

        return service

    def company_context(
        self,
        query: str,
        trade_date: str,
        ticker: str = "",
        company_name: str = "",
        top_k: int = 5,
    ) -> str:
        documents = self.retrieval.search_company_documents(
            query=query,
            trade_date=trade_date,
            ticker=ticker,
            company_name=company_name,
            top_k=top_k,
        )
        return self.integration.build_context(documents)

    def policy_context(
        self,
        query: str,
        trade_date: str,
        industry: str = "",
        top_k: int = 5,
    ) -> str:
        documents = self.retrieval.search_policy_knowledge(
            query=query,
            trade_date=trade_date,
            industry=industry,
            top_k=top_k,
        )
        return self.integration.build_context(documents)


_service: Optional[RAGService] = None
_service_lock = Lock()


def get_rag_service() -> RAGService:
    """返回进程内共享的 RAG 服务实例。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RAGService.from_config()
    return _service


def reset_rag_service() -> None:
    """清除共享实例，供配置切换和测试使用。"""
    global _service
    with _service_lock:
        _service = None
