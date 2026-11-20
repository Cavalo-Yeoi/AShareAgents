
import os
import math
import requests
from langchain_core.documents import Document


class BailianReranker:
    """百炼云端 Reranker 适配器。"""

    def __init__(self):
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.url = os.environ["BAILIAN_RERANK_URL"]
        self.model = os.getenv(
            "RAG_RERANKER_MODEL", "qwen3-rerank"
        )

    def score(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        """返回与原始文档顺序一致的相关性得分。"""

        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": texts,
                "top_n": len(texts),
                "return_documents": False,
            },
            timeout=(5, 30),
        )

        response.raise_for_status()

        results = response.json()["results"]

        if len(results) != len(texts):
            raise ValueError("Reranker 返回数量不匹配")

        # API 返回的是排名顺序，
        # 但原工作流需要的是输入文档顺序。
        scores = [None] * len(texts)

        for item in results:
            index = item["index"]
            score = float(item["relevance_score"])

            if (
                type(index) is not int
                or not 0 <= index < len(texts)
                or scores[index] is not None
            ):
                raise ValueError("Reranker 返回索引异常")

            if not math.isfinite(score):
                raise ValueError("Reranker 返回非法评分")

            scores[index] = score

        if any(s is None for s in scores):
            raise ValueError("Reranker 缺少文档评分")

        return scores
