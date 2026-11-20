
import os
import requests


class BailianEmbeddings:
    """百炼云端 Embedding 适配器。"""

    def __init__(self):
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        base_url = os.environ["BAILIAN_EMBEDDING_BASE_URL"]

        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = os.getenv(
            "RAG_EMBEDDING_MODEL", "text-embedding-v4"
        )
        self.dimension = int(
            os.getenv("RAG_EMBEDDING_DIMENSION", "512")
        )
        self.batch_size = 10

    def _embed_batch(self, texts):
        """调用云端 API，返回与输入顺序一致的向量。"""

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimension,
                "encoding_format": "float",
            },
            timeout=(5, 30),
        )
        response.raise_for_status()

        items = response.json()["data"]

        if len(items) != len(texts):
            raise ValueError("Embedding 返回数量不匹配")

        vectors = [None] * len(texts)

        for item in items:
            index = item["index"]
            vector = item["embedding"]

            if (
                not isinstance(index, int)
                or not 0 <= index < len(texts)
                or vectors[index] is not None
            ):
                raise ValueError("Embedding 返回索引异常")

            if len(vector) != self.dimension:
                raise ValueError("Embedding 向量维度异常")

            vectors[index] = vector

        if any(v is None for v in vectors):
            raise ValueError("Embedding 缺少返回向量")

        return vectors

    def embed_documents(self, texts):
        """批量生成文档向量。"""
        if not texts:
            return []

        vectors = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return vectors

    def embed_query(self, text):
        """生成查询向量。"""
        return self._embed_batch([text])[0]
