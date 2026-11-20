
"""AShareAgents 轻量级 SQLite 知识向量存储。"""

import heapq
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


class SQLiteKnowledgeStore:
    """使用 SQLite 保存知识块，使用 NumPy 检索向量。"""

    def __init__(self, path: str, dimension: int = 512):
        self.path = Path(path)
        self.dimension = int(dimension)

        if self.dimension <= 0:
            raise ValueError("向量维度必须为正整数")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")

            db.execute("""
                CREATE TABLE IF NOT EXISTS rag_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            row = db.execute(
                "SELECT value FROM rag_settings "
                "WHERE key = 'dimension'"
            ).fetchone()

            if row is not None and int(row[0]) != self.dimension:
                raise ValueError(
                    "SQLite 知识库向量维度与模型配置不匹配"
                )

            db.execute(
                "INSERT OR IGNORE INTO rag_settings "
                "VALUES ('dimension', ?)",
                (str(self.dimension),),
            )

            db.execute("""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    publish_ts INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)

            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_doc "
                "ON rag_chunks(document_id)"
            )

            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_filter "
                "ON rag_chunks(knowledge_type, ticker, publish_ts)"
            )

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(str(self.path), timeout=10)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _vector(self, value):
        vec = np.asarray(value, dtype=np.float32)

        if vec.ndim != 1 or len(vec) != self.dimension:
            raise ValueError(
                f"向量必须为 {self.dimension} 维"
            )

        if (
            not np.all(np.isfinite(vec))
            or np.linalg.norm(vec) == 0
        ):
            raise ValueError("向量包含非法数值或为零向量")

        return vec

    @staticmethod
    def _filters(
        knowledge_type="",
        ticker="",
        company_name="",
        industry="",
        publish_before_ts=None,
        document_type="",
        report_period="",
        publish_after_ts=None,
        effective_on=None,
        exclude_expired=False,
        date_verified_only=False,
    ):
        parts, params = [], []

        fields = (
            ("knowledge_type", knowledge_type),
            ("ticker", ticker),
            ("company_name", company_name if not ticker else ""),
            ("industry", industry),
        )

        for name, value in fields:
            if value:
                parts.append(f"{name} = ?")
                params.append(value)

        if publish_before_ts is not None:
            parts.append("publish_ts <= ?")
            params.append(int(publish_before_ts))

        if publish_after_ts is not None:
            parts.append("publish_ts >= ?")
            params.append(int(publish_after_ts))

        # A zero placeholder or explicitly unverified date is never eligible
        # for a point-in-time query. Legacy records with real dates remain valid.
        if publish_before_ts is not None or publish_after_ts is not None or date_verified_only:
            parts.append("publish_ts > 0")
            parts.append("COALESCE(json_extract(metadata, '$.date_status'), '') NOT IN (?, ?, ?)")
            params.extend(["unknown", "unverified", "date_unverified"])
        if date_verified_only:
            parts.append("json_extract(metadata, '$.date_status') IN (?, ?, ?)")
            params.extend(["verified", "official_verified", "verified_official"])

        for name, value in (("document_type", document_type), ("report_period", report_period)):
            if value:
                parts.append(f"json_extract(metadata, '$.{name}') = ?")
                params.append(value)

        if effective_on is not None:
            day = date.fromisoformat(str(effective_on)).isoformat()
            parts.append("COALESCE(json_extract(metadata, '$.effective_date'), '') != ''")
            parts.append("json_extract(metadata, '$.effective_date') <= ?")
            params.append(day)
        elif exclude_expired:
            # Use the requested publication cutoff when supplied, otherwise today.
            day = (datetime.fromtimestamp(int(publish_before_ts), tz=timezone.utc).date()
                   if publish_before_ts is not None else datetime.now(timezone.utc).date()).isoformat()
        if exclude_expired:
            parts.append("(COALESCE(json_extract(metadata, '$.expire_date'), '') = '' "
                         "OR json_extract(metadata, '$.expire_date') > ?)")
            params.append(day)

        where = (
            " WHERE " + " AND ".join(parts)
            if parts else ""
        )

        return where, params

    def upsert_documents(
        self,
        documents,
        vectors,
        replace_parents=True,
    ):
        """写入文档块和向量，支持按文档 ID 替换旧块。"""

        if len(documents) != len(vectors):
            raise ValueError("文档和向量数量不一致")

        if not documents:
            return 0

        rows = []

        for doc, value in zip(documents, vectors):
            meta = dict(doc.metadata)

            if not meta.get("chunk_id") or not meta.get("document_id"):
                raise ValueError("缺少 chunk_id 或 document_id")

            if not meta.get("knowledge_type") or "publish_ts" not in meta:
                raise ValueError("缺少知识类型或发布日期时间戳")

            vec = self._vector(value)

            rows.append((
                str(meta["chunk_id"]),
                str(meta["document_id"]),
                str(meta["knowledge_type"]),
                str(meta.get("ticker", "")),
                str(meta.get("company_name", "")),
                str(meta.get("industry", "")),
                int(meta["publish_ts"]),
                doc.page_content,
                vec.tobytes(),
                json.dumps(meta, ensure_ascii=False),
            ))

        with self._db() as db:
            if replace_parents:
                ids = sorted({row[1] for row in rows})

                db.executemany(
                    "DELETE FROM rag_chunks WHERE document_id = ?",
                    [(doc_id,) for doc_id in ids],
                )

            db.executemany("""
                INSERT OR REPLACE INTO rag_chunks (
                    chunk_id, document_id, knowledge_type,
                    ticker, company_name, industry,
                    publish_ts, content, vector, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

        return len(rows)

    def query_documents(
        self,
        limit=1000,
        knowledge_type="",
        ticker="",
        company_name="",
        industry="",
        publish_before_ts=None,
        document_type="",
        report_period="",
        publish_after_ts=None,
        effective_on=None,
        exclude_expired=False,
        date_verified_only=False,
    ):
        """按条件读取文档，供 BM25 使用。"""

        if limit is not None and limit <= 0:
            return []

        where, args = self._filters(
            knowledge_type, ticker, company_name,
            industry, publish_before_ts,
            document_type, report_period, publish_after_ts,
            effective_on, exclude_expired, date_verified_only,
        )

        with self._db() as db:
            records = db.execute(
                "SELECT chunk_id, content, metadata "
                "FROM rag_chunks" + where +
                " ORDER BY chunk_id" + (" LIMIT ?" if limit is not None else ""),
                [*args, int(limit)] if limit is not None else args,
            ).fetchall()

        return [
            Document(
                page_content=content,
                metadata={
                    **json.loads(meta),
                    "chunk_id": chunk_id,
                },
            )
            for chunk_id, content, meta in records
        ]

    def search(
        self,
        query_vector,
        limit=5,
        knowledge_type="",
        ticker="",
        company_name="",
        industry="",
        publish_before_ts=None,
        document_type="",
        report_period="",
        publish_after_ts=None,
        effective_on=None,
        exclude_expired=False,
        date_verified_only=False,
    ):
        """余弦相似度检索，返回分数最高的 Top-K 文档。"""

        if limit <= 0:
            return []

        query = self._vector(query_vector)
        query_norm = float(np.linalg.norm(query))

        where, args = self._filters(
            knowledge_type, ticker, company_name,
            industry, publish_before_ts,
            document_type, report_period, publish_after_ts,
            effective_on, exclude_expired, date_verified_only,
        )

        top = []

        with self._db() as db:
            cursor = db.execute(
                "SELECT chunk_id, content, metadata, vector "
                "FROM rag_chunks" + where,
                args,
            )

            for chunk_id, content, metadata, blob in cursor:
                vec = np.frombuffer(blob, dtype=np.float32)

                score = float(
                    np.dot(query, vec) /
                    (query_norm * np.linalg.norm(vec))
                )

                item = (score, chunk_id, content, metadata)

                if len(top) < limit:
                    heapq.heappush(top, item)
                elif item > top[0]:
                    heapq.heapreplace(top, item)

        return [
            Document(
                page_content=content,
                metadata={
                    **json.loads(metadata),
                    "chunk_id": chunk_id,
                    "score": score,
                },
            )
            for score, chunk_id, content, metadata
            in sorted(top, reverse=True)
        ]
