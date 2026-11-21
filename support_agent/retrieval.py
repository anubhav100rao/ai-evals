"""Tiny RAG stack over the policy knowledge base, fully traced with OpenInference span kinds:

    kb_retrieval (RETRIEVER)
     ├── embed_query (EMBEDDING)      local hashing embedder, 256-d
     └── vector_search                cosine similarity over pre-embedded policy docs
    rerank (RERANKER)                 keyword/tag cross-scorer, keeps top 3

No embedding API is available on our endpoint, so the embedder is a deterministic
feature-hashing model — good enough to demo retrieval quality and its observability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re

from openinference.semconv.trace import (
    DocumentAttributes as D,
    EmbeddingAttributes as E,
    OpenInferenceSpanKindValues,
    RerankerAttributes as R,
    SpanAttributes as S,
)
from opentelemetry import trace

EMBED_MODEL = "shopnova-hash-embed-256"
RERANK_MODEL = "shopnova-keyword-reranker-v1"
DIM = 256
_STOP = set("a an the is are was were be to of and or for in on at my me i you your it this that do does can how what when with".split())
_tracer = trace.get_tracer("shopnova.retrieval")


def _tokens(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in _STOP]
    return [re.sub(r"(ing|ed|es|s)$", "", t) if len(t) > 4 else t for t in toks]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    toks = _tokens(text)
    for t in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class PolicyIndex:
    def __init__(self, policies: list[dict]):
        self.docs = policies
        self.vectors = [embed(f"{p['title']} {' '.join(p['tags'])} {p['body']}") for p in policies]

    async def search(self, query: str, top_k: int = 5, top_n: int = 3) -> list[dict]:
        candidates = await self._retrieve(query, top_k)
        return await self._rerank(query, candidates, top_n)

    async def _retrieve(self, query: str, top_k: int) -> list[dict]:
        with _tracer.start_as_current_span("kb_retrieval") as span:
            span.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
            span.set_attribute(S.INPUT_VALUE, query)
            with _tracer.start_as_current_span("embed_query") as es:
                es.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value)
                es.set_attribute(S.EMBEDDING_MODEL_NAME, EMBED_MODEL)
                await asyncio.sleep(0.03)
                qv = embed(query)
                es.set_attribute(f"{S.EMBEDDING_EMBEDDINGS}.0.{E.EMBEDDING_TEXT}", query)
                es.set_attribute(f"{S.EMBEDDING_EMBEDDINGS}.0.{E.EMBEDDING_VECTOR}", [round(v, 4) for v in qv])
            with _tracer.start_as_current_span("vector_search") as vs:
                vs.set_attribute("vector_store.name", "policy-kb")
                vs.set_attribute("vector_store.size", len(self.docs))
                vs.set_attribute("vector_store.top_k", top_k)
                await asyncio.sleep(0.02)
                scored = sorted(((_cos(qv, v), p) for v, p in zip(self.vectors, self.docs)), key=lambda x: -x[0])[:top_k]
            results = [{**p, "score": round(s, 4)} for s, p in scored]
            for i, d in enumerate(results):
                pre = f"{S.RETRIEVAL_DOCUMENTS}.{i}"
                span.set_attribute(f"{pre}.{D.DOCUMENT_ID}", d["id"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_CONTENT}", d["body"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_SCORE}", d["score"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_METADATA}", json.dumps({"title": d["title"]}))
            span.set_attribute(S.OUTPUT_VALUE, json.dumps([d["id"] for d in results]))
            return results

    async def _rerank(self, query: str, docs: list[dict], top_n: int) -> list[dict]:
        with _tracer.start_as_current_span("rerank") as span:
            span.set_attribute(S.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value)
            span.set_attribute(R.RERANKER_QUERY, query)
            span.set_attribute(R.RERANKER_MODEL_NAME, RERANK_MODEL)
            span.set_attribute(R.RERANKER_TOP_K, top_n)
            q = query.lower()
            qt = set(_tokens(query))
            rescored = []
            for i, d in enumerate(docs):
                pre = f"{R.RERANKER_INPUT_DOCUMENTS}.{i}"
                span.set_attribute(f"{pre}.{D.DOCUMENT_ID}", d["id"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_CONTENT}", d["body"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_SCORE}", d["score"])
                tag_hits = sum(1 for t in d["tags"] if t in q)
                overlap = len(qt & set(_tokens(d["title"] + " " + d["body"]))) / (len(qt) or 1)
                rescored.append({**d, "score": round(0.5 * d["score"] + 0.3 * min(tag_hits, 3) / 3 + 0.2 * overlap, 4)})
            await asyncio.sleep(0.04)
            rescored.sort(key=lambda d: -d["score"])
            out = rescored[:top_n]
            for i, d in enumerate(out):
                pre = f"{R.RERANKER_OUTPUT_DOCUMENTS}.{i}"
                span.set_attribute(f"{pre}.{D.DOCUMENT_ID}", d["id"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_CONTENT}", d["body"])
                span.set_attribute(f"{pre}.{D.DOCUMENT_SCORE}", d["score"])
            return out
