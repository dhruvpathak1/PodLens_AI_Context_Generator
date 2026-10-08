"""Ask the episode: grounded Q&A over the transcript with clickable, validated timestamp citations.

Context strategy (hybrid):
- Short transcripts (<= ASK_FULL_CONTEXT_CHARS): the whole timestamped transcript goes to the model.
  No retrieval step, so nothing relevant can be missed.
- Long transcripts: ~60 s overlapping chunks are embedded once (OpenAI embeddings via LangChain),
  cached in memory keyed by a transcript fingerprint, and the top-K chunks for the question are
  sent. Retrieval is isolated in `_EpisodeIndex` so a persistent vector store (sqlite-vec,
  pgvector) can replace the in-memory cache when episodes are persisted.

Answering: ChatPromptTemplate | ChatOpenAI.with_structured_output(AskAnswer) returns
{found, answer, citations[{start_sec, quote}]}. Citations are validated server-side: any
timestamp not present in the context that was sent is dropped; the rest snap to real segment
starts, so a citation can never point at an invented moment.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level copies of the settings so tests can tweak them without rebuilding Settings.
ASK_MODEL = settings.ask_model
EMBED_MODEL = settings.embed_model
ASK_FULL_CONTEXT_CHARS = settings.ask_full_context_chars
ASK_CHUNK_SEC = settings.ask_chunk_sec
ASK_CHUNK_OVERLAP_SEC = settings.ask_chunk_overlap_sec
ASK_TOP_K = settings.ask_top_k
MAX_QUESTION_CHARS = 500
MAX_HISTORY_TURNS = 6
MAX_ENTITIES_IN_PROMPT = 200
CITATION_TOLERANCE_SEC = 3.0
INDEX_CACHE_SIZE = 16


# ---------- Structured output ----------


class Citation(BaseModel):
    """A transcript line that supports part of an answer."""

    start_sec: float = Field(description="The t= value of the transcript line that supports the claim. Copy it exactly.")
    quote: str = Field(description="Short verbatim excerpt (at most 20 words) from that line.")


class AskAnswer(BaseModel):
    """Structured answer returned by the model."""

    found: bool = Field(description="True only if the transcript excerpts actually answer the question.")
    answer: str = Field(
        description="Direct answer in 1-5 sentences of plain text. If found is false, say it is not discussed "
        "in this episode and, if useful, mention the closest related thing that is."
    )
    citations: list[Citation] = Field(
        description="1-4 transcript lines that support the answer, in episode order. Empty if found is false."
    )


SYSTEM_PROMPT = """You answer questions about one podcast episode using ONLY the transcript excerpts provided.

Rules:
- Ground every claim in the excerpts. Do not use outside knowledge, even if you know the answer.
- Cite the [t=...] value of each line you rely on, copied exactly.
- If the excerpts do not answer the question, set found=false and say it is not discussed in this
  episode. Never guess.
- Be direct and concise. Plain text, no markdown headings.
- The transcript comes from speech recognition; read past obvious transcription errors and use the
  canonical entity spellings listed."""

HUMAN_PROMPT = """Episode: {source_label}

Entities detected in this episode:
{entities}

Earlier in this conversation:
{history}

Transcript excerpts (t = seconds from the start of the episode):
{context}

Question: {question}"""


class AskNotConfigured(RuntimeError):
    """Raised when `OPENAI_API_KEY` is missing (mapped to HTTP 503 by the API layer)."""


def ask_configured() -> bool:
    """True when the OpenAI key needed for Q&A is present."""
    return settings.openai_configured


# ---------- Context building ----------


def _line(seg: dict[str, Any]) -> str:
    """Render one segment as a `[t=<start>] text` prompt line."""
    return f"[t={float(seg['start']):.1f}] {str(seg['text']).strip()}"


def _clean_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop empty segments and sort the rest by start time."""
    out = [s for s in segments if str(s.get("text", "")).strip()]
    return sorted(out, key=lambda s: float(s.get("start", 0.0)))


@dataclass
class _Chunk:
    """A ~60 s retrieval window made of consecutive segments."""

    start: float
    end: float
    segments: list[dict[str, Any]]

    @property
    def text(self) -> str:
        """Prompt-ready text for this window (one `[t=..]` line per segment)."""
        return "\n".join(_line(s) for s in self.segments)


def build_chunks(
    segments: list[dict[str, Any]], chunk_sec: float = ASK_CHUNK_SEC, overlap_sec: float = ASK_CHUNK_OVERLAP_SEC
) -> list[_Chunk]:
    """Time windows of ~chunk_sec; each new window re-includes the last overlap_sec of the previous one."""
    segs = _clean_segments(segments)
    chunks: list[_Chunk] = []
    i = 0
    while i < len(segs):
        start = float(segs[i]["start"])
        j = i
        while j < len(segs) and float(segs[j]["start"]) < start + chunk_sec:
            j += 1
        j = max(j, i + 1)
        window = segs[i:j]
        chunks.append(_Chunk(start=start, end=float(window[-1].get("end", window[-1]["start"])), segments=window))
        if j >= len(segs):
            break
        # next window starts at the first segment inside the overlap tail
        tail_start = float(segs[j - 1]["start"]) - overlap_sec
        k = j - 1
        while k > i + 1 and float(segs[k - 1]["start"]) >= tail_start:
            k -= 1
        i = max(k, i + 1)
    return chunks


def _fingerprint(segments: list[dict[str, Any]]) -> str:
    """SHA-256 over segment starts and text; identifies a transcript for the index cache."""
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{float(s.get('start', 0.0)):.2f}|{s.get('text', '')}\n".encode("utf-8"))
    return h.hexdigest()


def _normalize(v: list[float]) -> list[float]:
    """Scale a vector to unit length so a dot product equals cosine similarity."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class _EpisodeIndex:
    """In-memory embedding index for one transcript. Swap for a persistent vector store later."""

    def __init__(self, chunks: list[_Chunk], vectors: list[list[float]]):
        """Store chunks with their (normalized) embedding vectors, index-aligned."""
        self.chunks = chunks
        self.vectors = [_normalize(v) for v in vectors]

    def top_k(self, query_vec: list[float], k: int) -> list[_Chunk]:
        """Return the `k` most similar chunks, re-sorted into episode order."""
        q = _normalize(query_vec)
        scored = sorted(
            ((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self.vectors)), reverse=True
        )
        picked = sorted(i for _, i in scored[:k])
        return [self.chunks[i] for i in picked]


_index_cache: "OrderedDict[str, _EpisodeIndex]" = OrderedDict()


async def _get_index(segments: list[dict[str, Any]], embedder: Any) -> _EpisodeIndex:
    """Return the cached index for this transcript, embedding its chunks on first use (LRU of 16)."""
    key = _fingerprint(segments)
    idx = _index_cache.get(key)
    if idx is not None:
        _index_cache.move_to_end(key)
        return idx
    chunks = build_chunks(segments)
    vectors = await embedder.aembed_documents([c.text for c in chunks])
    idx = _EpisodeIndex(chunks, vectors)
    _index_cache[key] = idx
    while len(_index_cache) > INDEX_CACHE_SIZE:
        _index_cache.popitem(last=False)
    return idx


def _merge_chunks(chunks: list[_Chunk]) -> list[dict[str, Any]]:
    """Union of segments from the selected chunks, in time order, without duplicates."""
    seen: set[tuple[float, str]] = set()
    out: list[dict[str, Any]] = []
    for c in chunks:
        for s in c.segments:
            key = (float(s["start"]), str(s["text"]))
            if key not in seen:
                seen.add(key)
                out.append(s)
    return sorted(out, key=lambda s: float(s["start"]))


def _render_context(segs: list[dict[str, Any]]) -> str:
    """Join segments into prompt lines, inserting `…` where there is a gap of more than 5 s."""
    lines: list[str] = []
    prev_end: float | None = None
    for s in segs:
        if prev_end is not None and float(s["start"]) - prev_end > 5:
            lines.append("…")
        lines.append(_line(s))
        prev_end = float(s.get("end", s["start"]))
    return "\n".join(lines)


def _render_entities(entities: list[dict[str, Any]]) -> str:
    """Unique `TYPE: text` list (capped) so the model uses canonical spellings."""
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for e in entities:
        text = str(e.get("text", "")).strip()
        etype = str(e.get("type", "MISC")).strip().upper()
        key = (etype, text.lower())
        if text and key not in seen:
            seen.add(key)
            out.append(f"{etype}: {text}")
        if len(out) >= MAX_ENTITIES_IN_PROMPT:
            break
    return "; ".join(out) if out else "(none)"


def _render_history(history: list[dict[str, Any]]) -> str:
    """Last `MAX_HISTORY_TURNS` chat turns as `User:` / `Assistant:` lines for follow-ups."""
    turns = history[-MAX_HISTORY_TURNS:]
    if not turns:
        return "(no earlier questions)"
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {str(t.get('content', '')).strip()[:1200]}")
    return "\n".join(lines)


def validate_citations(raw: list[Citation], context_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only citations that point at a line actually sent to the model; snap to its start."""
    starts = [float(s["start"]) for s in context_segments]
    texts = {float(s["start"]): str(s["text"]).strip() for s in context_segments}
    out: list[dict[str, Any]] = []
    used: set[float] = set()
    for c in raw:
        if not starts:
            break
        nearest = min(starts, key=lambda s: abs(s - float(c.start_sec)))
        if abs(nearest - float(c.start_sec)) > CITATION_TOLERANCE_SEC or nearest in used:
            continue
        used.add(nearest)
        quote = c.quote.strip() or texts[nearest]
        out.append({"start_sec": nearest, "quote": quote[:220]})
    return sorted(out, key=lambda x: x["start_sec"])


# ---------- Chain ----------


def _build_chain():
    """Create `prompt | ChatOpenAI.with_structured_output(AskAnswer)` (strict JSON schema)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=ASK_MODEL, max_retries=2, timeout=90)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm.with_structured_output(AskAnswer, method="json_schema", strict=True)


def _build_embedder():
    """Create the OpenAI embeddings client used for long-episode retrieval."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL, max_retries=2)


async def answer_question(
    question: str,
    segments: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    source_label: str | None = None,
    chain: Any = None,
    embedder: Any = None,
) -> dict[str, Any]:
    """Answer a question about the episode using only its transcript.

    Args:
        question: User question (1-500 characters).
        segments: Whisper segments `{id, start, end, text}`.
        entities: Extracted entities, passed to the model for canonical spellings.
        history: Earlier `{role, content}` turns, used for follow-up questions.
        source_label: Episode name shown to the model.
        chain / embedder: Injected LangChain runnables (tests pass stubs).

    Returns:
        `{answer, found, citations, mode, model, unverified}`; `mode` is "full" or "retrieval".

    Raises:
        ValueError: empty/too-long question or empty transcript.
        AskNotConfigured: no OpenAI key.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
    segs = _clean_segments(segments)
    if not segs:
        raise ValueError("Transcript is empty.")

    if chain is None or embedder is None:
        if not ask_configured():
            raise AskNotConfigured("OPENAI_API_KEY is not set on the API server.")
    history = history or []

    full_chars = sum(len(_line(s)) + 1 for s in segs)
    if full_chars <= ASK_FULL_CONTEXT_CHARS:
        mode = "full"
        context_segments = segs
    else:
        mode = "retrieval"
        embedder = embedder or _build_embedder()
        index = await _get_index(segs, embedder)
        # Fold the previous user turn into the query so follow-ups ("what happened next?") retrieve well.
        prev_user = next((t.get("content", "") for t in reversed(history) if t.get("role") == "user"), "")
        query = f"{prev_user}\n{question}".strip()
        qvec = await embedder.aembed_query(query)
        context_segments = _merge_chunks(index.top_k(qvec, ASK_TOP_K))

    chain = chain or _build_chain()
    result: AskAnswer = await chain.ainvoke(
        {
            "source_label": (source_label or "podcast episode").strip()[:200],
            "entities": _render_entities(entities or []),
            "history": _render_history(history),
            "context": _render_context(context_segments),
            "question": question,
        }
    )

    citations = validate_citations(result.citations, context_segments) if result.found else []
    return {
        "answer": result.answer.strip(),
        "found": bool(result.found),
        "citations": citations,
        "mode": mode,
        "model": ASK_MODEL,
        "unverified": bool(result.found and not citations),
    }
