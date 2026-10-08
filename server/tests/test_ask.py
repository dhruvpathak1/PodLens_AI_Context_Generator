"""Ask service: context modes, citation validation, history and validation (LLM stubbed)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.services import ask

SEGS = [{"id": i, "start": i * 5.0, "end": i * 5.0 + 5, "text": f"Line {i}."} for i in range(40)]
SEGS[10]["text"] = "George Schaller published The Serengeti Lion in 1972."


def run(question, segments=SEGS, chain=None, embedder=object(), **kw):
    """Call `ask.answer_question` synchronously with test defaults."""
    return asyncio.run(ask.answer_question(question, segments, chain=chain, embedder=embedder, **kw))


def test_full_mode_validates_citations() -> None:
    """Short transcripts use full context and keep only valid citations."""
    seen = {}

    def fake(inp):
        """Stand-in for the LLM call used by the test above."""
        prompt = ChatPromptTemplate.from_messages([("system", ask.SYSTEM_PROMPT), ("human", ask.HUMAN_PROMPT)])
        seen["prompt"] = prompt.invoke(inp).to_messages()[1].content
        return ask.AskAnswer(found=True, answer="In 1972.", citations=[
            ask.Citation(start_sec=50.0, quote="published The Serengeti Lion in 1972"),
            ask.Citation(start_sec=50.4, quote="duplicate line"),
            ask.Citation(start_sec=9999, quote="invented"),
        ])

    res = run("When did Schaller publish?", chain=RunnableLambda(fake),
              entities=[{"type": "PERSON", "text": "George Schaller"}],
              history=[{"role": "user", "content": "Who is Schaller?"}, {"role": "assistant", "content": "A biologist."}])
    assert res["mode"] == "full" and not res["unverified"]
    assert [c["start_sec"] for c in res["citations"]] == [50.0]
    assert "User: Who is Schaller?" in seen["prompt"] and "PERSON: George Schaller" in seen["prompt"]


def test_not_found_drops_citations() -> None:
    """An answer marked not found carries no citations."""
    chain = RunnableLambda(lambda _: ask.AskAnswer(found=False, answer="Not discussed in this episode.",
                                                   citations=[ask.Citation(start_sec=0, quote="x")]))
    res = run("Price of tea?", chain=chain)
    assert res["found"] is False and res["citations"] == []


def test_retrieval_mode_caches_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long transcripts use retrieval and reuse the cached embedding index."""
    monkeypatch.setattr(ask, "ASK_FULL_CONTEXT_CHARS", 300)
    long = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 10, "text": ("lions " if i == 37 else "filler ") * 8} for i in range(60)]

    class FakeEmbedder:
        """Deterministic embeddings that count calls (no network)."""
        doc_calls = 0

        async def aembed_documents(self, texts):
            """Embed documents as fixed vectors."""
            FakeEmbedder.doc_calls += 1
            return [[t.count("lions"), 1.0] for t in texts]

        async def aembed_query(self, _):
            """Embed a query as a fixed vector."""
            return [1.0, 0.0]

    captured = {}

    def fake(inp):
        """Stand-in for the LLM call used by the test above."""
        captured["ctx"] = inp["context"]
        return ask.AskAnswer(found=True, answer="At 370 s.", citations=[ask.Citation(start_sec=370.2, quote="lions")])

    emb = FakeEmbedder()
    first = run("lions?", long, chain=RunnableLambda(fake), embedder=emb)
    run("lions again?", long, chain=RunnableLambda(fake), embedder=emb)
    assert first["mode"] == "retrieval" and first["citations"] == [{"start_sec": 370.0, "quote": "lions"}]
    assert "[t=370.0]" in captured["ctx"] and FakeEmbedder.doc_calls == 1


@pytest.mark.parametrize("question", ["", "x" * 501])
def test_rejects_bad_questions(question: str) -> None:
    """Empty or oversized questions are rejected."""
    with pytest.raises(ValueError):
        run(question, chain=RunnableLambda(lambda _: None))


def test_chunks_overlap() -> None:
    """Retrieval chunks overlap so answers at chunk edges are not lost."""
    long = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 10, "text": "x"} for i in range(60)]
    chunks = ask.build_chunks(long)
    assert [(c.start, c.end) for c in chunks[:2]] == [(0.0, 60.0), (40.0, 100.0)]
