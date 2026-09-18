from src.rag_pipeline import SAFE_FALLBACK, answer_query
from src.retriever import Evidence, RetrievalResult


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, query, top_k=None):
        return self.result


class FailingLLM:
    def __init__(self):
        self.called = False

    def generate(self, messages, temperature=0):
        self.called = True
        raise AssertionError("LLM must not be called without sufficient evidence")


class FakeLLM:
    def generate(self, messages, temperature=0):
        return "Grounded advisory"


def test_pipeline_returns_safe_fallback_without_calling_llm():
    llm = FailingLLM()
    result = answer_query(
        "unknown pest",
        retriever=FakeRetriever(RetrievalResult(evidence=[])),
        llm=llm,
        threshold=0.35,
    )

    assert result.answer == SAFE_FALLBACK
    assert result.evidence_status == "insufficient"
    assert not llm.called


def test_pipeline_generates_with_retrieved_context_and_sources():
    evidence = Evidence(
        text="Use documented biological management.",
        metadata={"title": "Official Guide", "page": 3},
        distance=0.2,
    )
    result = answer_query(
        "manage pest sustainably",
        retriever=FakeRetriever(RetrievalResult(evidence=[evidence])),
        llm=FakeLLM(),
        threshold=0.35,
    )

    assert result.answer == "Grounded advisory"
    assert result.evidence_status == "sufficient"
    assert result.sources == [{"title": "Official Guide", "page": 3}]
