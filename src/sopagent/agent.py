"""Retrieval, entity linking, intent routing, and four ways to answer a runbook question."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .corpus import INCIDENTS, SERVICES, TEAMS, KnowledgeBase

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.ids = list(docs)
        self.tf = [Counter(tokens(docs[d])) for d in self.ids]
        self.length = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.length) / max(1, len(self.length))
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = tokens(query)
        scored = []
        for doc_id, tf, length in zip(self.ids, self.tf, self.length, strict=True):
            if allowed is not None and doc_id not in allowed:
                continue
            score = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    score += (
                        self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
                    )
            if score > 0:
                scored.append((doc_id, score))
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]


INTENT_CUES = (  # checked in order; written as general synonyms, not tuned on the evaluation phrasing
    ("next_step", ("after", "next", "then", "following")),
    ("dependency_owner", ("depends on", "depend on", "relies on", "rely on", "upstream")),
    ("escalation", ("escalat", "unresolved", "cannot resolve", "can't resolve", "paged")),
    ("first_step", ("first", "begin", "start", "initial")),
)


def route(question: str) -> str:
    lowered = question.lower()
    for intent, cues in INTENT_CUES:
        if any(cue in lowered for cue in cues):
            return intent
    return "unknown"


def link(question: str) -> dict[str, str | None]:
    """Exact-name entity linking for services, incidents and teams."""
    lowered = question.lower()
    service = next((s for s in sorted(SERVICES, key=len, reverse=True) if s in lowered), None)
    incident = next((i for i in INCIDENTS if i in lowered), None)
    team = next((t for t in TEAMS if f"the {t} team" in lowered), None)
    return {"service": service, "incident": incident, "team": team}


@dataclass
class Answer:
    text: str | None  # None means the agent abstained
    citations: list[str] = field(default_factory=list)
    route: str = ""


def _step_after(steps: tuple[str, ...], question: str) -> str | None:
    """The step following the one quoted in the question (best token overlap)."""
    q = set(tokens(question))
    overlaps = [len(q & set(tokens(step))) for step in steps[:-1]]
    if not overlaps or max(overlaps) == 0:
        return None
    return steps[overlaps.index(max(overlaps)) + 1]


class Agent:
    name = "base"

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.index = BM25({d: rb.text for d, rb in kb.runbooks.items()})

    def refresh(self) -> None:
        self.index = BM25({d: rb.text for d, rb in self.kb.runbooks.items()})

    def answer(self, question: str) -> Answer:
        raise NotImplementedError

    def _read_runbook(self, doc_id: str, intent: str, question: str) -> str | None:
        rb = self.kb.runbooks[doc_id]
        if intent == "next_step":
            return _step_after(rb.steps, question)
        if intent == "dependency_owner":
            return rb.owner  # the only owner a runbook states is its own service's
        if intent == "escalation":
            return rb.owner  # "page the <owner> on-call": the owner, not the escalation target
        return rb.steps[0]


class NaiveRAG(Agent):
    """Top BM25 document for the whole question; always answers."""

    name = "BM25 RAG"

    def answer(self, question: str) -> Answer:
        intent = route(question)
        hits = self.index.search(question, k=1)
        if not hits:
            return Answer(None, [], intent)
        doc_id = hits[0][0]
        return Answer(self._read_runbook(doc_id, intent, question), [doc_id], intent)


class FilteredRAG(Agent):
    """BM25 restricted to runbooks of the linked service (metadata filtering), still document-only."""

    name = "BM25 + entity filter"

    def answer(self, question: str) -> Answer:
        intent = route(question)
        entities = link(question)
        if entities["service"] is None:
            return Answer(None, [], intent)
        allowed = {d for d, rb in self.kb.runbooks.items() if rb.service == entities["service"]}
        hits = self.index.search(question, k=1, allowed=allowed)
        if not hits:
            return Answer(None, [], intent)
        doc_id = hits[0][0]
        return Answer(self._read_runbook(doc_id, intent, question), [doc_id], intent)


class GraphOnly(Agent):
    """Relations from the graph; no access to runbook content."""

    name = "graph only"

    def answer(self, question: str) -> Answer:
        intent = route(question)
        entities = link(question)
        service = entities["service"]
        if service is None or service not in self.kb.owned_by:
            return Answer(None, [], intent)
        return _graph_answer(self.kb, intent, service) or Answer(None, [], intent)


def _graph_answer(kb: KnowledgeBase, intent: str, service: str) -> Answer | None:
    if intent == "dependency_owner":
        deps = kb.depends_on.get(service, ())
        if len(deps) != 1:
            return Answer(None, [], intent)  # zero or several dependencies: the question is ambiguous
        owner = kb.owned_by[deps[0]]
        return Answer(owner, [f"{service} depends_on {deps[0]}", f"{deps[0]} owned_by {owner}"], intent)
    if intent == "escalation":
        team = kb.owned_by[service]
        target = kb.escalates_to[team]
        return Answer(target, [f"{service} owned_by {team}", f"{team} escalates_to {target}"], intent)
    return None


class HybridAgent(Agent):
    """Route by intent: relations from the graph, procedures from the runbook the graph points to.

    Abstains when the service is unknown or no runbook exists for the
    (service, incident) pair, instead of answering from a near-duplicate.
    """

    name = "hybrid (graph + runbooks)"

    def answer(self, question: str) -> Answer:
        intent = route(question)
        entities = link(question)
        service = entities["service"]
        if service is None or service not in self.kb.owned_by:
            return Answer(None, [], intent)
        graph = _graph_answer(self.kb, intent, service)
        if graph is not None:
            return graph
        incident = entities["incident"]
        doc_id = self.kb.runbook_for.get((service, incident)) if incident else None
        if doc_id is None:
            return Answer(None, [], intent)
        text = self._read_runbook(doc_id, "next_step" if intent == "next_step" else "first_step", question)
        return Answer(text, [doc_id, f"{service} runbook_for {incident}"], intent)


AGENTS = (NaiveRAG, FilteredRAG, GraphOnly, HybridAgent)
