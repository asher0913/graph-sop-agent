from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import re


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


class BM25Index:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.term_counts = [Counter(tokens(doc.text)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(term for counts in self.term_counts for term in counts)

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        scored = []
        for doc, counts, length in zip(self.documents, self.term_counts, self.lengths):
            score = 0.0
            for term in tokens(query):
                if not counts[term]:
                    continue
                idf = math.log(1 + (len(self.documents) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                tf = counts[term]
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / max(1, self.avg_length)))
            if score:
                scored.append((doc, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add(self, source: str, relation: str, target: str) -> None:
        self.edges[source.lower()].append((relation, target.lower()))

    def walk(self, source: str, relations: tuple[str, ...]) -> list[str]:
        frontier = [source.lower()]
        for relation in relations:
            frontier = [target for node in frontier for edge, target in self.edges.get(node, []) if edge == relation]
        return frontier


class SOPAgent:
    INTENTS = {
        "exception": {"delay", "damaged", "missing", "exception"},
        "handoff": {"handoff", "owner", "team", "escalate"},
        "procedure": {"procedure", "steps", "process", "how"},
    }

    def __init__(self, graph: KnowledgeGraph, documents: list[Document]) -> None:
        self.graph = graph
        self.documents = list(documents)
        self.index = BM25Index(self.documents)

    def intent(self, query: str) -> str:
        query_tokens = set(tokens(query))
        scores = {name: len(words & query_tokens) for name, words in self.INTENTS.items()}
        return max(scores, key=scores.get) if max(scores.values()) else "procedure"

    def graph_query(self, query: str, intent: str) -> list[str]:
        entities = [node for node in self.graph.edges if node in query.lower()]
        if not entities:
            return []
        relation = "escalates_to" if intent in {"exception", "handoff"} else "next_step"
        return self.graph.walk(entities[0], (relation,))

    def answer(self, query: str) -> dict[str, object]:
        intent = self.intent(query)
        with ThreadPoolExecutor(max_workers=2) as pool:
            graph_future = pool.submit(self.graph_query, query, intent)
            docs_future = pool.submit(self.index.search, query, 3)
            graph_hits = graph_future.result()
            doc_hits = docs_future.result()
        citations = [doc.doc_id for doc, _ in doc_hits]
        facts = [f"graph:{item}" for item in graph_hits] + [doc.text for doc, _ in doc_hits]
        return {
            "intent": intent,
            "plan": ["graph_query", "bm25_search", "merge_evidence"],
            "answer": " ".join(facts) if facts else "No grounded answer found.",
            "citations": citations,
            "graph_entities": graph_hits,
        }

    def learn(self, document: Document, accepted: bool) -> bool:
        if not accepted or not document.text.strip() or any(d.doc_id == document.doc_id for d in self.documents):
            return False
        self.documents.append(document)
        self.index = BM25Index(self.documents)
        return True


def fixture() -> SOPAgent:
    graph = KnowledgeGraph()
    graph.add("damaged parcel", "escalates_to", "claims team")
    graph.add("customs hold", "escalates_to", "trade compliance")
    graph.add("label creation", "next_step", "carrier pickup")
    documents = [
        Document("sop-1", "Damage", "For a damaged parcel, capture photos, quarantine the item, and open a claim."),
        Document("sop-2", "Customs", "For a customs delay, validate commodity codes and contact trade compliance."),
        Document("sop-3", "Pickup", "After label creation, schedule carrier pickup and record the tracking number."),
    ]
    return SOPAgent(graph, documents)


def demo() -> dict[str, object]:
    agent = fixture()
    queries = ["How do I handle a damaged parcel exception?", "Who owns a customs hold escalation?", "What steps follow label creation?"]
    answers = [agent.answer(query) for query in queries]
    return {
        "queries": len(queries),
        "citation_coverage": sum(bool(answer["citations"]) for answer in answers) / len(answers),
        "intents": [answer["intent"] for answer in answers],
        "graph_hits": sum(bool(answer["graph_entities"]) for answer in answers),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
