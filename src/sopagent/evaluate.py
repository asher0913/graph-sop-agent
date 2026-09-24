"""Answer accuracy, abstention and citation quality; knowledge ingestion with review gates."""

from __future__ import annotations

import random
from collections import defaultdict

from .agent import AGENTS, HybridAgent
from .corpus import STEPS, KnowledgeBase, Question, Runbook, build_kb, build_questions, render

KINDS = ("first_step", "next_step", "dependency_owner", "escalation", "unanswerable")


def _norm(text: str | None) -> str | None:
    return None if text is None else text.strip().rstrip(".").lower()


def evaluate_agent(agent, questions: list[Question]) -> dict:
    per_kind: dict[str, list[bool]] = defaultdict(list)
    answered = hallucinated = evidence_hits = 0
    for q in questions:
        a = agent.answer(q.text)
        if q.answer is None:
            correct = a.text is None
            hallucinated += a.text is not None
        else:
            correct = _norm(a.text) == _norm(q.answer)
        per_kind[q.kind].append(correct)
        if a.text is not None and q.answer is not None:
            answered += 1
            evidence_hits += set(q.evidence) <= set(a.citations)
    unanswerable = len(per_kind["unanswerable"])
    return {
        "accuracy": sum(sum(v) for v in per_kind.values()) / len(questions),
        **{kind: sum(per_kind[kind]) / len(per_kind[kind]) for kind in KINDS if per_kind[kind]},
        "answered_the_unanswerable": hallucinated / unanswerable if unanswerable else 0.0,
        "citations_cover_gold_evidence": evidence_hits / answered if answered else 0.0,
    }


def compare(seed: int = 0) -> dict:
    kb, uncovered = build_kb(seed)
    questions = build_questions(kb, uncovered)
    out = {
        "knowledge_base": {
            "services": len(kb.owned_by),
            "runbooks": len(kb.runbooks),
            "uncovered_pairs": len(uncovered),
            "questions": len(questions),
        }
    }
    for phrasing in ("dev", "paraphrase"):
        subset = [q for q in questions if q.phrasing == phrasing]
        out[phrasing] = {cls.name: evaluate_agent(cls(kb), subset) for cls in AGENTS}
    return out


def propose_runbooks(kb: KnowledgeBase, uncovered: list[tuple[str, str]], n: int = 40, seed: int = 3) -> list[dict]:
    """Runbooks drafted from postmortems. Some are not approved; some state the wrong owner."""
    rng = random.Random(seed)
    proposals = []
    for s, incident in rng.sample(uncovered, k=min(n, len(uncovered))):
        owner = kb.owned_by[s]
        roll = rng.random()
        stated_owner = rng.choice([t for t in kb.escalates_to if t != owner]) if roll < 0.15 else owner
        steps = tuple(step.format(s=s) for step in STEPS[incident])
        proposals.append(
            {
                "service": s,
                "incident": incident,
                "stated_owner": stated_owner,
                "approved": roll >= 0.15 and rng.random() < 0.8,
                "steps": steps,
                "text": render(s, incident, stated_owner, steps, kb.depends_on[s]),
            }
        )
    return proposals


def ingest(kb: KnowledgeBase, proposals: list[dict]) -> dict:
    """Accept a proposal only if a reviewer approved it and it agrees with the graph.

    A proposal whose stated owner contradicts the graph is quarantined even if
    approved: one of the two sources is wrong and a human has to decide which.
    """
    counts = {"accepted": 0, "not_approved": 0, "quarantined_conflict": 0}
    for p in proposals:
        key = (p["service"], p["incident"])
        if p["stated_owner"] != kb.owned_by[p["service"]]:
            status = "quarantined_conflict"
        elif not p["approved"]:
            status = "not_approved"
        else:
            status = "accepted"
            doc_id = f"rb-{p['service']}-{p['incident'].replace(' ', '-')}"
            kb.runbooks[doc_id] = Runbook(doc_id, p["service"], p["incident"], p["stated_owner"], p["steps"], p["text"])
            kb.runbook_for[key] = doc_id
        counts[status] += 1
        kb.history.append({"service": p["service"], "incident": p["incident"], "status": status})
    return counts


def evolution(seed: int = 0) -> dict:
    kb, uncovered = build_kb(seed)
    proposals = propose_runbooks(kb, uncovered)
    questions = [
        Question(
            f"new-{i}",
            "first_step",
            f"What is the first step for {p['incident']} on {p['service']}?",
            p["steps"][0],
            (),
        )
        for i, p in enumerate(proposals)
    ]
    agent = HybridAgent(kb)

    def outcome(label: str) -> dict:
        answers = [agent.answer(q.text) for q in questions]
        return {
            "stage": label,
            "answered_correctly": sum(
                _norm(a.text) == _norm(q.answer) for a, q in zip(answers, questions, strict=True)
            ),
            "abstained": sum(a.text is None for a in answers),
            "answered_wrongly": sum(
                a.text is not None and _norm(a.text) != _norm(q.answer) for a, q in zip(answers, questions, strict=True)
            ),
        }

    before = outcome("before ingestion")
    counts = ingest(kb, proposals)
    agent.refresh()
    after = outcome("after ingestion")
    return {"proposals": len(proposals), "review": counts, "questions_on_new_pairs": [before, after]}
