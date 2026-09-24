"""A seeded operations knowledge base: a service graph, runbooks, and labelled questions.

The runbooks are deliberately repetitive: every "high latency" runbook reads
almost the same whatever the service, so for a (service, incident) pair with
no runbook there is always a near-duplicate that *looks* like an answer.
Ownership, dependencies and escalation live in the graph; a runbook only
mentions its own service's owner, so questions that need two hops cannot be
answered from any single document.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SERVICES = (
    "checkout-api", "payments-api", "payments-db", "search-index", "orders-queue", "inventory-svc",
    "auth-gateway", "notifications", "billing-worker", "cart-cache", "recommendations", "shipping-api",
    "catalog-db", "pricing-engine", "fraud-scorer", "ledger-db", "email-relay", "image-cdn",
    "session-store", "analytics-etl", "returns-api", "loyalty-svc", "tax-calculator", "warehouse-sync",
)  # fmt: skip
TEAMS = ("checkout", "payments", "search", "fulfilment", "identity", "platform", "data", "growth")
INCIDENTS = ("high latency", "error spike", "replication lag", "disk full", "certificate expiry", "queue backlog")

STEPS = {
    "high latency": (
        "Check p99 latency for {s} on the service dashboard",
        "Compare current traffic for {s} with the weekly baseline",
        "Inspect slow traces for {s} in the tracing tool",
        "Scale out {s} by two replicas",
        "Enable request shedding on {s}",
        "Confirm p99 latency for {s} is back under the SLO",
    ),
    "error spike": (
        "Open the error-rate panel for {s}",
        "Identify the most frequent error code returned by {s}",
        "Check whether a deploy of {s} happened in the last hour",
        "Roll back the latest deploy of {s}",
        "Purge poisoned cache entries used by {s}",
        "Verify the error rate for {s} is below one percent",
    ),
    "replication lag": (
        "Read the replica lag metric for {s}",
        "Check write volume on the primary for {s}",
        "Pause batch jobs that write to {s}",
        "Restart the replication worker for {s}",
        "Rebuild the lagging replica of {s} from a snapshot",
        "Confirm replica lag for {s} is under ten seconds",
    ),
    "disk full": (
        "Check disk usage on every node of {s}",
        "Find the largest directories on the full node of {s}",
        "Rotate and compress logs on {s}",
        "Expand the volume attached to {s}",
        "Move cold data of {s} to object storage",
        "Confirm free disk on {s} is above twenty percent",
    ),
    "certificate expiry": (
        "Check the certificate expiry date for {s}",
        "Confirm the renewal job for {s} ran last night",
        "Request a new certificate for {s} from the internal CA",
        "Deploy the renewed certificate to {s}",
        "Restart the TLS terminators in front of {s}",
        "Verify clients can complete a handshake with {s}",
    ),
    "queue backlog": (
        "Check queue depth and consumer lag for {s}",
        "Check consumer error logs for {s}",
        "Scale up consumers of {s}",
        "Move poison messages from {s} to the dead-letter queue",
        "Replay the dead-letter queue of {s} after the fix",
        "Confirm consumer lag for {s} is back to normal",
    ),
}


@dataclass(frozen=True)
class Runbook:
    doc_id: str
    service: str
    incident: str
    owner: str
    steps: tuple[str, ...]
    text: str
    version: int = 1


@dataclass
class KnowledgeBase:
    owned_by: dict[str, str]
    depends_on: dict[str, tuple[str, ...]]
    escalates_to: dict[str, str]
    runbooks: dict[str, Runbook]  # doc id -> runbook
    runbook_for: dict[tuple[str, str], str]  # (service, incident) -> doc id
    history: list[dict] = field(default_factory=list)  # accepted and rejected changes


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str  # first_step | next_step | dependency_owner | escalation | unanswerable
    text: str
    answer: str | None  # None: the right response is to abstain
    evidence: tuple[str, ...]  # doc ids or graph edges that justify the answer
    phrasing: str = "dev"


def render(service: str, incident: str, owner: str, steps: tuple[str, ...], deps: tuple[str, ...]) -> str:
    lines = [
        f"Runbook: {incident} on {service}",
        f"Owner: {owner} team. Depends on: {', '.join(deps) if deps else 'nothing'}.",
        "Steps:",
        *[f"{i}. {step}." for i, step in enumerate(steps, 1)],
        f"Escalation: if unresolved after 30 minutes, page the {owner} on-call.",
    ]
    return "\n".join(lines)


def build_kb(seed: int = 0, coverage: float = 0.6) -> tuple[KnowledgeBase, list[tuple[str, str]]]:
    """Return the knowledge base and the (service, incident) pairs left without a runbook."""
    rng = random.Random(seed)
    owned_by = {s: rng.choice(TEAMS) for s in SERVICES}
    order = list(SERVICES)
    rng.shuffle(order)
    depends_on = {}
    for i, s in enumerate(order):  # a DAG: only depend on services later in the order
        later = order[i + 1 :]
        depends_on[s] = tuple(sorted(rng.sample(later, k=min(len(later), rng.choice([0, 1, 1, 2]))))) if later else ()
    escalates_to = {}
    for team in TEAMS:
        escalates_to[team] = "incident-command" if team == "platform" else rng.choice(["platform", "incident-command"])
    runbooks, runbook_for, uncovered = {}, {}, []
    for s in SERVICES:
        for incident in INCIDENTS:
            if rng.random() > coverage:
                uncovered.append((s, incident))
                continue
            steps = tuple(step.format(s=s) for step in STEPS[incident])
            doc_id = f"rb-{s}-{incident.replace(' ', '-')}"
            runbooks[doc_id] = Runbook(
                doc_id, s, incident, owned_by[s], steps, render(s, incident, owned_by[s], steps, depends_on[s])
            )
            runbook_for[(s, incident)] = doc_id
    return KnowledgeBase(owned_by, depends_on, escalates_to, runbooks, runbook_for), uncovered


TEMPLATES = {
    "dev": {
        "first_step": "What is the first step for {i} on {s}?",
        "next_step": "During {i} on {s}, what comes after: {step}?",
        "dependency_owner": "Which team owns the service that {s} depends on?",
        "escalation": "If the owner of {s} cannot resolve an incident, who do they escalate to?",
    },
    "paraphrase": {
        "first_step": "How should I begin handling {i} for {s}?",
        "next_step": "I've finished '{step}' for the {s} {i} — what's next?",
        "dependency_owner": "Who is responsible for whatever {s} relies on?",
        "escalation": "Where does an unresolved {s} incident go after its owning team?",
    },
}


def build_questions(kb: KnowledgeBase, uncovered: list[tuple[str, str]], n_per_kind: int = 60, seed: int = 1):
    rng = random.Random(seed)
    questions: list[Question] = []
    pairs = sorted(kb.runbook_for)
    single_dep = sorted(s for s, deps in kb.depends_on.items() if len(deps) == 1)
    for phrasing, templates in TEMPLATES.items():
        for i in range(n_per_kind):
            s, incident = rng.choice(pairs)
            rb = kb.runbooks[kb.runbook_for[(s, incident)]]
            text = templates["first_step"].format(i=incident, s=s)
            questions.append(Question(f"{phrasing}-first-{i}", "first_step", text, rb.steps[0], (rb.doc_id,), phrasing))

            k = rng.randrange(len(rb.steps) - 1)
            text = templates["next_step"].format(i=incident, s=s, step=rb.steps[k].lower())
            questions.append(
                Question(f"{phrasing}-next-{i}", "next_step", text, rb.steps[k + 1], (rb.doc_id,), phrasing)
            )

            s = rng.choice(single_dep)
            dep = kb.depends_on[s][0]
            owner = kb.owned_by[dep]
            evidence = (f"{s} depends_on {dep}", f"{dep} owned_by {owner}")
            text = templates["dependency_owner"].format(s=s)
            questions.append(Question(f"{phrasing}-depowner-{i}", "dependency_owner", text, owner, evidence, phrasing))

            s = rng.choice(SERVICES)
            team = kb.owned_by[s]
            target = kb.escalates_to[team]
            evidence = (f"{s} owned_by {team}", f"{team} escalates_to {target}")
            text = templates["escalation"].format(s=s)
            questions.append(Question(f"{phrasing}-esc-{i}", "escalation", text, target, evidence, phrasing))

            s, incident = rng.choice(uncovered)
            if rng.random() < 0.3:
                s = rng.choice(["ledger-archiver", "mobile-bff", "geo-router"])  # not in the graph at all
            text = templates["first_step"].format(i=incident, s=s)
            questions.append(Question(f"{phrasing}-none-{i}", "unanswerable", text, None, (), phrasing))
    return questions
