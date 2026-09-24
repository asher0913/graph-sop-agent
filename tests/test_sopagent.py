import pytest

from sopagent.agent import BM25, FilteredRAG, GraphOnly, HybridAgent, NaiveRAG, link, route
from sopagent.cli import main
from sopagent.corpus import SERVICES, build_kb, build_questions
from sopagent.evaluate import compare, evolution, ingest, propose_runbooks


@pytest.fixture(scope="module")
def kb_and_uncovered():
    return build_kb(0)


def test_dependency_graph_is_acyclic(kb_and_uncovered):
    kb, _ = kb_and_uncovered
    seen, stack = set(), set()

    def visit(s):
        assert s not in stack, "cycle"
        if s in seen:
            return
        stack.add(s)
        for d in kb.depends_on[s]:
            visit(d)
        stack.discard(s)
        seen.add(s)

    for s in SERVICES:
        visit(s)


def test_questions_are_well_formed(kb_and_uncovered):
    kb, uncovered = kb_and_uncovered
    questions = build_questions(kb, uncovered)
    assert {q.kind for q in questions} == {"first_step", "next_step", "dependency_owner", "escalation", "unanswerable"}
    for q in questions:
        if q.kind == "unanswerable":
            assert q.answer is None
            service = link(q.text)["service"]
            incident = link(q.text)["incident"]
            assert (service, incident) not in kb.runbook_for


@pytest.mark.parametrize(
    "text, intent",
    [
        ("What is the first step for disk full on ledger-db?", "first_step"),
        ("During error spike on cart-cache, what comes after: roll back the latest deploy?", "next_step"),
        ("Which team owns the service that checkout-api depends on?", "dependency_owner"),
        ("If the owner of image-cdn cannot resolve an incident, who do they escalate to?", "escalation"),
        ("Tell me about ledger-db", "unknown"),
    ],
)
def test_router(text, intent):
    assert route(text) == intent


def test_linker_prefers_the_longest_service_name():
    assert link("what about payments-db replication lag")["service"] == "payments-db"
    assert link("nothing here")["service"] is None


def test_bm25_ranks_exact_matches_first():
    index = BM25({"a": "disk full on ledger-db", "b": "high latency on checkout-api", "c": "disk usage report"})
    assert index.search("ledger-db disk full")[0][0] == "a"
    assert index.search("checkout-api", allowed={"a", "c"}) == []


def test_hybrid_uses_the_graph_for_two_hops(kb_and_uncovered):
    kb, _ = kb_and_uncovered
    service = next(s for s, deps in kb.depends_on.items() if len(deps) == 1)
    dep = kb.depends_on[service][0]
    answer = HybridAgent(kb).answer(f"Which team owns the service that {service} depends on?")
    assert answer.text == kb.owned_by[dep]
    assert f"{service} depends_on {dep}" in answer.citations
    naive = NaiveRAG(kb).answer(f"Which team owns the service that {service} depends on?")
    assert naive.citations and naive.citations[0].startswith("rb-")  # a runbook, not the graph


def test_hybrid_abstains_where_naive_rag_invents(kb_and_uncovered):
    kb, uncovered = kb_and_uncovered
    service, incident = next((s, i) for s, i in uncovered if any(k[0] == s for k in kb.runbook_for))
    question = f"What is the first step for {incident} on {service}?"
    assert HybridAgent(kb).answer(question).text is None
    assert NaiveRAG(kb).answer(question).text is not None
    assert GraphOnly(kb).answer(question).text is None
    assert FilteredRAG(kb).answer(question).text is not None  # filtered to the service, wrong incident


def test_headline_numbers():
    report = compare(0)
    dev = report["dev"]
    assert dev["hybrid (graph + runbooks)"]["accuracy"] == 1.0
    assert dev["BM25 RAG"]["answered_the_unanswerable"] == 1.0
    assert dev["graph only"]["first_step"] == 0.0
    # Documented weakness: "go after its owning team" is routed to next_step.
    assert report["paraphrase"]["hybrid (graph + runbooks)"]["escalation"] == 0.0


def test_ingestion_gates(kb_and_uncovered):
    kb, uncovered = build_kb(0)
    proposals = propose_runbooks(kb, uncovered)
    counts = ingest(kb, proposals)
    assert sum(counts.values()) == len(proposals)
    for p in proposals:
        accepted = (p["service"], p["incident"]) in kb.runbook_for
        assert accepted == (p["approved"] and p["stated_owner"] == kb.owned_by[p["service"]])


def test_evolution_never_answers_wrongly():
    report = evolution(0)
    before, after = report["questions_on_new_pairs"]
    assert before["answered_correctly"] == 0 and after["answered_correctly"] == report["review"]["accepted"]
    assert before["answered_wrongly"] == after["answered_wrongly"] == 0


def test_cli(tmp_path, capsys):
    assert main(["ask", "Which team owns the service that checkout-api depends on?"]) == 0
    assert main(["evaluate", "--out", str(tmp_path / "e.json")]) == 0
    assert main(["evolve"]) == 0
    assert "hybrid" in capsys.readouterr().out
