# Graph SOP Agent

[![CI](https://github.com/asher0913/graph-sop-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/asher0913/graph-sop-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-none-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An on-call assistant that answers questions about incident runbooks by combining a **service
graph** (who owns what, what depends on what, where incidents escalate) with the **runbook
documents** themselves, and that refuses when the knowledge base has no answer. It is compared
with document-only retrieval on 600 labelled questions, and it ingests new runbooks only through
a review gate that checks them against the graph.

```text
$ sop-agent ask "What is the first step for disk full on payments-db?"
BM25 RAG                   answer='Check p99 latency for payments-db on the service dashboard'
                           cites=['rb-payments-db-high-latency']
hybrid (graph + runbooks)  answer=None      # there is no disk-full runbook for payments-db

$ sop-agent ask "Which team owns the service that checkout-api depends on?"
BM25 RAG                   answer='data'     # checkout-api's own owner, read off its runbook
hybrid (graph + runbooks)  answer='platform'
                           cites=['checkout-api depends_on analytics-etl', 'analytics-etl owned_by platform']
```

## Results

24 services, 8 teams, 76 runbooks covering 60% of (service, incident) pairs. 300 questions per
phrasing set: first steps, "what comes after …", the owner of a service's dependency (two hops),
the escalation target of a service's owner (two hops), and questions about runbooks that do not
exist (the right answer is to decline).

| Agent | Overall | First step | Next step | Dependency owner | Escalation | Declines when it should |
|---|---:|---:|---:|---:|---:|---:|
| BM25 RAG | 42.7% | 100% | 100% | 13.3% | 0% | **0%** |
| BM25 + entity filter | 50.3% | 100% | 100% | 13.3% | 0% | 38.3% |
| graph only | 60.0% | 0% | 0% | 100% | 100% | 100% |
| **hybrid (graph + runbooks)** | **100%** | 100% | 100% | 100% | 100% | 100% |

- **Retrieval is not the problem; knowing when there is no answer is.** With exact service and
  incident names in the question, BM25 finds the right runbook every time. But when the runbook
  does not exist it returns the nearest one, and because runbooks are near-duplicates across
  incidents, the answer reads perfectly plausibly: 100% of unanswerable questions get one.
  Filtering to the named service only helps when the service does not exist at all (the 38%);
  a known service always has some runbook to fall back on.
- **Two-hop questions need the graph.** A runbook states its own service's owner; the owner of a
  dependency or an escalation target is in no single document. Document-only agents answer with
  the wrong team (the 13% are cases where the two teams happen to coincide).
- **The hybrid routes by intent**: graph traversal for relational questions, and for procedures
  only the runbook the graph links to that exact (service, incident) pair. Every answer cites the
  runbook or the graph edges it used.

### Paraphrased questions

The same question types reworded ("How should I begin handling …", "Who is responsible for
whatever X relies on?", "Where does an unresolved X incident go after its owning team?"):

| Agent | Overall | Dependency owner | Escalation | Declines when it should |
|---|---:|---:|---:|---:|
| BM25 RAG | 44.0% | 23.3% | 0% | 0% |
| **hybrid** | **79.3%** | 100% | **0%** | 100% |

The drop is one routing failure, and it is instructive: "go **after** its owning team" contains
a next-step cue, so every paraphrased escalation question is routed to the procedure path and
fails. Keyword routing breaks on a single word; a learned intent classifier (or an LLM router
constrained to the four intents) is the obvious replacement, and this set is the regression test
for it. The cue lists and both phrasing sets were written by the same author, so this measures
robustness to rewording, not generalisation to real users.

### Growing the knowledge base safely

40 runbooks are drafted from postmortems for pairs the knowledge base does not cover. A draft is
accepted only if a reviewer approved it **and** its stated owner agrees with the graph; a
disagreement is quarantined, because one of the two sources is wrong and a human must decide which.

| | Answered correctly | Declined | Answered wrongly |
|---|---:|---:|---:|
| Before ingestion | 0 | 40 | 0 |
| After ingestion (23 accepted, 11 not approved, 6 quarantined) | **23** | 17 | **0** |

## Usage

```bash
pip install -e '.[dev]'

sop-agent ask "If the owner of image-cdn cannot resolve an incident, who do they escalate to?"
sop-agent evaluate --out results/evaluation.json
sop-agent evolve --out results/evolution.json
```

## Tests

`pytest -q` runs 15 tests: the dependency graph is acyclic, unanswerable questions really have no
runbook, routing for each intent, longest-name entity linking, BM25 ranking and filtering, two-hop
answers with graph citations, declining where document-only agents answer, the headline numbers
(including the documented paraphrase failure), the ingestion gates, and the CLI.

## Limitations

- The knowledge base and questions are synthetic, and entity linking is exact-name only; real
  on-call questions use nicknames, typos and partial service names.
- Routing is keyword based by design, to make its brittleness measurable.
- Runbook steps are read verbatim; there is no generation, so there is no hallucination inside an
  answer, only in the choice of source.

## License

MIT
