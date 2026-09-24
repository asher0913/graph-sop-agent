"""``sop-agent ask | evaluate | evolve``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import HybridAgent, NaiveRAG
from .corpus import build_kb
from .evaluate import compare, evolution


def _ask(args) -> int:
    kb, _ = build_kb(args.seed)
    for agent in (NaiveRAG(kb), HybridAgent(kb)):
        a = agent.answer(args.question)
        print(f"{agent.name:<28} route={a.route:<17} answer={a.text!r}\n{'':<28} cites={a.citations}")
    return 0


def _evaluate(args) -> int:
    report = compare(args.seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["knowledge_base"]))
    for phrasing in ("dev", "paraphrase"):
        print(f"\n{phrasing} phrasing")
        print("| Agent | Overall | First step | Next step | Dependency owner | Escalation | Abstains when it should |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, m in report[phrasing].items():
            keys = ("accuracy", "first_step", "next_step", "dependency_owner", "escalation", "unanswerable")
            print(f"| {name} | " + " | ".join(f"{100 * m[k]:.1f}%" for k in keys) + " |")
    return 0


def _evolve(args) -> int:
    report = evolution(args.seed)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sop-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question with the naive and the hybrid agent")
    ask.add_argument("question")
    ask.add_argument("--seed", type=int, default=0)
    ask.set_defaults(func=_ask)
    ev = sub.add_parser("evaluate", help="all agents on development and paraphrased questions")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out")
    ev.set_defaults(func=_evaluate)
    evo = sub.add_parser("evolve", help="review-gated ingestion of new runbooks")
    evo.add_argument("--seed", type=int, default=0)
    evo.add_argument("--out")
    evo.set_defaults(func=_evolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
