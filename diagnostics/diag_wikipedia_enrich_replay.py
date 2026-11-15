"""
Diagnostic: diag_wikipedia_enrich_replay.py
============================================
Phase 4 of the Wikipedia integration (§14.21): before LOCALIST_WIKIPEDIA_ENRICH
gets a default, replay real stored requests through the exact rule the live
code uses (localist.wikipedia_match.decide) against live Wikipedia, and
measure how often a background summary would attach — and, with labels, how
often the attached article is the wrong one.

Two sets, from the chat_turns table:
  A  turns where web_search or research actually fired (the case the flag
     affects). The query is the one recorded in workflow_steps when present,
     else derived from the user's message the way _run_web_search derives it.
  B  every user message, derived the same way, as if it had been a web
     search — a stress test for wrong attachments on everyday phrasing.
  C  a fixed synthetic battery (SYNTHETIC_* below, no personal data):
     encyclopedic searches with the article each should attach, and
     searches where any summary would be wrong or useless. Labelled by
     construction, so its numbers can go in the committed report. Written
     before the first replay run and not tuned against afterwards.

Labelling: run once with --details, read the attached rows, write a labels
JSON {"<query>": "right" | "wrong" | "unhelpful"}, rerun with --labels.
"right" = the article is what the query is about and its summary helps;
"unhelpful" = right article but adds nothing (counted separately);
"wrong" = a different thing than asked.

Personal data: the details and labels files hold the user's own requests —
keep them OUTSIDE the repo (the script refuses paths inside it). Only
aggregates print, and only aggregates belong in diagnostics/reports/.

READ-ONLY on the database (sqlite mode=ro). One Wikipedia API request per
unique query, sequential, 0.25 s apart, compliant User-Agent. Not part of
the pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_wikipedia_enrich_replay.py \
        --details /path/outside/repo/wiki_replay.jsonl \
        [--labels /path/outside/repo/wiki_labels.json] [--db PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND / "src"))

from dotenv import dotenv_values  # noqa: E402

from localist import wikipedia_match  # noqa: E402
from localist.mcp_server import wikipedia  # noqa: E402
from localist.mcp_tool_dispatcher import _WEB_SEARCH_FILLER_PREFIXES  # noqa: E402

# Set C — (query, expected title substring, case-insensitive).
SYNTHETIC_POSITIVES: list[tuple[str, str]] = [
    ("who was Ada Lovelace", "Ada Lovelace"),
    ("history of the transistor", "Transistor"),
    ("what is CRISPR", "CRISPR"),
    ("Mercury planet facts", "Mercury (planet)"),
    ("how does photosynthesis work", "Photosynthesis"),
    ("Treaty of Versailles", "Treaty of Versailles"),
    ("Rust programming language", "Rust (programming language)"),
    ("Battle of Hastings", "Battle of Hastings"),
    ("Marie Curie", "Marie Curie"),
    ("what is a black hole", "Black hole"),
    ("who invented the telephone", "telephone"),
    ("Great Barrier Reef", "Great Barrier Reef"),
    ("what is quantum entanglement", "Quantum entanglement"),
    ("Apollo 11 mission", "Apollo 11"),
    ("the Byzantine Empire", "Byzantine Empire"),
    ("what is the Turing test", "Turing test"),
    ("Mount Kilimanjaro", "Kilimanjaro"),
    ("explain the French Revolution", "French Revolution"),
    ("who is Ursula K. Le Guin", "Ursula K. Le Guin"),
    ("what is a transformer neural network", "Transformer"),
    ("Linux kernel", "Linux kernel"),
    ("what is inflation in economics", "Inflation"),
    ("the Silk Road", "Silk Road"),
    ("Higgs boson", "Higgs boson"),
    ("who was Nikola Tesla", "Nikola Tesla"),
    ("what is the Krebs cycle", "Citric acid cycle"),
    ("Hubble Space Telescope", "Hubble Space Telescope"),
    ("Magna Carta", "Magna Carta"),
    ("what is a solid-state drive", "Solid-state drive"),
    ("Python programming language history", "Python (programming language)"),
]
SYNTHETIC_NEGATIVES: list[str] = [
    "best pizza near me",
    "iPhone 17 price",
    "weather in Boston this weekend",
    "how to fix a leaky faucet",
    "python sort list of dicts by key",
    "Cyberpunk 2077 update 2.5 patch notes",
    "flights to Tokyo next week",
    "Notion pricing tiers",
    "Apple earnings Q3",
    "Mercury retrograde dates",
    "Jaguar top speed",
    "how long to boil an egg",
    "Xbox Game Pass games leaving this month",
    "is Costco open on Sunday",
    "ollama api structured outputs example",
    "convert 5 miles to kilometers",
    "M4 Mac mini vs M4 Pro benchmark",
    "Taylor Swift tour dates",
    "Tesla stock",
    "error ECONNREFUSED localhost 8003",
    "gift ideas for a 10 year old",
    "Svelte 5 runes migration guide",
    "who won the game last night",
    "how many calories in a banana",
    "Amazon return policy",
    "GTA 6 release date",
    "best mechanical keyboard 2026",
    "translate thank you to Japanese",
    "Hacker News top stories",
    "llama.cpp metal backend flags",
]

_QUERY_PARAM_RE = re.compile(r"^query=(['\"])(.*)\1$", re.DOTALL)


def _outside_repo(path: Path) -> Path:
    path = path.expanduser().resolve()
    if REPO_ROOT in path.parents or path == REPO_ROOT:
        sys.exit(f"refusing {path}: personal data must stay outside the repo")
    return path


def derive_query(instruction: str) -> str:
    """Same derivation as MCPToolDispatcher._run_web_search."""
    derived = instruction.strip()
    for filler in _WEB_SEARCH_FILLER_PREFIXES:
        if derived.lower().startswith(filler):
            derived = derived[len(filler):]
            break
    return derived[:120]


def load_sets(db: Path) -> tuple[list[str], list[str]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT task_id, role, content, metadata_json FROM chat_turns ORDER BY created_at"
    ).fetchall()
    conn.close()

    user_by_task = {t: c for t, r, c, _ in rows if r == "user"}
    set_a: list[str] = []
    for task_id, role, _content, meta_json in rows:
        if role != "assistant":
            continue
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            continue
        fired = meta.get("tools_fired") or []
        if not ({"web_search", "research"} & set(fired)):
            continue
        query = ""
        for step in meta.get("workflow_steps") or []:
            if str(step.get("tool_name", "")).startswith("web_search"):
                m = _QUERY_PARAM_RE.match(step.get("parameters", ""))
                if m:
                    query = m.group(2)
                    break
        if not query and task_id in user_by_task:
            query = derive_query(user_by_task[task_id])
        if query:
            set_a.append(query)

    set_b = [derive_query(c) for t, r, c, _ in rows if r == "user" and c.strip()]
    dedupe = lambda xs: list(dict.fromkeys(x for x in xs if x.strip()))  # noqa: E731
    return dedupe(set_a), dedupe(set_b)


async def replay(queries: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, q in enumerate(queries, 1):
        try:
            res = await wikipedia.wikipedia_search(q)
            best = res.get("best_match") or {}
            title = best.get("title", "")
            dec = (
                wikipedia_match.decide(q, title, bool(best.get("disambiguation")))
                if best else wikipedia_match.MatchDecision(False, "no_hit", 0.0)
            )
            out[q] = {"query": q, "title": title, "url": best.get("url", ""),
                      "attach": dec.attach, "reason": dec.reason, "coverage": dec.coverage}
        except ValueError as exc:
            out[q] = {"query": q, "title": "", "url": "", "attach": False,
                      "reason": "api_error", "coverage": 0.0, "error": str(exc)}
        if i % 25 == 0:
            print(f"  … {i}/{len(queries)}", file=sys.stderr)
        await asyncio.sleep(0.25)
    return out


def summarize(name: str, queries: list[str], results: dict[str, dict], labels: dict[str, str]) -> None:
    rows = [results[q] for q in queries]
    attached = [r for r in rows if r["attach"]]
    reasons = Counter(r["reason"] for r in rows)
    print(f"\n== Set {name}: {len(rows)} queries ==")
    print(f"  attached: {len(attached)} ({len(attached) / max(len(rows), 1):.0%})")
    print("  not attached, by reason: " + ", ".join(
        f"{k} {v}" for k, v in reasons.most_common() if k != "match"))
    if labels:
        judged = [labels.get(r["query"]) for r in attached]
        c = Counter(j or "unlabelled" for j in judged)
        print(f"  attached, labelled: right {c['right']}, unhelpful {c['unhelpful']}, "
              f"wrong {c['wrong']}, unlabelled {c['unlabelled']}")
        missed = [r for r in rows if not r["attach"] and labels.get(r["query"]) == "missed"]
        if missed:
            print(f"  labelled 'missed' (should have attached): {len(missed)}")


def summarize_synthetic(results: dict[str, dict]) -> None:
    print(f"\n== Set C (synthetic, labelled by construction) ==")
    pos_attached = pos_right = 0
    for q, expected in SYNTHETIC_POSITIVES:
        r = results[q]
        if r["attach"]:
            pos_attached += 1
            if expected.lower() in r["title"].lower():
                pos_right += 1
    neg_attached = [q for q in SYNTHETIC_NEGATIVES if results[q]["attach"]]
    n_pos, n_neg = len(SYNTHETIC_POSITIVES), len(SYNTHETIC_NEGATIVES)
    print(f"  positives: attached {pos_attached}/{n_pos}, of which the expected article "
          f"{pos_right}, a different one {pos_attached - pos_right}")
    print(f"  negatives: attached {len(neg_attached)}/{n_neg} (should be 0)")
    for q, expected in SYNTHETIC_POSITIVES:
        r = results[q]
        mark = ("ok " if expected.lower() in r["title"].lower() else "WRONG") if r["attach"] else "miss"
        print(f"    + {mark:<5} {q!r} -> {r['title']!r} ({r['reason']}, cov {r['coverage']})")
    for q in SYNTHETIC_NEGATIVES:
        r = results[q]
        mark = "ATTACHED" if r["attach"] else "ok"
        print(f"    - {mark:<8} {q!r} -> {r['title']!r} ({r['reason']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--details", required=True, type=Path, help="JSONL of per-query results (outside the repo)")
    ap.add_argument("--labels", type=Path, help="labels JSON (outside the repo)")
    ap.add_argument("--db", type=Path, help="chat database; default LOCALIST_MEMORY_DB from backend/.env")
    ap.add_argument("--limit", type=int, default=0, help="cap set B (0 = all)")
    args = ap.parse_args()

    details = _outside_repo(args.details)
    labels: dict[str, str] = {}
    if args.labels:
        labels = json.loads(_outside_repo(args.labels).read_text())

    db = args.db or Path(os.path.expanduser(
        (dotenv_values(BACKEND / ".env").get("LOCALIST_MEMORY_DB") or "").strip('"')
    ))
    if not db.is_file():
        sys.exit(f"database not found: {db}")

    set_a, set_b = load_sets(db)
    if args.limit:
        set_b = set_b[: args.limit]

    cached: dict[str, dict] = {}
    if details.is_file():
        for line in details.read_text().splitlines():
            row = json.loads(line)
            cached[row["query"]] = row
    set_c = [q for q, _ in SYNTHETIC_POSITIVES] + SYNTHETIC_NEGATIVES
    todo = [q for q in dict.fromkeys(set_a + set_b + set_c) if q not in cached]
    print(f"Set A {len(set_a)}, set B {len(set_b)}; {len(todo)} new Wikipedia lookups "
          f"({len(cached)} cached in the details file).", file=sys.stderr)
    cached.update(asyncio.run(replay(todo)))
    details.parent.mkdir(parents=True, exist_ok=True)
    details.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cached.values()))

    print(f"Rule: localist.wikipedia_match (MIN_COVERAGE={wikipedia_match.MIN_COVERAGE})")
    summarize("A (web_search/research turns)", set_a, cached, labels)
    summarize("B (every user request, as a search)", set_b, cached, labels)
    summarize_synthetic(cached)


if __name__ == "__main__":
    main()
