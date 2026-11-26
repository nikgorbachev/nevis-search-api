"""Run with the deterministic demo loaded: docker compose exec -T api python - < scripts/evaluate.py"""

import json
import os
from statistics import median
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import urlopen

from sqlalchemy.orm import Session

from app.db import engine
from app.models import Document
from app.search import COSINE_FLOOR, IDENTIFIER, SEMANTIC_WEIGHT, candidates, fuse, semantic_candidates

# Title-level relevance judgments. Multiple clients can have the same acceptable title.
# A blank title is a judged no-result query. Freeze these sets before tuning.
DEV = [
    ("address proof", "November electricity bill"),
    ("ZK-4729", "Family transition memorandum"),
    ("2.5%", "Annual advisory fee notice"),
    ("when can I sell my shares after vesting", "Restricted stock vesting memo"),
    ("give money to the foundation in installments", "Foundation grant schedule"),
    ("courier tracking", "Courier pickup receipt"),
    ("penguin incubation temperature", ""),
    ("ZK-999999", ""),
    ("asteroid mining royalties", ""),
]
HELD_OUT = [
    ("which document shows where the household lives", "November electricity bill"),
    ("25%", "Portfolio concentration worksheet"),
    ("pay for school next academic year", "Tuition reserve worksheet"),
    ("estate inventory", "Estate inventory reconciliation"),
    ("conditions on payment after selling the company", "Deferred consideration briefing"),
    ("instructions for entering the building", "Office access instructions"),
    ("when do I need cleared funds for my property purchase", "Property completion cash plan"),
    ("fungal growth on an oak tree", ""),
    ("QQ-0047", ""),
    ("%_", ""),
]


def evaluate(cases, weights, floor, details=False):
    first = five = false_positives = positives = negatives = 0
    for query, title, lexical, semantic, dense_baseline in cases:
        weight = weights if isinstance(weights, float) else (
            0.3 if len(query.split()) <= 2 else 0.5 if len(query.split()) <= 5 else 0.7
        )
        rows = fuse(lexical, dense_baseline if weight == 1 else semantic, weight, floor)
        titles = [r["title"] for r in rows]
        if title:
            positives += 1
            first += title in titles[:1]
            five += title in titles[:5]
        else:
            negatives += 1
            false_positives += bool(titles)
        if details:
            print(f"  {query!r} -> {titles[0] if titles else '(empty)'}")
    return f"{first}/{positives}  {five}/{positives}  {false_positives}/{negatives}"


def check_api(session):
    def search(query, kind):
        params = urlencode({"q": query, "type": kind})
        with urlopen("http://127.0.0.1:8000/search?" + params, timeout=60) as response:
            return json.load(response)

    for query, email in [
        ("NevisWealth", "jordan.doe@neviswealth.example.com"),
        ("Marin Reed", "marin.reed@wealth-demo.example.com"),
        ("Maren Reed", "maren.reed@wealth-demo.example.com"),
        ("Elena Muller", "elena.muller@wealth-demo.example.com"),
        ("Elna Muller", "elena.muller@wealth-demo.example.com"),
    ]:
        assert search(query, "client")[0]["email"] == email, query
    print("Client top-1 checks: 5/5")

    for query, title, phrase in [
        ("address proof", "November electricity bill", "12 Sample Lane"),
        ("ZK-4729", "Family transition memorandum", "ZK-4729"),
        ("2.5%", "Annual advisory fee notice", "2.5%"),
        ("25%", "Portfolio concentration worksheet", "25%"),
    ]:
        hit = search(query, "document")[0]
        ev = hit["evidence"]
        original = getattr(session.get(Document, hit["id"]), ev["source"])
        assert original[ev["start"]:ev["start"] + len(ev["text"])] == ev["text"], query
        assert hit["title"] == title and phrase in ev["text"], query
        if IDENTIFIER.fullmatch(query):
            assert query in [ev["text"][s["start"]:s["end"]] for s in ev["highlights"]], query
    print("Top-result evidence checks: 4/4")

    timings = []
    for q, _ in HELD_OUT:
        start = perf_counter()
        search(q, "document")
        timings.append((perf_counter() - start) * 1000)
    print(f"Warm document search HTTP median ({len(timings)} queries): {median(timings):.0f} ms")


def main():
    with Session(engine) as session:
        for label, queries in (("development", DEV), ("held-out", HELD_OUT)):
            if os.getenv("EVAL_SPLIT", label) != label:
                continue
            cases, timings = [], []
            for query, target in queries:
                start = perf_counter()
                lexical, semantic = candidates(session, query)
                timings.append((perf_counter() - start) * 1000)
                dense = semantic_candidates(session, query) if IDENTIFIER.fullmatch(query) else semantic
                cases.append((query, target, lexical, semantic, dense))
            print(label, "Hit@1  Hit@5  false positives / negatives")
            for weight in (0.0, 1.0, 0.3, 0.5, 0.7, "adaptive"):
                print(weight, evaluate(cases, weight, COSINE_FLOOR))
            if label == "development":
                for query, _, _, semantic, _ in cases:
                    print(query, "best cosine", round(semantic[0]["score"], 3) if semantic else "n/a")
                for floor in (0.50, 0.55, 0.60, 0.65, 0.70):
                    print("floor", floor, evaluate(cases, SEMANTIC_WEIGHT, floor))
            evaluate(cases, SEMANTIC_WEIGHT, COSINE_FLOOR, details=True)
            print(f"Median retrieval time: {median(timings):.0f} ms")

        if not os.getenv("EVAL_SPLIT"):
            check_api(session)


if __name__ == "__main__":
    main()
