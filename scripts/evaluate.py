"""3-Fold Retrieval Evaluation Suite (QAD-2 methodology) for Nevis Search API.

Measures:
- Step 1 (Parent Discovery): Doc Hit@1, Doc Hit@5, Doc MRR
- Step 2 (Passage Retrieval): Segment Hit@1, Deduplicated Graded Segment NDCG@5
- Step 3 (Cue Localization): Exact Highlight Rate, Verbatim Offset Integrity
- Typo Resilience (Documents & Clients)
- True Negative Precision (Out-of-Domain, Missing/Near-Miss IDs, SQL Wildcards, Near-Domain Financial)
- Stage-by-stage Latency Breakdown & Live HTTP API Checks

Run from repository root:
  docker compose exec -T api python - < scripts/evaluate.py
"""

import json
import math
import re
from statistics import median
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import urlopen

from sqlalchemy.orm import Session

from app.db import engine
from app.embedding import QUERY_PREFIX, embed
from app.models import Document
from app.normalize import fold_accents, normalize_text
from app.search import (
    COSINE_FLOOR,
    IDENTIFIER,
    SEMANTIC_ONLY_FLOOR,
    SEMANTIC_WEIGHT,
    _INFLECTION_QUERY,
    _LEXICAL_INDEXED,
    _SEMANTIC_HNSW,
    _expand_alias,
    candidates,
    client_hits,
    correct_query,
    evidence,
    fuse,
    semantic_candidates,
)

# Graded segment-level & highlight-level benchmark
# (group, query, [(title_prefix, expected_segment_substr, rel_grade)], expected_highlights_or_None)
DOC_SUITE = [
    # Group A: Exact Identifiers & Numbers (10)
    ("id_num", "ZK-4729", [("Family transition memorandum", "Form ZK-4729", 3)], ["ZK-4729"]),
    ("id_num", "zk-4729", [("Family transition memorandum", "Form ZK-4729", 3)], ["ZK-4729"]),
    ("id_num", "POA-8814", [("Irrevocable Trust Deed Summary", "POA-8814", 3)], ["POA-8814"]),
    ("id_num", "CAP-3092", [("Irrevocable Trust Deed Summary", "CAP-3092", 3)], ["CAP-3092"]),
    ("id_num", "ELEC-5197", [("November electricity bill", "ELEC-5197", 3)], ["ELEC-5197"]),
    ("id_num", "2.5%", [("Annual advisory fee notice", "2.5%", 3)], ["2.5%"]),
    ("id_num", "25%", [("Portfolio concentration worksheet", "25%", 3)], ["25%"]),
    ("id_num", "12.5%", [("Irrevocable Trust Deed Summary", "12.5%", 3)], ["12.5%"]),
    ("id_num", "1,250 EUR", [("Recurring cash transfer instructions", "1,250 EUR", 3)], ["1,250 EUR"]),
    ("id_num", "1.5%", [("OCR Scan — Municipal Water & Power Statement", "1.5 %", 3)], ["1.5 %"]),

    # Group B: Brands, Keywords & Multi-Chunk Localization (8)
    ("brand_kw", "North Quay Electricity", [("November electricity bill", "North Quay Electricity", 3)], ["North Quay Electricity"]),
    ("brand_kw", "portfolio", [("Portfolio snapshot", "Portfolio", 3), ("Portfolio concentration worksheet", "portfolio", 3)], ["portfolio"]),
    ("brand_kw", "courier tracking", [("Courier pickup receipt", "Tracking code", 3)], ["courier", "tracking"]),
    ("brand_kw", "estate inventory", [("Estate inventory reconciliation", "inventory", 3)], ["estate", "inventory"]),
    ("brand_kw", "successor custodian transfer paperwork", [("Family transition memorandum", "successor custodian", 3)], ["successor", "custodian", "transfer"]),
    ("brand_kw", "Northstar Infrastructure Fund capital call", [("Irrevocable Trust Deed Summary", "CAP-3092", 3)], ["Northstar", "Infrastructure", "Capital", "call"]),
    ("brand_kw", "tax-loss harvesting wash-sale", [("Investment Policy Statement", "tax-loss harvesting", 3)], ["tax", "harvesting", "wash"]),
    ("brand_kw", "tobacco thermal coal excluded", [("Investment Policy Statement", "thermal coal", 3)], ["tobacco", "thermal", "coal"]),

    # Group C: Semantic / Conceptual Paraphrases & Dirty OCR (10)
    ("semantic", "address proof", [("November electricity bill", "12 Sample Lane", 3), ("OCR Scan — Municipal Water & Power Statement", "44 Cedar Ridge", 2)], None),
    ("semantic", "proof of residence", [("November electricity bill", "12 Sample Lane", 3)], None),
    ("semantic", "power of attorney", [("Irrevocable Trust Deed Summary", "power of attorney", 3), ("Investment Policy Statement", "power of attorney", 2)], ["power", "attorney"]),
    ("semantic", "when can I sell my shares after vesting", [("Restricted stock vesting memo", "tranche releases", 3), ("Portfolio concentration worksheet", "sold after vesting", 2)], ["shares", "vesting"]),
    ("semantic", "give money to the foundation in installments", [("Foundation grant schedule", "grant installment", 3), ("Gift letter extract", "annual installments", 2)], ["foundation", "installments"]),
    ("semantic", "pay for school next academic year", [("Tuition reserve worksheet", "next academic year", 3)], ["academic", "year"]),
    ("semantic", "instructions for entering the building", [("Office access instructions", "courtyard entrance", 3)], None),
    ("semantic", "when do I need cleared funds for my property purchase", [("Property completion cash plan", "cleared funds", 3)], ["cleared", "funds"]),
    ("semantic", "EUR 142.80 municipal electricity water", [("OCR Scan — Municipal Water & Power Statement", "EUR142.80", 3)], None),
    ("semantic", "W-9 taxpayer identification number", [("Irrevocable Trust Deed Summary", "W-9", 3), ("Investment Policy Statement", "W-9", 2)], ["W-9", "taxpayer"]),

    # Group D: Typos & Compact IDs (7)
    ("typo", "portfolo", [("Portfolio snapshot", "Portfolio", 3), ("Portfolio concentration worksheet", "portfolio", 3)], ["portfolio"]),
    ("typo", "electrcity bill", [("November electricity bill", "Electricity", 3)], ["electricity", "bill"]),
    ("typo", "adres proof", [("November electricity bill", "12 Sample Lane", 3)], None),
    ("typo", "vestng shares", [("Restricted stock vesting memo", "shares", 3)], ["vesting", "shares"]),
    ("typo", "cuorier tracking", [("Courier pickup receipt", "Tracking", 3)], ["courier", "tracking"]),
    ("typo", "ZK4729", [("Family transition memorandum", "ZK-4729", 3)], ["ZK-4729"]),
    ("typo", "indemnity escow", [("Deferred consideration briefing", "escrow", 3), ("Adviser meeting notes", "indemnity escrow", 2)], ["indemnity", "escrow"]),
]

NEGATIVES = [
    ("ood", "penguin incubation temperature"),
    ("ood", "asteroid mining royalties"),
    ("ood", "fungal growth on an oak tree"),
    ("id_miss", "ZK-999999"),
    ("id_miss", "QQ-0047"),
    ("id_miss", "ELEC-9999"),
    ("id_miss", "ZK-4728"),
    ("id_miss", "99.9%"),
    ("wildcard", "%_"),
    ("wildcard", "%%"),
    ("near_domain", "cryptocurrency staking yield"),
    ("near_domain", "margin call liquidation notice"),
    ("near_domain", "options Greeks delta hedging"),
    ("near_domain", "Swiss numbered bank account IBAN"),
]


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate_mode(session: Session, weight: float, floor: float = COSINE_FLOOR, sem_only: float = SEMANTIC_ONLY_FLOOR):
    by_group: dict[str, list] = {}
    misses: list[str] = []

    for grp, q, targets, exp_hl in DOC_SUITE:
        lex, sem, corrected_q, _ = candidates(session, q)
        if weight == 1.0 and IDENTIFIER.fullmatch(normalize_text(q)):
            dense, _ = semantic_candidates(session, q)
            rows = fuse(lex, dense, weight=weight, floor=floor, semantic_only_floor=sem_only)
        else:
            rows = fuse(lex, sem, weight=weight, floor=floor, semantic_only_floor=sem_only)

        inflections: set[str] = set()
        if rows:
            words = {
                fold_accents(w)
                for row in rows
                for w in re.findall(r"[^\W_]+", row["title"] + " " + row["passage"])
            }
            inflections = set(
                session.scalars(_INFLECTION_QUERY, {"terms": fold_accents(corrected_q), "words": list(words)})
            )

        doc_ranks: list[int] = []
        seg_rels: list[int] = []
        seen_gt: set[int] = set()
        top1_seg_hit = False
        top1_hl_exact = None
        top1_offset_ok = None

        for idx, row in enumerate(rows[:5]):
            ev = evidence(q, row, inflections, corrected_q=corrected_q)
            best_doc_rel = 0
            best_seg_rel = 0
            best_gt_idx = None
            for gi, (t_pref, seg_sub, rel) in enumerate(targets):
                if row["title"] == t_pref or row["title"].startswith(t_pref):
                    best_doc_rel = max(best_doc_rel, rel)
                    if seg_sub.lower() in ev.text.lower() or seg_sub.lower() in row["passage"].lower():
                        if rel > best_seg_rel:
                            best_seg_rel = rel
                            best_gt_idx = gi
            if best_doc_rel >= 2:
                doc_ranks.append(idx + 1)
            if best_gt_idx is not None and best_gt_idx not in seen_gt:
                seen_gt.add(best_gt_idx)
                seg_rels.append(best_seg_rel)
            else:
                seg_rels.append(0)

            if idx == 0:
                top1_seg_hit = best_seg_rel >= 2
                orig = getattr(session.get(Document, row["id"]), ev.source)
                top1_offset_ok = orig[ev.start:ev.start + len(ev.text)] == ev.text
                hl_texts = [ev.text[sp.start:sp.end].lower() for sp in ev.highlights]
                if exp_hl is not None:
                    top1_hl_exact = bool(hl_texts) and any(
                        any(eh.lower() in ht or ht in eh.lower() for ht in hl_texts)
                        for eh in exp_hl
                    )

        doc_h1 = 1 if (doc_ranks and doc_ranks[0] == 1) else 0
        doc_h5 = 1 if doc_ranks else 0
        doc_mrr = (1.0 / doc_ranks[0]) if doc_ranks else 0.0
        ideal = sorted([t[2] for t in targets], reverse=True)[:5]
        ndcg5 = dcg(seg_rels) / dcg(ideal) if dcg(ideal) > 0 else 0.0
        by_group.setdefault(grp, []).append((doc_h1, doc_h5, doc_mrr, int(top1_seg_hit), ndcg5, top1_hl_exact, top1_offset_ok))
        if not doc_h1 or not top1_seg_hit:
            misses.append(f"  MISS [{grp}] {q!r} -> {rows[0]['title'] if rows else '(empty)'!r} (seg_hit={top1_seg_hit})")

    print(f"\n=== MODE weight={weight} (floor={floor}, sem_only={sem_only}) ===")
    all_rows = []
    for grp, vals in by_group.items():
        all_rows.extend(vals)
        n = len(vals)
        hl_vals = [v[5] for v in vals if v[5] is not None]
        hl_str = f"{sum(hl_vals)}/{len(hl_vals)}" if hl_vals else "n/a"
        print(
            f"  {grp:10s} (n={n:2d}): "
            f"DocH@1={sum(v[0] for v in vals)/n:.3f}  "
            f"DocH@5={sum(v[1] for v in vals)/n:.3f}  "
            f"MRR={sum(v[2] for v in vals)/n:.3f}  "
            f"SegH@1={sum(v[3] for v in vals)/n:.3f}  "
            f"SegNDCG@5={sum(v[4] for v in vals)/n:.3f}  "
            f"CueHL={hl_str}"
        )
    no_typo = [v for g, vs in by_group.items() if g != "typo" for v in vs]
    for label, subset in [(f"NON-TYPO ({len(no_typo)})", no_typo), (f"ALL+TYPO ({len(all_rows)})", all_rows)]:
        n = len(subset)
        hl_vals = [v[5] for v in subset if v[5] is not None]
        off_vals = [v[6] for v in subset if v[6] is not None]
        print(
            f"  {label:14s}: "
            f"DocH@1={sum(v[0] for v in subset)/n:.3f}  "
            f"DocH@5={sum(v[1] for v in subset)/n:.3f}  "
            f"MRR={sum(v[2] for v in subset)/n:.3f}  "
            f"SegH@1={sum(v[3] for v in subset)/n:.3f}  "
            f"SegNDCG@5={sum(v[4] for v in subset)/n:.3f}  "
            f"CueHL={sum(hl_vals)}/{len(hl_vals)}  "
            f"OffOK={sum(off_vals)}/{len(off_vals)}"
        )
    if weight == SEMANTIC_WEIGHT and misses:
        print("  Notable misses on shipped mode:")
        for line in misses:
            print(line)


def check_negatives(session: Session):
    print(f"\n=== NEGATIVES BY SUBCLASS (w={SEMANTIC_WEIGHT}, floor={COSINE_FLOOR}, sem_only={SEMANTIC_ONLY_FLOOR}) ===")
    by_sub: dict[str, list] = {}
    for sub, q in NEGATIVES:
        lex, sem, _, _ = candidates(session, q)
        rows = fuse(lex, sem, weight=SEMANTIC_WEIGHT, floor=COSINE_FLOOR, semantic_only_floor=SEMANTIC_ONLY_FLOOR)
        by_sub.setdefault(sub, []).append((q, len(rows), len(lex), round(sem[0]["score"], 3) if sem else None))
    total_fp = total_n = 0
    for sub, items in by_sub.items():
        fps = sum(1 for _, n, _, _ in items if n > 0)
        total_fp += fps
        total_n += len(items)
        print(f"  {sub:12s}: FP={fps}/{len(items)}  {items}")
    print(f"  TOTAL NEGATIVES: FP={total_fp}/{total_n} (TNR={(total_n - total_fp)/total_n:.1%})")


def check_live_api(session: Session):
    def search(query: str, kind: str | None = None):
        params = {"q": query}
        if kind:
            params["type"] = kind
        with urlopen("http://127.0.0.1:8000/search?" + urlencode(params), timeout=30) as response:
            return json.load(response)

    client_checks = [
        ("NevisWealth", "jordan.doe@neviswealth.example.com"),
        ("neviswelth", "jordan.doe@neviswealth.example.com"),
        ("welth", None),  # must return >= 1 wealth/neviswealth email match
        ("Marin Reed", "marin.reed@wealth-demo.example.com"),
        ("Maren Reed", "maren.reed@wealth-demo.example.com"),
        ("Elena Muller", "elena.muller@wealth-demo.example.com"),
        ("Elna Muller", "elena.muller@wealth-demo.example.com"),
        ("Elna", "elena.muller@wealth-demo.example.com"),
        ("Muler", "elena.muller@wealth-demo.example.com"),
        ("Marn Red", None),  # must find Marin Reed or Maren Reed
        ("philanthropy", None),  # must find charitable-giving clients via description semantics
        ("exiting a family company", None),  # must find business-transition clients
    ]
    passed = 0
    for q, expected_email in client_checks:
        hits = search(q, "client")
        if expected_email:
            ok = bool(hits) and hits[0]["email"] == expected_email
        else:
            ok = bool(hits)
        if not ok:
            print(f"  CLIENT CHECK FAIL: q={q!r} expected={expected_email!r} got={[h['email'] for h in hits[:3]]}")
        else:
            passed += 1
    print(f"\nClient checks (exact + accent + typos + semantic desc): {passed}/{len(client_checks)}")

    ev_checks = [
        ("address proof", "November electricity bill", "12 Sample Lane"),
        ("ZK-4729", "Family transition memorandum", "ZK-4729"),
        ("ZK4729", "Family transition memorandum", "ZK-4729"),
        ("2.5%", "Annual advisory fee notice", "2.5%"),
        ("25%", "Portfolio concentration worksheet", "25%"),
        ("POA-8814", "Irrevocable Trust Deed Summary", "POA-8814"),
    ]
    ev_passed = 0
    for q, title, phrase in ev_checks:
        hits = search(q, "document")
        assert hits, f"No hits for {q}"
        hit = hits[0]
        ev = hit["evidence"]
        original = getattr(session.get(Document, hit["id"]), ev["source"])
        assert original[ev["start"]:ev["start"] + len(ev["text"])] == ev["text"], q
        assert hit["title"] == title and phrase in ev["text"], (q, hit["title"], ev["text"])
        assert hit["summary"], f"Missing summary for {q}"
        ev_passed += 1
    print(f"Top-result evidence & summary checks: {ev_passed}/{len(ev_checks)}")

    timings = []
    for _, q, _, _ in DOC_SUITE[:15]:
        t0 = perf_counter()
        search(q, "document")
        timings.append((perf_counter() - t0) * 1000)
    print(f"Warm document search HTTP median ({len(timings)} queries): {median(timings):.1f} ms (p95: {sorted(timings)[int(len(timings)*0.95)]:.1f} ms)")


def check_latency_breakdown(session: Session):
    print("\n=== STAGE LATENCY BREAKDOWN (warm) ===")
    for q in ["address proof", "when can I sell my shares after vesting", "portfolio", "portfolo", "ZK-4729", "NevisWealth"]:
        q_clean = normalize_text(q)
        t0 = perf_counter()
        client_hits(session, q)
        t_cli = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        corrected, _ = correct_query(session, q_clean)
        t_corr = (perf_counter() - t0) * 1000

        rtxt, is_alias = _expand_alias(corrected)
        needle = fold_accents(corrected)
        like_needle = "%" + needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        t0 = perf_counter()
        session.execute(
            _LEXICAL_INDEXED,
            {"terms": fold_accents(rtxt), "needle": needle, "like_needle": like_needle, "is_alias": is_alias},
        ).mappings().all()
        t_lex = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        vec = embed([QUERY_PREFIX + rtxt])[0].tolist()
        t_emb = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        session.execute(_SEMANTIC_HNSW, {"vector": str(vec)}).mappings().all()
        t_vec = (perf_counter() - t0) * 1000

        print(
            f"  q={q!r:42s} client={t_cli:5.1f}ms spell={t_corr:4.1f}ms "
            f"lex_gin={t_lex:5.1f}ms embed={t_emb:5.1f}ms hnsw={t_vec:5.1f}ms"
        )


def main():
    with Session(engine) as session:
        for w in (0.0, 1.0, SEMANTIC_WEIGHT, 0.5):
            evaluate_mode(session, w)
        check_negatives(session)
        check_latency_breakdown(session)
        check_live_api(session)


if __name__ == "__main__":
    main()
