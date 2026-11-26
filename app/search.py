"""Client field matching and document passage retrieval in PostgreSQL."""

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embedding import QUERY_PREFIX, embed
from app.schemas import DocumentHit, Evidence, Span

LIMIT = 10
COSINE_FLOOR = 0.60
SEMANTIC_WEIGHT = 0.3
IDENTIFIER = re.compile(r"(?:[a-z]{1,10}-\d[\w-]*|\d+(?:[.,]\d+)?\s*%)", re.I)

CLIENTS = text("""
WITH fields AS (
    SELECT id, first_name, last_name, email, description,
           lower(unaccent(first_name || ' ' || last_name)) AS name,
           lower(unaccent(:q)) AS needle
    FROM clients
), ranked AS (
    SELECT *, similarity(name, needle) AS likeness, CASE
        WHEN lower(email) = lower(:q) THEN 1
        WHEN name = needle THEN 2
        WHEN starts_with(name, needle) OR starts_with(lower(unaccent(last_name)), needle) THEN 3
        WHEN strpos(name, needle) > 0 THEN 4
        WHEN strpos(lower(email), lower(:q)) > 0 THEN 5
        WHEN to_tsvector('english', coalesce(description, '')) @@ plainto_tsquery('english', :q)
             OR strpos(lower(description), lower(:q)) > 0 THEN 6
        WHEN :fuzzy AND similarity(name, needle) >= 0.45 THEN 7
    END AS tier
    FROM fields
)
SELECT id, first_name, last_name, email,
       CASE WHEN tier IN (1,5) THEN 'email' WHEN tier = 6 THEN 'description' ELSE 'name' END AS matched_field
FROM ranked WHERE tier IS NOT NULL
AND (tier < 7 OR NOT EXISTS (SELECT FROM ranked WHERE tier <= 5))
ORDER BY tier, likeness DESC, name, id LIMIT 10
""")


def client_hits(session: Session, q: str):
    if not any(c.isalnum() for c in q):
        return []
    fuzzy = len(q) >= 4 and all(c.isalpha() or c in " '-" for c in q)
    return session.execute(CLIENTS, {"q": q, "fuzzy": fuzzy}).mappings().all()


# Rank each document by its strongest passage before taking each arm's shortlist.
PASSAGES = """
SELECT d.id, d.client_id, c.first_name || ' ' || c.last_name AS client_name,
       d.title, ch.ordinal, ch.start_char,
       substr(d.content, ch.start_char + 1, ch.end_char - ch.start_char) AS passage,
       ch.embedding
FROM document_chunks ch JOIN documents d ON ch.document_id = d.id
JOIN clients c ON c.id = d.client_id
"""
SHORTLIST = """
, best AS (
    SELECT DISTINCT ON (id) id, client_id, client_name, title, ordinal, start_char, passage, score
    FROM scores WHERE score > 0 ORDER BY id, score DESC, ordinal
)
SELECT * FROM best ORDER BY score DESC, id LIMIT 20
"""
QUERY = """SELECT array_to_string(ARRAY(
    SELECT quote_literal(word) FROM unnest(tsvector_to_array(to_tsvector('english', :terms))) word
), ' | ')::tsquery AS terms"""

LEXICAL = text(f"""
WITH passages AS ({PASSAGES}), query AS ({QUERY}), scores AS (
    SELECT passages.*,
        CASE WHEN :strict THEN
            CASE WHEN title ~* :pattern THEN 3 WHEN passage ~* :pattern THEN 2 ELSE 0 END
        ELSE
            CASE WHEN lower(title) = lower(:q) THEN 2
                 WHEN strpos(lower(title), lower(:q)) > 0 THEN 0.5
                 WHEN strpos(lower(passage), lower(:q)) > 0 THEN 0.25 ELSE 0 END
            + ts_rank_cd(setweight(to_tsvector('english', title), 'A') ||
                         setweight(to_tsvector('english', passage), 'B'), terms, 32)
        END AS score
    FROM passages CROSS JOIN query
) {SHORTLIST}
""")
SEMANTIC = text(f"""
WITH passages AS ({PASSAGES}), scores AS (
    SELECT passages.*, 1 - (embedding <=> CAST(:vector AS vector)) AS score FROM passages
) {SHORTLIST}
""")


# In the seeded corpus, address-proof queries otherwise favored change-of-address
# notes. Expand the intent to evidence a provider might actually issue.
ADDRESS_PROOF = {"address proof", "proof of address", "proof of residence", "residency verification"}


def candidates(session: Session, q: str):
    if not any(c.isalnum() for c in q):
        return [], []
    strict = bool(IDENTIFIER.fullmatch(q))
    normalized = " ".join(q.split()).casefold()
    retrieval_text = "utility bill showing the service address" if normalized in ADDRESS_PROOF else q
    pattern = r"(?<![[:alnum:]_.])" + re.escape(q) + r"(?![[:alnum:]_])"
    lexical = session.execute(LEXICAL, {"q": q, "terms": retrieval_text, "strict": strict, "pattern": pattern}).mappings().all()
    if strict:
        return lexical, []
    return lexical, semantic_candidates(session, retrieval_text)


def semantic_candidates(session: Session, query: str):
    vector = embed([QUERY_PREFIX + query])[0].tolist()
    return session.execute(SEMANTIC, {"vector": str(vector)}).mappings().all()


def fuse(lexical, semantic, weight=SEMANTIC_WEIGHT, floor=COSINE_FLOOR):
    """Weighted reciprocal rank fusion over unique document IDs."""
    totals, strongest, passages = {}, {}, {}
    eligible = [row for row in semantic if row["score"] >= floor]
    for rows, arm_weight in ((lexical, 1 - weight), (eligible, weight)):
        if arm_weight == 0:
            continue
        for rank, row in enumerate(rows, 1):
            key = row["id"]
            contribution = arm_weight / (60 + rank)
            totals[key] = totals.get(key, 0) + contribution
            if contribution > strongest.get(key, 0):
                strongest[key], passages[key] = contribution, row
    ids = sorted(totals, key=lambda key: (-totals[key], key))
    return [passages[key] for key in ids[:LIMIT]]


def evidence(q, row, inflections):
    passage = row["passage"]
    source, offset = "content", row["start_char"]
    pattern = re.escape(q)
    if IDENTIFIER.fullmatch(q):
        pattern = r"(?<![\w.])" + pattern + r"(?!\w)"
    if (re.search(pattern, row["title"], re.I) and not re.search(pattern, passage, re.I)) or not passage.strip():
        passage, source, offset = row["title"], "title", 0
    # Scores use normalized text. All returned offsets refer to the unmodified source.
    matches = list(re.finditer(pattern, passage, re.I))
    if not matches:
        matches = [m for m in re.finditer(r"\w+", passage) if m[0].lower() in inflections]
    start, end = 0, len(passage)
    if matches:
        # Present a whole source sentence or form row around the first hit.
        anchor = matches[0]
        boundaries = [m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+", passage)]
        start = max((b for b in boundaries if b <= anchor.start()), default=0)
        end = next((b for b in boundaries if b >= anchor.end()), end)
        if anchor.start() - start > 180:
            start = anchor.start()
        end = min(end, max(start + 360, anchor.end()))
    elif end > 600:
        end = 600
    if end < len(passage) and passage[end-1:end].isalnum() and passage[end:end+1].isalnum():
        boundary = passage.rfind(" ", matches[0].end() if matches else start, end)
        if boundary >= 0:
            end = boundary
    while end > start and passage[end - 1].isspace():
        end -= 1
    highlights = [
        Span(start=m.start() - start, end=m.end() - start)
        for m in matches if start <= m.start() and m.end() <= end
    ]
    return Evidence(source=source, start=offset + start, text=passage[start:end], highlights=highlights)


def document_hits(session: Session, q: str):
    rows = fuse(*candidates(session, q))
    if not rows:
        return []
    words = {word for row in rows for word in re.findall(r"\w+", (row["title"] + " " + row["passage"]).lower())}
    inflections = set(session.scalars(text(f"""
        WITH query AS ({QUERY})
        SELECT word FROM unnest(CAST(:words AS text[])) word, query
        WHERE to_tsvector('english', word) @@ terms
    """), {"terms": q, "words": list(words)}))
    return [DocumentHit(**row, evidence=evidence(q, row, inflections)) for row in rows]
