"""Indexed hybrid search with OpenSearch-style Levenshtein/trigram fuzziness and verbatim evidence."""

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embedding import QUERY_PREFIX, embed, tokenizer
from app.normalize import fold_accents, normalize_text, summarize_document
from app.schemas import ClientHit, DocumentHit, Evidence, Span

LIMIT = 10
COSINE_FLOOR = 0.60
SEMANTIC_ONLY_FLOOR = 0.69
CLIENT_DESC_SEMANTIC_FLOOR = 0.68
SEMANTIC_WEIGHT = 0.3

# Matches hyphenated form codes (ZK-4729), compact alphanumeric codes (ZK4729), and percentages (2.5%).
IDENTIFIER = re.compile(
    r"(?:[a-z]{1,10}-\d[\w-]*|[a-z]{2,10}\d{3,}[\w-]*|\d+(?:[.,]\d+)?\s*%)",
    re.I,
)
COMPACT_CODE = re.compile(r"^([a-z]{2,10})(\d{3,}[\w-]*)$", re.I)

# Curated wealth-management domain aliases (applied after spell correction).
DOMAIN_ALIASES: list[tuple[set[str], str]] = [
    (
        {"address proof", "proof of address", "proof of residence", "residency verification"},
        "utility bill showing the service address electricity bill council tax",
    ),
    (
        {"kyc", "know your customer", "identity verification"},
        "identity verification passport onboarding kyc compliance",
    ),
    (
        {"poa", "power of attorney"},
        "power of attorney successor trustee legal authority",
    ),
    (
        {"rmd", "required minimum distribution"},
        "pension drawdown required minimum distribution retirement withdrawal",
    ),
    (
        {"w9", "w-9", "tax id", "tin"},
        "tax identification number w-9 taxpayer tin",
    ),
]


def _levenshtein_py(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


_EXACT_VOCAB = text("SELECT word FROM search_vocab WHERE word = ANY(CAST(:tokens AS text[]))")
_FUZZY_VOCAB = text("""
SELECT word,
       levenshtein_less_equal(word, :tok, :max_d) AS dist,
       similarity(word, :tok) AS sim,
       doc_freq
FROM search_vocab
WHERE abs(length(word) - length(:tok)) <= :max_d
  AND levenshtein_less_equal(word, :tok, :max_d) <= :max_d
ORDER BY dist ASC, sim DESC, doc_freq DESC
LIMIT 8
""")


def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string alignment distance (treats adjacent transpositions like cuorier<->courier as 1)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def correct_query(session: Session, q_clean: str) -> tuple[str, dict[str, str]]:
    """OpenSearch AUTO-style Damerau-Levenshtein spell correction against corpus vocabulary.

    Only rewrites out-of-vocabulary words that split into multiple WordPiece subwords
    (protects valid English words like 'margin' or 'exiting' from false rewrites).
    Never modifies IDs or numbers.
    """
    if IDENTIFIER.fullmatch(q_clean):
        return q_clean, {}
    folded = fold_accents(q_clean)
    alpha_tokens = sorted({t for t in re.findall(r"[a-z]+", folded) if len(t) >= 4})
    if not alpha_tokens:
        return q_clean, {}
    known = set(session.scalars(_EXACT_VOCAB, {"tokens": alpha_tokens}))
    tok_model = tokenizer()
    replacements: dict[str, str] = {}
    for tok in alpha_tokens:
        if tok in known:
            continue
        # Skip valid single-token English words present in BGE's 30k WordPiece vocabulary.
        if len(tok_model.encode(tok, add_special_tokens=False).ids) == 1:
            continue
        max_d = 1 if len(tok) <= 4 else 2
        rows = session.execute(_FUZZY_VOCAB, {"tok": tok, "max_d": max_d}).all()
        best = None
        for word, _, sim, freq in rows:
            d = _damerau_levenshtein(word, tok)
            if d == 1 and sim >= 0.25:
                cand = (d, -sim, -freq, word)
            elif d == 2 and sim >= 0.36 and (len(tok) >= 6 or word[0] == tok[0]):
                cand = (d, -sim, -freq, word)
            else:
                continue
            if best is None or cand < best:
                best = cand
        if best is not None:
            replacements[tok] = best[3]
    if not replacements:
        return q_clean, {}

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        return replacements.get(fold_accents(raw), raw)

    corrected = re.sub(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", _replace, q_clean)
    return corrected, replacements


def _expand_alias(normalized_q: str) -> tuple[str, bool]:
    key = " ".join(normalized_q.split()).casefold()
    for triggers, expansion in DOMAIN_ALIASES:
        if key in triggers:
            return expansion, True
        # Catch minor 1-2 char typos on multi-word domain aliases (e.g. "adres proof").
        if len(key) >= 5:
            for trigger in triggers:
                if len(trigger) >= 5 and _levenshtein_py(key, trigger) <= 2:
                    return expansion, True
    return normalized_q, False


# --- Client Search ---

_CLIENTS_EXACT = text("""
WITH fields AS (
    SELECT id, first_name, last_name, email, description, social_links,
           lower(unaccent(first_name || ' ' || last_name)) AS name,
           lower(unaccent(last_name)) AS lname,
           lower(unaccent(:q)) AS needle,
           lower(unaccent(:corr)) AS corr_needle
    FROM clients
), ranked AS (
    SELECT *, similarity(name, needle) AS likeness, CASE
        WHEN lower(email) = lower(:q) THEN 1
        WHEN name = needle THEN 2
        WHEN starts_with(name, needle) OR starts_with(lname, needle) THEN 3
        WHEN strpos(name, needle) > 0 THEN 4
        WHEN strpos(lower(email), lower(:q)) > 0 THEN 5
        WHEN to_tsvector('english', coalesce(description, '')) @@ plainto_tsquery('english', :corr)
             OR strpos(lower(coalesce(description, '')), lower(:q)) > 0
             OR (:corr != :q AND strpos(lower(coalesce(description, '')), lower(:corr)) > 0) THEN 6
        WHEN :corr != :q AND (
             strpos(name, corr_needle) > 0 OR strpos(lower(email), lower(:corr)) > 0
        ) THEN 7
    END AS tier
    FROM fields
)
SELECT id, first_name, last_name, email, description, social_links, tier,
       CASE
           WHEN tier IN (1, 5) THEN 'email'
           WHEN tier = 6 THEN 'description'
           WHEN tier = 7 AND strpos(lower(email), lower(:corr)) > 0 THEN 'email'
           ELSE 'name'
       END AS matched_field
FROM ranked
WHERE tier IS NOT NULL
ORDER BY tier, likeness DESC, name, id
LIMIT :limit
""")

_CLIENTS_FUZZY = text("""
WITH qtokens AS (
    SELECT ARRAY(
        SELECT tok FROM unnest(regexp_split_to_array(lower(unaccent(:q)), '[^a-z0-9]+')) tok
        WHERE length(tok) >= 3
    ) AS qtoks
), candidates AS (
    SELECT c.id, c.first_name, c.last_name, c.email, c.description, c.social_links,
           lower(unaccent(c.first_name || ' ' || c.last_name)) AS name,
           regexp_split_to_array(lower(unaccent(c.first_name || ' ' || c.last_name)), '[^a-z0-9]+') AS name_toks,
           regexp_split_to_array(lower(c.email), '[^a-z0-9]+') AS email_toks,
           qtokens.qtoks
    FROM clients c CROSS JOIN qtokens
    WHERE cardinality(qtokens.qtoks) > 0
), scored AS (
    SELECT id, first_name, last_name, email, description, social_links, name,
           -- All query tokens must fuzzy-match at least one name token (OpenSearch AUTO edit distance).
           (
               SELECT bool_and(EXISTS (
                   SELECT 1 FROM unnest(name_toks) nt
                   WHERE length(nt) >= 3
                     AND levenshtein_less_equal(nt, qt, CASE WHEN length(qt) <= 5 THEN 1 ELSE 2 END)
                         <= CASE WHEN length(qt) <= 5 THEN 1 ELSE 2 END
               ))
               FROM unnest(qtoks) qt
           ) AS name_fuzzy,
           (
               SELECT bool_and(EXISTS (
                   SELECT 1 FROM unnest(email_toks) et
                   WHERE length(et) >= 3
                     AND levenshtein_less_equal(et, qt, CASE WHEN length(qt) <= 5 THEN 1 ELSE 2 END)
                         <= CASE WHEN length(qt) <= 5 THEN 1 ELSE 2 END
               ))
               FROM unnest(qtoks) qt
           ) AS email_fuzzy,
           greatest(word_similarity(lower(unaccent(:q)), name), similarity(name, lower(unaccent(:q)))) AS name_sim,
           word_similarity(lower(:q), lower(email)) AS email_sim
    FROM candidates
)
SELECT id, first_name, last_name, email, description, social_links,
       CASE WHEN name_fuzzy THEN 'name' ELSE 'email' END AS matched_field,
       greatest(name_sim, email_sim) AS likeness
FROM scored
WHERE name_fuzzy OR email_fuzzy
ORDER BY name_fuzzy DESC, likeness DESC, name, id
LIMIT :limit
""")

_CLIENTS_SEMANTIC_DESC = text("""
SELECT id, first_name, last_name, email, description, social_links,
       'description' AS matched_field,
       1 - (description_embedding <=> CAST(:vector AS vector)) AS score
FROM clients
WHERE description_embedding IS NOT NULL
ORDER BY description_embedding <=> CAST(:vector AS vector)
LIMIT :limit
""")


def client_hits(
    session: Session,
    q: str,
    limit: int = LIMIT,
    query_vector: list[float] | None = None,
    corrected_q: str | None = None,
) -> tuple[list[ClientHit], bool]:
    """Return ranked ClientHit list and whether a strong exact/prefix/substring match exists."""
    q_clean = normalize_text(q)
    if not any(c.isalnum() for c in q_clean):
        return [], False
    if corrected_q is None:
        corrected_q, _ = correct_query(session, q_clean)

    rows = list(
        session.execute(
            _CLIENTS_EXACT, {"q": q_clean, "corr": corrected_q, "limit": limit}
        ).mappings().all()
    )
    has_strong = any(r["tier"] <= 5 for r in rows)
    seen = {r["id"] for r in rows}

    # OpenSearch-style token-level Levenshtein fuzzy fallback when exact name/email missed.
    is_fuzzy_candidate = (
        not has_strong
        and not IDENTIFIER.fullmatch(q_clean)
        and len(q_clean) >= 4
        and any(c.isalpha() for c in q_clean)
    )
    if is_fuzzy_candidate and len(rows) < limit:
        fuzzy_rows = session.execute(_CLIENTS_FUZZY, {"q": q_clean, "limit": limit}).mappings().all()
        for fr in fuzzy_rows:
            if fr["id"] not in seen:
                seen.add(fr["id"])
                rows.append(fr)

    # Semantic description fallback (e.g. "philanthropy" -> "charitable commitment").
    if not has_strong and not IDENTIFIER.fullmatch(q_clean) and len(rows) < limit:
        vec = query_vector
        if vec is None:
            vec = embed([QUERY_PREFIX + corrected_q])[0].tolist()
        sem_rows = session.execute(
            _CLIENTS_SEMANTIC_DESC, {"vector": str(vec), "limit": limit}
        ).mappings().all()
        for sr in sem_rows:
            if sr["score"] >= CLIENT_DESC_SEMANTIC_FLOOR and sr["id"] not in seen:
                seen.add(sr["id"])
                rows.append(sr)

    hits = [
        ClientHit(
            id=r["id"],
            first_name=r["first_name"],
            last_name=r["last_name"],
            email=r["email"],
            description=r["description"],
            social_links=r["social_links"] or [],
            matched_field=r["matched_field"],
        )
        for r in rows[:limit]
    ]
    return hits, has_strong


# --- Document Search ---

_LEXICAL_STRICT = text("""
WITH matched AS (
    SELECT d.id, d.client_id, c.first_name || ' ' || c.last_name AS client_name,
           d.title, d.content, d.created_at, ch.ordinal, ch.start_char,
           substr(d.content, ch.start_char + 1, ch.end_char - ch.start_char) AS passage,
           CASE
               WHEN ch.normalized_title ~* :pattern THEN 3.0
               WHEN ch.normalized_passage ~* :pattern THEN 2.0
               ELSE 0.0
           END AS score
    FROM document_chunks ch
    JOIN documents d ON ch.document_id = d.id
    JOIN clients c ON c.id = d.client_id
    WHERE ch.normalized_title ~* :pattern OR ch.normalized_passage ~* :pattern
), best AS (
    SELECT DISTINCT ON (id) id, client_id, client_name, title, content, created_at, ordinal, start_char, passage, score
    FROM matched
    WHERE score > 0
    ORDER BY id, score DESC, ordinal
)
SELECT * FROM best ORDER BY score DESC, id LIMIT 20
""")

_LEXICAL_INDEXED = text("""
WITH qterms AS (
    SELECT ARRAY(
        SELECT DISTINCT word
        FROM unnest(tsvector_to_array(to_tsvector('english', :terms))) AS word
    ) AS words
), qts AS (
    SELECT words,
           cardinality(words) AS total_terms,
           CASE WHEN cardinality(words) = 0 THEN ''::tsquery
                ELSE array_to_string(ARRAY(SELECT quote_literal(w) FROM unnest(words) w), ' | ')::tsquery
           END AS tsq
    FROM qterms
), raw_hits AS (
    SELECT d.id, d.client_id, c.first_name || ' ' || c.last_name AS client_name,
           d.title, d.content, d.created_at, ch.ordinal, ch.start_char,
           substr(d.content, ch.start_char + 1, ch.end_char - ch.start_char) AS passage,
           ch.normalized_title, ch.normalized_passage, ch.search_vector,
           qts.words, qts.total_terms, qts.tsq
    FROM document_chunks ch
    JOIN documents d ON ch.document_id = d.id
    JOIN clients c ON c.id = d.client_id
    CROSS JOIN qts
    WHERE (qts.total_terms > 0 AND ch.search_vector @@ qts.tsq)
       OR ch.normalized_title LIKE :like_needle ESCAPE '\'
       OR ch.normalized_passage LIKE :like_needle ESCAPE '\'
), scored AS (
    SELECT id, client_id, client_name, title, content, created_at, ordinal, start_char, passage,
           CASE
               WHEN normalized_title = :needle THEN 2.0
               WHEN strpos(normalized_title, :needle) > 0 THEN 0.5
               WHEN strpos(normalized_passage, :needle) > 0 THEN 0.25
               ELSE 0.0
           END AS phrase_bonus,
           CASE WHEN total_terms > 0 THEN ts_rank_cd(search_vector, tsq, 32) ELSE 0.0 END AS fts_rank,
           total_terms,
           (
               SELECT count(*) FROM unnest(words) w
               WHERE search_vector @@ to_tsquery('english', quote_literal(w))
           ) AS matched_terms
    FROM raw_hits
), gated AS (
    SELECT *, (phrase_bonus + fts_rank) AS score
    FROM scored
    WHERE phrase_bonus > 0
       OR :is_alias
       OR total_terms <= 2
       OR matched_terms >= LEAST(total_terms, GREATEST(2, ceil(total_terms * 0.45)::int))
), best AS (
    SELECT DISTINCT ON (id) id, client_id, client_name, title, content, created_at, ordinal, start_char, passage, score
    FROM gated
    WHERE score > 0
    ORDER BY id, score DESC, ordinal
)
SELECT * FROM best ORDER BY score DESC, id LIMIT 20
""")

_SEMANTIC_HNSW = text("""
WITH nearest AS (
    SELECT ch.document_id, ch.ordinal, ch.start_char, ch.end_char,
           1 - (ch.embedding <=> CAST(:vector AS vector)) AS score
    FROM document_chunks ch
    ORDER BY ch.embedding <=> CAST(:vector AS vector)
    LIMIT 60
), joined AS (
    SELECT d.id, d.client_id, c.first_name || ' ' || c.last_name AS client_name,
           d.title, d.content, d.created_at, n.ordinal, n.start_char,
           substr(d.content, n.start_char + 1, n.end_char - n.start_char) AS passage,
           n.score
    FROM nearest n
    JOIN documents d ON n.document_id = d.id
    JOIN clients c ON c.id = d.client_id
), best AS (
    SELECT DISTINCT ON (id) id, client_id, client_name, title, content, created_at, ordinal, start_char, passage, score
    FROM joined
    WHERE score > 0
    ORDER BY id, score DESC, ordinal
)
SELECT * FROM best ORDER BY score DESC, id LIMIT 20
""")


def semantic_candidates(session: Session, query: str, query_vector: list[float] | None = None):
    vector = query_vector if query_vector is not None else embed([QUERY_PREFIX + query])[0].tolist()
    return list(session.execute(_SEMANTIC_HNSW, {"vector": str(vector)}).mappings().all()), vector


def candidates(session: Session, q: str, query_vector: list[float] | None = None):
    q_clean = normalize_text(q)
    if not any(c.isalnum() for c in q_clean):
        return [], [], q_clean, None

    strict = bool(IDENTIFIER.fullmatch(q_clean))
    if strict:
        folded = fold_accents(q_clean)
        pattern = r"(?<![[:alnum:]_.])" + re.escape(folded) + r"(?![[:alnum:]_])"
        lexical = list(session.execute(_LEXICAL_STRICT, {"pattern": pattern}).mappings().all())
        return lexical, [], q_clean, None

    corrected_q, _ = correct_query(session, q_clean)
    retrieval_text, is_alias = _expand_alias(corrected_q)
    needle = fold_accents(corrected_q)
    like_needle = "%" + needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    terms_folded = fold_accents(retrieval_text)

    lexical = list(
        session.execute(
            _LEXICAL_INDEXED,
            {"terms": terms_folded, "needle": needle, "like_needle": like_needle, "is_alias": is_alias},
        ).mappings().all()
    )
    semantic, vec = semantic_candidates(session, retrieval_text, query_vector=query_vector)
    return lexical, semantic, corrected_q, vec


def fuse(
    lexical,
    semantic,
    weight: float = SEMANTIC_WEIGHT,
    floor: float = COSINE_FLOOR,
    semantic_only_floor: float = SEMANTIC_ONLY_FLOOR,
    limit: int = LIMIT,
):
    """Weighted reciprocal rank fusion over unique document IDs with two-tier semantic gating."""
    totals: dict[str, float] = {}
    strongest: dict[str, float] = {}
    passages: dict[str, dict] = {}

    lex_ids = {row["id"] for row in lexical}
    eligible = [
        row for row in semantic
        if (row["id"] in lex_ids and row["score"] >= floor)
        or (row["id"] not in lex_ids and row["score"] >= semantic_only_floor)
    ]

    for rows, arm_weight in ((lexical, 1 - weight), (eligible, weight)):
        if arm_weight == 0:
            continue
        for rank, row in enumerate(rows, 1):
            key = row["id"]
            contribution = arm_weight / (60 + rank)
            totals[key] = totals.get(key, 0.0) + contribution
            if contribution > strongest.get(key, 0.0):
                strongest[key], passages[key] = contribution, row

    ids = sorted(totals, key=lambda key: (-totals[key], key))
    return [passages[key] for key in ids[:limit]]


def _raw_id_pattern(q_clean: str) -> str:
    m = COMPACT_CODE.fullmatch(q_clean)
    if m:
        return r"(?<![\w.])" + re.escape(m.group(1)) + r"[-\s]?" + re.escape(m.group(2)) + r"(?!\w)"
    if q_clean.endswith("%"):
        num = q_clean[:-1].strip()
        return r"(?<![\w.])" + re.escape(num) + r"\s*%(?!\w)"
    return r"(?<![\w.])" + re.escape(q_clean) + r"(?!\w)"


def _find_matches(text_str: str, q_clean: str, inflections: set[str], query_folded_tokens: set[str]):
    if IDENTIFIER.fullmatch(q_clean):
        pat = _raw_id_pattern(q_clean)
        return list(re.finditer(pat, text_str, re.I))

    matches = list(re.finditer(re.escape(q_clean), text_str, re.I))
    if matches:
        return matches

    # Accent-insensitive and English-stem token matching preserving exact raw character spans.
    token_matches = []
    for m in re.finditer(r"[^\W_]+", text_str):
        folded = fold_accents(m.group(0))
        if folded in inflections or folded in query_folded_tokens:
            token_matches.append(m)
    return token_matches


def evidence(q: str, row, inflections: set[str], corrected_q: str | None = None) -> Evidence:
    q_clean = normalize_text(corrected_q or q)
    query_folded_tokens = {t for t in re.findall(r"[a-z]+", fold_accents(q_clean)) if len(t) >= 3}

    passage = row["passage"]
    source, offset = "content", row["start_char"]

    title_matches = _find_matches(row["title"], q_clean, inflections, query_folded_tokens)
    passage_matches = _find_matches(passage, q_clean, inflections, query_folded_tokens)
    if (title_matches and not passage_matches) or not passage.strip():
        passage, source, offset = row["title"], "title", 0
        matches = title_matches
    else:
        matches = passage_matches

    start, end = 0, len(passage)
    if matches:
        # Score candidate sentences/rows by distinct matched query terms (Step 3 cue localization).
        boundaries = [0] + [m.end() for m in re.finditer(r"\n+|(?<=[.!?])\s+", passage)] + [len(passage)]
        best_score = (-1, 0)
        best_span = (0, len(passage))
        for b_start, b_end in zip(boundaries, boundaries[1:]):
            if b_start >= b_end:
                continue
            seg_hits = [m for m in matches if b_start <= m.start() and m.end() <= b_end]
            if not seg_hits:
                continue
            distinct_terms = len({fold_accents(m.group(0)) for m in seg_hits})
            score = (distinct_terms, -b_start)
            if score > best_score:
                best_score = score
                best_span = (b_start, b_end)

        start, end = best_span
        anchor = next((m for m in matches if start <= m.start() and m.end() <= end), matches[0])
        if anchor.start() - start > 180:
            start = anchor.start()
        end = min(end, max(start + 360, anchor.end()))
    elif end > 600:
        end = 600

    if end < len(passage) and passage[end - 1:end].isalnum() and passage[end:end + 1].isalnum():
        anchor_end = matches[0].end() if matches and start <= matches[0].start() <= end else start
        boundary = passage.rfind(" ", anchor_end, end)
        if boundary >= 0:
            end = boundary
    while end > start and passage[end - 1].isspace():
        end -= 1

    highlights = [
        Span(start=m.start() - start, end=m.end() - start)
        for m in matches if start <= m.start() and m.end() <= end
    ]
    return Evidence(source=source, start=offset + start, text=passage[start:end], highlights=highlights)


_INFLECTION_QUERY = text("""
WITH qterms AS (
    SELECT array_to_string(ARRAY(
        SELECT quote_literal(word) FROM unnest(tsvector_to_array(to_tsvector('english', :terms))) word
    ), ' | ')::tsquery AS terms
)
SELECT word FROM unnest(CAST(:words AS text[])) word, qterms
WHERE qterms.terms != ''::tsquery AND to_tsvector('english', word) @@ qterms.terms
""")


def document_hits(
    session: Session,
    q: str,
    limit: int = LIMIT,
    include_content: bool = False,
    query_vector: list[float] | None = None,
) -> tuple[list[DocumentHit], str, list[float] | None]:
    lexical, semantic, corrected_q, vec = candidates(session, q, query_vector=query_vector)
    rows = fuse(lexical, semantic, limit=limit)
    if not rows:
        return [], corrected_q, vec

    words = {
        fold_accents(w)
        for row in rows
        for w in re.findall(r"[^\W_]+", row["title"] + " " + row["passage"])
    }
    inflections = set(
        session.scalars(
            _INFLECTION_QUERY,
            {"terms": fold_accents(corrected_q), "words": list(words)},
        )
    )
    hits = [
        DocumentHit(
            id=row["id"],
            client_id=row["client_id"],
            client_name=row["client_name"],
            title=row["title"],
            created_at=row["created_at"],
            summary=summarize_document(row["title"], row["content"]),
            evidence=evidence(q, row, inflections, corrected_q=corrected_q),
            content=row["content"] if include_content else None,
        )
        for row in rows
    ]
    return hits, corrected_q, vec
