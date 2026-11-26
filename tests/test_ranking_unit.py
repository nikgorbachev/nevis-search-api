"""Unit tests for core ranking, two-tier semantic gating, Damerau-Levenshtein, and cue localization.

Runs automatically inside the API container (where SQLAlchemy/Pydantic are installed):
  docker compose exec -T api python -m unittest discover -s tests -v
"""

import os
import unittest

try:
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials
    from app.main import require_api_token
    from app.search import _damerau_levenshtein, _expand_alias, evidence, fuse
    HAS_APP_DEPS = True
except ModuleNotFoundError:
    HAS_APP_DEPS = False


@unittest.skipUnless(HAS_APP_DEPS, "Run inside the API container to exercise app.search unit tests")
class RankingUnitTests(unittest.TestCase):
    def test_damerau_levenshtein_treats_transposition_as_distance_one(self):
        self.assertEqual(_damerau_levenshtein("courier", "cuorier"), 1)
        self.assertEqual(_damerau_levenshtein("portfolio", "portfolo"), 1)
        self.assertEqual(_damerau_levenshtein("address", "adres"), 2)
        self.assertEqual(_damerau_levenshtein("wealth", "welth"), 1)

    def test_domain_alias_expansion_handles_exact_and_typo_variants(self):
        expanded, is_alias = _expand_alias("address proof")
        self.assertTrue(is_alias)
        self.assertIn("utility bill", expanded)

        typo_expanded, is_typo_alias = _expand_alias("adres proof")
        self.assertTrue(is_typo_alias)
        self.assertIn("utility bill", typo_expanded)

        plain, is_plain_alias = _expand_alias("quarterly portfolio snapshot")
        self.assertFalse(is_plain_alias)
        self.assertEqual(plain, "quarterly portfolio snapshot")

    def test_fuse_two_tier_semantic_floor_and_rrf_ordering(self):
        lex = [
            {"id": "doc-dual", "score": 0.8},
            {"id": "doc-lex-only", "score": 0.9},
        ]
        sem = [
            {"id": "doc-sem-weak", "score": 0.65},   # semantic-only below 0.69 -> rejected
            {"id": "doc-dual", "score": 0.62},       # dual-arm >= 0.60 -> admitted & boosted
            {"id": "doc-sem-strong", "score": 0.78}, # semantic-only >= 0.69 -> admitted
        ]
        fused = fuse(lex, sem, weight=0.3, floor=0.60, semantic_only_floor=0.69, limit=10)
        ids = [r["id"] for r in fused]
        self.assertEqual(ids[0], "doc-dual")
        self.assertIn("doc-lex-only", ids)
        self.assertIn("doc-sem-strong", ids)
        self.assertNotIn("doc-sem-weak", ids)

    def test_evidence_density_selection_accent_folding_and_compact_ids(self):
        row = {
            "title": "Family transition memorandum",
            "start_char": 100,
            "passage": (
                "Introductory sentence mentioning custodian once.\n"
                "The successor custodian requires Form ZK-4729 for Elena Müller."
            ),
        }
        # Density scoring picks sentence 2 (matches successor + custodian) over sentence 1.
        ev = evidence("successor custodian", row, {"successor", "custodian"})
        self.assertEqual(ev.source, "content")
        self.assertIn("successor custodian", ev.text)
        self.assertEqual(ev.start, 100 + row["passage"].index(ev.text))

        # Accent-insensitive highlight against raw "Elena Müller" preserves exact raw span.
        ev_accent = evidence("Elena Muller", row, {"elena", "muller"})
        hl_texts = [ev_accent.text[s.start:s.end] for s in ev_accent.highlights]
        self.assertEqual(hl_texts, ["Elena", "Müller"])

        # Compact ID "ZK4729" highlights raw "ZK-4729" verbatim.
        ev_id = evidence("ZK4729", row, set())
        id_hls = [ev_id.text[s.start:s.end] for s in ev_id.highlights]
        self.assertEqual(id_hls, ["ZK-4729"])

    def test_evidence_never_fabricates_highlights_on_semantic_paraphrase(self):
        row = {
            "title": "November electricity bill",
            "start_char": 0,
            "passage": "Service supplied to Jordan Example at 12 Sample Lane.",
        }
        ev = evidence("address proof", row, set())
        self.assertEqual(ev.highlights, [])
        self.assertIn("12 Sample Lane", ev.text)

    def test_optional_bearer_auth(self):
        prev = os.environ.pop("API_TOKEN", None)
        try:
            # Open when API_TOKEN is unset.
            self.assertIsNone(require_api_token(None))

            # Enforced when API_TOKEN is set.
            os.environ["API_TOKEN"] = "secret-demo-token"
            with self.assertRaises(HTTPException) as missing:
                require_api_token(None)
            self.assertEqual(missing.exception.status_code, 401)

            with self.assertRaises(HTTPException) as wrong:
                require_api_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong"))
            self.assertEqual(wrong.exception.status_code, 401)

            self.assertIsNone(
                require_api_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-demo-token"))
            )
        finally:
            if prev is None:
                os.environ.pop("API_TOKEN", None)
            else:
                os.environ["API_TOKEN"] = prev
