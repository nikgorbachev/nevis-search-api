"""Pure stdlib unit tests for the text normalization and extractive summary layer."""

import unittest

from app.normalize import (
    extract_vocab_words,
    fold_accents,
    normalize_for_index,
    normalize_text,
    summarize_document,
)


class NormalizeUnitTests(unittest.TestCase):
    def test_nfkc_ligatures_hyphenation_and_ocr_spacing(self):
        raw = "U T I L I T Y   B I L L\nThe ﬁnancial custod-\nian veriﬁed EUR142.80 at 1.5 %.\u200b"
        cleaned = normalize_text(raw)
        self.assertIn("UTILITY BILL", cleaned)
        self.assertIn("financial custodian verified", cleaned)
        self.assertIn("EUR 142.80", cleaned)
        self.assertIn("1.5%", cleaned)
        self.assertNotIn("\u200b", cleaned)

    def test_email_quotes_and_dividers_stripped(self):
        raw = "> > Hi team,\n-----Original Message-----\n> Please send the report.\n=========="
        cleaned = normalize_text(raw)
        self.assertNotIn(">", cleaned)
        self.assertNotIn("-----", cleaned)
        self.assertNotIn("====", cleaned)
        self.assertIn("Hi team,", cleaned)
        self.assertIn("Please send the report.", cleaned)

    def test_fold_accents_and_compact_form_codes(self):
        self.assertEqual(fold_accents("Elena Müller"), "elena muller")
        indexed = normalize_for_index("Form ZK-4729 and Account ELEC-5197 for Elena Müller")
        self.assertIn("zk-4729", indexed)
        self.assertIn("zk4729", indexed)
        self.assertIn("elec5197", indexed)
        self.assertIn("elena muller", indexed)

    def test_extract_vocab_words_filters_short_and_numeric_tokens(self):
        words = extract_vocab_words("Form ZK-4729 for Elena Müller: 2.5% fee on EUR 58.20")
        self.assertIn("elena", words)
        self.assertIn("muller", words)
        self.assertIn("form", words)
        self.assertNotIn("fee", words)  # len < 4
        self.assertNotIn("4729", words)

    def test_summarize_document_strips_email_headers_and_truncates_cleanly(self):
        raw = (
            "From: Bob <bob@example.com>\n"
            "Sent: Tuesday\n"
            "Subject: Re: Trust\n"
            "-----Original Message-----\n"
            "> > First substantive sentence about trust administration. "
            "Second sentence covering quarterly distributions. "
            "Third very long sentence that should not force mid-word truncation."
        )
        summary = summarize_document("Trust email", raw, max_chars=120)
        self.assertNotIn("From:", summary)
        self.assertNotIn("Subject:", summary)
        self.assertNotIn("> >", summary)
        self.assertTrue(summary.startswith("First substantive sentence"))
        self.assertLessEqual(len(summary), 120)

    def test_summarize_falls_back_to_title_when_content_empty(self):
        self.assertEqual(summarize_document("Quarterly Memo", "   "), "Quarterly Memo")
