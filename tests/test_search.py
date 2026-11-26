import json
import os
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

BASE = os.getenv("NEVIS_TEST_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("NEVIS_API_TOKEN")


def request(path, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    with urlopen(Request(BASE + path, data=payload, headers=headers), timeout=60) as res:
        return json.load(res)


def search(q, **options):
    return request("/search?" + urlencode({"q": q, **options}))


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serial = str(uuid4().int)[:12]
        tag = "".join(chr(ord("a") + int(d)) for d in serial[:6])
        cls.serial = serial
        cls.tag = tag
        cls.last_name = f"Nolan{tag}"
        cls.code = f"NEV-{serial}"
        cls.compact_code = f"NEV{serial}"
        cls.title_code = f"TTL-{serial}"
        cls.domain_token = f"MeridianWealth{tag}"
        cls.email = f"lucia.{serial}@{cls.domain_token.lower()}.example.com"
        client = request("/clients", {
            "first_name": "Lucía",
            "last_name": cls.last_name,
            "email": cls.email,
            "description": f"Vineyard owner and wine exporter {serial}",
            "social_links": ["https://example.com/lucia"],
        })
        cls.client_id = client["id"]
        cls.content = ("Review the quarterly statements. " * 140 +
                       f"\nCustody office 🗂 requests form {cls.code} before closing.")
        cls.doc = request(f"/clients/{cls.client_id}/documents",
                          {"title": "Transfer decision", "content": cls.content})
        cls.title_doc = request(f"/clients/{cls.client_id}/documents", {
            "title": f"Charter {cls.title_code}",
            "content": "Routine administrative filing.",
        })
        cls.bill_doc = request(f"/clients/{cls.client_id}/documents", {
            "title": f"Power statement {serial}",
            "content": f"Electricity utility bill for {cls.last_name}\nBilling period: October 2026\nAmount due: EUR 42.10",
        })
        cls.dirty_doc = request(f"/clients/{cls.client_id}/documents", {
            "title": f"Scanned notice {serial}",
            "content": (
                "From: Ops <ops@example.com>\n"
                "Subject: Re: Scan\n"
                "-----Original Message-----\n"
                f"> > U T I L I T Y   R E C O R D {serial}\n"
                f"> > The ﬁnancial custod-\nian veriﬁed account {cls.last_name} with rate 6.75 % and fee EUR91.40."
            ),
        })
        cls.rate_small = request(f"/clients/{cls.client_id}/documents", {
            "title": f"Fee schedule {serial}",
            "content": "Management fee is 7.25% per annum on eligible balances.",
        })
        cls.rate_large = request(f"/clients/{cls.client_id}/documents", {
            "title": f"Allocation limit {serial}",
            "content": "Single-position cap is 72.5% of the restricted sleeve.",
        })

    def test_public_contract_and_client_matching(self):
        hits = search(self.email, type="client")
        self.assertEqual(hits[0]["id"], self.client_id)
        self.assertEqual(hits[0]["matched_field"], "email")
        self.assertEqual(hits[0]["type"], "client")
        self.assertEqual(hits[0]["description"], f"Vineyard owner and wine exporter {self.serial}")
        self.assertEqual(hits[0]["social_links"], ["https://example.com/lucia"])

        # Case-insensitive email domain substring and typoed email domain.
        domain_hits = search(self.domain_token, type="client")
        self.assertEqual(domain_hits[0]["id"], self.client_id)
        self.assertEqual(domain_hits[0]["matched_field"], "email")

        typo_domain = self.domain_token.replace("Wealth", "Welth")
        typo_domain_hits = search(typo_domain, type="client")
        self.assertEqual(typo_domain_hits[0]["id"], self.client_id)

        # Accent-insensitive exact name + single-token & multi-token Levenshtein typo fallback.
        name_hits = search(f"Lucia {self.last_name}", type="client")
        self.assertEqual(name_hits[0]["id"], self.client_id)
        self.assertEqual(name_hits[0]["matched_field"], "name")

        typo_full = search(f"Lucia Nlan{self.last_name[5:]}", type="client")
        self.assertEqual(typo_full[0]["id"], self.client_id)

        typo_last_only = search(f"Nlan{self.last_name[5:]}", type="client")
        self.assertEqual(typo_last_only[0]["id"], self.client_id)

        # Description match.
        desc_hits = search(f"exporter {self.serial}", type="client")
        self.assertEqual(desc_hits[0]["id"], self.client_id)
        self.assertEqual(desc_hits[0]["matched_field"], "description")

    def test_long_document_compact_id_and_raw_offsets(self):
        # Exact hyphenated form code + compact form code without hyphen.
        for query in (self.code, self.compact_code):
            hits = search(query, type="document")
            self.assertEqual([doc["id"] for doc in hits], [self.doc["id"]])
            hit = hits[0]
            self.assertEqual(hit["type"], "document")
            self.assertTrue(hit["created_at"])
            self.assertTrue(hit["summary"])
            self.assertIsNone(hit["content"])
            ev = hit["evidence"]
            self.assertEqual(ev["source"], "content")
            self.assertEqual(self.content[ev["start"]:ev["start"] + len(ev["text"])], ev["text"])
            self.assertEqual([ev["text"][s["start"]:s["end"]] for s in ev["highlights"]], [self.code])

        # Optional include_content=true and GET /documents/{id}.
        with_content = search(self.code, type="document", include_content="true")
        self.assertEqual(with_content[0]["content"], self.content)
        fetched = request(f"/documents/{self.doc['id']}")
        self.assertEqual(fetched["content"], self.content)

    def test_dirty_ocr_normalization_preserves_raw_offsets_and_clean_summary(self):
        # Normalized indexing resolves ligature "ﬁnancial", line-wrapped "custod-\nian", and spaced "6.75 %".
        hits = search(f"financial custodian {self.last_name}", type="document")
        self.assertTrue(hits)
        target = next((h for h in hits if h["id"] == self.dirty_doc["id"]), None)
        self.assertIsNotNone(target)
        # Clean summary strips email headers and quote markers.
        self.assertNotIn("From:", target["summary"])
        self.assertNotIn("> >", target["summary"])
        # Evidence text still aligns verbatim to raw stored content.
        ev = target["evidence"]
        self.assertEqual(self.dirty_doc["content"][ev["start"]:ev["start"] + len(ev["text"])], ev["text"])

        # Spaced percentage "6.75 %" normalized to "6.75%".
        pct_hits = search("6.75%", type="document")
        self.assertIn(self.dirty_doc["id"], [h["id"] for h in pct_hits])

    def test_title_evidence_and_percentage_boundaries(self):
        title_hits = search(self.title_code, type="document")
        self.assertEqual([doc["id"] for doc in title_hits], [self.title_doc["id"]])
        ev = title_hits[0]["evidence"]
        self.assertEqual(ev["source"], "title")
        self.assertEqual(ev["start"], 0)
        self.assertEqual([ev["text"][s["start"]:s["end"]] for s in ev["highlights"]], [self.title_code])

        small_hits = search("7.25%", type="document")
        small_ids = [doc["id"] for doc in small_hits]
        self.assertIn(self.rate_small["id"], small_ids)
        self.assertNotIn(self.rate_large["id"], small_ids)
        small_ev = next(doc for doc in small_hits if doc["id"] == self.rate_small["id"])["evidence"]
        self.assertIn("7.25%", [small_ev["text"][s["start"]:s["end"]] for s in small_ev["highlights"]])

        large_hits = search("72.5%", type="document")
        large_ids = [doc["id"] for doc in large_hits]
        self.assertIn(self.rate_large["id"], large_ids)
        self.assertNotIn(self.rate_small["id"], large_ids)

    def test_semantic_address_proof_and_typo_alias(self):
        for query in ("address proof", "adres proof"):
            hits = search(query, type="document")
            self.assertTrue(hits, query)
            self.assertTrue(all(hit["type"] == "document" for hit in hits))
            self.assertIn(self.bill_doc["id"], [hit["id"] for hit in hits], query)

    def test_empty_matches_near_domain_negatives_and_validation(self):
        self.assertEqual(search("%_"), [])
        self.assertEqual(search("ABS-000000123456"), [])
        self.assertEqual(search("cryptocurrency staking yield", type="document"), [])
        self.assertEqual(search("options Greeks delta hedging", type="document"), [])
        self.assertEqual(search("Swiss numbered bank account IBAN", type="document"), [])
        for value in ("   ", "q" * 257, "a\x00"):
            with self.assertRaises(HTTPError) as error:
                search(value)
            self.assertEqual(error.exception.code, 422)
