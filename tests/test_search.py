import json
import os
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

BASE = os.getenv("NEVIS_TEST_URL", "http://127.0.0.1:8000")


def request(path, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    with urlopen(Request(BASE + path, data=payload, headers={"Content-Type": "application/json"}), timeout=60) as res:
        return json.load(res)


def search(q, **options):
    return request("/search?" + urlencode({"q": q, **options}))


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        serial = str(uuid4().int)[:12]
        tag = "".join(chr(ord("a") + int(d)) for d in serial[:6])
        cls.serial = serial
        cls.last_name = f"Nolan{tag}"
        cls.code = f"NEV-{serial}"
        cls.title_code = f"TTL-{serial}"
        cls.domain_token = f"MeridianWealth{tag}"
        cls.email = f"lucia.{serial}@{cls.domain_token.lower()}.example.com"
        client = request("/clients", {
            "first_name": "Lucía",
            "last_name": cls.last_name,
            "email": cls.email,
            "description": f"Wine exporter {serial}",
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

        # Case-insensitive email domain substring (e.g. NevisWealth).
        domain_hits = search(self.domain_token, type="client")
        self.assertEqual(domain_hits[0]["id"], self.client_id)
        self.assertEqual(domain_hits[0]["matched_field"], "email")

        # Accent-insensitive exact name and fuzzy typo fallback.
        name_hits = search(f"Lucia {self.last_name}", type="client")
        self.assertEqual(name_hits[0]["id"], self.client_id)
        self.assertEqual(name_hits[0]["matched_field"], "name")

        typo_hits = search(f"Lucia Nlan{self.last_name[5:]}", type="client")
        self.assertEqual(typo_hits[0]["id"], self.client_id)
        self.assertEqual(typo_hits[0]["matched_field"], "name")

        # Description match.
        desc_hits = search(f"exporter {self.serial}", type="client")
        self.assertEqual(desc_hits[0]["id"], self.client_id)
        self.assertEqual(desc_hits[0]["matched_field"], "description")

    def test_new_long_document_is_searchable_with_original_offsets(self):
        groups = search(self.code, view="grouped")
        self.assertEqual(groups["clients"], [])
        self.assertEqual([doc["id"] for doc in groups["documents"]], [self.doc["id"]])
        ev = groups["documents"][0]["evidence"]
        self.assertEqual(ev["source"], "content")
        self.assertEqual(self.content[ev["start"]:ev["start"] + len(ev["text"])], ev["text"])
        self.assertEqual([ev["text"][s["start"]:s["end"]] for s in ev["highlights"]], [self.code])

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

    def test_semantic_address_proof_finds_utility_bill(self):
        hits = search("address proof", type="document")
        self.assertTrue(hits)
        self.assertTrue(all(hit["type"] == "document" for hit in hits))
        self.assertIn(self.bill_doc["id"], [hit["id"] for hit in hits])

    def test_empty_matches_and_validation(self):
        self.assertEqual(search("%_"), [])
        self.assertEqual(search("ABS-000000123456"), [])
        for value in ("   ", "q" * 257, "a\x00"):
            with self.assertRaises(HTTPError) as error:
                search(value)
            self.assertEqual(error.exception.code, 422)
