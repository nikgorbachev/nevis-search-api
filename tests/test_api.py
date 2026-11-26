import json
import os
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("NEVIS_TEST_URL", "http://127.0.0.1:8000")


class CreateTests(unittest.TestCase):
    def post(self, path, payload):
        request = Request(
            BASE_URL + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_create_client_and_document(self):
        client_payload = {
            "first_name": "Jordan",
            "last_name": "Example",
            "email": "jordan@example.com",
        }
        status, client = self.post("/clients", client_payload)
        self.assertEqual(status, 201, client)
        self.assertEqual({key: client[key] for key in client_payload}, client_payload)
        self.assertIsNone(client["description"])
        self.assertEqual(client["social_links"], [])

        document_payload = {"title": "Utility bill", "content": "Amount due: EUR 58.20"}
        status, document = self.post(
            f"/clients/{client['id']}/documents", document_payload
        )
        self.assertEqual(status, 201, document)
        self.assertEqual(document["client_id"], client["id"])
        self.assertEqual(document["content"], document_payload["content"])
        self.assertTrue(document["created_at"])

    def test_missing_client(self):
        status, body = self.post(
            "/clients/missing-client/documents",
            {"title": "Utility bill", "content": "Example bill"},
        )
        self.assertEqual(status, 404, body)

    def test_invalid_requests(self):
        status, body = self.post(
            "/clients",
            {"first_name": "Jordan", "last_name": "Example", "email": "invalid"},
        )
        self.assertEqual(status, 422, body)

        status, body = self.post("/clients/missing-client/documents", {"title": "Bill"})
        self.assertEqual(status, 422, body)

    def test_missing_document_returns_404(self):
        try:
            urlopen(BASE_URL + "/documents/missing-doc", timeout=10)
            self.fail("Expected HTTP 404 for missing document")
        except HTTPError as error:
            self.assertEqual(error.code, 404)
