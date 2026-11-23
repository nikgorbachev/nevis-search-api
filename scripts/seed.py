"""Create a synthetic client and utility bill through the running API."""

import json
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    request = Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        if response.status != 201:
            raise RuntimeError(f"Expected HTTP 201 for {path}, got {response.status}")
        return json.load(response)


def main() -> None:
    client = post(
        "/clients",
        {
            "first_name": "Jordan",
            "last_name": "Example",
            "email": "jordan@example.com",
            "description": "Synthetic client for local testing",
            "social_links": ["https://example.com/jordan"],
        },
    )
    document = post(
        f"/clients/{client['id']}/documents",
        {
            "title": "November electricity bill",
            "content": "Electricity utility bill\nJordan Example\n12 Sample Lane\n"
            "Billing period: November 2026\nAmount due: EUR 58.20",
        },
    )
    print(json.dumps({"client": client, "document": document}, indent=2))


if __name__ == "__main__":
    main()
