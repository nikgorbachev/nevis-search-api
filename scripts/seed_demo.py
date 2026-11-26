"""Emit a repeatable PostgreSQL seed for a realistic wealth-management demo corpus.

Default scale is 120 synthetic clients (~1,600 documents with multi-chunk memos,
dirty OCR/email records, and routine operational files). Supports --clients N
(e.g. --clients 1000 for firm-scale load testing).

Run from the repo root:
  python3 scripts/seed_demo.py | docker compose exec -T db psql -X -U nevis -d nevis -v ON_ERROR_STOP=1
"""

import argparse
import csv
import sys
from datetime import date, timedelta
from random import Random
from uuid import NAMESPACE_URL, uuid5

FIRST_NAMES = "Avery,Leila,Sofia,Amir,Nora,Theo,Maya,Julian,Amara,Elias,Inez,Noah,Sana,Oscar,Rina,Felix,Clara,Nadia,Robin,Mina,Henrik,Priya,Mateo,Chloe,David".split(",")
LAST_NAMES = "Bennett,Diaz,Shah,Nielsen,Klein,Laurent,Patel,Navarro,Reed,Moreau,Silva,Cheng,Avery,Okafor,Chen,Rossi,Morgan,Silva-Brooks,Ito,Rocha,Lindqvist,Kowalski,Alvarez,Fischer,Paterson".split(",")

PROFILES = [
    ("retirement", "Preparing for retirement and reviewing recurring pension withdrawals."),
    ("business transition", "An owner planning a gradual exit from a family business."),
    ("education", "Balancing private education funding with a longer-term savings plan."),
    ("inheritance", "Organizing paperwork after an inheritance and consolidating estate accounts."),
    ("international relocation", "Preparing for a move abroad and tracking obligations in two currencies."),
    ("first home", "Building a cash reserve for a residential property purchase."),
    ("philanthropy", "Planning how to fund a multi-year charitable commitment through a donor fund."),
    ("equity compensation", "Reviewing concentrated employer shares, RSUs, and vesting schedules."),
]

TOPICS = {
    "retirement": "drawdown schedule and pension dates",
    "business transition": "sale proceeds and indemnity escrow",
    "education": "tuition installments and the cash reserve",
    "inheritance": "the estate inventory and transfer timing",
    "international relocation": "currency exposure and the move date",
    "first home": "the housing deposit and savings target",
    "philanthropy": "grant installments and the remaining pledge",
    "equity compensation": "vesting dates and exposure to employer stock",
}

SPECIALIST_NOTES = {
    "retirement": (
        "Pension drawdown schedule",
        "The first pension payment is expected in April. Hold six months of ordinary spending in cash while the provider verifies the account. "
        "Record statutory payments separately from transfers from the investment account. Required minimum distribution rules apply from age 73.",
    ),
    "business transition": (
        "Deferred consideration briefing",
        "The buyer proposes three installments following completion. Part of the second installment will remain in escrow for warranty claims. "
        "The adviser should review spending assumptions against the net amount released after capital gains tax.",
    ),
    "education": (
        "Tuition reserve worksheet",
        "The deposit for next academic year is due at acceptance. Put the first two installments in short-term deposits and retain the balance until enrollment is confirmed. "
        "Travel and accommodation sit in separate budget lines.",
    ),
    "inheritance": (
        "Estate inventory reconciliation",
        "Compare the beneficiary schedule with the executor's inventory. Some holdings are still recorded under the deceased owner's name. "
        "Archive the full statement and file the valuation note separately.",
    ),
    "international relocation": (
        "Relocation and currency handover",
        "Keep the first month of rent and the security deposit in the destination currency. Confirm whether the existing bank will retain the account after the address change. "
        "Store the destination bank instructions alongside the relocation checklist.",
    ),
    "first home": (
        "Property completion cash plan",
        "The solicitor has requested cleared funds two business days before completion. The deposit, transfer taxes, and moving expenses have separate allocations. "
        "Keep the short-term reserve available until the completion date is confirmed.",
    ),
    "philanthropy": (
        "Foundation grant schedule",
        "The first grant installment depends on receipt of the signed award letter. Schedule the remaining installments annually after the committee meeting. "
        "Keep the donation acknowledgement in the tax folder.",
    ),
    "equity compensation": (
        "Restricted stock vesting memo",
        "The next tranche releases during a trading blackout. Partner with payroll to confirm shares withheld for tax. "
        "Review single-stock exposure after the dealing window reopens.",
    ),
}


def stable_id(kind: str, key: object) -> str:
    return str(uuid5(NAMESPACE_URL, f"nevis-search-api/demo/v2/{kind}/{key}"))


def long_ips_document(index: int, name: str, currency: str, balance: int, profile: str) -> tuple[str, str]:
    """Realistic multi-section Investment Policy Statement (3-4 chunks) for segment-level evaluation."""
    risk_band = ["Conservative", "Balanced", "Growth", "Dynamic Growth"][index % 4]
    eq_wt = [35, 55, 65, 80][index % 4]
    fi_wt = 95 - eq_wt
    return (
        f"Investment Policy Statement | WM-{index:04d}",
        "\n\n".join([
            f"Section 1 — Client Mandate & Objectives\n"
            f"This Investment Policy Statement governs the discretionary advisory portfolio for {name} (Account WM-{index:04d}). "
            f"The household profile centers on {profile}, with an indicative investable balance of {balance:,} {currency}. "
            f"Primary objectives are real capital preservation after inflation and disciplined liquidity management ahead of planned household cash flows. "
            f"The assigned risk classification is {risk_band}, reviewed annually or upon any material change in family circumstances.",
            f"Section 2 — Strategic Asset Allocation & Rebalancing Bands\n"
            f"Target allocation is set at {eq_wt}% global equities, {fi_wt}% investment-grade fixed income, and 5% cash equivalents. "
            f"Portfolio weights are monitored monthly and rebalanced automatically whenever any asset class drifts by more than 5 percentage points from target. "
            f"Single-issuer equity exposure may not exceed 10% of total liquid net worth without written client acknowledgement. "
            f"Low-cost index funds and direct sovereign bonds are preferred to minimize fee drag across taxable accounts.",
            f"Section 3 — Tax Efficiency, Liquidity & Restrictions\n"
            f"Advisers will apply tax-loss harvesting quarterly where realized losses can offset short-term capital gains without violating wash-sale windows. "
            f"At least twelve months of scheduled distributions must remain in money-market instruments or short-dated treasury bills. "
            f"Holdings in tobacco, thermal coal, and unhedged leveraged ETFs are excluded under the client's mandate restrictions. "
            f"All cross-border transfers require pre-clearance with the operations desk and updated W-9 or CRS self-certification on file.",
            f"Section 4 — Governance, Reporting & Custody\n"
            f"Quarterly performance reports benchmark net-of-fee returns against a blended {eq_wt}/{fi_wt} MSCI ACWI and Bloomberg Global Aggregate index. "
            f"Fee billing is calculated quarterly in arrears based on average daily assets under management. "
            f"Any change to authorized signatories, power of attorney designations, or successor trustee records must be countersigned by compliance before execution.",
        ]),
    )


def dirty_email_document(index: int, name: str, topic: str, currency: str, cash: int) -> tuple[str, str]:
    """Realistic messy forwarded email thread with headers, quote markers, and OCR-style formatting."""
    return (
        f"Fwd: Quick question on WM-{index:04d} paperwork",
        f"From: {name} <client{index}@mail.example.com>\n"
        f"Sent: Tuesday, 14 October 2025 09:42\n"
        f"To: Advisory Desk <advisers@neviswealth.example.com>\n"
        f"Subject: Fwd: Re: {topic}\n"
        f"-----Original Message-----\n"
        f"> > Hi team,\n"
        f"> > Please keep {currency}{cash:,} ready while we review {topic}.\n"
        f"> > Also attaching the ﬁnancial statement and updated KYC conﬁrmation.\n"
        f"> > Let me know if the custod-\nian needs a wet signature.\n"
        f"==============================\n"
        f"Best regards,\n{name}",
    )


def routine_documents(index: int, name: str, profile: str):
    rng = Random(6100 + index)
    topic = TOPICS[profile]
    currency = rng.choice(["EUR", "GBP", "USD", "CHF"])
    starting_balance = rng.randrange(45_000, 1_200_000, 500)
    for period in range(2 + index % 3):
        day = date(2025, 8, 15) + timedelta(days=30 * period + index % 12)
        balance = starting_balance + rng.randrange(-4_500, 9_000, 100)
        cash = balance // rng.randint(8, 17)
        yield (
            f"Portfolio snapshot | {day:%b %Y}",
            f"Prepared for {name} as of {day:%d %B %Y}.\n\n"
            f"Reporting currency: {currency}\nAccount reference: WM-{index:04d}\n"
            f"Portfolio market value: {balance:,} {currency}\nAvailable cash: {cash:,} {currency}\n"
            f"Open planning item: {topic}. Market values are indicative until custodian reconciliation.",
        )
        note = rng.choice(
            [
                f"Agreed to wait for a revised timeline before moving funds. Review {topic} before our next call.",
                f"Client asked for a written explanation of {topic}. The adviser will send a draft after checking the custodian statement.",
                f"Updated our assumptions for {topic}. Client will email the supporting paperwork before the review.",
                f"Discussed two scenarios for {topic}. Current instructions remain in force pending confirmation.",
            ]
        )
        yield (
            f"Adviser meeting notes | {day:%d %b %Y}",
            f"{name} / planning review\n\n{note}\n\nFollow-up owner: client service. "
            f"Next review: {(day + timedelta(days=21)):%d %B %Y}.",
        )
        if (index + period) % 2 == 0:
            yield (
                f"Custodian reconciliation | {day:%b %Y}",
                "OPERATIONS WORKSHEET\n"
                f"Account\tWM-{index:04d}\nCurrency\t{currency}\n"
                f"Statement cash\t{cash:,}\nLedger cash\t{cash:,}\nDifference\t0\n"
                f"Checked on\t{day.isoformat()}\nStatus\tClosed",
            )

    yield (
        "Risk questionnaire extract",
        f"Client: {name}\nTime horizon: {rng.randint(3, 20)} years\n"
        f"Liquidity reserve: {cash:,} {currency}\n"
        f"Primary consideration: {topic}. Adviser to confirm assumptions at annual review.",
    )
    yield (
        "Onboarding checklist",
        f"Household: {name}\nReference: WM-{index:04d}\n"
        "Identity confirmed via passport scan (KYC complete). Mailing details entered into the client register. "
        "Pending items are tracked in the planning notes.",
    )

    # Multi-section IPS for 20% of clients (creates real multi-chunk segment competition).
    if index % 5 == 0:
        yield long_ips_document(index, name, currency, starting_balance, profile)

    # Messy forwarded client email for 25% of clients (exercises normalization layer).
    if index % 4 == 0:
        yield dirty_email_document(index, name, topic, currency, cash)

    one_offs = [
        ("Storage inventory photograph notes", "Folder labels visible include 2022, Taxes, and Household. Photograph filed with the intake packet."),
        ("Courier pickup receipt", f"Sealed envelope collected at reception. Tracking code: CR-{index:05d}. Contents were recorded as correspondence."),
        ("Event registration confirmation", "Booked a seat at the Thursday estate-planning seminar. Accessibility request forwarded to the venue."),
        ("Office access instructions", "Use the courtyard entrance after 16:00. The lift is beside the concierge desk. Visitor badge required."),
        ("Warranty card transcription", "Equipment serial recorded on the lower panel. Registration status pending. Original card held by the client."),
        ("Translated handwritten note", "Please leave the signed copy with reception and call before sending the original overseas."),
        ("Archived travel itinerary", "Evening rail service departs from platform four. Seat reservation is on the second page. Booking reference retained for expenses."),
        ("Workshop seating sketch", "Front row reserved for speakers. Mark the east door as the accessible entrance. Keep the folding tables near the wall."),
    ]
    yield rng.choice(one_offs)

    if index % 3 == 0:
        yield (
            "Miscellaneous scanning note",
            f"Packet WM-{index:04d} includes a folded cover page, two envelopes, and a handwritten annotation. Scan delivered to the service team.",
        )

    title, body = SPECIALIST_NOTES[profile]
    yield title, f"Planning note for {name}\n\n{body}"


# Stable v1 UUIDs preserved for the 4 anchor clients and their flagship fixtures so README examples remain deterministic.
def legacy_anchor_id(kind: str, key: object) -> str:
    return str(uuid5(NAMESPACE_URL, f"nevis-search-api/demo/v1/{kind}/{key}"))


ANCHORS = [
    (
        "Jordan", "Example", "jordan.doe@neviswealth.example.com", "Annual planning and onboarding documentation.",
        [
            ("November electricity bill", "North Quay Electricity\nAccount: ELEC-5197\nBill issued: 12 November 2026\n\n"
             "Service supplied to Jordan Example at 12 Sample Lane, Harbor View. The service location is the same as the billing location.\n\n"
             "Meter 4311 | Billing period 1–30 November | Amount due EUR 58.20"),
            ("Invoice approval checklist", "Supplier invoices are approved against the purchase order and submitted to accounts payable. "
             "Enter the invoice total and document the reconciliation. The procurement contact is stored on the vendor profile."),
            ("Electricity rates comparison", "Compare estimated energy prices for three districts using standing fees and per-unit charges. "
             "The figures describe tariffs for a sample household and the worksheet contains no customer account."),
        ],
    ),
    (
        "Elena", "Müller", "elena.muller@wealth-demo.example.com", "Business transition and family reserve planning.",
        [
            ("Family transition memorandum", "\n\n".join([
                "Engagement scope\nThe family asked for a single written record of decisions made during the first two planning meetings. The scope covers the timing of the business sale, the cash reserve and the paperwork needed for a transfer of ownership.",
                "Sale proceeds\nThe working estimate assumes a staggered payment. The first installment would be held in an interest-bearing account pending settlement. The family wants to earmark enough money to cover school fees and ordinary household expenses through the following autumn.",
                "Liquidity plan\nThe household intends to defer any large purchase until the second installment is confirmed. The advisers agreed to keep a rolling forecast with quarterly dates and possible early withdrawals. Local tax payments and administrative costs should be recorded as separate items.",
                "Custody and transfer\nThe successor custodian will accept the transfer only after receiving the signed letter of direction. Form ZK-4729 must accompany the transfer to the successor custodian. The original certification will remain in the client file with the adviser’s signed checklist.",
                "Review schedule\nOperations will confirm receipt of the paperwork before the next meeting. The agenda includes a review of beneficiary designations and a discussion of whether surplus cash should be assigned to an education reserve.",
            ])),
            ("Transfer checklist draft", "Check the destination custodian, attach the signed client instruction, and confirm that the current provider has acknowledged the request. The product team will review this generic checklist next month."),
        ],
    ),
    (
        "Marin", "Reed", "marin.reed@wealth-demo.example.com", "Concentrated employer stock and retirement transition.",
        [
            ("Annual advisory fee notice", "The annual advisory fee is 2.5% of assets on the covered account. Fees are assessed quarterly in arrears. The estimate excludes taxes and third-party custody charges."),
            ("Portfolio concentration worksheet", "Employer stock share: 25% of the model portfolio. The worksheet tracks price sensitivity and shares to be sold after vesting."),
            ("Recurring cash transfer instructions", "Please send 1,250 EUR on the first business day of each month to cover household spending following retirement. Retain a full year of planned withdrawals in reserve."),
        ],
    ),
    (
        "Maren", "Reed", "maren.reed@wealth-demo.example.com", "Charitable giving and a planned relocation.",
        [
            ("Gift letter extract", "The trustees agreed to fund the education grant in five annual installments. Retain a signed copy in the foundation register."),
            ("Household contact update", "Use the temporary correspondence location after the move. This note records a request for an update. The onboarding team still needs a supporting statement from a service provider."),
        ],
    ),
    (
        "Robert", "Vance", "bob.vance@vance-holdings.example.com", "Family office governance, irrevocable trust administration, and private equity commitments.",
        [
            ("Irrevocable Trust Deed Summary", "\n\n".join([
                "Article I — Trust Establishment & Grantor\n"
                "The Vance Family Irrevocable Trust was settled on 10 March 2022 by Robert \"Bob\" Vance for the benefit of his spouse and two adult children. "
                "The trust holds minority operating shares in Vance Family Holdings LLC alongside a diversified liquid investment portfolio managed under a conservative mandate. "
                "All trust accounting is maintained on a calendar-year basis with annual K-1 schedules issued by external tax counsel.",
                "Article II — Trustees & Power of Attorney\n"
                "Harbor Fiduciary Services serves as corporate co-trustee alongside Claire Vance as family trustee. "
                "Under Section 4.2, a durable power of attorney executed under form POA-8814 authorizes Claire Vance to execute routine custody instructions during international travel. "
                "Any replacement of the successor trustee requires unanimous written consent of the adult beneficiaries.",
                "Article III — Distributions & Tax Compliance\n"
                "Discretionary distributions for health, education, maintenance, and support are reviewed quarterly by the distribution committee. "
                "Before releasing distributions exceeding 50,000 USD, operations must verify current taxpayer identification number (Form W-9) records and confirm absence of outstanding capital calls. "
                "Annual tax-loss harvesting reports are archived with the family office audit file.",
                "Article IV — Private Market Commitments\n"
                "The trust maintains a 2,500,000 USD limited partner commitment to Northstar Infrastructure Fund IV. "
                "Capital call notice CAP-3092 requests 12.5% of unfunded commitment (312,500 USD) wired within ten business days to the fund escrow account. "
                "Recallable distributions are tracked separately from ordinary income.",
            ])),
            ("OCR Scan — Municipal Water & Power Statement",
             "M U N I C I P A L   U T I L I T Y   B I L L\n"
             "Account Ref: UTIL-7741 | Service Period: October 2026\n"
             "Customer: Robert Vance, 44 Cedar Ridge Road, Northbrook\n"
             "Note: ﬁnal meter reading veriﬁed on site. Elec-\ntricity and water charges total EUR142.80.\n"
             "Late fee rate: 1.5 % per month after due date."),
        ],
    ),
]


def corpus(client_count: int):
    """Yield synthetic clients with their text documents."""
    for index in range(1, client_count + 1):
        if index <= len(ANCHORS):
            first, last, email, description, extra = ANCHORS[index - 1]
            profile = ["first home", "business transition", "retirement", "philanthropy", "inheritance"][index - 1]
            id_fn = legacy_anchor_id if index <= 4 else stable_id
        else:
            offset = index - len(ANCHORS) - 1
            first = FIRST_NAMES[offset % len(FIRST_NAMES)]
            last = LAST_NAMES[(offset // len(FIRST_NAMES) + offset % len(FIRST_NAMES)) % len(LAST_NAMES)]
            email = f"{first.lower()}.{last.lower()}.{index:04d}@wealth-demo.example.com"
            profile, description = PROFILES[index % len(PROFILES)]
            extra = []
            id_fn = stable_id
        name = f"{first} {last}"
        client_id = id_fn("client", index)
        yield (
            (client_id, first, last, email, description, "[]"),
            [
                (id_fn("document", f"{index}/{number}"), client_id, title, content)
                for number, (title, content) in enumerate([*extra, *routine_documents(index, name, profile)], 1)
            ],
        )


def copy_rows(table: str, rows):
    print(f"COPY {table} FROM STDIN WITH (FORMAT CSV);")
    csv.writer(sys.stdout, lineterminator="\n", quoting=csv.QUOTE_ALL).writerows(rows)
    print(r"\.")


def main():
    parser = argparse.ArgumentParser(description="Emit deterministic PostgreSQL seed SQL for Nevis Search API.")
    parser.add_argument("--clients", type=int, default=120, help="Number of synthetic clients to generate (default: 120)")
    args = parser.parse_args()

    clients, documents = zip(*corpus(args.clients))
    print("BEGIN;")
    print("CREATE TEMP TABLE demo_clients (id text, first_name text, last_name text, email text, description text, social_links json) ON COMMIT DROP;")
    print("CREATE TEMP TABLE demo_documents (id text, client_id text, title text, content text) ON COMMIT DROP;")
    copy_rows("demo_clients (id, first_name, last_name, email, description, social_links)", clients)
    copy_rows("demo_documents (id, client_id, title, content)", (doc for group in documents for doc in group))
    print("""
INSERT INTO clients (id, first_name, last_name, email, description, social_links)
SELECT id, first_name, last_name, email, description, social_links FROM demo_clients
ON CONFLICT (id) DO NOTHING;

INSERT INTO documents (id, client_id, title, content)
SELECT id, client_id, title, content FROM demo_documents
ON CONFLICT (id) DO NOTHING;

SELECT count(*) AS demo_clients FROM clients JOIN demo_clients USING (id);
SELECT count(*) AS demo_documents FROM documents JOIN demo_documents USING (id);
COMMIT;
""")


if __name__ == "__main__":
    main()
