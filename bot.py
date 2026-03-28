#!/usr/bin/env python3
"""Slack bot that answers Metabase questions using Claude + raw SQL."""

import json
import os
import re
import logging

import anthropic
import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

logging.basicConfig(level=logging.INFO)

# -- Config ------------------------------------------------------------------

METABASE_URL = os.environ["METABASE_URL"]
METABASE_API_KEY = os.environ["METABASE_API_KEY"]
MB_HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

app = App(token=os.environ["SLACK_BOT_TOKEN"])
claude = anthropic.Anthropic()

INTERNAL_ORGS = (
    "c93177dc-55ee-4219-8c15-d1612e121e91",
    "9ca37ea9-1122-429b-9b60-f36f62fbf3dc",
)

SCHEMA = """
## PostgreSQL (database_id=3) — Linkup DB

### users
id (uuid), created_at (timestamp), name (varchar), email (varchar), referral_source (enum), use_case (enum), last_active_organization_id (uuid)

### organizations
id (uuid), created_at (timestamp), name (varchar), stripe_customer_id (text), is_partner (bool), is_paying_customer (bool), slug (text)

### members
id (uuid), organization_id (uuid), user_id (uuid), role (enum), created_at (timestamp)

### search_logs
id (uuid), created_at (timestamp), question (varchar), answer (varchar), time (int4, latency ms), organization_id (uuid), service_account_id (uuid), depth (varchar: fast/standard/deep/research), output_type (varchar), iterations (int4), from_date (timestamp), to_date (timestamp), include_images (bool), brain_version (varchar), request_id (varchar), include_inline_citations (bool), max_results (int4)
Note: only ~7 days retained. Use Lakehouse for historical.

### service_accounts
id (uuid), created_at (timestamp), key (uuid), name (varchar), organization_id (uuid), budget_period (enum), budget (float8)

### wallets
id (uuid), balance (numeric), organization_id (uuid), auto_refill_enabled (bool), notification_threshold (numeric)

### wallet_transactions
id (uuid), amount (numeric), type (enum: deposit, automatic_top_up, api_usage:search, api_usage:fetch, api_usage:research, api_usage:x402, monthly_gift), status (enum: completed, failed, pending), wallet_id (uuid), search_log_id (uuid), created_at (timestamp), service_account_id (uuid)
Note: 24M+ rows, Feb 2025+. Use for historical query counts & revenue.

### teams
id (uuid), name (varchar), organization_id (uuid)

### team_members
id (uuid), team_id (uuid), user_id (uuid)

## Databricks Lakehouse (database_id=5)

### bronze.api_search_logs
Same schema as PG search_logs but historical (June 2025+, 21M+ rows). No org names — only organization_id.

### gold.fetch_logs
id, organization_id, service_account_id, url, render_js, extract_images, created_at

### gold.search_logs
Same as bronze but cleaned.

## Important notes
- Internal org IDs to ALWAYS exclude from rankings: {internal_orgs}
- wallet_transactions.type is an enum — use IN (...) not LIKE
- Cross-database joins not possible
- For historical query counts: join wallet_transactions (type IN ('api_usage:search','api_usage:fetch','api_usage:research')) with wallets → organizations
- Pre-aggregate in CTEs for large tables
""".format(internal_orgs=", ".join(INTERNAL_ORGS))

SYSTEM_PROMPT = f"""You are a data analyst bot for Linkup. You answer questions by writing SQL queries against our Metabase instance.

{SCHEMA}

When the user asks a question:
1. Write a SQL query to answer it.
2. Return ONLY a JSON object (no markdown fences) with:
   {{"database": 3, "sql": "SELECT ..."}}
   Use database 3 for Postgres, 5 for Lakehouse.

Rules:
- Always exclude internal orgs from rankings/aggregations.
- Use parameterized intervals like INTERVAL 'N days'.
- Keep queries efficient — use LIMIT, CTEs, avoid SELECT *.
- For "last N days" on Postgres search_logs, remember only ~7 days exist.
- If the question cannot be answered with SQL, return: {{"error": "explanation"}}
"""

SUMMARY_PROMPT = """You are a concise data analyst. Given the user's question and query results, write a short Slack message with the answer. Use markdown formatting suitable for Slack (bold with *, code blocks with ```, no tables — use aligned text or bullet points instead). Keep it brief and insightful."""


# -- Metabase ----------------------------------------------------------------

def run_query(sql: str, database: int = 3) -> tuple[list[str], list[list]]:
    resp = requests.post(
        f"{METABASE_URL}/dataset",
        headers=MB_HEADERS,
        json={"database": database, "type": "native", "native": {"query": sql}},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    cols = [c["name"] for c in data["cols"]]
    return cols, data["rows"]


# -- Claude ------------------------------------------------------------------

def ask_claude(messages: list, system: str = "") -> str:
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def generate_sql(question: str) -> dict:
    raw = ask_claude(
        [{"role": "user", "content": question}],
        system=SYSTEM_PROMPT,
    )
    # Strip markdown fences if Claude adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


def summarize_results(question: str, cols: list, rows: list) -> str:
    # Truncate to avoid token limits
    preview = rows[:50]
    data_str = json.dumps(
        [dict(zip(cols, r)) for r in preview], indent=2, default=str
    )
    if len(rows) > 50:
        data_str += f"\n... and {len(rows) - 50} more rows"

    return ask_claude(
        [
            {
                "role": "user",
                "content": f"Question: {question}\n\nQuery returned {len(rows)} rows. Data:\n{data_str}",
            }
        ],
        system=SUMMARY_PROMPT,
    )


# -- Slack handlers ----------------------------------------------------------

@app.event("app_mention")
def handle_mention(event, say):
    question = re.sub(r"<@[A-Z0-9]+>\s*", "", event["text"]).strip()
    if not question:
        say("Ask me a question about our data! e.g. _top orgs by query volume this week_")
        return

    say(f":mag: Looking into: _{question}_")

    try:
        result = generate_sql(question)

        if "error" in result:
            say(f":warning: {result['error']}")
            return

        sql = result["sql"]
        db = result.get("database", 3)
        cols, rows = run_query(sql, database=db)

        if not rows:
            say(":shrug: Query returned no results.")
            return

        summary = summarize_results(question, cols, rows)
        say(summary)

    except json.JSONDecodeError:
        say(":x: I couldn't generate a valid query for that. Try rephrasing?")
    except requests.exceptions.HTTPError as e:
        say(f":x: Query failed: `{e}`")
    except Exception as e:
        logging.exception("Error handling question")
        say(f":x: Something went wrong: `{e}`")


@app.event("message")
def handle_dm(event, say):
    # Handle DMs (no mention needed)
    if event.get("channel_type") == "im" and not event.get("bot_id"):
        handle_mention(event, say)


# -- Main --------------------------------------------------------------------

def main():
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    print("Bot is running!")
    handler.start()


if __name__ == "__main__":
    main()
