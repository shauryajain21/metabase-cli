#!/usr/bin/env python3
"""Agentic Slack bot for querying Linkup's Metabase instance.

Features:
- Thread-based conversations with full context from Slack API
- Two-loop agentic system: SQL retry (inner) + quality gate (outer)
- Thinking indicator that updates in-place
- Fuzzy org name matching with auto-discovery
"""

import json
import os
import re
import logging
from datetime import datetime, timezone

import anthropic
import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -- Config ------------------------------------------------------------------

METABASE_URL = os.environ["METABASE_URL"]
METABASE_API_KEY = os.environ["METABASE_API_KEY"]
MB_HEADERS = {"x-api-key": METABASE_API_KEY, "Content-Type": "application/json"}

app = App(token=os.environ["SLACK_BOT_TOKEN"])
claude = anthropic.Anthropic()

MAX_SQL_RETRIES = 3

INTERNAL_ORGS = (
    "c93177dc-55ee-4219-8c15-d1612e121e91",
    "9ca37ea9-1122-429b-9b60-f36f62fbf3dc",
)

# Bot user ID — set at startup
BOT_USER_ID = None

# Store last SQL per thread for "explain" command
thread_sql: dict[str, str] = {}

SCHEMA = """
## PostgreSQL (database_id=3) — Linkup DB

### users
id (uuid), created_at (timestamp), name (varchar), email (varchar), referral_source (enum), use_case (enum), last_active_organization_id (uuid)

### organizations
id (uuid), created_at (timestamp), name (varchar), stripe_customer_id (text), is_partner (bool), is_paying_customer (bool), slug (text)

### members
id (uuid), organization_id (uuid), user_id (uuid), role (enum), created_at (timestamp)

### search_logs
id (uuid), created_at (timestamp), question (varchar), answer (varchar), time (int4, latency ms), organization_id (uuid), service_account_id (uuid), traces (varchar, JSON array of tool calls — contains step-by-step execution: tool names, params, results), depth (varchar: fast/standard/deep/research), output_type (varchar), iterations (int4), from_date (timestamp), to_date (timestamp), include_images (bool), brain_version (varchar), request_id (varchar), include_inline_citations (bool), max_results (int4)
Note: only ~7 days retained. Use Lakehouse for historical. The traces column contains the raw execution trace as a JSON array — use it when users ask for traces, tool calls, or execution details.

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


def build_system_prompt() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You are a data analyst bot for Linkup. You answer questions by writing SQL queries against our Metabase instance.

Today's date is {today} (UTC). Use this for any relative date calculations (e.g. "this week", "last month").

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
- When searching by org name, use ILIKE with wildcards for fuzzy matching. Start broad (e.g. '%car%' not '%cargo%') to catch similar names.
- If a UUID is provided, use exact match. If a name is provided, use ILIKE.
- Users often approximate org names. "cargo" might mean an org named "Carg". Always use the closest match you find.
- If the question cannot be answered with SQL, return: {{"error": "explanation"}}
"""


SUMMARY_PROMPT = """Format query results as a clean Slack message. Strict rules:

- Just the data. No commentary, no opinions, no suggestions, no "worth checking", no exclamation marks.
- No emojis. Ever.
- No preamble like "Here are the results".
- For lists, use this exact format with a code block:
```
1. Org Name         1,012,789
2. Another Org        104,252
3. Third Org           84,031
```
- Right-align numbers in code blocks for readability.
- Add a one-line bold title before the code block summarizing what the data shows, e.g. *Top orgs by query volume (last 7 days)*
- For single-value answers, just state the value.
- For user lookups, use key: value on separate lines.
- For monetary values, include currency symbol and commas.
- If showing more than 20 rows, show top 15 and add a summary line for the rest.
- If the org name in the data doesn't exactly match what the user typed (fuzzy match), note it naturally, e.g. *Last 10 queries by Carg (closest match for "cargo")*
- Keep it minimal."""


# -- Metabase ----------------------------------------------------------------

def run_query(sql: str, database: int = 3) -> tuple[list[str], list[list], str | None]:
    """Execute SQL. Returns (cols, rows, error_message)."""
    try:
        resp = requests.post(
            f"{METABASE_URL}/dataset",
            headers=MB_HEADERS,
            json={"database": database, "type": "native", "native": {"query": sql}},
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") == "failed":
            return [], [], body.get("error", "Unknown query error")
        data = body["data"]
        cols = [c["name"] for c in data["cols"]]
        return cols, data["rows"], None
    except requests.exceptions.Timeout:
        return [], [], "Query timed out (>60s). Try a narrower date range or add LIMIT."
    except requests.exceptions.HTTPError as e:
        return [], [], str(e)
    except Exception as e:
        return [], [], str(e)


# -- Claude ------------------------------------------------------------------

def ask_claude(messages: list, system: str = "") -> str:
    resp = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def parse_json_response(raw: str) -> dict:
    """Parse Claude's JSON response, stripping markdown fences."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


def summarize_results(question: str, sql: str, cols: list, rows: list) -> str:
    """Generate a Slack-formatted summary of query results."""
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
                "content": (
                    f"Question: {question}\n\n"
                    f"SQL used:\n```sql\n{sql}\n```\n\n"
                    f"Query returned {len(rows)} rows. Data:\n{data_str}"
                ),
            }
        ],
        system=SUMMARY_PROMPT,
    )


# -- SQL Agent Loop ----------------------------------------------------------

def run_sql_loop(messages: list) -> tuple[str | None, str | None, list[str], list[list]]:
    """
    Inner agentic loop: generate SQL, run it, retry on errors/empty results.
    Returns (error, sql, cols, rows).
    """
    agent_messages = list(messages)
    system = build_system_prompt()

    for attempt in range(1, MAX_SQL_RETRIES + 1):
        logging.info(f"SQL attempt {attempt}/{MAX_SQL_RETRIES}")

        raw = ask_claude(agent_messages, system=system)
        agent_messages.append({"role": "assistant", "content": raw})

        try:
            result = parse_json_response(raw)
        except json.JSONDecodeError:
            feedback = 'Your response was not valid JSON. Return ONLY: {"database": 3, "sql": "SELECT ..."}'
            agent_messages.append({"role": "user", "content": feedback})
            continue

        if "error" in result:
            return result["error"], None, [], []

        sql = result["sql"]
        db = result.get("database", 3)
        cols, rows, query_error = run_query(sql, database=db)

        if query_error:
            feedback = (
                f"SQL error: {query_error}\n\n"
                f"Query was:\n```sql\n{sql}\n```\n\n"
                f"Fix the query and try again. Return only the JSON object."
            )
            agent_messages.append({"role": "user", "content": feedback})
            logging.info(f"SQL error on attempt {attempt}: {query_error}")
            continue

        if not rows:
            # On first empty result for org-related queries, discover real names
            if attempt == 1 and ("organization" in sql.lower() or "org" in sql.lower()):
                discovery_sql = "SELECT name FROM organizations WHERE name IS NOT NULL ORDER BY name"
                _, org_rows, _ = run_query(discovery_sql, database=3)
                org_list = ", ".join(r[0] for r in org_rows[:100] if r[0])
                extra = f"\n\nExisting org names: {org_list}"
            else:
                extra = ""

            feedback = (
                f"Query returned 0 rows:\n```sql\n{sql}\n```\n\n"
                f"Try a different approach:\n"
                f"- User may have misspelled the name. Try shorter substrings: '%car%' instead of '%cargo%'\n"
                f"- Widen date ranges if needed\n"
                f"- Try alternate tables (wallet_transactions for historical data){extra}\n\n"
                f"Return only the JSON object."
            )
            agent_messages.append({"role": "user", "content": feedback})
            logging.info(f"Empty results on attempt {attempt}, retrying...")
            continue

        return None, sql, cols, rows

    return "Couldn't find results after multiple attempts.", None, [], []


# -- Thread Context ----------------------------------------------------------

def get_thread_ts(event: dict) -> str:
    """Get thread_ts to reply in. Uses existing thread or starts a new one."""
    return event.get("thread_ts", event["ts"])


def fetch_thread_context(channel: str, thread_ts: str) -> list[dict]:
    """Fetch thread messages from Slack API and convert to Claude messages."""
    try:
        result = app.client.conversations_replies(
            channel=channel, ts=thread_ts, limit=20
        )
    except Exception as e:
        logging.warning(f"Failed to fetch thread context: {e}")
        return []

    messages = []
    for msg in result.get("messages", [])[:-1]:  # exclude triggering message
        text = re.sub(r"<@[A-Z0-9]+>\s*", "", msg.get("text", "")).strip()
        if not text:
            continue
        is_bot = msg.get("bot_id") or msg.get("user") == BOT_USER_ID
        role = "assistant" if is_bot else "user"
        messages.append({"role": role, "content": text})

    # Collapse consecutive same-role messages
    collapsed = []
    for msg in messages:
        if collapsed and collapsed[-1]["role"] == msg["role"]:
            collapsed[-1]["content"] += "\n" + msg["content"]
        else:
            collapsed.append(msg)

    return collapsed


# -- Message Helpers ---------------------------------------------------------

def post_thinking(channel: str, thread_ts: str) -> str | None:
    """Post a thinking indicator. Returns the message ts for later update."""
    try:
        result = app.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text="Thinking..."
        )
        return result["ts"]
    except Exception as e:
        logging.warning(f"Failed to post thinking indicator: {e}")
        return None


def update_message(channel: str, ts: str | None, text: str):
    """Update an existing message, or post a new one if ts is None."""
    if ts is None:
        return
    try:
        app.client.chat_update(channel=channel, ts=ts, text=text)
    except Exception as e:
        logging.warning(f"Failed to update message: {e}")
        try:
            app.client.chat_postMessage(channel=channel, text=text)
        except Exception:
            pass


# -- Main Handler ------------------------------------------------------------

def handle_question(event, say):
    channel = event["channel"]
    question = re.sub(r"<@[A-Z0-9]+>\s*", "", event["text"]).strip()
    thread_ts = get_thread_ts(event)

    if not question:
        say(text="Ask me a question about our data.", thread_ts=thread_ts)
        return

    # Handle "explain" — show the SQL + plain-English reasoning
    if question.lower().strip() in ("explain", "show sql", "why", "how"):
        last_sql = thread_sql.get(thread_ts)
        if not last_sql:
            say(text="No previous query in this thread to explain.", thread_ts=thread_ts)
            return

        explanation = ask_claude(
            [{"role": "user", "content": f"Explain this SQL query in plain English — what it does, why each filter/join is there, and any assumptions made:\n\n```sql\n{last_sql}\n```"}],
            system="You explain SQL queries to non-technical users. Be concise — 3-5 bullet points max. Use Slack mrkdwn formatting. No emojis.",
        )
        say(text=f"{explanation}\n\n```sql\n{last_sql}\n```", thread_ts=thread_ts)
        return

    # Post thinking indicator
    thinking_ts = post_thinking(channel, thread_ts)

    try:
        # Fetch thread context from Slack
        context_messages = fetch_thread_context(channel, thread_ts)
        messages = context_messages + [{"role": "user", "content": question}]

        # Check if user wants raw output
        is_raw = bool(re.search(r'\braw\b', question, re.IGNORECASE))

        # SQL agent loop with retries
        error, sql, cols, rows = run_sql_loop(messages)

        if error:
            update_message(channel, thinking_ts, error)
            return

        # Store SQL for "explain" command
        if sql:
            thread_sql[thread_ts] = sql

        if is_raw:
            # Upload raw data as a JSON file in the thread
            raw_data = []
            for r in rows:
                row = {}
                for col, val in zip(cols, r):
                    # Parse JSON strings (e.g. traces column) into real objects
                    if isinstance(val, str) and val.startswith(("[", "{")):
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    row[col] = val
                raw_data.append(row)
            raw_json = json.dumps(raw_data, indent=2, default=str, ensure_ascii=False)

            update_message(channel, thinking_ts, f"Uploading raw data ({len(rows)} rows)...")

            try:
                app.client.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    content=raw_json,
                    filename="raw_results.json",
                    title=f"Raw query results ({len(rows)} rows)",
                    initial_comment=f"```sql\n{sql}\n```",
                )
                # Delete the thinking message since the file upload has its own comment
                try:
                    app.client.chat_delete(channel=channel, ts=thinking_ts)
                except Exception:
                    pass
            except Exception as e:
                logging.warning(f"File upload failed: {e}")
                # Fallback: post truncated raw JSON in message
                truncated = raw_json[:3900] + ("\n..." if len(raw_json) > 3900 else "")
                update_message(channel, thinking_ts, f"```json\n{truncated}\n```")
        else:
            # Summarize and send
            summary = summarize_results(question, sql, cols, rows)
            update_message(channel, thinking_ts, summary)

    except Exception as e:
        logging.exception("Error handling question")
        update_message(channel, thinking_ts, f"Something went wrong: `{e}`")


# -- Slack Event Handlers ----------------------------------------------------

@app.event("app_mention")
def handle_mention(event, say):
    handle_question(event, say)


@app.event("message")
def handle_dm(event, say):
    if event.get("channel_type") == "im" and not event.get("bot_id"):
        handle_question(event, say)


@app.error
def global_error_handler(error, body, logger):
    logger.exception(f"Unhandled error: {error}")


# -- Main --------------------------------------------------------------------

def main():
    global BOT_USER_ID
    auth = app.client.auth_test()
    BOT_USER_ID = auth["user_id"]
    logging.info(f"Bot user ID: {BOT_USER_ID}")

    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    print("Bot is running!")
    handler.start()


if __name__ == "__main__":
    main()
