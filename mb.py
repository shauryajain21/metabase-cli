#!/usr/bin/env python3
"""Simple CLI for querying Linkup's Metabase instance."""

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

URL = os.environ["METABASE_URL"]
API_KEY = os.environ["METABASE_API_KEY"]
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# Internal orgs to exclude from rankings
INTERNAL_ORGS = (
    "c93177dc-55ee-4219-8c15-d1612e121e91",
    "9ca37ea9-1122-429b-9b60-f36f62fbf3dc",
)
INTERNAL_FILTER = " AND ".join(f"sl.organization_id != '{o}'" for o in INTERNAL_ORGS)

# -- Metabase helpers --------------------------------------------------------

def run_query(sql: str, database: int = 3) -> tuple[list[str], list[list]]:
    """Execute a native SQL query. Returns (column_names, rows)."""
    resp = requests.post(
        f"{URL}/dataset",
        headers=HEADERS,
        json={"database": database, "type": "native", "native": {"query": sql}},
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    cols = [c["name"] for c in data["cols"]]
    return cols, data["rows"]


def print_table(cols: list[str], rows: list[list]):
    """Pretty-print results as a table."""
    if not rows:
        print("No results.")
        return
    print(tabulate(rows, headers=cols, tablefmt="simple"))
    print(f"\n({len(rows)} rows)")


# -- Commands ----------------------------------------------------------------

def cmd_sql(args):
    """Run raw SQL against a database."""
    cols, rows = run_query(args.query, database=args.db)
    if args.json:
        print(json.dumps([dict(zip(cols, r)) for r in rows], indent=2, default=str))
    else:
        print_table(cols, rows)


def cmd_top_orgs(args):
    """Top orgs by query volume."""
    sql = f"""
    SELECT o.name AS org, COUNT(*) AS queries
    FROM search_logs sl
    JOIN organizations o ON o.id = sl.organization_id
    WHERE sl.created_at >= NOW() - INTERVAL '{args.days} days'
      AND {INTERNAL_FILTER}
    GROUP BY o.name
    ORDER BY queries DESC
    LIMIT {args.limit}
    """
    cols, rows = run_query(sql)
    print(f"Top orgs by query volume (last {args.days} days)\n")
    print_table(cols, rows)


def cmd_lookup(args):
    """Look up a user by email."""
    sql = f"""
    SELECT u.id, u.name, u.email, u.created_at,
           o.id AS org_id, o.name AS org_name,
           o.is_paying_customer, w.balance AS wallet_balance
    FROM users u
    LEFT JOIN members m ON m.user_id = u.id
    LEFT JOIN organizations o ON o.id = m.organization_id
    LEFT JOIN wallets w ON w.organization_id = o.id
    WHERE u.email = '{args.email}'
    """
    cols, rows = run_query(sql)
    if not rows:
        print(f"No user found with email: {args.email}")
        return
    for col, val in zip(cols, rows[0]):
        print(f"  {col}: {val}")


def cmd_queries(args):
    """Show recent queries for an org (by name or ID)."""
    if "-" in args.org and len(args.org) > 30:
        org_filter = f"sl.organization_id = '{args.org}'"
    else:
        org_filter = f"o.name ILIKE '%{args.org}%'"

    sql = f"""
    SELECT sl.created_at, sl.question, sl.depth, sl.output_type, sl.time AS latency_ms
    FROM search_logs sl
    JOIN organizations o ON o.id = sl.organization_id
    WHERE {org_filter}
    ORDER BY sl.created_at DESC
    LIMIT {args.limit}
    """
    cols, rows = run_query(sql)
    if args.json:
        print(json.dumps([dict(zip(cols, r)) for r in rows], indent=2, default=str))
    else:
        print_table(cols, rows)


def cmd_revenue(args):
    """Revenue summary over a period."""
    sql = f"""
    WITH rev AS (
        SELECT
            DATE_TRUNC('{args.group}', wt.created_at) AS period,
            SUM(CASE WHEN wt.type IN ('deposit', 'automatic_top_up') THEN wt.amount ELSE 0 END) AS deposits,
            SUM(CASE WHEN wt.type LIKE 'api_usage:%' THEN ABS(wt.amount) ELSE 0 END) AS usage
        FROM wallet_transactions wt
        JOIN wallets w ON w.id = wt.wallet_id
        WHERE wt.status = 'completed'
          AND wt.created_at >= NOW() - INTERVAL '{args.days} days'
          AND w.organization_id NOT IN {str(INTERNAL_ORGS).replace("'", "''")}
        GROUP BY period
    )
    SELECT period, deposits, usage FROM rev ORDER BY period DESC
    """
    # Simpler approach without the tuple issue
    sql = f"""
    SELECT
        DATE_TRUNC('{args.group}', wt.created_at) AS period,
        ROUND(SUM(CASE WHEN wt.type IN ('deposit', 'automatic_top_up') THEN wt.amount ELSE 0 END)::numeric, 2) AS deposits,
        ROUND(SUM(CASE WHEN wt.type LIKE 'api_usage:%%' THEN ABS(wt.amount) ELSE 0 END)::numeric, 2) AS usage
    FROM wallet_transactions wt
    JOIN wallets w ON w.id = wt.wallet_id
    WHERE wt.status = 'completed'
      AND wt.created_at >= NOW() - INTERVAL '{args.days} days'
    GROUP BY period
    ORDER BY period DESC
    """
    cols, rows = run_query(sql)
    print(f"Revenue ({args.group}ly, last {args.days} days)\n")
    print_table(cols, rows)


def cmd_slow(args):
    """Slowest queries in the last N days."""
    sql = f"""
    SELECT sl.created_at, o.name AS org, sl.depth, sl.time AS latency_ms,
           LEFT(sl.question, 120) AS question
    FROM search_logs sl
    JOIN organizations o ON o.id = sl.organization_id
    WHERE sl.created_at >= NOW() - INTERVAL '{args.days} days'
      AND sl.time IS NOT NULL
      AND {INTERNAL_FILTER}
    ORDER BY sl.time DESC
    LIMIT {args.limit}
    """
    cols, rows = run_query(sql)
    print(f"Slowest queries (last {args.days} days)\n")
    print_table(cols, rows)


def cmd_depth(args):
    """Search depth distribution."""
    sql = f"""
    SELECT sl.depth, COUNT(*) AS count,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
    FROM search_logs sl
    WHERE sl.created_at >= NOW() - INTERVAL '{args.days} days'
      AND {INTERNAL_FILTER}
    GROUP BY sl.depth
    ORDER BY count DESC
    """
    cols, rows = run_query(sql)
    print(f"Depth distribution (last {args.days} days)\n")
    print_table(cols, rows)


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Linkup Metabase CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # sql
    p = sub.add_parser("sql", help="Run raw SQL")
    p.add_argument("query", help="SQL query string")
    p.add_argument("--db", type=int, default=3, help="Database ID (3=Postgres, 5=Lakehouse)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_sql)

    # top-orgs
    p = sub.add_parser("top-orgs", help="Top orgs by query volume")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_top_orgs)

    # lookup
    p = sub.add_parser("lookup", help="Look up user by email")
    p.add_argument("email", help="User email")
    p.set_defaults(func=cmd_lookup)

    # queries
    p = sub.add_parser("queries", help="Recent queries for an org")
    p.add_argument("org", help="Org name (partial match) or org UUID")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_queries)

    # revenue
    p = sub.add_parser("revenue", help="Revenue summary")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--group", default="week", choices=["day", "week", "month"])
    p.set_defaults(func=cmd_revenue)

    # slow
    p = sub.add_parser("slow", help="Slowest queries")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_slow)

    # depth
    p = sub.add_parser("depth", help="Search depth distribution")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(func=cmd_depth)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
