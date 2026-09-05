"""Shared setup for the examples.

Point them at your database once:

    export POSTGRES_URI=postgresql://localhost:5432/postgres

Every example uses its own realm, so they never collide and can be run in any
order. None of them needs an LLM or any network service beyond PostgreSQL.
"""
import os
import time

DSN = os.getenv("POSTGRES_URI", "postgresql://localhost:5432/postgres")


def fresh_realm(base: str) -> str:
    """A realm nobody has written to yet, so each run starts clean."""
    return f"{base}_{int(time.time())}"


def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
