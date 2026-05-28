"""Checkpointer factory – return MemorySaver or SqliteSaver based on env config."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_checkpointer():
    """Return a LangGraph checkpointer.

    If the SQLITE_DB_PATH environment variable is set, a SqliteSaver backed by
    that path is returned so that conversation memory persists across restarts.
    A new SQLite connection is created each call to avoid cross-thread sharing.

    Otherwise, an in-memory MemorySaver is used (useful for development /
    stateless deployments).

    Environment variables:
        SQLITE_DB_PATH  – path to the SQLite database file
                          (e.g. "./memory/agent_memory.db")
    """
    db_path = os.getenv("SQLITE_DB_PATH", "")

    if db_path:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # Create a fresh connection per call so each thread has its own handle.
        conn = sqlite3.connect(db_path, check_same_thread=True)
        return SqliteSaver(conn)

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
