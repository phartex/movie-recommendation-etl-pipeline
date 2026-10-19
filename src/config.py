"""Shared pipeline configuration — the DATABASE_URL and the one SQLAlchemy engine.

Engine created at import time, once per process: schema.py, load.py and
(Phase 6) main.py all share it. Never create engines inside functions —
each engine owns a connection pool; one per process is the rule.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/movie_etl",
)

engine = create_engine(DATABASE_URL)
