"""Pipeline orchestrator — Phase 6. Run the entire ETL with one command:

    python src/main.py

Flow (each arrow is a stage boundary main.py owns):

    EXTRACT -> GATE 1 -> TRANSFORM -> GATE 2 -> SCHEMA -> STAGE -> PROMOTE

    EXTRACT      raw files become DataFrames — no logic, no trust yet
    GATE 1       validate_raw: structure, nulls, ranges, foreign keys
                 any ERROR here exits non-zero BEFORE the database is touched
    TRANSFORM    business rules: preferences, thresholds, ranking, top 3
    GATE 2       validate_recommendations: the recommendation contract
    SCHEMA       create_tables (idempotent — IF NOT EXISTS)
    STAGE        validated rows mirror into staging_* tables
    PROMOTE      staging -> star schema, one transaction

Design rule: validation failures stop the pipeline before any write, so
"bad data loaded" is not a failure mode this pipeline can produce.
"""

import logging
import sys
import time
from pathlib import Path

from config import engine
from extract import extract_movies, extract_ratings, extract_users
from load import load_staging, promote_to_final
from schema import create_tables
from transform import (
    build_recommendations,
    build_user_preferences,
    calculate_movie_statistics,
    transform_movie_genres,
    transform_movies,
    transform_ratings,
    transform_users,
)
from validate import enforce, validate_raw, validate_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("movie_etl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_FILES = {
    "ratings": DATA_DIR / "u.data",
    "movies": DATA_DIR / "u.item",
    "users": DATA_DIR / "u.user",
}


def print_summary(
    raw_counts: dict[str, int],
    staging_counts: dict[str, int],
    final_counts: dict[str, int],
    elapsed_seconds: float,
) -> None:
    """The Phase 7 reconciliation panel: one place to see what happened."""
    rule = "=" * 62
    print(f"\n{rule}")
    print("  PIPELINE SUMMARY")
    print(rule)

    print("  Extracted (rows read from raw files):")
    for name, count in raw_counts.items():
        print(f"    {name:<26}{count:>10,}")

    print("  Staged (passed both validation gates):")
    for name, count in staging_counts.items():
        print(f"    {name:<26}{count:>10,}")

    print("  Loaded (PostgreSQL star schema):")
    for name, count in final_counts.items():
        print(f"    {name:<26}{count:>10,}")

    print(f"\n  Elapsed: {elapsed_seconds:.1f}s")
    print(rule)
    print("  DONE — inspect in pgAdmin: movie_etl -> public -> Tables")
    print(rule + "\n")


def main() -> None:
    started = time.perf_counter()

    # ── EXTRACT ───────────────────────────────────────────────────────────
    logger.info("EXTRACT: reading raw files from %s", DATA_DIR)
    missing_files = [str(path) for path in RAW_FILES.values() if not path.exists()]
    if missing_files:
        logger.error("missing raw file(s): %s", ", ".join(missing_files))
        sys.exit(1)

    raw_users = extract_users(str(RAW_FILES["users"]))
    raw_ratings = extract_ratings(str(RAW_FILES["ratings"]))
    raw_movies = extract_movies(str(RAW_FILES["movies"]))
    logger.info(
        "extracted %s users, %s ratings, %s movies",
        f"{len(raw_users):,}",
        f"{len(raw_ratings):,}",
        f"{len(raw_movies):,}",
    )
    raw_counts = {
        "u.user (users)": len(raw_users),
        "u.data (ratings)": len(raw_ratings),
        "u.item (movies)": len(raw_movies),
    }

    # ── GATE 1: raw data ──────────────────────────────────────────────────
    logger.info("VALIDATE: gate 1 — raw data checks")
    enforce(validate_raw(raw_ratings, raw_movies, raw_users))

    # ── TRANSFORM ─────────────────────────────────────────────────────────
    logger.info("TRANSFORM: cleaning and standardising")
    users = transform_users(raw_users)
    ratings = transform_ratings(raw_ratings)
    movies = transform_movies(raw_movies)

    logger.info("TRANSFORM: deriving genres, statistics, preferences")
    movie_genres = transform_movie_genres(movies)
    movie_statistics = calculate_movie_statistics(ratings)
    preferences = build_user_preferences(ratings, movie_genres)

    logger.info(
        "TRANSFORM: building recommendations (threshold, anti-join, top 3)"
    )
    recommendations = build_recommendations(
        preferences,
        movie_genres,
        ratings,
        movie_statistics,
    )
    logger.info("produced %s recommendations", f"{len(recommendations):,}")

    # ── GATE 2: recommendation contract ───────────────────────────────────
    logger.info("VALIDATE: gate 2 — recommendation rules")
    enforce(validate_recommendations(recommendations, ratings, movies))

    # ── SCHEMA ────────────────────────────────────────────────────────────
    logger.info("SCHEMA: creating tables if missing")
    create_tables(engine)

    # ── STAGE ─────────────────────────────────────────────────────────────
    logger.info("LOAD: mirroring validated data into staging tables")
    staging_counts = load_staging(engine, users, ratings, movies)
    for table, count in staging_counts.items():
        logger.info("  %s: %s rows", table, f"{count:,}")

    # ── PROMOTE ───────────────────────────────────────────────────────────
    logger.info("LOAD: promoting staging -> star schema (one transaction)")
    final_counts = promote_to_final(engine, movie_genres, recommendations)
    for table, count in final_counts.items():
        logger.info("  %s: %s rows", table, f"{count:,}")

    print_summary(
        raw_counts,
        staging_counts,
        final_counts,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
