"""Load transformed DataFrames into PostgreSQL — Phase 5, staging edition.

Rules of the road:

  ORDER     parents before children: dims -> bridge -> facts. A foreign-key
            row referencing a missing parent is rejected by the engine,
            so load order mirrors (in reverse) the dependency order of
            schema.py's DDL.
  TWO STEP  load_staging writes raw validated rows into staging_* tables;
            promote_to_final moves them into the star schema. Validation
            (main.py) runs BEFORE load_staging, so bad data never reaches
            the database at all; a failed promotion rolls back, and staging
            keeps the rejected rows for debugging.
  ATOMIC    each step is ONE transaction (engine.begin): either every
            table in that step loads, or none does.
  IDEMPOTENT re-running never duplicates. Strategy matches data shape:
            reference data -> UPSERT, bridge pairs -> DO NOTHING,
            recomputed snapshot -> TRUNCATE + INSERT (see load_recommendations).

Promotion is set-based SQL (INSERT ... SELECT from staging), not row-by-row
pandas: the database moves data to itself faster than any driver loop, and
the statement reads as a transformation, which is exactly what it is.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from transform import GENRE_COLUMNS

TABLES = (
    "dim_users",
    "dim_movies",
    "dim_genres",
    "dim_movie_genres",
    "fact_recommendations",
)

STAGING_TABLES = (
    "staging_users",
    "staging_ratings",
    "staging_movies",
)

# ── staging SQL: mirror the validated DataFrames, row for row ────────────

TRUNCATE_STAGING_USER_SQL = "TRUNCATE TABLE staging_users"
TRUNCATE_STAGING_RATING_SQL = "TRUNCATE TABLE staging_ratings"
TRUNCATE_STAGING_MOVIE_SQL = "TRUNCATE TABLE staging_movies"

INSERT_STAGING_USER_SQL = """
    INSERT INTO staging_users (user_id, age, gender, occupation, zip_code)
    VALUES (:user_id, :age, :gender, :occupation, :zip_code)
"""

INSERT_STAGING_RATING_SQL = """
    INSERT INTO staging_ratings (user_id, movie_id, rating, timestamp)
    VALUES (:user_id, :movie_id, :rating, :timestamp)
"""

# staging_movies carries every u.item column, so its INSERT is built from
# the same list the DDL uses. Bind params are positional (c0, c1, ...) because
# genre names are not legal bind identifiers ("Children's", "Film-Noir",
# "unknown"); the DataFrame columns are renamed to match right before insert.
STAGING_MOVIE_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_COLUMNS,
]
INSERT_STAGING_MOVIE_SQL = (
    "INSERT INTO staging_movies ({}) VALUES ({})".format(
        ", ".join(f'"{column}"' for column in STAGING_MOVIE_COLUMNS),
        ", ".join(f":c{index}" for index in range(len(STAGING_MOVIE_COLUMNS))),
    )
)

# ── promotion SQL: staging -> star schema, one statement per dimension ───

# D1 DECISION visible: only the four surviving columns are promoted —
# video_release_date stays in staging_movies, never reaches dim_movies.
PROMOTE_USERS_SQL = """
    INSERT INTO dim_users (user_id, age, gender, occupation, zip_code)
    SELECT user_id, age, gender, occupation, zip_code
    FROM staging_users
    ON CONFLICT (user_id) DO UPDATE SET
        age        = EXCLUDED.age,
        gender     = EXCLUDED.gender,
        occupation = EXCLUDED.occupation,
        zip_code   = EXCLUDED.zip_code
"""

PROMOTE_MOVIES_SQL = """
    INSERT INTO dim_movies (movie_id, title, release_date, imdb_url)
    SELECT movie_id, title, release_date, imdb_url
    FROM staging_movies
    ON CONFLICT (movie_id) DO UPDATE SET
        title        = EXCLUDED.title,
        release_date = EXCLUDED.release_date,
        imdb_url     = EXCLUDED.imdb_url
"""

INSERT_GENRE_SQL = """
    INSERT INTO dim_genres (genre_name)
    VALUES (:genre_name)
    ON CONFLICT (genre_name) DO NOTHING
"""

INSERT_BRIDGE_SQL = """
    INSERT INTO dim_movie_genres (movie_id, genre_id)
    VALUES (:movie_id, :genre_id)
    ON CONFLICT (movie_id, genre_id) DO NOTHING
"""

# Snapshot strategy: TRUNCATE empties the table inside our transaction, so
# re-runs REPLACE last run's truth instead of upserting beside it. An
# ON CONFLICT upsert cannot delete a movie that fell off the top-3 list —
# stale rows would survive every load.
TRUNCATE_FACT_SQL = "TRUNCATE TABLE fact_recommendations"

INSERT_FACT_SQL = """
    INSERT INTO fact_recommendations (user_id, movie_id, rank, score, rating_count)
    VALUES (:user_id, :movie_id, :rank, :score, :rating_count)
"""


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as plain-Python dicts.

    The classic integration seam: psycopg2 cannot adapt numpy.int64/float64
    or pandas NaT — it raises "can't adapt type 'int64'". DB drivers speak
    Python, pandas speaks numpy; conversion happens exactly here, once.
    NaN/NaT become None so SQLAlchemy writes SQL NULL rather than crashing.
    """
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None                # NaN / NaT -> SQL NULL
            elif isinstance(value, np.generic):
                clean[key] = value.item()        # numpy scalar -> python scalar
            else:
                clean[key] = value
        records.append(clean)
    return records


def _insert(conn: Connection, sql: str, df: pd.DataFrame) -> int:
    """Bulk-insert every row of df in a single executemany call. Returns row count."""
    rows = _records(df)
    if rows:
        conn.execute(text(sql), rows)  # one round trip, N rows: "bulk insert"
    return len(rows)


def load_staging(
    engine: Engine,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> dict[str, int]:
    """Mirror the validated DataFrames into staging_* tables — atomically.

    TRUNCATE + INSERT per table: re-runs replace the previous mirror,
    never append beside it. Runs AFTER validation (main.py guarantees it),
    so staging always holds only data that passed both gates.
    """
    with engine.begin() as conn:
        conn.execute(text(TRUNCATE_STAGING_USER_SQL))
        conn.execute(text(TRUNCATE_STAGING_RATING_SQL))
        conn.execute(text(TRUNCATE_STAGING_MOVIE_SQL))

        counts = {
            "staging_users": _insert(conn, INSERT_STAGING_USER_SQL, users),
            "staging_ratings": _insert(conn, INSERT_STAGING_RATING_SQL, ratings),
            "staging_movies": _insert(
                conn,
                INSERT_STAGING_MOVIE_SQL,
                _positional_params(movies),
            ),
        }

    return counts


def _positional_params(movies: pd.DataFrame) -> pd.DataFrame:
    """Reorder/label movie columns to match INSERT_STAGING_MOVIE_SQL's :c0, :c1..."""
    staged = movies[STAGING_MOVIE_COLUMNS].copy()
    staged.columns = [f"c{index}" for index in range(len(STAGING_MOVIE_COLUMNS))]
    return staged


def load_genres(conn: Connection) -> int:
    """Seed the 19 genre names from transform's single source of truth (GENRE_COLUMNS)."""
    # Surrogate ids come from the database (SERIAL); only names are inserted,
    # so re-runs hit the UNIQUE(name) constraint and do nothing — ids stable.
    return _insert(conn, INSERT_GENRE_SQL, pd.DataFrame({"genre_name": GENRE_COLUMNS}))


def load_movie_genres(conn: Connection, movie_genres: pd.DataFrame) -> int:
    """Map genre NAMES -> ids (D2: mapping happens here, at the DB boundary), then insert pairs."""
    genre_map = pd.read_sql(
        text("SELECT genre_id, genre_name FROM dim_genres"), conn,
    )
    bridge = movie_genres.merge(genre_map, left_on="genre", right_on="genre_name")

    # Loud check at the seam: if a name failed to map, rows would silently
    # vanish here. Crash instead — wrong-but-quiet is the enemy.
    if len(bridge) != len(movie_genres):
        raise ValueError(
            f"genre mapping lost rows: {len(movie_genres)} in, {len(bridge)} mapped — "
            "a genre name is missing from dim_genres"
        )

    return _insert(conn, INSERT_BRIDGE_SQL, bridge[["movie_id", "genre_id"]])


def load_recommendations(conn: Connection, recommendations: pd.DataFrame) -> int:
    """Replace the recommendation snapshot: TRUNCATE + INSERT (see module docstring)."""
    conn.execute(text(TRUNCATE_FACT_SQL))

    facts = recommendations[
        ["user_id", "movie_id", "rank", "avg_rating", "rating_count"]
    ].rename(columns={"avg_rating": "score"})  # transform says avg_rating, the fact table says score

    return _insert(conn, INSERT_FACT_SQL, facts)


def promote_to_final(
    engine: Engine,
    movie_genres: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> dict[str, int]:
    """Promote staging into the star schema — ONE transaction, FK-safe order.

    dims are populated with set-based INSERT ... SELECT from staging;
    the bridge and facts are derived DataFrames that never live in staging
    (they are produced by transform, validated by Gate 2, then written).
    If anything fails, the transaction rolls back: final tables keep last
    run's truth, staging keeps this run's rows for debugging.
    """
    with engine.begin() as conn:  # ONE transaction: all five tables or none
        conn.execute(text(PROMOTE_USERS_SQL))
        conn.execute(text(PROMOTE_MOVIES_SQL))
        load_genres(conn)
        load_movie_genres(conn, movie_genres)
        load_recommendations(conn, recommendations)

        counts = {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in TABLES
        }

    return counts
