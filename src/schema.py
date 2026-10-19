"""PostgreSQL schema for the movie recommendation pipeline.

Structure only — schema.py creates empty tables; load.py (Phase 5) fills them.
Every constraint here is the permanent form of a check that pandas could only
enforce for a single run:

    PRIMARY KEY  == your drop_duplicates, enforced forever
    FOREIGN KEY  == your Phase 2 orphan-ID checks, enforced forever
    CHECK        == your business rules, enforced forever

Tables are created in dependency order: staging first (no dependencies),
then dimensions, then the bridge, then facts — PostgreSQL validates foreign
keys at CREATE time, so a referenced table must already exist. All statements
run in ONE transaction: either the whole schema exists, or none of it does.
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

DDL_STATEMENTS = [
    # ── staging: the audit layer between validation and the star schema ──
    # Staging holds EXACTLY what the pipeline validated, row for row.
    # load.py promotes staging -> dims inside one transaction, so:
    #   * validation fails before staging   -> database untouched
    #   * promotion fails midway            -> transaction rolls back,
    #     staging keeps the rejected data for debugging
    # Deliberately bare: no PK/FK/CHECK. Staging is a mirror of the
    # pipeline's output, not a model of the business — constraints live
    # where the model lives (the star schema below).
    """
    CREATE TABLE IF NOT EXISTS staging_users (
        user_id     INTEGER,
        age         SMALLINT,
        gender      CHAR(1),
        occupation  VARCHAR(32),
        zip_code    VARCHAR(10)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS staging_ratings (
        user_id    INTEGER,
        movie_id   INTEGER,
        rating     SMALLINT,
        timestamp  TIMESTAMPTZ
    )
    """,
    # All 19 genre flags are staged too: staging is a faithful dump, so a
    # future re-check of genre assignment is possible without re-extracting.
    # Genre names containing apostrophes/hyphens ("Children's", "Film-Noir")
    # need quoted identifiers — a small taste of real-world DDL.
    """
    CREATE TABLE IF NOT EXISTS staging_movies (
        movie_id           INTEGER,
        title              TEXT,
        release_date       DATE,
        video_release_date DATE,
        imdb_url           TEXT,
        "unknown"          SMALLINT,
        "Action"           SMALLINT,
        "Adventure"        SMALLINT,
        "Animation"        SMALLINT,
        "Children's"       SMALLINT,
        "Comedy"           SMALLINT,
        "Crime"            SMALLINT,
        "Documentary"      SMALLINT,
        "Drama"            SMALLINT,
        "Fantasy"          SMALLINT,
        "Film-Noir"        SMALLINT,
        "Horror"           SMALLINT,
        "Musical"          SMALLINT,
        "Mystery"          SMALLINT,
        "Romance"          SMALLINT,
        "Sci-Fi"           SMALLINT,
        "Thriller"         SMALLINT,
        "War"              SMALLINT,
        "Western"          SMALLINT
    )
    """,

    # ── dimensions: the things we describe and filter by ─────────────────

    # zip_code is VARCHAR, never INTEGER: leading zeros are real ("02139").
    # Profile columns stay nullable: they mirror u.user exactly as validated
    # — NOT NULL is a data-quality promise; only make promises you've verified.
    """
    CREATE TABLE IF NOT EXISTS dim_users (
        user_id     INTEGER PRIMARY KEY,
        age         SMALLINT,
        gender      CHAR(1),
        occupation  VARCHAR(32),
        zip_code    VARCHAR(10)
    )
    """,

    # D1 DECISION: video_release_date is intentionally ABSENT — verified 100%
    # empty in u.item. A documented drop, not an accidental omission.
    """
    CREATE TABLE IF NOT EXISTS dim_movies (
        movie_id     INTEGER PRIMARY KEY,
        title        VARCHAR(200) NOT NULL,   -- trailing whitespace stripped in transform
        release_date DATE,                    -- NaT -> NULL (documented policy)
        imdb_url     TEXT
    )
    """,

    # D2 DECISION: surrogate key. Our transform speaks genre NAMES; load.py
    # will map name -> genre_id when it inserts dim_genres. The genre list
    # lives in exactly ONE place: GENRE_COLUMNS in transform.py.
    """
    CREATE TABLE IF NOT EXISTS dim_genres (
        genre_id     SERIAL PRIMARY KEY,
        genre_name   VARCHAR(20) NOT NULL UNIQUE
    )
    """,

    # ── bridge: movies <-> genres is many-to-many ────────────────────────
    # The composite PRIMARY KEY rejects the exact duplicate (movie, genre)
    # pairs that forced us to drop_duplicates() back in build_recommendations.
    # genre_id must be INTEGER to match dim_genres.genre_id (SERIAL = INTEGER).
    """
    CREATE TABLE IF NOT EXISTS dim_movie_genres (
        movie_id   INTEGER NOT NULL REFERENCES dim_movies(movie_id),
        genre_id   INTEGER NOT NULL REFERENCES dim_genres(genre_id),
        PRIMARY KEY (movie_id, genre_id)
    )
    """,

    # PostgreSQL does NOT auto-index foreign keys: the composite PK only
    # serves movie_id-first lookups. "Which movies are in genre X?" needs this.
    """
    CREATE INDEX IF NOT EXISTS idx_movie_genres_genre
        ON dim_movie_genres (genre_id)
    """,

    # ── fact: one row = one recommendation the pipeline produced ─────────
    # D3 DECISION: natural PK (user_id, rank), because
    #   * GET /recommendations/{user_id} is served by this index as-is
    #     (user_id leads the key — same lesson as sorting before head():
    #     leading order decides what the structure is good for)
    #   * Phase 8 idempotent reloads become ON CONFLICT DO UPDATE for free
    #   * choosing a surrogate id instead would leave that API query
    #     needing its own secondary index
    # The score CHECK guards against bugs in OUR OWN transform output —
    # never blindly trust even code you wrote yourself.
    """
    CREATE TABLE IF NOT EXISTS fact_recommendations (
        user_id        INTEGER      NOT NULL REFERENCES dim_users(user_id),
        movie_id       INTEGER      NOT NULL REFERENCES dim_movies(movie_id),
        rank           SMALLINT     NOT NULL CHECK (rank BETWEEN 1 AND 3),
        score          NUMERIC(3,2) NOT NULL CHECK (score BETWEEN 1.00 AND 5.00),
        rating_count   INTEGER      NOT NULL,   -- provenance: why the score was trusted
        recommended_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, rank),
        UNIQUE (user_id, movie_id)              -- the same movie twice for one user: impossible
    )
    """,
]


def create_tables(engine: Engine) -> None:
    """Create every pipeline table — idempotent (IF NOT EXISTS), atomic (one transaction)."""
    with engine.begin() as conn:
        for statement in DDL_STATEMENTS:
            conn.execute(text(statement))


if __name__ == "__main__":
    from config import engine

    create_tables(engine)
    print("schema created — inspect it: psql movie_etl")
