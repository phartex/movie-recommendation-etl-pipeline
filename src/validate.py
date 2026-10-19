"""Validation layer — Phase 2 (raw data) and Phase 7 (recommendations).

Never blindly trust your input data. Two gates run in main.py:

    GATE 1  validate_raw            after extract, before transform
    GATE 2  validate_recommendations after transform, before any database write

Severity model:

    ERROR    data is wrong in a way that corrupts results (bad ratings,
             orphan foreign keys, watched movies in recommendations).
             The pipeline prints the report and exits non-zero BEFORE
             touching PostgreSQL.

    WARNING  data is imperfect but survivable (missing release dates,
             missing IMDb URLs, unexpected row counts). Logged, pipeline
             continues — because real-world data is rarely perfect and a
             pipeline that stops on every nullable column never finishes.

A report collects EVERY check's outcome before anyone acts on it: a data
engineer wants the full list of what failed, not just the first failure.
"""

import sys

import pandas as pd

from transform import MIN_RATING_COUNT

# Expected MovieLens 100K sizes — WARNING only, not ERROR: a different
# extract (a newer MovieLens dump) is a surprise, not corruption.
EXPECTED_RATINGS = 100_000
EXPECTED_MOVIES = 1_682
EXPECTED_USERS = 943


class ValidationReport:
    """Collects errors and warnings from one validation gate."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def print(self) -> None:
        status = "VALIDATION PASSED" if self.ok else "VALIDATION FAILED"
        rule = "=" * 62
        print(f"\n{rule}")
        print(f"  {self.stage}: {status}")
        print(rule)

        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            for message in self.errors:
                print(f"  X {message}")

        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for message in self.warnings:
                print(f"  ! {message}")

        if self.ok:
            print("  All checks passed.")
        else:
            print("\n  Pipeline stopped.")
            print("  Nothing was loaded into PostgreSQL.")
        print(rule + "\n")


def enforce(report: ValidationReport) -> None:
    """Print the report; exit(1) if any ERROR was recorded."""
    report.print()
    if not report.ok:
        sys.exit(1)


def validate_raw(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
) -> ValidationReport:
    """GATE 1 — structural and referential checks on the raw extracts."""

    report = ValidationReport("RAW DATA")

    required_columns = {
        "ratings": ratings,
        "movies": movies,
        "users": users,
    }
    expected = {
        "ratings": ["user_id", "movie_id", "rating", "timestamp"],
        "movies": ["movie_id", "title", "release_date", "imdb_url"],
        "users": ["user_id", "age", "gender", "occupation", "zip_code"],
    }

    # -- 1. schema check: without columns, every later check is impossible
    columns_ok = True
    for name, frame in required_columns.items():
        missing = [c for c in expected[name] if c not in frame.columns]
        if missing:
            columns_ok = False
            report.error(f"{name}: missing required columns {missing}")

    # -- 2. row counts (WARNING severity, see EXPECTED_* comment)
    if len(ratings) != EXPECTED_RATINGS:
        report.warning(
            f"ratings: expected {EXPECTED_RATINGS:,} rows, found {len(ratings):,}"
        )
    if len(movies) != EXPECTED_MOVIES:
        report.warning(
            f"movies: expected {EXPECTED_MOVIES:,} rows, found {len(movies):,}"
        )
    if len(users) != EXPECTED_USERS:
        report.warning(f"users: expected {EXPECTED_USERS:,} rows, found {len(users):,}")

    if not columns_ok:
        # Dependent checks would raise KeyError on absent columns —
        # the report already explains why we are stopping here.
        return report

    # -- 3. critical nulls: a null key means the row cannot be joined
    critical_nulls = {
        "ratings": ["user_id", "movie_id", "rating", "timestamp"],
        "movies": ["movie_id", "title"],
        "users": ["user_id"],
    }
    for name, frame in (("ratings", ratings), ("movies", movies), ("users", users)):
        for column in critical_nulls[name]:
            nulls = int(frame[column].isna().sum())
            if nulls:
                report.error(f"{name}.{column}: {nulls} null value(s)")

    # -- 4. rating values must satisfy the business rule 1..5
    invalid_ratings = ratings[~ratings["rating"].between(1, 5)]
    if len(invalid_ratings):
        offenders = sorted(invalid_ratings["rating"].unique().tolist())
        report.error(
            f"ratings.rating: {len(invalid_ratings)} row(s) outside 1..5 "
            f"(distinct invalid values: {offenders})"
        )

    # -- 5. duplicate ratings: same user, movie and instant, twice
    duplicates = ratings.duplicated(subset=["user_id", "movie_id", "timestamp"])
    duplicate_count = int(duplicates.sum())
    if duplicate_count:
        report.error(f"ratings: {duplicate_count} duplicate rating record(s)")

    # -- 6. foreign keys: every rating must point at a real user and movie
    orphan_movies = ~ratings["movie_id"].isin(movies["movie_id"])
    orphan_movie_count = int(orphan_movies.sum())
    if orphan_movie_count:
        bad_ids = sorted(ratings.loc[orphan_movies, "movie_id"].unique().tolist())[:10]
        report.error(
            f"ratings.movie_id: {orphan_movie_count} rating(s) reference "
            f"movies missing from u.item (e.g. {bad_ids})"
        )

    orphan_users = ~ratings["user_id"].isin(users["user_id"])
    orphan_user_count = int(orphan_users.sum())
    if orphan_user_count:
        bad_ids = sorted(ratings.loc[orphan_users, "user_id"].unique().tolist())[:10]
        report.error(
            f"ratings.user_id: {orphan_user_count} rating(s) reference "
            f"users missing from u.user (e.g. {bad_ids})"
        )

    # -- 7. tolerable gaps: WARNING severity (see module docstring)
    missing_dates = int(movies["release_date"].isna().sum())
    if missing_dates:
        report.warning(f"movies.release_date: {missing_dates} missing value(s)")

    missing_urls = int(movies["imdb_url"].isna().sum())
    if missing_urls:
        report.warning(f"movies.imdb_url: {missing_urls} missing value(s)")

    return report


def validate_recommendations(
    recommendations: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> ValidationReport:
    """GATE 2 — business-rule checks on the transformed output.

    These are the executable form of the recommendation contract:
    never recommend a watched movie, never trust an under-watched movie,
    never give a user more than three ranked slots.
    """

    report = ValidationReport("RECOMMENDATIONS")

    required = ["user_id", "movie_id", "rank", "avg_rating", "rating_count"]
    missing = [c for c in required if c not in recommendations.columns]
    if missing:
        report.error(f"recommendations: missing required columns {missing}")
        return report

    if recommendations.empty:
        report.error("recommendations: no rows produced")
        return report

    # -- 1. the anti-join in transform must have removed every watched movie
    watched = ratings[["user_id", "movie_id"]].drop_duplicates()
    seen = recommendations.merge(watched, on=["user_id", "movie_id"])
    if len(seen):
        report.error(
            f"{len(seen)} recommendation(s) are movies the user already rated"
        )

    # -- 2. the 50-rating trust threshold must hold for every row
    under_watched = recommendations[recommendations["rating_count"] < MIN_RATING_COUNT]
    if len(under_watched):
        report.error(
            f"{len(under_watched)} recommendation(s) below the "
            f"{MIN_RATING_COUNT}-rating threshold"
        )

    # -- 3. at most 3 per user, ranks exactly 1..n with no gaps or repeats
    per_user = recommendations.groupby("user_id")
    group_sizes = per_user.size()
    oversize = group_sizes[group_sizes > 3]
    if len(oversize):
        report.error(
            f"{len(oversize)} user(s) received more than 3 recommendations"
        )

    bad_ranks = 0
    for _, user_ranks in per_user["rank"]:   # iterates (user_id, ranks) pairs
        ranks = sorted(user_ranks.tolist())
        if ranks != list(range(1, len(ranks) + 1)):
            bad_ranks += 1
    if bad_ranks:
        report.error(
            f"{bad_ranks} user(s) have ranks that are not exactly 1..n"
        )

    # -- 4. the UNIQUE (user_id, movie_id) contract, checked before the DB
    duplicates = recommendations.duplicated(subset=["user_id", "movie_id"])
    duplicate_count = int(duplicates.sum())
    if duplicate_count:
        report.error(f"recommendations: {duplicate_count} duplicate user/movie pair(s)")

    # -- 5. every recommended movie must exist in the movie dimension
    orphan_count = int((~recommendations["movie_id"].isin(movies["movie_id"])).sum())
    if orphan_count:
        report.error(f"recommendations: {orphan_count} movie_id(s) missing from movies")

    # -- 6. score range: guards the NUMERIC(3,2) CHECK constraint upstream
    out_of_range = recommendations[~recommendations["avg_rating"].between(1.0, 5.0)]
    if len(out_of_range):
        report.error(f"{len(out_of_range)} recommendation(s) with score outside 1.00..5.00")

    return report
