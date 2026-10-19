import pandas as pd

def transform_ratings(ratings :pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the ratings dataset."""

    ratings = ratings.copy()

    ratings["user_id"] = ratings["user_id"].astype("int64")
    ratings["movie_id"] = ratings["movie_id"].astype("int64")
    ratings["rating"] = ratings["rating"].astype("int8")

    ratings["timestamp"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        utc=True,
    )

    return ratings


def transform_movies(movies :pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise movie metadata."""

    movies = movies.copy()


    movies["movie_id"] = movies["movie_id"].astype("int64")

    # u.item pads 3 titles with trailing spaces before the "|" delimiter.
    movies["title"] = movies["title"].str.strip()

    movies["release_date"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce",
    )

    return movies


def transform_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise user profiles.

    zip_code is cast to a string deliberately, never to int: US zip codes
    such as 02139 (Cambridge, MA) lose their leading zero the moment they
    pass through a numeric type.
    """

    users = users.copy()

    users["user_id"] = users["user_id"].astype("int64")
    users["age"] = users["age"].astype("int8")
    users["gender"] = users["gender"].str.strip()
    users["occupation"] = users["occupation"].str.strip()

    # "string" (pandas dtype) keeps the dtype explicit even when empty;
    # str.strip() removes any padding the raw file may carry.
    users["zip_code"] = users["zip_code"].astype("string").str.strip()

    return users


def calculate_movie_statistics(
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rating statistics for each movie."""

    return (
        ratings
        .groupby("movie_id")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )


GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

MIN_PREFERENCE_RATING = 4
TOP_GENRES_PER_USER = 3
MIN_RATING_COUNT = 50


def transform_movie_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """Convert the movie genre flags into one row per movie/genre."""

    genre_data = movies[["movie_id", *GENRE_COLUMNS]].copy()

    genre_data = genre_data.melt(
        id_vars=["movie_id"],
        value_vars=GENRE_COLUMNS,
        var_name="genre",
        value_name="is_genre",
    )

    return genre_data[genre_data["is_genre"] == 1][
        ["movie_id", "genre"]
    ].reset_index(drop=True)

def build_user_preferences(ratings : pd.DataFrame, movie_genres : pd.DataFrame) -> pd.DataFrame:
    """Find each user's preferred genres based on their high ratings."""

    # Step 1: each rating expands to one row per genre of that movie
    rated_genres =  ratings.merge(movie_genres ,on="movie_id" , how = "inner")

    #Step 2: keep only strong positive signals
    high_ratings = rated_genres[rated_genres["rating"] >= MIN_PREFERENCE_RATING]

    # Step 3: count high-rating movies per (user, genre) pair
    genre_counts = (
        high_ratings
        .groupby(["user_id","genre"])
        .agg(high_rating_count = ("rating", "count"))
        .reset_index()
    )

    # Step 4: rank each user's genres by popularity, keep the top 3.
    # Ties broken alphabetically by genre name so the output is deterministic.
    genre_counts = genre_counts.sort_values(
        ["user_id", "high_rating_count", "genre"],
        ascending=[True, False, True],
    )

    return (
        genre_counts
        .groupby("user_id")
        .head(TOP_GENRES_PER_USER)
        .reset_index(drop=True)
    )

def build_recommendations(
    preferences: pd.DataFrame,
    movie_genres: pd.DataFrame,
    ratings: pd.DataFrame,
    movie_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Produce top-3 movie recommendations for every user."""

    # Step 1: favourite genres -> candidate (user_id, movie_id) pairs
    # expect 1,204,491 rows (one user matches a movie once per shared genre)
    candidates = preferences.merge(movie_genres, on="genre")

    # Step 2: one row per (user, movie)
    # expect 1,033,262 rows
    # genre/high_rating_count belonged to the user-genre pair and are now
    # meaningless per movie, so they go; dedupe prevents the same movie
    # appearing twice in one user's top 3
    candidates = candidates.drop(columns=["genre", "high_rating_count"])
    candidates = candidates.drop_duplicates(["user_id", "movie_id"])

    # Step 3: anti-join — drop movies the user has already watched
    # expect 954,153 rows
    watched = ratings[["user_id", "movie_id"]].drop_duplicates()
    candidates = candidates.merge(
        watched, on=["user_id", "movie_id"], how="left", indicator=True,
    )
    candidates = candidates[candidates["_merge"] == "left_only"].drop(columns="_merge")

    # Step 4: Rule B gate — only movies with enough ratings to trust
    # expect 322,422 rows (598 distinct movies survive)
    candidates = candidates.merge(movie_statistics, on="movie_id")
    candidates = candidates[candidates["rating_count"] >= MIN_RATING_COUNT]

    # Step 5: deterministic ranking, then top 3 per user
    # expect 2,826 rows = 942 users x 3
    candidates = candidates.sort_values(
        ["user_id", "avg_rating", "rating_count", "movie_id"],
        ascending=[True, False, False, True],
    )
    top3 = candidates.groupby("user_id").head(3).copy()

    # rank 1..3 within each user — rows are already in ranked order,
    # so a simple position counter gives each row its place
    top3["rank"] = top3.groupby("user_id").cumcount() + 1

    return top3.reset_index(drop=True)


