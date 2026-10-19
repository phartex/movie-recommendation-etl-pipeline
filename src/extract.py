import pandas as pd


def extract_ratings(file_path: str) -> pd.DataFrame:
    """Extract raw movie ratings from u.data."""

    columns = [
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
    ]

    return pd.read_csv(
        file_path,
        sep="\t",
        names=columns,
    )


def extract_movies(file_path: str) -> pd.DataFrame:
    """Extract raw movie metadata from u.item."""

    genre_columns = [
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

    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *genre_columns,
    ]

    return pd.read_csv(
        file_path,
        sep="|",
        names=columns,
        encoding="latin-1",
    )


def extract_users(file_path: str) -> pd.DataFrame:
    """Extract raw user profiles from u.user.

    u.user is pipe-delimited with no header row:
        user_id|age|gender|occupation|zip_code
    Example: 1|24|M|technician|85711

    zip_code stays a string in the raw read — leading zeros are real
    ("02139"), and a naive int read would silently destroy them.
    """

    columns = [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    ]

    return pd.read_csv(
        file_path,
        sep="|",
        names=columns,
        dtype={"zip_code": "string"},
    )
