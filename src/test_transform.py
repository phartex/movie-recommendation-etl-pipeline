import pandas as pd

from extract import extract_movies, extract_ratings
from transform import (
    calculate_movie_statistics,
    transform_movies,
    transform_ratings,
    build_user_preferences,
    build_recommendations
)
from transform import transform_movie_genres
# 1. Extract the raw data
ratings = extract_ratings("data/u.data")
movies = extract_movies("data/u.item")

# 2. Transform the data
clean_ratings = transform_ratings(ratings)
clean_movies = transform_movies(movies)

# 3. Build the movie-genre bridge table
movie_genres = transform_movie_genres(clean_movies)

# 3. Calculate movie statistics
movie_statistics = calculate_movie_statistics(clean_ratings)

preferences = build_user_preferences(clean_ratings, movie_genres)
recommendations = build_recommendations(
    preferences, movie_genres, clean_ratings, movie_statistics,
)

print("=== USER 196'S RECOMMENDATIONS ===")
u196 = recommendations[recommendations["user_id"] == 196]
print(u196.merge(clean_movies[["movie_id", "title"]], on="movie_id"))

print("\n=== SHAPE ===")
print(recommendations.shape)  # must be (2826, 6)
print("users:", recommendations["user_id"].nunique())  # must be 942
