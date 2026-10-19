from extract import extract_movies, extract_ratings


ratings = extract_ratings("data/u.data")
movies = extract_movies("data/u.item")

print("Ratings:")
print(ratings.head())

print("\nMovies:")
print(movies.head())

print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)