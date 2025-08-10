import pandas as pd

# Load the CSV
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")

# Filter Hindi movies
hindi_movies = movies_df[movies_df['original_language'] == 'xx']

# Show list of Hindi movie titles
print("Hindi movies in the dataset:")
print(hindi_movies['title'].tolist())

# Optional: Count Hindi movies
print(f"\nNumber of Hindi movies: {len(hindi_movies)}")
