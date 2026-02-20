import pandas as pd
import ast
import os


# -----------------------------
# Load and Merge Raw Datasets
# -----------------------------
def load_and_merge():
    movies = pd.read_csv("data/raw/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/raw/tmdb_5000_credits.csv")

    # Merge datasets
    movies = movies.merge(credits, left_on='id', right_on='movie_id')

    # Select only necessary columns
    movies = movies[['id', 'title_x', 'genres', 'keywords', 'overview', 'cast']]

    # Rename title column
    movies.rename(columns={'title_x': 'title'}, inplace=True)

    # Drop missing values
    movies.dropna(inplace=True)

    return movies


# -----------------------------
# Convert JSON Columns
# -----------------------------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


def clean_data(movies):

    # Convert JSON string columns into lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    # Extract top 3 cast members
    movies['cast'] = movies['cast'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)[:3]]
    )

    # Remove spaces in names (important for vectorization)
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create tags column
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast']

    # Convert list to string and lowercase
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    return movies


# -----------------------------
# Save Processed Dataset
# -----------------------------
def save_processed(movies):

    # Create processed folder if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned dataset
    movies.to_csv("data/processed/movies_cleaned.csv", index=False)

    print("✅ movies_cleaned.csv saved successfully.")