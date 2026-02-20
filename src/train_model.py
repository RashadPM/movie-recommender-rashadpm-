import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def train():

    print("📂 Loading processed dataset...")
    movies = pd.read_csv("data/processed/movies_cleaned.csv")

    # Make sure no NaN values
    movies['tags'] = movies['tags'].fillna("")

    print("🧠 Applying TF-IDF vectorization...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])

    print("📊 Calculating cosine similarity...")
    similarity = cosine_similarity(tfidf_matrix)

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    print("💾 Saving model files...")

    # Save similarity matrix
    with open("models/similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    # Save movies dataframe as CSV
    movies.to_csv("models/movies.csv", index=False)

    print("✅ Model training completed successfully!")