import streamlit as st
import pandas as pd
import requests
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎬 AI Movie Recommender")
st.markdown("Find movies similar to your favorite ones.")

API_KEY = os.getenv("TMDB_API_KEY")

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
    credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[[
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]]

    movies.dropna(inplace=True)

    # Convert string to list safely
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Extract genre names safely
    def extract_genres(obj):
        try:
            return [i["name"] for i in obj]
        except:
            return []

    movies["genres"] = movies["genres"].apply(extract_genres)

    # Extract keyword names
    movies["keywords"] = movies["keywords"].apply(
        lambda x: [i["name"] for i in x] if isinstance(x, list) else []
    )

    # Extract top 3 cast
    movies["cast"] = movies["cast"].apply(
        lambda x: [i["name"] for i in x[:3]] if isinstance(x, list) else []
    )

    # Extract director
    def fetch_director(obj):
        try:
            for i in obj:
                if i["job"] == "Director":
                    return i["name"]
        except:
            return ""
        return ""

    movies["director"] = movies["crew"].apply(fetch_director)

    # Create tags
    movies["tags"] = (
        movies["overview"].fillna("") + " " +
        movies["genres"].astype(str) + " " +
        movies["keywords"].astype(str) + " " +
        movies["cast"].astype(str) + " " +
        movies["director"].fillna("")
    )

    movies = movies[[
        "movie_id",
        "title",
        "overview",
        "genres",
        "tags"
    ]]

    return movies


# -----------------------------
# BUILD SIMILARITY MATRIX
# -----------------------------
@st.cache_data
def build_similarity(movies):
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(movies["tags"])
    similarity = cosine_similarity(vectors)
    return similarity


movies = load_and_prepare_data()
similarity = build_similarity(movies)

# -----------------------------
# FETCH POSTER
# -----------------------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={API_KEY}"
        data = requests.get(url).json()

        if data.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        return None
    return None


# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = []

    for i in movie_list:
        movie_data = movies.iloc[i[0]]
        poster = fetch_poster(movie_data.movie_id)

        recommended.append({
            "title": movie_data.title,
            "poster": poster,
            "score": round(i[1] * 100, 2),
            "overview": movie_data.overview if pd.notna(movie_data.overview) else "",
            "genres": ", ".join(movie_data.genres) if isinstance(movie_data.genres, list) else ""
        })

    return recommended


# -----------------------------
# UI SECTION
# -----------------------------
selected_movie = st.selectbox(
    "Search for a movie",
    movies["title"].values
)

if st.button("Recommend"):

    with st.spinner("Generating recommendations..."):
        results = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(len(results)):
        with cols[i]:

            if results[i]["poster"]:
                st.image(results[i]["poster"])

            st.markdown(f"### {results[i]['title']}")
            st.markdown(f"⭐ {results[i]['score']}% Match")

            # Genre
            if results[i]["genres"]:
                st.caption(f"🎭 {results[i]['genres']}")

            # Overview
            if results[i]["overview"]:
                st.write(results[i]["overview"][:200] + "...")