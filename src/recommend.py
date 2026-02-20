import pickle
import pandas as pd


def recommend(movie_name, top_n=5):

    # Load models INSIDE the function
    similarity = pickle.load(open("models/similarity.pkl", "rb"))
    movies = pd.read_csv("models/movies.csv")

    movie_name = movie_name.lower()

    matches = movies[movies['title'].str.lower() == movie_name]

    if matches.empty:
        return ["❌ Movie not found in database."]

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:top_n+1]

    recommended_movies = [movies.iloc[i[0]].title for i in movies_list]

    return recommended_movies