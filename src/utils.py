import pandas as pd
import numpy as np
import tmdbsimple as tmdb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src import config


tmdb.API_KEY = config.TMDB_API_KEY


def get_unique_list_values(series):
    return set(i for x in series.to_list() for i in eval(x))


def get_filter_values(data):
    cast = sorted(list(get_unique_list_values(data.cast)))
    director = sorted(list(data.director.apply(str).unique()))
    genres = sorted(list(get_unique_list_values(data.genres)))
    return cast, director, genres


def apply_filters(data, cast, director, genres, sort_column="imdb_score"):
    cast_mask = data["cast"].apply(lambda x: bool(set(eval(x)).intersection(set(cast))))
    director_mask = data["director"].isin(director)
    genre_mask = data["genres"].apply(lambda x: bool(set(eval(x)).intersection(set(genres))))
    filtered_films = data[cast_mask & director_mask & genre_mask]
    filtered_films = filtered_films.sort_values(by=sort_column, ascending=False)
    return filtered_films


def replace_spaces_with_underscores(x):
    if isinstance(x, list):
        return [str(text).lower().replace(" ", "_") for text in x]
    elif isinstance(x, str):
        return x.lower().replace(" ", "_")
    else:
        return ""


@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data


@st.cache
def load_similarity_matrices():
    cast_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "cast_similarity.csv").set_index(
        "id"
    )
    director_similarity = pd.read_csv(
        config.BASE_SIMILARITY_PATH + "director_similarity.csv"
    ).set_index("id")
    overview_similarity = pd.read_csv(
        config.BASE_SIMILARITY_PATH + "overview_similarity.csv"
    ).set_index("id")
    keywords_similarity = pd.read_csv(
        config.BASE_SIMILARITY_PATH + "keywords_similarity.csv"
    ).set_index("id")
    user_embedding_similarity = pd.read_csv(
        config.BASE_SIMILARITY_PATH + "collaborative_similarity.csv"
    )
    user_embedding_similarity.index = user_embedding_similarity.columns
    return (
        cast_similarity,
        director_similarity,
        keywords_similarity,
        overview_similarity,
        user_embedding_similarity,
    )


def get_vectorized_text_array(dataframe: pd.DataFrame, column: str, tfidf_vectorizer: bool = False):
    if tfidf_vectorizer:
        vectorizer = TfidfVectorizer(stop_words="english")
    else:
        vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(dataframe[column])
    return X.toarray().astype("float32")


def get_similarity_matrix(array, index):
    X = pd.DataFrame(cosine_similarity(array, array), index=index, columns=index)
    return X


def generate_weighted_similarity_matrix(arrays: list, weights: list):
    df = arrays[0]
    indices = df.index.values
    arrays = [np.array(array, dtype="float32") for array in arrays]
    weighted_similarity_matrix = np.average(np.array(arrays), axis=0, weights=weights)
    return pd.DataFrame(weighted_similarity_matrix, index=indices, columns=indices)


def get_recommendations(
    films: pd.DataFrame, titles: list, similarity_matrix: pd.DataFrame, top_n: int
):
    film_indices = films.loc[films.title.isin(titles), "id"].values
    closest_films = similarity_matrix[film_indices].mean(axis=1)
    closest_films = closest_films.drop(labels=film_indices)
    closest_films = closest_films.sort_values(ascending=False)
    closest_films = closest_films[:top_n]
    id_title_map = dict(zip(films.id, films.title))
    closest_films = [id_title_map[idx] for idx in closest_films.index.values]
    return closest_films
