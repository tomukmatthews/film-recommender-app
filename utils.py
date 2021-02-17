import pandas as pd
import numpy as np
import requests
import tmdbsimple as tmdb
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import config

tmdb.API_KEY = config.TMDB_API_KEY


def replace_spaces_with_underscores(x):
    if isinstance(x, list):
        return [str(text).lower().replace(" ", "_") for text in x]
    elif isinstance(x, str):
        return x.lower().replace(" ", "_")
    else:
        return ""


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data


@st.cache
def load_similarity_matrices():
    cast_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "cast_similarity.csv").set_index("id")
    director_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "director_similarity.csv").set_index("id")
    overview_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "overview_similarity.csv").set_index("id")
    keywords_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "keywords_similarity.csv").set_index("id")
    user_embedding_similarity = pd.read_csv(config.BASE_SIMILARITY_PATH + "collaborative_similarity.csv")
    user_embedding_similarity.index = user_embedding_similarity.columns
    return cast_similarity, director_similarity, keywords_similarity, overview_similarity, user_embedding_similarity


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


def weighted_rating(x, m, C):
    v = x["vote_count"]
    R = x["vote_average"]
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


def assign_poster_path(row):
    poster_path = row.poster_path
    poster_path_updated = True
    try:
        response = requests.get(config.POSTER_BASE_URL + poster_path)
        Image.open(BytesIO(response.content))
    except:
        try:
            movie = tmdb.Movies(int(row.id))
            response = movie.info()
            poster_path = movie.poster_path
        except Exception as e:
            print(e)
            poster_path_updated = False
    return poster_path, poster_path_updated


def update_poster_paths(dataframe, runtime_seconds):
    films_updated = dataframe[dataframe["poster_path_updated"]]
    films_not_updated = dataframe[~dataframe["poster_path_updated"]]

    with tqdm(total=films_not_updated.shape[0]) as pbar:
        start = time.time()
        for row in films_not_updated.itertuples():
            if time.time() <= start + runtime_seconds:
                pbar.update(1)
                (
                    films_not_updated.at[row.Index, "poster_path"],
                    films_not_updated.at[row.Index, "poster_path_updated"],
                ) = assign_poster_path(row)
            else:
                break
    dataframe = pd.concat([films_updated, films_not_updated])
    return dataframe


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


def get_recommendations(films: pd.DataFrame, titles: list, similarity_matrix: pd.DataFrame, top_n: int):
    film_indices = films.loc[films.title.isin(titles), "id"].values
    closest_films = similarity_matrix[film_indices].mean(axis=1)
    closest_films = closest_films.drop(labels=film_indices)
    closest_films = closest_films.sort_values(ascending=False)
    closest_films = closest_films[:top_n]
    id_title_map = dict(zip(films.id, films.title))
    closest_films = [id_title_map[idx] for idx in closest_films.index.values]
    return closest_films


def add_film_posters(streamlit, data: pd.DataFrame, num_rows: int, posters_per_row: int):
    posters = data.poster_path.head(n=num_rows * posters_per_row).to_list()
    titles = data.title.head(n=num_rows * posters_per_row).to_list()
    directors = data.director.head(n=num_rows * posters_per_row).to_list()
    release_date = data.release_date.head(n=num_rows * posters_per_row).to_list()
    cast = data.cast.head(n=num_rows * posters_per_row).to_list()
    keywords = data.keywords.head(n=num_rows * posters_per_row).to_list()
    rating = data.vote_average.head(n=num_rows * posters_per_row).to_list()

    posters = [config.POSTER_BASE_URL + poster for poster in posters]

    for row in range(num_rows):
        cols = streamlit.beta_columns(posters_per_row)
        all_posters_shown = len(posters) < row * posters_per_row
        if all_posters_shown:
            break
        for idx in range(posters_per_row):
            all_posters_shown = idx + (row * posters_per_row) == len(posters)
            if not all_posters_shown:
                col = cols[idx]
                index = idx + (row * posters_per_row)
                # streamlit.write(index)
                col.image(posters[index], use_column_width=True)
                with col.beta_expander(f"{titles[index]}: {rating[index]}"):
                    streamlit.markdown(f"**Director:** {directors[index]}")
                    cast_string = cast[index].replace("[", "").replace("]", "").replace("'", "")
                    keywords_string = keywords[index].replace("[", "").replace("]", "").replace("'", "")
                    streamlit.write(f"**Cast:** {cast_string}")
                    streamlit.write(f"**Date:** {release_date[index]}")
                    streamlit.write(f"**Keywords:** {keywords_string}")
                col.text("")
            else:
                break


def add_text(streamlit, text_list):
    if not isinstance(text_list, list):
        text_list = list(text_list)
    for text in text_list:
        streamlit.markdown(f"**{text}**")
    return streamlit


def add_parameter_controls(streamlit, min_value: float, max_value: float, default_value: float):
    col1, col2 = streamlit.sidebar.beta_columns(2)
    director = col1.slider(label="Director", min_value=min_value, max_value=max_value, value=default_value)
    cast = col1.slider(label="Cast", min_value=min_value, max_value=max_value, value=default_value)
    keywords = col2.slider(label="Keywords", min_value=min_value, max_value=max_value, value=default_value)
    overview = col2.slider(label="Overview", min_value=min_value, max_value=max_value, value=default_value)
    user = streamlit.sidebar.slider(
        label="Similar User Preferences", min_value=min_value, max_value=max_value * 2, value=default_value * 2
    )
    return cast, director, keywords, overview, user

