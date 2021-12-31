from io import BytesIO
import time

import pandas as pd
from tqdm import tqdm
import tmdbsimple as tmdb
from PIL import Image

import config

tmdb.API_KEY = config.TMDB_API_KEY


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three.
        # If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


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
