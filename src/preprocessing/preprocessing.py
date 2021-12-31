import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import tmdbsimple as tmdb
import time
from PIL import Image
from io import BytesIO

from utils import get_director, get_list
import config

tmdb.API_KEY = config.TMDB_API_KEY


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


# Load keywords and credits
credits = pd.read_csv("data/credits.csv")
keywords = pd.read_csv("data/keywords.csv")
metadata = pd.read_csv("data/movies_metadata.csv")
updated_poster_paths = pd.read_csv("data/updated_poster_paths.csv")

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords["id"] = keywords["id"].astype("int")
credits["id"] = credits["id"].astype("int")
metadata["id"] = metadata["id"].astype("int")

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on="id")
metadata = metadata.merge(keywords, on="id")

metadata = metadata.drop(["poster_path"], axis=1)
metadata = metadata.merge(updated_poster_paths, on="id")

features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

metadata["director"] = metadata["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
metadata[features] = metadata[features].applymap(get_list)

film_features = metadata[config.USEFUL_COLUMNS]
film_features = film_features.drop_duplicates(subset="id", keep="first")
film_features["imdb_score"] = film_features.apply(
    weighted_rating, axis=1, args=(config.m, config.C)
)
film_features = film_features.sort_values(by="imdb_score", ascending=False)

film_features["popularity"] = film_features["popularity"].astype("float32")
film_features["scaled_popularity"] = (
    film_features["popularity"] - film_features["popularity"].mean()
) / film_features["popularity"].std()
film_features["scaled_imdb_score"] = (
    film_features["imdb_score"] - film_features["imdb_score"].mean()
) / film_features["imdb_score"].std()
film_features["overall_score"] = (
    film_features["scaled_imdb_score"] + film_features["scaled_popularity"]
)
film_features = film_features[~film_features["original_language"].isin(config.EXCLUDE_LANGUAGES)]
film_features = film_features[film_features["poster_path_updated"]]
film_features = film_features[film_features.release_date > config.MIN_FILM_DATE]
film_features = film_features.sort_values(by="overall_score", ascending=False)
film_features = film_features.drop(
    ["scaled_imdb_score", "scaled_popularity", "overall_score"], axis=1
)

film_features = film_features.head(n=config.NUM_FILMS_TO_KEEP)

assert len(film_features) == film_features.id.nunique(), "Contains duplicate movies"

film_features.to_csv("film_features/film_features.csv", index=False)
