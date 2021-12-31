import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    replace_spaces_with_underscores,
    get_vectorized_text_array,
    generate_weighted_similarity_matrix,
    get_recommendations,
    get_similarity_matrix,
)
import config

data = pd.read_csv("film_features/film_features.csv")

data[["director", "overview"]] = data[["director", "overview"]].astype("str")

text_cols = ["keywords", "cast", "genres", "director"]
text_list_cols = ["keywords", "cast", "genres"]

data[text_list_cols] = data[text_list_cols].applymap(literal_eval)
data[text_cols] = data[text_cols].applymap(replace_spaces_with_underscores)
data["overview"] = data["overview"].apply(lambda x: x.lower())
data[text_list_cols] = data[text_list_cols].applymap(lambda x: " ".join(x))

X_cast = get_vectorized_text_array(dataframe=data, column="cast")
X_director = get_vectorized_text_array(dataframe=data, column="director")
X_keywords = get_vectorized_text_array(dataframe=data, column="keywords")
X_overview = get_vectorized_text_array(dataframe=data, column="overview", tfidf_vectorizer=True)

cast_similarity = get_similarity_matrix(array=X_cast, index=data.id)
director_similarity = get_similarity_matrix(array=X_director, index=data.id)
keywords_similarity = get_similarity_matrix(array=X_keywords, index=data.id)
overview_similarity = get_similarity_matrix(array=X_overview, index=data.id)

cast_similarity.to_csv(config.BASE_SIMILARITY_PATH + "cast_similarity.csv")
director_similarity.to_csv(config.BASE_SIMILARITY_PATH + "director_similarity.csv")
keywords_similarity.to_csv(config.BASE_SIMILARITY_PATH + "keywords_similarity.csv")
overview_similarity.to_csv(config.BASE_SIMILARITY_PATH + "overview_similarity.csv")
