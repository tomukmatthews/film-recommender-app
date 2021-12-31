import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

import config

data = pd.read_csv("film_features/film_features.csv")
ratings = pd.read_csv("data/ratings.csv")
ratings = ratings[ratings["movieId"].isin(data.id)]
valid_users = ratings.userId.value_counts() > 1
ratings = ratings[ratings["userId"].isin(valid_users.index.values)]
user_film_matrix = pd.crosstab(
    index=ratings.userId, columns=ratings.movieId, values=ratings.rating, aggfunc="mean"
).fillna(0)
film_user_matrix = user_film_matrix.T
film_ids = film_user_matrix.reset_index().movieId.to_list()
SVD = TruncatedSVD(n_components=25)
X = SVD.fit_transform(film_user_matrix)
collaborative_similarity = cosine_similarity(X, X)
user_embedding_similarity = pd.DataFrame(collaborative_similarity, index=film_ids, columns=film_ids)

# There are some with no views from users that must be added to the similarity matrix
missing_indices = list(set(data.id.values) - set(user_embedding_similarity.index.values))
missing_indices_df_cols = pd.DataFrame(columns=[str(id) for id in missing_indices])
missing_indices_df_index = pd.DataFrame(index=[str(id) for id in missing_indices])
collaborative_similarity_df = pd.concat(
    [user_embedding_similarity, missing_indices_df_index], axis=0
)
collaborative_similarity_df = pd.concat(
    [collaborative_similarity_df, missing_indices_df_cols], axis=1
)
collaborative_similarity_df = collaborative_similarity_df.fillna(0)
collaborative_similarity_df.to_csv(
    config.BASE_SIMILARITY_PATH + "collaborative_similarity.csv", index=False
)
