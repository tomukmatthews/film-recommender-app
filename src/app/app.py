from pandas.api.types import CategoricalDtype
import streamlit as st

from app.ui_utils import display_film_posters, display_parameter_controls
from utils import (
    load_data,
    load_similarity_matrices,
    generate_weighted_similarity_matrix,
    get_recommendations,
    get_filter_values,
    apply_filters,
)
import config

st.set_page_config(layout="wide")

data = load_data(config.DATA_PATH)

filtered_films = data.copy(deep=True)

# Sidebar
st.sidebar.title("Enter films you like then click personal recommendations")
liked_films = st.sidebar.multiselect("", data["title"])
film_display_option = st.sidebar.empty()
film_selector = 0
option = film_display_option.radio(
    label="Show me",
    options=["Top Films", "Personal Recommendations"],
    index=film_selector,
)

# Main
st.title("Film Recommender System")
cast_options, director_options, genres_options = get_filter_values(filtered_films)
cast_default, director_default, genres_default = get_filter_values(data)
cast_filter, director_filter, genre_filter = st.beta_columns(3)
cast = cast_filter.multiselect("Filter by cast", cast_options)
director = director_filter.multiselect("Filter by director", director_options)
genres = genre_filter.multiselect("Filter by genre", genres_options)
cast = cast_default if not cast else cast
director = director_default if not director else director
genres = genres_default if not genres else genres

if option == "Top Films":
    filtered_films = apply_filters(
        data=data, cast=cast, director=director, genres=genres, sort_column="imdb_score"
    )
    filtered_films = filtered_films.loc[filtered_films["vote_count"] >= config.m]
    display_film_posters(
        streamlit=st,
        data=filtered_films,
        num_rows=config.NUM_POSTER_ROWS,
        posters_per_row=config.POSTERS_PER_ROW,
    )

elif option == "Personal Recommendations":
    if len(liked_films) == 0:
        st.sidebar.write("Please select at least one film for recommendations.")
    else:
        (
            cast_similarity,
            director_similarity,
            keywords_similarity,
            overview_similarity,
            user_embedding_similarity,
        ) = load_similarity_matrices()

        st.sidebar.write("Choose recommendation focus")
        (
            cast_weight,
            director_weight,
            keywords_weight,
            overview_weight,
            user_embedding_weight,
        ) = display_parameter_controls(
            streamlit=st,
            min_value=config.PARAMETER_CONTROL_MIN,
            max_value=config.PARAMETER_CONTROL_MAX,
            default_value=config.PARAMETER_CONTROL_DEFAULT,
        )
        with st.sidebar.beta_expander("Click to see how this works:"):
            st.write(config.APP_EXPLANATION_1)
            st.write(config.APP_EXPLANATION_2)
            st.write(config.SOURCE_CODE_LINK)
        sim_matrices = [
            cast_similarity,
            director_similarity,
            keywords_similarity,
            overview_similarity,
            user_embedding_similarity,
        ]
        sim_weights = [
            cast_weight,
            director_weight,
            keywords_weight,
            overview_weight,
            user_embedding_weight,
        ]
        sim_mat = generate_weighted_similarity_matrix(arrays=sim_matrices, weights=sim_weights)
        recommendations = get_recommendations(
            films=data,
            titles=liked_films,
            similarity_matrix=sim_mat,
            top_n=config.POSTERS_PER_ROW * config.NUM_POSTER_ROWS,
        )
        filtered_films = data[data["title"].isin(recommendations)]

        recommendation_order = CategoricalDtype(recommendations, ordered=True)
        filtered_films["title"] = filtered_films["title"].astype(recommendation_order)
        filtered_films = filtered_films.sort_values("title")

        display_film_posters(
            streamlit=st,
            data=filtered_films,
            num_rows=config.NUM_POSTER_ROWS,
            posters_per_row=config.POSTERS_PER_ROW,
        )
