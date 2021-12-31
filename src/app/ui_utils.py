import pandas as pd

from src import config


def display_film_posters(streamlit, data: pd.DataFrame, num_rows: int, posters_per_row: int):
    posters = data.poster_path.head(n=num_rows * posters_per_row).to_list()
    titles = data.title.head(n=num_rows * posters_per_row).to_list()
    directors = data.director.head(n=num_rows * posters_per_row).to_list()
    release_date = data.release_date.head(n=num_rows * posters_per_row).to_list()
    cast = data.cast.head(n=num_rows * posters_per_row).to_list()
    keywords = data.keywords.head(n=num_rows * posters_per_row).to_list()
    rating = data.vote_average.head(n=num_rows * posters_per_row).to_list()

    posters = [config.POSTER_BASE_URL + poster for poster in posters]

    for row in range(num_rows):
        cols = streamlit.columns(posters_per_row)
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
                with col.expander(f"{titles[index]}: {rating[index]}"):
                    streamlit.markdown(f"**Director:** {directors[index]}")
                    cast_string = cast[index].replace("[", "").replace("]", "").replace("'", "")
                    keywords_string = (
                        keywords[index].replace("[", "").replace("]", "").replace("'", "")
                    )
                    streamlit.write(f"**Cast:** {cast_string}")
                    streamlit.write(f"**Date:** {release_date[index]}")
                    streamlit.write(f"**Keywords:** {keywords_string}")
                col.text("")
            else:
                break


def display_text(streamlit, text_list):
    if not isinstance(text_list, list):
        text_list = list(text_list)
    for text in text_list:
        streamlit.markdown(f"**{text}**")
    return streamlit


def display_parameter_controls(streamlit, min_value: float, max_value: float, default_value: float):
    col1, col2 = streamlit.sidebar.columns(2)
    director = col1.slider(
        label="Director", min_value=min_value, max_value=max_value, value=default_value
    )
    cast = col1.slider(label="Cast", min_value=min_value, max_value=max_value, value=default_value)
    keywords = col2.slider(
        label="Keywords", min_value=min_value, max_value=max_value, value=default_value
    )
    overview = col2.slider(
        label="Overview", min_value=min_value, max_value=max_value, value=default_value
    )
    user = streamlit.sidebar.slider(
        label="Similar User Preferences",
        min_value=min_value,
        max_value=max_value * 2,
        value=default_value * 2,
    )
    return cast, director, keywords, overview, user
