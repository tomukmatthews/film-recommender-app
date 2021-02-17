MIN_FILM_DATE = "01/01/1970"
EXCLUDE_LANGUAGES = ["hi"]
USEFUL_COLUMNS = [
    "id",
    "imdb_id",
    "title",
    "adult",
    "original_language",
    "poster_path",
    "poster_path_updated",
    "release_date",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "overview",
    "keywords",
    "cast",
    "director",
    "genres",
    "popularity",
]
GET_VALID_POSTER_PATHS = False
NUM_FILMS_TO_KEEP = 3500

# Main
C = 5.6  # C = data["vote_average"].mean()
m = 156  # m = data["vote_count"].quantile(0.90)

IMAGE_WIDTH = 175
NUM_POSTER_ROWS = 15
POSTERS_PER_ROW = 8
POSTER_BASE_URL = "https://image.tmdb.org/t/p/original/"
TMDB_API_KEY = "a1278aeed02f77120a749d2b13b14cec"
DATA_PATH = "film_features/film_features.csv"
BASE_SIMILARITY_PATH = "similarity_matrices/"
