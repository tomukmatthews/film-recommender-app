import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from preprocessing.utils import update_poster_paths

try:
    dataframe = pd.read_csv("data/updated_poster_paths.csv")
except:
    dataframe = pd.read_csv("data/movies_metadata.csv")
    # Remove rows with bad IDs.
    dataframe = dataframe.drop([19730, 29503, 35587])
    dataframe["id"] = dataframe["id"].astype("int")
    dataframe = dataframe[["id", "poster_path"]]
    dataframe["poster_path_updated"] = False

updated_poster_paths = update_poster_paths(dataframe=dataframe, runtime_seconds=500)
updated_poster_paths.to_csv("data/updated_poster_paths.csv", index=False)
