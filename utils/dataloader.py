import os
import requests
import zipfile
import pandas as pd
import re
import numpy as np
import sys
sys.path.append('.')
class DataFormat:
    def __init__(
        self,
        sep,
        data_path,
        item_sep=None,
        item_path=None,
        user_features_path=None
    ):
        """MovieLens data format container as a different size of MovieLens data file
        has a different format

        Args:
            sep (str): Rating data delimiter
            data_path (str): Rating data path within the original zip file
            item_sep (str): Item data delimiter
            item_path (str): Item data path within the original zip file
            user_features_path (str): user features data path within the original zip file
        """

        # Rating file
        self._sep = sep
        self._data_path = data_path

        # Item file
        self._item_sep = item_sep
        self._item_path = item_path

        # User features file, only for 100k
        self._user_features_path = user_features_path

    @property
    def separator(self):
        return self._sep

    @property
    def data_path(self):
        return self._data_path

    @property
    def item_separator(self):
        return self._item_sep

    @property
    def item_path(self):
        return self._item_path
    
    @property
    def user_features_path(self):
        return self._user_features_path

DATA_FORMAT = {
    "100k": DataFormat("\t", "ml-100k/u.data", "|", "ml-100k/u.item", "ml-100k/u.user"),
    "1m": DataFormat(
        "::", "ml-1m/ratings.dat",  "::", "ml-1m/movies.dat", ""
    )
}

def maybe_download_and_unzip(url, dest_path):
    dirs, file_name = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    zip_path = os.path.join(dirs, file_name)
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dirs)

    os.remove(zip_path)

def download_movielens(size):
    if size not in DATA_FORMAT:
        raise ValueError(f"Invalid size: {size}.")

    url = "https://files.grouplens.org/datasets/movielens/ml-" + size + ".zip"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(current_dir)
    dest_path = os.path.join(data_dir, "data", f"{size}.zip")
    maybe_download_and_unzip(url, dest_path)

def maybe_download(size, data_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(current_dir)

    if data_name == "u.data":
        file_name = DATA_FORMAT[size].data_path
    elif data_name == "u.item":
        file_name = DATA_FORMAT[size].item_path
    elif size == "100k" and data_name == "u.user":
        file_name = DATA_FORMAT[size].user_features_path
    else:
        print("data name error")

    filepath = os.path.join(data_dir, "data", file_name)
    if not os.path.exists(filepath):
        download_movielens(size)
    else:
        print("file is already exist")
    return filepath

# To load UserId, ItemId, Rating and Timestamp from u.data
def load_data_df(size):
    COL_USER = "UserId"
    COL_ITEM = "ItemId"
    COL_RATING = "Rating"
    COL_TIMESTAMP = "Timestamp"
    filepath = maybe_download(size, "u.data")
    print(filepath)
    data = pd.read_csv(filepath, sep=DATA_FORMAT[size].separator, names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP])
    return data

#################load_data_df use case example ###########
# data = load_data_df("1m") # "100k or 1m"
# print(data)
#################

# 100K data genres index to string mapper. For 1m, 10m, and 20m, the genres labels are already in the dataset.
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
def load_item_df(size, item_datapath, item_col, title_col=None, genres_col=None, year_col=None):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [item_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend((range(5, 24)))  # genres columns
        else:
            item_header.append(genres_col)
            usecols.append(2)  # genres column

    item_df = pd.read_csv(
        item_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=None,
        encoding="ISO-8859-1",
    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:
        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df

#################load_item_df use case example ###########
# use example, set the cols you want, if you not set title_col, it will not return

# filepath = maybe_download("1m", "u.item")
# items = load_item_df("1m", filepath,
#         item_col="item",
#         title_col="title",
#         genres_col="genres",
#         year_col="year")
# print(items)

#################

def load_user_features(path):
    """Load user features
    """
    data = pd.read_csv(
        path,
        delimiter="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return data

#################load_user_features use case example ###########

# filepath = maybe_download("100k", "u.user")
# items = load_user_features(filepath)
# print(items)

#################









