import os
from typing import Optional
import requests
import zipfile
import pandas as pd
import re
import numpy as np
import sys
# import tqdm
# sys.path.append('.')


class MovieLensConfig:
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

    COLUMN_NAMES = {
        'user': 'user_id',
        'item': 'item_id',
        'rating': 'rating',
        'timestamp': 'timestamp',
        'title': 'title',
        'genres': 'genres',
        'year': 'year',
        # User features
        'age': 'age',
        'gender': 'gender',
        'occupation': 'occupation',
        'zipcode': 'zip_code'
    }

     # Age groupings for 1M dataset
    AGE_GROUPS = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    
    # Occupation mappings
    OCCUPATIONS_100K = {
        "administrator": 0,
        "artist": 1,
        "doctor": 2,
        "educator": 3,
        "engineer": 4,
        "entertainment": 5,
        "executive": 6,
        "healthcare": 7,
        "homemaker": 8,
        "lawyer": 9,
        "librarian": 10,
        "marketing": 11,
        "none": 12,
        "other": 13,
        "programmer": 14,
        "retired": 15,
        "salesman": 16,
        "scientist": 17,
        "student": 18,
        "technician": 19,
        "writer": 20
    }

    OCCUPATIONS_1M = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }
    
    BASE_URL = "https://files.grouplens.org/datasets/movielens/ml-{size}.zip"
    

class DataFormat:
    def __init__(
        self,
        ratings_separator: str,
        ratings_path: str,
        items_separator: Optional[str] = None,
        items_path: Optional[str] = None,
        users_separator: Optional[str] = None,
        user_features_path: Optional[str] = None
    ):
        self.ratings_separator = ratings_separator
        self.ratings_path = ratings_path
        self.items_separator = items_separator
        self.items_path = items_path
        self.users_separator = users_separator
        self.user_features_path = user_features_path


class DataLoader:
    DATA_FORMATS = {
        "100k": DataFormat(
            ratings_separator="\t",
            ratings_path="ml-100k/u.data",
            items_separator="|",
            items_path="ml-100k/u.item",
            users_separator="|",
            user_features_path="ml-100k/u.user"
        ),
        "1m": DataFormat(
            ratings_separator="::",
            ratings_path="ml-1m/ratings.dat",
            items_separator="::",
            items_path="ml-1m/movies.dat",
            users_separator="::",
            user_features_path="ml-1m/users.dat"
        )
    }

    def __init__(self, size: str) -> None:
        if size not in self.DATA_FORMATS:
            raise ValueError(f"Invalid size: {size}. Supported sizes: {list(self.DATA_FORMATS.keys())}")
        
        self.size = size
        self.data_format = self.DATA_FORMATS[size]
        self.data_dir = self._get_data_dir()

    def _get_data_dir(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(current_dir), "data")
    
    def _download_and_extract(self, url: str, dest_path: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        print(f"Downloading dataset from {url}...")
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise an exception for error HTTP statuses
                with open(dest_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            with zipfile.ZipFile(dest_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(dest_path))

            print("Dataset downloaded and extracted successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
        except zipfile.BadZipfile as e:
            print(f"Error extracting zip file: {e}")
        finally:
            os.remove(dest_path)

    def _check_data_and_download(self, filename: str) -> str:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            url = MovieLensConfig.BASE_URL.format(size=self.size)
            dest_path = os.path.join(self.data_dir, f"{self.size}.zip")
            self._download_and_extract(url, dest_path)
        return filepath

    def load_ratings(self) -> pd.DataFrame:
        """        
        Returns:
            pandas dataFrame: user_id, movie_id, rating, timestamp
        """
        filepath = self._check_data_and_download(self.data_format.ratings_path)
        
        df =  pd.read_csv(
            filepath,
            sep=self.data_format.ratings_separator,
            names=['user','item','rating','timestamp'],
            engine="python"
        )
        return df
    
    def load_user_features(self, 
                           convert_age_to_range: bool = False, 
                           convert_occupation_to_code: bool = False) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: DataFrame containing user demographic information with columns:
                          ['userid', 'age', 'gender', 'occupation', 'zipcode']
        """
        if not self.data_format.user_features_path:
            raise ValueError(f"User features not available for {self.size} dataset")
            
        filepath = self._check_data_and_download(self.data_format.user_features_path)
        
        # Column name order
        # 100K dataset -> user id | age | gender | occupation | zip code
        # 1M dataset -> UserID::Gender::Age::Occupation::Zip-code
        if self.size == "100k":
            column_order = ['userid', 'age', 'gender', 'occupation', 'zipcode']
        else:
            column_order = ['userid', 'gender', 'age', 'occupation', 'zipcode']
        
        df = pd.read_csv(
            filepath,
            sep=self.data_format.users_separator,
            names=column_order,
            engine="python"
        )

        # handle age
        if convert_age_to_range:
            if self.size == "100k":
                age_bins = [0, 17, 24, 34, 44, 49, 55, 200]
                age_labels = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
                df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
            else:
                df['age'] = df['age'].map(MovieLensConfig.AGE_GROUPS)
        
        # handle occupation
        if convert_occupation_to_code: 
            if self.size == "100k":
                df['occupation'] = df['occupation'].map(MovieLensConfig.OCCUPATIONS_100K)
        else: 
            if self.size == "1m":
                df['occupation'] = df['occupation'].map(MovieLensConfig.OCCUPATIONS_1M)

        df = df[['userid', 'age', 'gender', 'occupation', 'zipcode']]        
        
        return df
    
    def load_items(self, 
                   process_title: bool = True, 
                   process_year: bool = True, 
                   process_genres: bool = True, 
                   genres_as_binary: bool = True) -> pd.DataFrame:
        """
        Loads the movie items data into a unified DataFrame with consistent column names and order.

        Args:
            process_title (bool): Whether to include the 'title' column.
            process_year (bool): Whether to include the 'year' column.
            process_genres (bool): Whether to include the 'genrers' columns.
            genres_as_binary (bool): if process as column features.

        Returns:
            pd.DataFrame: DataFrame containing input flags
        """
        if not self.data_format.items_path:
            raise ValueError(f"Movie data not available for {self.size} dataset")
            
        filepath = self._check_data_and_download(self.data_format.items_path)
        
        # Define base columns
        final_columns = ['movie_id']

        if process_title:
            final_columns.append('title')
        if process_year:
            final_columns.append('year')

        if self.size == "100k":
            # For 100K format: movie_id|movie_title|release_date|video_release_date|IMDb_URL|unknown|Action|Adventure|...

            genre_cols = [f'genre_{genre}' for genre in MovieLensConfig.GENRES]
            df = pd.read_csv(
                filepath,
                sep=self.data_format.items_separator,
                names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols,
                encoding="ISO-8859-1",
                engine="python"
            )
            
            if process_year:
                df['year'] = df['movie_title'].str.extract(r'\((\d{4})\)').astype("Int64")
            if process_title:
                df['title'] = df['movie_title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)


            if process_genres:
                if genres_as_binary:
                    final_columns.extend([f'genre_{genre}' for genre in MovieLensConfig.GENRES])
                else:
                    # Convert binary columns to pipe-separated string
                    genre_cols = [f'genre_{genre}' for genre in MovieLensConfig.GENRES]
                    df['genres'] = df[genre_cols].apply(
                        lambda x: '|'.join([MovieLensConfig.GENRES[i] for i, v in enumerate(x) if v == 1]),
                        axis=1
                    )
                    final_columns.append('genres')

        else:
            # 1m format -> MovieID::Title::Genres            
            df = pd.read_csv(
                filepath,
                sep=self.data_format.items_separator,
                names=['movie_id', 'title', 'genres'],
                encoding="ISO-8859-1",
                engine="python"
            )
            
            if process_year:
                df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype("Int64")
            if process_title:
                df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
            if process_genres:
                if genres_as_binary:
                    # Convert pipe-separated genres to binary columns
                    for genre in MovieLensConfig.GENRES:
                        df[f'genre_{genre}'] = df['genres'].str.contains(genre).astype(int)
                    final_columns.extend([f'genre_{genre}' for genre in MovieLensConfig.GENRES])
                else:
                    final_columns.append('genres')

        return df[final_columns]


# -------------------- Use Case Examples --------------------
if __name__ == "__main__":
    # Initialize loader
    loader_100k = DataLoader("100k")
    loader_1m = DataLoader("1m")


    ##########################################
    # load ratings
    ratings_100k = loader_100k.load_ratings()
    print("Ratings 100K:")
    print(ratings_100k.head())

    ratings_1m = loader_1m.load_ratings()
    print("\nRatings 1M:")
    print(ratings_1m.head())

    ##########################################
    # load user features
    # 100k senario (age: numerical, occupation: labels)
    # # No conversion
    # user_features_100k = loader_100k.load_user_features(convert_age_to_range=False,convert_occupation_to_code=False)
    # print("User Features (100K) - No Conversion:")
    # print(user_features_100k.head())
    
    # Convert age to range
    user_features_100k_age = loader_100k.load_user_features(convert_age_to_range=True,convert_occupation_to_code=False)
    print("\nUser Features (100K) - Age Converted to Range:")
    print(user_features_100k_age.head())
    
    # # Convert occupation to numerical codes
    # user_features_100k_occ = loader_100k.load_user_features(convert_age_to_range=False,convert_occupation_to_code=True)
    # print("\nUser Features (100K) - Occupation Converted to Codes:")
    # print(user_features_100k_occ.head())

    # # convert age to range, and occupation to codes
    # user_features_100k_both = loader_100k.load_user_features(convert_age_to_range=True,convert_occupation_to_code=True)
    # print("\nUser Features (100K) - Both Age and Occupation Converted:")
    # print(user_features_100k_both.head())
    
    # 1m senario (age: range, occupation: numarical)
    # # No conversion
    # user_features_1m = loader_1m.load_user_features(convert_age_to_range=True,convert_occupation_to_code=True)
    # print("\nUser Features (1M) - No Conversion:")
    # print(user_features_1m.head())
    
    # #  Convert age to numarical
    # user_features_1m_age_num = loader_1m.load_user_features(convert_age_to_range=False,convert_occupation_to_code=True)
    # print("\nUser Features (1M) - Occupations as Descriptions:")
    # print(user_features_1m_age_num.head())

    # Convert occupations to labels
    user_features_1m_desc = loader_1m.load_user_features(convert_age_to_range=True, convert_occupation_to_code=False)
    print("\nUser Features (1M) - Occupations as Descriptions:")
    print(user_features_1m_desc.head())
    
    # # convert range to age, and codes to occupation
    # user_features_1m_both = loader_1m.load_user_features(convert_age_to_range=False,convert_occupation_to_code=False)
    # print("\nUser Features (1M) - Occupations as Codes:")
    # print(user_features_1m_both.head())

    ##########################################
    # Load movie information

    movies_df_100k_all = loader_100k.load_items(process_title=True, process_year=True, process_genres=True, genres_as_binary=True)
    print("\nMovies Data (100k) - All Columns:")
    print(movies_df_100k_all.head())

    movies_df_1m = loader_1m.load_items(process_title=True, process_year=True, process_genres=True, genres_as_binary=True)
    print("\nMovies Data (1M):")
    print(movies_df_1m.head())










