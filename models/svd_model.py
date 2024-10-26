from .base_model import BaseModel
from utils.dataloader import load_data_df, load_item_df, load_user_features, DATA_FORMAT, maybe_download

import os
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, Union
from typeguard import typechecked

import torch
import torch.nn as nn
from surprise import SVD as surSVD
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split


@typechecked
class SVDModel:
    def __init__(
        self,
        size: str,  # "100k" or "1m"
        n_factors: int = 100,
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        reg_all: float = 0.02, # regularization 
        rating_scale: Tuple[int, int] = (1, 5) # min to max rating
    ) -> None:
        
        if size not in DATA_FORMAT:
            raise ValueError(f"Invalid size: {size}. Choose from {list(DATA_FORMAT.keys())}.")

        self.size = size
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_all = reg_all
        self.rating_scale = rating_scale

        self.model = surSVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            reg_all=self.reg_all
        )

        self.ratings = self._load_ratings()
        self.items = self._load_items()
        self.user_features = self._load_user_features()

        # Rename columns to match Surprise's expectations
        self.ratings.rename(columns={'UserId': 'user', 'MovieId': 'item', 'Rating': 'rating'}, inplace=True)


    # TODO
    def _load_ratings(self) -> pd.DataFrame:
        ratings = load_data_df(self.size)
        return ratings

    def _load_items(self) -> pd.DataFrame:
        items = load_item_df(
            size=self.size,
            item_datapath=maybe_download(self.size, "u.item"),
            item_col="MovieId",
            title_col="Title",
            genres_col="Genres",
            year_col="Year"
        )
        return items

    def _load_user_features(self) -> Union[pd.DataFrame, None]:
        user_features = load_user_features(maybe_download("100k", "u.user"))
        return user_features


    def train(self) -> None:
        # prepare surprise dataset
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(self.ratings[['user', 'item', 'rating']], reader)
        self.trainset = data.build_full_trainset()

        self.model.fit(self.trainset)

    def predict(self, user: Union[int, str], item: Union[int, str]) -> float:
        prediction = self.model.predict(str(user), str(item))
        return prediction.est

    def evaluate(self) -> Dict[str, float]:
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(self.ratings[['user', 'item', 'rating']], reader)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        self.model.fit(train_data)
        predictions = self.model.test(test_data)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        return {'RMSE': rmse, 'MAE': mae}