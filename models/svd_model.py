
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, Union
from typeguard import typechecked

from surprise import SVD as surSVD
from surprise import Dataset, Reader
from utils.data_split import python_random_split
from utils.dataloader import DataLoader
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
        
        self.data_loader = DataLoader(size=size)
        if size not in self.data_loader.DATA_FORMATS:
            raise ValueError(f"Invalid size: {size}. Choose from {list(self.data_loader.DATA_FORMATS.keys())}.")

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

    def prepare_training_data(self) -> None:
        data = self.data_loader.load_ratings()
        data = data[['user', 'item', 'rating']]

        self.train_data, self.test_data = python_random_split(data, 0.75)
        self.train_pre, self.test_pre = (lambda x, y: (x, y))(self.train_data, self.test_data)

        reader = Reader(rating_scale=(1, 5))
        self.trainset = Dataset.load_from_df(self.train_data, reader).build_full_trainset()

    def train(self) -> None:
        self.model.fit(self.trainset)

    def predict(self) -> pd.DataFrame:

        data = self.test_pre
        predictions = [
           self.model.predict(getattr(row, "user"), getattr(row, "item"))
        #    self.model.predict(getattr(row, "UserId"), getattr(row, "ItemId"))
            for row in data.itertuples()
        ]
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(
            index=str, columns={"uid": "user", "iid": "item", "est": "prediction"}
            # index=str, columns={"uid": "UserId", "iid": "ItemId", "est": "prediction"}
        )
        return predictions.drop(["details", "r_ui"], axis="columns")

    def recommend_k_svd(self) -> pd.DataFrame:
        data =  data = self.train_pre
        preds_lst = []
        users = data["user"].unique()
        items = data["item"].unique()
        # users = data["UserId"].unique()
        # items = data["ItemId"].unique()

        for user in users:
            for item in items:
                preds_lst.append([user, item, self.model.predict(user, item).est])

        all_predictions = pd.DataFrame(data=preds_lst, columns=["user", "item", "prediction"])
        # all_predictions = pd.DataFrame(data=preds_lst, columns=["UserId", "ItemId", "prediction"])

        # mark which user-item pairs belong to the actual test set in the merge operation, 
        # so that user-item pairs that only appear in the prediction set but not the actual test set 
        # can be easily distinguished in the screening.
        tempdf = pd.concat(
            [
                data[["user", "item"]],
                # data[["UserId", "ItemId"]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=["user", "item"], how="outer")
        # merged = pd.merge(tempdf, all_predictions, on=["UserId", "ItemId"], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)