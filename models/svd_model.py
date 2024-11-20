
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict, Union
from typeguard import typechecked

from surprise import SVD as surSVD
from surprise import Dataset, Reader

@typechecked
class SVDModel:
    def __init__(
        self,
        train_set:pd.DataFrame,
        test_set:pd.DataFrame,
        n_factors: int = 100,
        n_epochs: int = 20,
        learning_rate: float = 0.005,
        reg_all: float = 0.02, # regularization 
        rating_scale: Tuple[int, int] = (1, 5) # min to max rating
    ) -> None:
        
        self.train_set = train_set
        self.test_set = test_set
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
        self.train_pre, self.test_pre = (lambda x, y: (x, y))(self.train_set, self.test_set)
        reader = Reader(rating_scale=(1, 5))

        self.trainset = Dataset.load_from_df(self.train_set, reader).build_full_trainset()

    def train(self) -> None:
        self.model.fit(self.trainset)

    def predict(self) -> pd.DataFrame:
        data = self.test_pre
        predictions = [
           self.model.predict(getattr(row, "user"), getattr(row, "item"))
            for row in data.itertuples()
        ]
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(
            index=str, columns={"uid": "user", "iid": "item", "est": "prediction"}
        )
        return predictions.drop(["details", "r_ui"], axis="columns")

    def recommend_k_svd(self) -> pd.DataFrame:
        data = self.train_pre
        preds_lst = []
        users = data["user"].unique()
        items = data["item"].unique()

        for user in users:
            for item in items:
                preds_lst.append([user, item, self.model.predict(user, item).est])

        all_predictions = pd.DataFrame(data=preds_lst, columns=["user", "item", "prediction"])

        # mark which user-item pairs belong to the actual test set in the merge operation, 
        # so that user-item pairs that only appear in the prediction set but not the actual test set 
        # can be easily distinguished in the screening.
        tempdf = pd.concat(
            [
                data[["user", "item"]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=["user", "item"], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)