from abc import ABC, abstractmethod
from typing import Any, Union, Dict
import pandas as pd

class BaseModel(ABC):

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, user: Union[str,int], item:Union[str, int]) -> float:
        pass

    @abstractmethod
    def evaluate(self, test_data:pd.DataFrame) -> Dict[str, float]:
        pass