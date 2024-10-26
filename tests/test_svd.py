# tests/test_svd_model.py

import unittest
import pandas as pd
from models.svd_model import SVDModel

class TestSVDModel(unittest.TestCase):
    def test_svd_model_with_user_features(self):
        # Test with "100k" dataset
        svd_model = SVDModel(size="100k", n_factors=10, n_epochs=5)
        svd_model.train()
        self.assertIsNotNone(svd_model.user_features)
        self.assertIsInstance(svd_model.user_features, pd.DataFrame)

    # def test_svd_model_without_user_features(self):
    #     # Test with "1m" dataset
    #     svd_model = SVDModel(size="1m", n_factors=10, n_epochs=5)
    #     svd_model.train()
    #     self.assertIsNone(svd_model.user_features)

if __name__ == '__main__':
        unittest.main()