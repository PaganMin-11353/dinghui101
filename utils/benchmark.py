from utils.metrics import calculate_rating_metrics, calculate_ranking_metrics
from models.svd_model import SVDModel

def svd_model_benchmark(size):
    svd_model = SVDModel(size, n_factors=200, n_epochs=30)
    svd_model.prepare_training_data()
    svd_model.train()
    predictions = svd_model.predict()

    ratings = calculate_rating_metrics(svd_model.test_pre, predictions)

    top_k_scores = svd_model.recommend_k_svd()
    rankings = calculate_ranking_metrics(svd_model.test_pre, top_k_scores, 10)

    return ratings, rankings