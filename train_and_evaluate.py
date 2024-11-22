from utils.metrics import calculate_rating_metrics, calculate_ranking_metrics
from models.svd_model import SVDModel
from models.lightgcn_model import LightGCNModel
from models.lightgcn_model_exp import LightGCNModel2
from models.recommender import RecommenderModel
import os

def svd_model_train_and_evaluate(train_set, test_set):
    svd_model = SVDModel(train_set=train_set, test_set=test_set, n_factors=200, n_epochs=30)
    svd_model.prepare_training_data()
    svd_model.train()
    predictions = svd_model.predict()

    ratings = calculate_rating_metrics(svd_model.test_pre, predictions)

    top_k_scores = svd_model.recommend_k_svd()
    rankings = calculate_ranking_metrics(svd_model.test_pre, top_k_scores, 10)

    return ratings, rankings

def lgcn_model_train_and_evaluate(train_set, test_set):
    lgcn_model = LightGCNModel(
        train_set=train_set,
        test_set=test_set,
        num_layers=3,
        num_negatives=4,
        embedding_dim=64,
        learning_rate=0.01,
        epochs=1000,
        batch_size=1024,
        device='cpu'
    )

    lgcn_model.prepare_training_data()
    lgcn_model.train()
    predictions = lgcn_model.predict()

    ratings = calculate_rating_metrics(lgcn_model.test_pre, predictions)

    top_k_scores = lgcn_model.recommend_k()
    rankings = calculate_ranking_metrics(lgcn_model.test_pre, top_k_scores, 10)
    return ratings, rankings

def lgcn_model_train_and_evaluate_2(train_set, test_set):
    lgcn_model2 = LightGCNModel2(
        train_set=train_set,
        test_set=test_set,
        num_layers=3,
        num_negatives=4,
        embedding_dim=64,
        learning_rate=0.01,
        epochs=1000,
        batch_size=1024,
        paths = {"umam": os.path.join("data", "umam_embeddings.pt"), 
                "umdm":os.path.join("data", "umdm_embeddings.pt"),
                "umum":os.path.join("data", "umum_embeddings.pt"),
                "user_content": os.path.join("data", "user_content_based_embeddings.pt"),
                "item_content": os.path.join("data", "movie_genre_hot_embeddings.pt"),
                "user_pretrained": os.path.join("data", "pretrain_user_embeddings.pt"),
                "item_pretrained": os.path.join("data", "pretrain_item_embeddings.pt")},
        device='cpu'
    )

    lgcn_model2.prepare_training_data()
    lgcn_model2.train()
    predictions = lgcn_model2.predict()

    ratings = calculate_rating_metrics(lgcn_model2.test_pre, predictions)

    top_k_scores = lgcn_model2.recommend_k()
    rankings = calculate_ranking_metrics(lgcn_model2.test_pre, top_k_scores, 10)
    return ratings, rankings

def recommender_train_and_evaluate(train_set, test_set):
    rec_model = RecommenderModel(
        train_set=train_set,
        test_set=test_set,
        num_layers=3,
        num_negatives=4,
        embedding_dim=64,
        learning_rate=0.01,
        epochs=500,
        batch_size=1024,
        paths = {"umam": os.path.join("data", "umam_embeddings.pt"), 
                "umdm":os.path.join("data", "umdm_embeddings.pt"),
                "umum":os.path.join("data", "umum_embeddings.pt"),
                "user_content": os.path.join("data", "user_content_based_embeddings.pt"),
                "item_content": os.path.join("data", "movie_genre_hot_embeddings.pt"),
                "user_pretrained": os.path.join("data", "pretrain_user_embeddings.pt"),
                "item_pretrained": os.path.join("data", "pretrain_item_embeddings.pt")},
        device='cpu'
    )
    rec_model.prepare_training_data()
    rec_model.train()
    predictions = rec_model.predict()

    ratings = calculate_rating_metrics(rec_model.test_pre, predictions)

    top_k_scores = rec_model.recommend_k()
    rankings = calculate_ranking_metrics(rec_model.test_pre, top_k_scores, 10)
    return ratings, rankings