import numpy as np
import pandas as pd
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

def merge_rating_correct_pred(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
):
    """Join truth and prediction data frames on userID and itemID and return the true
    and predicted rated with the correct index

    Returns:
        numpy.ndarray: Array with the true ratings
        numpy.ndarray: Array with the predicted ratings

    """

    # pd.merge will apply suffixes to columns which have the same name across both dataframes
    suffixes = ["_true", "_pred"]
    rating_true_pred = pd.merge(
        rating_true, rating_pred, on=[col_user, col_item], suffixes=suffixes
    )
    if col_rating in rating_pred.columns:
        col_rating = col_rating + suffixes[0]
    if col_prediction in rating_true.columns:
        col_prediction = col_prediction + suffixes[1]
    return rating_true_pred[col_rating], rating_true_pred[col_prediction]

def rmse(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
):
    """Calculate Root Mean Squared Error
    Returns:
        float: Root mean squared error
    """

    y_true, y_pred = merge_rating_correct_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
):
    """Calculate Mean Absolute Error.
        Returns:
            float: Mean Absolute Error.
    """

    y_true, y_pred = merge_rating_correct_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_error(y_true, y_pred)

def rsquared(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
):
    """Calculate R squared
    Returns:
        float: R squared (min=0, max=1).
    """

    y_true, y_pred = merge_rating_correct_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return r2_score(y_true, y_pred)


def exp_var(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
):
    """Calculate explained variance.
    Returns:
        float: Explained variance (min=0, max=1).
    """

    y_true, y_pred = merge_rating_correct_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return explained_variance_score(y_true, y_pred)

def get_top_k_items(dataframe, col_user="UserID", col_rating="Rating", k=10):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.
    Returns:
        pandas.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.sort_values([col_user, col_rating], ascending=[True, False])
            .groupby(col_user, as_index=False)
            .head(k)
            .reset_index(drop=True)
        )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items

def merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_prediction,
    relevancy_method,
    k=10,
    threshold=10,
    **_,
):
    """Filter truth and prediction data frames on common users
    Returns:
        pandas.DataFrame, pandas.DataFrame, int: DataFrame of recommendation hits, sorted by `col_user` and `rank`
        DataFrame of hit counts vs actual relevant items per user number of unique user ids
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)

    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method == "by_threshold":
        top_k = threshold
    elif relevancy_method is None:
        top_k = None
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items(
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg(hit="count"),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            actual="count",
        ),
        on=col_user,
    )

    return df_hit, df_hit_count, n_users

def _get_reciprocal_rank(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_prediction="prediction",
    relevancy_method="top_k",
    k=10,
    threshold=10,
):
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return None, n_users

    # calculate reciprocal rank of items for each user and sum them up
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"rr": "sum"}).reset_index()

    return pd.merge(df_hit_sorted, df_hit_count, on=col_user), n_users

def map(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_prediction="prediction",
    relevancy_method="top_k",
    k=10,
    threshold=10,
    **_,
):
    """Mean Average Precision for top k prediction items
    Returns:
        float: MAP (min=0, max=1)
    """
    df_merge, n_users = _get_reciprocal_rank(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_merge is None:
        return 0.0
    else:
        return (df_merge["rr"] / df_merge["actual"]).sum() / n_users

def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_rating="Rating",
    col_prediction="prediction",
    relevancy_method="top_k",
    k=10,
    threshold=10,
    score_type="binary",
    discfun_type="loge",
    **_,
):
    """Normalized Discounted Cumulative Gain (nDCG).
    Returns:
        float: nDCG at k (min=0, max=1).
    """
    df_hit, _, _ = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_dcg = df_hit.merge(rating_pred, on=[col_user, col_item]).merge(
        rating_true, on=[col_user, col_item], how="outer", suffixes=("_left", None)
    )

    if score_type == "binary":
        df_dcg["rel"] = 1
    elif score_type == "raw":
        df_dcg["rel"] = df_dcg[col_rating]
    elif score_type == "exp":
        df_dcg["rel"] = 2 ** df_dcg[col_rating] - 1
    else:
        raise ValueError("score_type must be one of 'binary', 'raw', 'exp'")

    if discfun_type == "loge":
        discfun = np.log
    elif discfun_type == "log2":
        discfun = np.log2
    else:
        raise ValueError("discfun_type must be one of 'loge', 'log2'")

    # Calculate the actual discounted gain for each record
    df_dcg["dcg"] = df_dcg["rel"] / discfun(1 + df_dcg["rank"])

    # Calculate the ideal discounted gain for each record
    df_idcg = df_dcg.sort_values([col_user, col_rating], ascending=False)
    df_idcg["irank"] = df_idcg.groupby(col_user, as_index=False, sort=False)[
        col_rating
    ].rank("first", ascending=False)
    df_idcg["idcg"] = df_idcg["rel"] / discfun(1 + df_idcg["irank"])

    # Calculate the actual DCG for each user
    df_user = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})

    # Calculate the ideal DCG for each user
    df_user = df_user.merge(
        df_idcg.groupby(col_user, as_index=False, sort=False)
        .head(k)
        .groupby(col_user, as_index=False, sort=False)
        .agg({"idcg": "sum"}),
        on=col_user,
    )

    # DCG over IDCG is the normalized DCG
    df_user["ndcg"] = df_user["dcg"] / df_user["idcg"]
    return df_user["ndcg"].mean()

def precision_at_k(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_prediction="prediction",
    relevancy_method="top_k",
    k=10,
    threshold=10,
    **_,
):
    """Precision at K.
    Returns:
        float: precision at k (min=0, max=1)
    """
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users

def recall_at_k(
    rating_true,
    rating_pred,
    col_user="UserId",
    col_item="ItemId",
    col_prediction="prediction",
    relevancy_method="top_k",
    k=10,
    threshold=10,
    **_,
):
    """Recall at K.
    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than
        k items exist for a user in rating_true.
    """
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users

def calculate_rating_metrics(test, predictions):
    return {
        "RMSE": rmse(test, predictions),
        "MAE": mae(test, predictions),
        "R2": rsquared(test, predictions),
        "Explained Variance": exp_var(test, predictions),
    }

def calculate_ranking_metrics(train, predictions, k=10):
    return {
        "MAP": map(train, predictions, k=k),
        "nDCG@k": ndcg_at_k(train, predictions, k=k),
        "Precision@k": precision_at_k(train, predictions, k=k),
        "Recall@k": recall_at_k(train, predictions, k=k),
    }