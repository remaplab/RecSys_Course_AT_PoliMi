import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps
from loguru import logger
from scipy.stats import kurtosis, skew
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm, trange

from Data_manager.competition import load, load_raw
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrid import (
    ScoresMultipleHybridRecommender,
    UserWideHybridRecommender,
)
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

USE = "training"
EXPERIMENT = "final"
NUMBER_FOLDS = 10
CUTOFF = 50

TOP_POPULAR_THRESHOLDS = (10, 100, 1000)

ITEM_LATENT_DIMENSIONS = 10
USER_LATENT_DIMENSIONS = 10

RECOMMENDATION_MODELS_TO_USE = (
    70,
    71,
    72,
    73,
)
SCORE_MODELS_TO_USE = (
    20,
    21,
    22,
    23,
)
USE_SCORE_HYBRID = True
USE_USER_HYBRID = True
USER_WIDE_HYBRID_BEGIN = 30

OUTPUT_PATH = Path() / f"ranker_{USE}_data_{EXPERIMENT}.parquet"

MODELS_BASE_DIR = Path() / "models"
TRAIN_MODELS_BASE_DIR = MODELS_BASE_DIR / "train"
TRAIN_MODELS_MAP_DIR = TRAIN_MODELS_BASE_DIR / "map"
TRAIN_MODELS_RECALL_DIR = TRAIN_MODELS_BASE_DIR / "recall"
SUBMISSION_MODELS_MAP_DIR = MODELS_BASE_DIR / "all" / "map" / "renamed"
SUBMISSION_MODELS_RECALL_DIR = Path(
    str(SUBMISSION_MODELS_MAP_DIR).replace("map", "recall")
)

NUMBER_GROUPS_USER_WIDE_HYBRID = 10

MULTIPLE_SCORE_HYBRID_WEIGHTS = {
    50: 0.253770701546336,
    51: 0.10324855050317669,
}

SVD_FIT_PARAMS = {
    "num_factors": 350,
}

ROWS_PER_FOLD = 900_000

RNG = np.random.default_rng(42)


def build_user_wide_hybrid(urm: sps.csr_matrix, models: dict[str, BaseRecommender]):
    profile_lengths = np.ediff1d(urm.indptr)
    sorted_users = np.argsort(profile_lengths)
    block_size = len(sorted_users) // NUMBER_GROUPS_USER_WIDE_HYBRID
    group_users = {}
    for group in range(NUMBER_GROUPS_USER_WIDE_HYBRID + 1):
        group_users[group] = sorted_users[group * block_size : (group + 1) * block_size]
    group_recommenders = {
        group: models.pop(str(USER_WIDE_HYBRID_BEGIN + group))
        for group in range(NUMBER_GROUPS_USER_WIDE_HYBRID + 1)
    }
    return UserWideHybridRecommender(urm, group_users, group_recommenders)


def build_score_hybrid(urm: sps.csr_matrix, models: dict[str, BaseRecommender]):
    recommenders = [
        models.pop(str(index)) for index in MULTIPLE_SCORE_HYBRID_WEIGHTS.keys()
    ]
    weights = list(MULTIPLE_SCORE_HYBRID_WEIGHTS.values())
    return ScoresMultipleHybridRecommender(urm, recommenders, weights)


def urm_df_to_csr(
    urm_df: pd.DataFrame, number_users: int, number_items: int
) -> sps.csr_matrix:
    return sps.csr_matrix(
        (urm_df.data, (urm_df.user_id, urm_df.item_id)),
        shape=(number_users, number_items),
    )


def row_statistics(df: pd.DataFrame, like: str) -> pd.DataFrame:
    df = df.filter(like=like)
    statistics_df = pd.DataFrame([], index=df.index)
    statistics_df[f"{like}_mean"] = df.mean(axis="columns")
    statistics_df[f"{like}_std"] = df.std(axis="columns")
    statistics_df[f"{like}_min"] = df.min(axis="columns")
    statistics_df[f"{like}_max"] = df.max(axis="columns")
    statistics_df[f"{like}_kurtosis"] = kurtosis(df, axis=1)
    statistics_df[f"{like}_skew"] = skew(df, axis=1)
    return statistics_df


def compute_base_dataset(
    number_users: int,
    recommendation_models: dict[str, BaseRecommender],
    score_models: dict[str, BaseRecommender],
    cutoff: int,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    dataset = pd.DataFrame(index=range(0, number_users), columns=["ItemID"])
    dataset.index.name = "UserID"

    recommendations_list = []
    recommenders_list = []
    rank_list = []
    for user_id in trange(number_users, desc="User (candidate)"):
        user_recommendations = []
        user_recommenders = []
        user_rankings = []
        for name, recommender in recommendation_models.items():
            user_recommendations.extend(
                recommender.recommend(
                    user_id,
                    cutoff=cutoff,
                    remove_seen_flag=True,
                )
            )
            user_recommenders.extend([name] * cutoff)
            user_rankings.extend(list(range(cutoff)))
        recommendations_list.append(user_recommendations)
        recommenders_list.append(user_recommenders)
        rank_list.append(user_rankings)

    dataset["ItemID"] = recommendations_list
    dataset["Recommender"] = recommenders_list
    dataset["Ranking"] = rank_list

    exploded_recommender = dataset["Recommender"].explode()
    exploded_ranking = dataset["Ranking"].explode()
    dataset = dataset.explode("ItemID")
    dataset["Recommender"] = exploded_recommender
    dataset["Ranking"] = exploded_ranking.astype("int")

    recommender_agreement = (
        dataset.reset_index()[["UserID", "ItemID"]]
        .groupby(["UserID", "ItemID"])
        .value_counts()
    )
    dataset["recommender_agreement"] = recommender_agreement.loc[
        list(zip(dataset.index, dataset["ItemID"]))
    ].to_numpy()

    for user_id in tqdm(dataset.index.unique(), desc="User (score)"):
        for rec_label, rec_instance in score_models.items():
            item_list = dataset.loc[user_id, "ItemID"].to_list()

            all_item_scores = rec_instance._compute_item_score(
                [user_id], items_to_compute=item_list
            )

            dataset.loc[user_id, f"score_{rec_label}"] = all_item_scores[0, item_list]

    score_statistics = row_statistics(dataset, "score")
    dataset = pd.concat([dataset, score_statistics], axis="columns")

    dataset = dataset.reset_index()
    dataset = dataset.rename(columns={"index": "UserID"})

    if fold is not None:
        dataset["fold"] = fold

    return dataset


def add_labels(training_df: pd.DataFrame, correct_recommendations_df: pd.DataFrame):
    training_df = training_df.merge(
        correct_recommendations_df,
        on=["UserID", "ItemID"],
        how="left",
        indicator="Exist",
    )
    training_df["Label"] = training_df["Exist"] == "both"
    training_df = training_df.drop(columns=["Exist"])

    if training_df.shape[0] > ROWS_PER_FOLD:
        logger.debug("Reducing dataset from {} to {} rows", training_df.shape[0], ROWS_PER_FOLD)
        logger.debug("Original positive rate {}", training_df["Label"].mean())
        positive_indices = training_df[training_df["Label"]].index
        negative_indices = training_df[~training_df["Label"]].index
        negative_sampled_indices = RNG.choice(
            negative_indices, ROWS_PER_FOLD - len(positive_indices), replace=False
        )
        training_df = training_df.loc[positive_indices.union(negative_sampled_indices)]
        logger.debug("New positive rate {}", training_df["Label"].mean())

    return training_df


def compute_correct_recommendations(
    urm_val: sps.csr_matrix,
) -> pd.DataFrame:
    urm_val_coo = sps.coo_matrix(urm_val)
    return pd.DataFrame({"UserID": urm_val_coo.row, "ItemID": urm_val_coo.col})


def load_models_fold(
    urm: sps.csr_matrix,
    fold_dir: Path,
    use_only: Optional[list[str]] = None,
    with_user_hybrid: bool = False,
    with_score_hybrid: bool = False,
):
    all_models = load_models_all(fold_dir)
    models = all_models
    if use_only is not None:
        models = {str(index): models[str(index)] for index in use_only}
    if with_user_hybrid and "user_wide_hybrid" not in models:
        user_wide_hybrid = build_user_wide_hybrid(urm, all_models)
        models["user_wide_hybrid"] = user_wide_hybrid
    if with_score_hybrid and "score_hybrid" not in models:
        score_hybrid = build_score_hybrid(urm, all_models)
        models["score_hybrid"] = score_hybrid

    return models


def load_models_all(dir_: Path) -> dict[str, BaseRecommender]:
    return {path.stem: pickle.load(path.open("rb")) for path in dir_.glob("*.pkl")}


def compute_training_dataset(
    number_users: int,
    number_items: int,
    urm_df: pd.DataFrame,
    folds: Optional[int] = None,
):
    if folds is None:
        split = ShuffleSplit(1, test_size=0.2, random_state=42)
    else:
        split = KFold(folds, shuffle=True, random_state=42)

    fold_training_datasets: dict[int, pd.DataFrame] = {}
    for i, (train_indices, val_indices) in tqdm(
        enumerate(split.split(urm_df)),
        total=folds,
        desc="Fold",
    ):
        fold_urm_train = urm_df_to_csr(
            urm_df.iloc[train_indices], number_users, number_items
        )
        fold_urm_val = urm_df_to_csr(
            urm_df.iloc[val_indices], number_users, number_items
        )

        fold_recommendation_models_dir = TRAIN_MODELS_RECALL_DIR
        fold_score_models_dir = TRAIN_MODELS_MAP_DIR
        if folds is not None:
            fold_recommendation_models_dir /= str(i)
            fold_score_models_dir /= str(i)

        recommendation_models = load_models_fold(
            fold_urm_train,
            fold_recommendation_models_dir,
            use_only=RECOMMENDATION_MODELS_TO_USE,
            with_user_hybrid=False,
            with_score_hybrid=False,
        )
        logger.debug("Recommendation models: {}", recommendation_models.keys())
        score_models = load_models_fold(
            fold_urm_train,
            fold_score_models_dir,
            use_only=SCORE_MODELS_TO_USE,
            with_user_hybrid=USE_USER_HYBRID,
            with_score_hybrid=USE_SCORE_HYBRID,
        )
        logger.debug("Score models: {}", score_models.keys())

        fold_training_dataset = compute_base_dataset(
            number_users,
            recommendation_models,
            score_models,
            CUTOFF,
            fold=i,
        )
        correct_recommendations_df = compute_correct_recommendations(fold_urm_val)
        fold_training_dataset = add_labels(
            fold_training_dataset, correct_recommendations_df
        )
        fold_training_datasets[i] = fold_training_dataset

    return pd.concat(fold_training_datasets.values())


def compute_submission_dataset(number_users: int):
    recommendation_models = load_models_all(SUBMISSION_MODELS_RECALL_DIR)
    score_models = load_models_all(SUBMISSION_MODELS_MAP_DIR)
    return compute_base_dataset(
        number_users,
        recommendation_models,
        score_models,
        CUTOFF,
    )


def compute_dataset(
    use: Literal["training", "submission"],
    number_users: int,
    number_items: int,
    urm_df: pd.DataFrame,
    folds: Optional[int] = None,
):
    if use == "training":
        return compute_training_dataset(number_users, number_items, urm_df, folds)
    elif use == "submission":
        return compute_submission_dataset(number_users)


def add_features(dataset: pd.DataFrame, urm: sps.csr_matrix, icm: sps.csr_matrix):
    svd = PureSVDRecommender(urm)
    svd.fit(**SVD_FIT_PARAMS)
    # Item features

    ## Item popularity
    item_popularity = np.ediff1d(sps.csc_matrix(urm).indptr)

    dataset["item_popularity"] = item_popularity[
        dataset["ItemID"].to_numpy().astype(int)
    ]

    ## Distance to closest items
    item_similarity = Compute_Similarity(icm.T).compute_similarity()
    item_similarity

    mean_item_similarity_dict = {i: row.mean() for i, row in enumerate(item_similarity)}
    mean_item_similarity: pd.DataFrame = pd.Series(mean_item_similarity_dict).to_frame(
        name="item_similarity"
    )
    mean_item_similarity

    dataset = dataset.join(mean_item_similarity, on="ItemID")

    ## Singular vectors
    for i in range(ITEM_LATENT_DIMENSIONS):
        dataset[f"item_svd_{i}"] = svd.ITEM_factors[
            dataset["ItemID"].to_numpy().astype(int), i
        ]

    ## Autoencoder embeddings
    encoder_embeddings: pd.DataFrame = pd.read_csv("reduced_features.csv", index_col=0)
    enencoder_embeddings = encoder_embeddings.rename(
        columns=lambda x: f"item_autoencoder_{x}"
    )
    dataset = dataset.join(enencoder_embeddings, on="ItemID")

    # User features

    ## User popularity
    user_popularity = np.ediff1d(sps.csr_matrix(urm).indptr)

    dataset["user_profile_len"] = user_popularity[
        dataset["UserID"].to_numpy().astype(int)
    ]

    ## User popularity bias
    # (measure of how much popularity influences the user)
    item_popularity_ranking = item_popularity.argsort()[::-1]
    item_popularity_ranking

    item_id_df = urm_df[["user_id", "item_id"]]
    item_id_df

    for k in TOP_POPULAR_THRESHOLDS:
        top_k_popular = item_popularity_ranking[:k]
        item_id_df.loc[item_id_df["item_id"].isin(top_k_popular), f"top_{k}"] = 1
    item_id_df = item_id_df.fillna(0)
    item_id_df

    user_top_k_df = item_id_df.groupby("user_id").aggregate(
        {f"top_{k}": "sum" for k in TOP_POPULAR_THRESHOLDS}
    )
    user_top_k_df

    dataset = dataset.join(user_top_k_df, on="UserID")

    ## Distance to closest users
    user_similarity = Compute_Similarity(urm.T).compute_similarity()
    user_similarity

    mean_user_similarity_dict = {i: row.mean() for i, row in enumerate(user_similarity)}
    mean_user_similarity: pd.DataFrame = pd.Series(mean_user_similarity_dict).to_frame(
        name="user_similarity"
    )
    mean_user_similarity

    dataset = dataset.join(mean_user_similarity, on="UserID")

    ## Singular vectors
    for i in range(USER_LATENT_DIMENSIONS):
        dataset[f"user_svd_{i}"] = svd.USER_factors[
            dataset["UserID"].to_numpy().astype(int), i
        ]

    return dataset


if __name__ == "__main__":
    icm_df, urm_df = load_raw()
    number_users = urm_df["user_id"].nunique()
    number_items = icm_df["item_id"].nunique()

    icm_matrix, urm_all, *_ = load()

    dataset = compute_dataset(
        use=USE,
        number_users=number_users,
        number_items=number_items,
        urm_df=urm_df,
        folds=NUMBER_FOLDS,
    )

    dataset = add_features(dataset, urm_all, icm_matrix)

    for categorical_column in ("UserID", "ItemID", "Recommender"):
        dataset[categorical_column] = dataset[categorical_column].astype("category")

    dataset.to_parquet(OUTPUT_PATH)
