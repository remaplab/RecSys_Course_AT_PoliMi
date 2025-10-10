import pickle
from pathlib import Path
from multiprocessing import cpu_count

import scipy.sparse as sps
from sklearn.model_selection import KFold

from Data_manager.competition import load_raw
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommenderICM import RP3betaRecommenderICM
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import (
    ItemKNN_CFCBF_Hybrid_Recommender,
)
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import (
    SLIMElasticNetRecommender,
)

N_FOLDS = 10
MODEL_DIR = Path() / "models" / "train" / "recall"

HYPERPARAMETERS: dict[int, dict] = {
    # User-wide hybrid 1 (0-10)
    # 0: {
    #     "topK": 22,
    #     "alpha": 0.015137951778257512,
    #     "normalize_similarity": True,
    #     "implicit": True,
    # },
    # 1: {
    #     "topK": 5,
    #     "shrink": 774,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.01,
    # },
    # 2: {
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.2626851799303072,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.1560410093044209,
    # },
    # 3: {
    #     "topK": 1000,
    #     "alpha": 1.8920160119169898,
    #     "beta": 0.4950301468130674,
    #     "delta": 0.30908791366521954,
    #     "normalize_similarity": True,
    # },
    # 4: {
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.06864228467890522,
    # },
    # 5: {
    #     "topK": 7,
    #     "shrink": 293,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0732688773175534,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.23668747670276377,
    # },
    # 6: {
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.06565478344525211,
    # },
    # 7: {
    #     "topK": 5,
    #     "shrink": 0,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0329315091653946,
    #     "feature_weighting": "BM25",
    #     "ICM_weight": 0.16124267891305158,
    # },
    # 8: {
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.171628301912052,
    # },
    # 9: {
    #     "topK": 10,
    #     "alpha": 0.35225624527493254,
    #     "normalize_similarity": True,
    #     "implicit": True,
    # },
    # 10: {
    #     "topK": 44,
    #     "shrink": 473,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.35983197418129564,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.12542629369630146,
    # },
    # 20: {  # Item KNN CF+CBF
    #     "topK": 96,
    #     "shrink": 966,
    #     "similarity": "cosine",
    #     "normalize": True,
    #     "feature_weighting": "BM25",
    #     "ICM_weight": 0.015154282137075726,
    # },
    # 21: {  # SLIM ElasticNet
    #     "l1_ratio": 0.4408355927953408,
    #     "alpha": 0.00013519978876092592,
    #     "positive_only": False,
    #     "topK": 59,
    #     "do_feature_selection": True,
    # },
    # 22: {  # RP3 ICM
    #     "topK": 11,
    #     "alpha": 1.9811525250064195,
    #     "beta": 0.6832513917848906,
    #     "delta": 0.0037274512973076712,
    #     "normalize_similarity": True,
    #     "implicit": True,
    #     "min_rating": 1.0,
    # },
    # 23: {  # Item KNN CF
    #     "topK": 5,
    #     "shrink": 224,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    # },
    # User-wide hybrid 2 (30-40)
    # 30: {
    #     "topK": 1000,
    #     "l1_ratio": 0.0036552968571563925,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 31: {
    #     "topK": 1000,
    #     "l1_ratio": 0.0036439600383419896,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 32: {
    #     "topK": 1000,
    #     "l1_ratio": 0.01294361044706415,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 33: {
    #     "topK": 469,
    #     "l1_ratio": 0.0025724182700638666,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 34: {
    #     "topK": 1000,
    #     "l1_ratio": 0.012451061879323577,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 35: {
    #     "topK": 1000,
    #     "l1_ratio": 0.0037651439623475717,
    #     "alpha": 0.001,
    #     "positive_only": False,
    #     "do_feature_selection": True,
    # },
    # 36: {
    #     "topK": 1000,
    #     "l1_ratio": 0.009466188626970398,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 37: {
    #     "topK": 196,
    #     "l1_ratio": 0.019833595367995636,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 38: {
    #     "topK": 145,
    #     "l1_ratio": 2.6489644774823373e-05,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 39: {
    #     "topK": 866,
    #     "l1_ratio": 0.019729118757762613,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 40: {
    #     "topK": 44,
    #     "shrink": 473,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.35983197418129564,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.12542629369630146,
    # },
    # Score hybrid
    # 50: {  # RP3 ICM
    #     "topK": 79,
    #     "alpha": 0.7864757238135991,
    #     "beta": 0.443333110568691,
    #     "delta": 0.7593249588588719,
    #     "min_rating": 0.008553401844836345,
    #     "implicit": True,
    #     "normalize_similarity": True,
    # },
    # 51: {  # SLIM ElasticNet
    #     "l1_ratio": 0.04077479852537514,
    #     "alpha": 0.0004098922954204119,
    #     "positive_only": True,
    #     "topK": 144,
    #     "do_feature_selection": True,
    # },
    # Recall@10 optimised models
    # 60: {  # SLIM ElasticNet
    #     "topK": 1000,
    #     "l1_ratio": 0.009196376132404047,
    #     "alpha": 0.001,
    #     "positive_only": True,
    #     "do_feature_selection": True,
    # },
    # 61: {  # Item KNN CF+CBF
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.0,
    #     "feature_weighting": "TF-IDF",
    #     "ICM_weight": 0.1918507776404466,
    # },
    # 62: {  # Item KNN CF
    #     "topK": 5,
    #     "shrink": 1000,
    #     "similarity": "asymmetric",
    #     "normalize": True,
    #     "asymmetric_alpha": 0.12250234857130494,
    #     "feature_weighting": "TF-IDF",
    # },
    # 63: {  # RP3 ICM
    #     "topK": 556,
    #     "alpha": 2.0,
    #     "beta": 0.43088991464943555,
    #     "delta": 0.0,
    #     "normalize_similarity": True,
    # },
    # 64: {},  # Top Popular
    # Recall@50 optimised models
    70: {  # Item KNN CF
        "topK": 152,
        "shrink": 1000,
        "similarity": "asymmetric",
        "normalize": True,
        "asymmetric_alpha": 0.2696194971486583,
        "feature_weighting": "TF-IDF",
    },
    71: {  # Item KNN CF+CBF
        "topK": 43,
        "shrink": 937,
        "similarity": "asymmetric",
        "normalize": True,
        "asymmetric_alpha": 0.45165257781514373,
        "feature_weighting": "TF-IDF",
        "ICM_weight": 0.3749251701759684,
    },
    72: {  # SLIM ElasticNet
        "l1_ratio": 0.14747318214902194,
        "alpha": 0.00043480530562990655,
        "positive_only": False,
        "topK": 75,
        "do_feature_selection": True,
    },
    73: {  # RP3 ICM
        "topK": 351,
        "alpha": 1.8421852767137328,
        "beta": 0.26321333084561177,
        "delta": 0.27195701300859715,
        "normalize_similarity": True,
    },
}


def recommender_factory(urm, icm) -> dict[int, BaseRecommender]:
    return {
        # User-wide hybrid 1 (0-10)
        # 0: P3alphaRecommender(urm),
        # 1: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 2: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 3: RP3betaRecommenderICM(urm, icm),
        # 4: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 5: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 6: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 7: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 8: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 9: P3alphaRecommender(urm),
        # 10: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 20: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 21: SLIMElasticNetRecommender(urm),
        # 22: RP3betaRecommenderICM(urm, icm),
        # 23: ItemKNNCFRecommender(urm),
        # User-wide hybrid 2 (30-40)
        # 30: SLIMElasticNetRecommender(urm),
        # 31: SLIMElasticNetRecommender(urm),
        # 32: SLIMElasticNetRecommender(urm),
        # 33: SLIMElasticNetRecommender(urm),
        # 34: SLIMElasticNetRecommender(urm),
        # 35: SLIMElasticNetRecommender(urm),
        # 36: SLIMElasticNetRecommender(urm),
        # 37: SLIMElasticNetRecommender(urm),
        # 38: SLIMElasticNetRecommender(urm),
        # 39: SLIMElasticNetRecommender(urm),
        # 40: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # Score hybrid
        # 50: RP3betaRecommenderICM(urm, icm),
        # 51: SLIMElasticNetRecommender(urm),
        # Recall@10 optimised models
        # 60: SLIMElasticNetRecommender(urm),
        # 61: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        # 62: ItemKNNCFRecommender(urm),
        # 63: RP3betaRecommenderICM(urm, icm),
        # 64: TopPop(urm),
        # Recall@50 optimised models
        70: ItemKNNCFRecommender(urm),
        71: ItemKNN_CFCBF_Hybrid_Recommender(urm, icm),
        72: SLIMElasticNetRecommender(urm),
        73: RP3betaRecommenderICM(urm, icm),
    }


if __name__ == "__main__":
    from datetime import datetime
    from concurrent.futures import ProcessPoolExecutor

    icm_df, urm_df = load_raw()
    num_users = urm_df["user_id"].nunique()
    num_items = urm_df["item_id"].nunique()
    num_features = icm_df["feature_id"].nunique()

    icm = sps.csr_matrix(
        (icm_df.data, (icm_df.item_id, icm_df.feature_id)),
        shape=(num_items, num_features),
    )

    def train_fold(i, train_indices, icm, num_users, num_items, urm_df):
        fold_dir = MODEL_DIR / str(i)
        fold_dir.mkdir(exist_ok=True)

        fold_urm_df = urm_df.iloc[train_indices]
        fold_urm = sps.csr_matrix(
            (fold_urm_df["data"], (fold_urm_df["user_id"], fold_urm_df["item_id"])),
            shape=(num_users, num_items),
        )

        fold_recommenders = recommender_factory(fold_urm, icm)
        for j, (key, recommender) in enumerate(fold_recommenders.items()):
            print(
                f"Fold {str(i).zfill(2)} Recommender {str(j).zfill(2)} {datetime.now()}"
            )
            recommender.fit(**HYPERPARAMETERS[key])
            with (fold_dir / f"{key}.pkl").open("wb") as f:
                pickle.dump(recommender, f)

    with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
        futures = [
            executor.submit(
                train_fold,
                i,
                train_indices,
                icm.copy(),
                num_users,
                num_items,
                urm_df.copy(),
            )
            for i, (train_indices, _) in enumerate(
                KFold(N_FOLDS, shuffle=True, random_state=42).split(urm_df)
            )
        ]
        for future in futures:
            future.result()
