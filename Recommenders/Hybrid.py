import numpy.linalg as LA
import scipy.sparse as sps

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import (
    BaseItemSimilarityMatrixRecommender,
)

# recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
# recommender_object.fit(new_similarity)

# result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
# result_df


class ScoresHybridRecommender(BaseRecommender):
    """
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2
    may come from algorithms trained on different loss functions.
    In this case, normalising scores is advisable (by default, no normalisation is applied.)
    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super().__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.norm = None

    def fit(self, norm=None, alpha=0.5):
        self.alpha = alpha
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        if self.norm is not None:
            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(
                        self.norm
                    )
                )
            item_weights_1 /= norm_item_weights_1

            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(
                        self.norm
                    )
                )
            item_weights_2 /= norm_item_weights_2

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


class UserWideHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "UserWideHybridRecommender"

    def __init__(
        self,
        URM_train,
        group_users: dict,
        group_recommenders: dict[int, BaseRecommender],
        verbose=True,
    ):
        super().__init__(URM_train, verbose=verbose)
        self.user_groups = {
            user: group for group, users in group_users.items() for user in users
        }
        self.group_recommenders = group_recommenders

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        for user_id in user_id_array:
            user_group = self.user_groups[user_id]
            return self.group_recommenders[user_group]._compute_item_score(
                user_id_array, items_to_compute
            )


class ScoresMultipleHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Hybrid recommender that combines multiple recommenders
    through a weighted sum of their predictions
    """

    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(
        self, URM_train, recommender_list, recommender_weights=None, verbose=True
    ):
        super().__init__(URM_train, verbose=verbose)

        if recommender_weights is None:
            recommender_weights = [1 / len(recommender_list)] * len(recommender_list)

        if len(recommender_list) != len(recommender_weights):
            raise ValueError("Number of recommenders and weights must match")

        self.recommender_list = recommender_list
        self.recommender_weights = recommender_weights

        # Normalize weights to sum to 1
        total_weight = sum(self.recommender_weights)
        if total_weight > 0:
            self.recommender_weights = [
                w / total_weight for w in self.recommender_weights
            ]

        if self.verbose:
            print(
                f"{self.RECOMMENDER_NAME}: Initialized with {len(recommender_list)} recommenders"
            )

    def fit(self):
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Fitting - No additional training needed")
        pass  # Base recommenders are already fitted

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Initialize the score matrix
        item_scores = None

        # Combine predictions from all recommenders
        for idx, recommender in enumerate(self.recommender_list):
            current_score = recommender._compute_item_score(
                user_id_array, items_to_compute
            )

            if item_scores is None:
                item_scores = current_score * self.recommender_weights[idx]
            else:
                item_scores += current_score * self.recommender_weights[idx]

        return item_scores
