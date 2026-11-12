#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gab
"""

import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
import time
import sys

class RP3betaRecommenderICM(BaseItemSimilarityMatrixRecommender):
    """ RP3beta recommender with item content information """

    RECOMMENDER_NAME = "RP3betaRecommenderICM"

    def __init__(self, URM_train, ICM, verbose=True):
        super(RP3betaRecommenderICM, self).__init__(URM_train, verbose=verbose)
        self.ICM = check_matrix(ICM, 'csr')
    
    def __str__(self):
        return "RP3betaICM(alpha={}, beta={}, delta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(
            self.alpha, self.beta, self.delta, self.min_rating, self.topK, self.implicit, self.normalize_similarity)

    def fit(self, alpha=1., beta=0.6, delta=0.5, min_rating=0, topK=100, implicit=False, normalize_similarity=True):
        """
        Trains the recommender.
        
        Args:
            alpha: float
                Power coefficient for the final rating matrix
            beta: float
                Penalization coefficient for item popularity
            delta: float
                Weight for ICM path (0 to 1). Higher values give more weight to ICM paths
            min_rating: float
                Minimum rating to consider
            topK: int
                Number of top items to keep per similarity row
            implicit: bool
                If True, ratings are treated as binary
            normalize_similarity: bool
                If True, similarity matrix will be normalized
        """
        
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Normalize the URM considering only the target user
        Pui = normalize(self.URM_train, norm='l1', axis=1)
        
        # Normalize the ICM path: Item -> Feature
        Pif = normalize(self.ICM, norm='l1', axis=1)
        # And Feature -> Item (transpose)
        Pfi = normalize(self.ICM.T, norm='l1', axis=1)

        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Normalized URM and ICM")

        # Compute the degree for each item
        item_degree = np.array(self.URM_train.sum(axis=0)).ravel()
        item_degree_inv = np.zeros_like(item_degree, dtype=np.float32)
        nonzero_mask = item_degree != 0.0
        item_degree_inv[nonzero_mask] = np.power(item_degree[nonzero_mask], -self.beta)

        # Initialize similarity builder
        similarity_builder = Incremental_Similarity_Builder(self.n_items, 
                                                         initial_data_block=self.n_items*self.topK,
                                                         dtype=np.float32)

        start_time = time.time()
        start_time_batch = start_time
        processedItems = 0

        # Compute similarities blockwise
        block_size = 1000

        for current_item_start_position in range(0, self.n_items, block_size):
            
            if current_item_start_position + block_size > self.n_items:
                block_size = self.n_items - current_item_start_position

            # Compute similarity for a block of items
            # Path 1: U -> I -> U -> I  (original RP3beta path)
            path1_block = self.URM_train.T[current_item_start_position:current_item_start_position + block_size].dot(Pui)
            
            # Path 2: I -> F -> I  (feature path)
            path2_block = Pif[current_item_start_position:current_item_start_position + block_size].dot(Pfi)
            
            # Combine both paths
            similarity_block = (1 - self.delta) * path1_block + self.delta * path2_block

            if self.alpha != 1.:
                similarity_block = similarity_block.power(self.alpha)

            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_size):
                item_idx = current_item_start_position + row_in_block

                # Apply the degree normalization
                this_item_weights = similarity_block[row_in_block, :] * item_degree_inv
                this_item_weights[item_idx] = 0.0

                # Get the top-K items
                relevant_items_partition = np.argpartition(-this_item_weights, self.topK-1, axis=0)[:self.topK]
                this_item_weights = this_item_weights[relevant_items_partition]

                # Remove zeros
                nonzero_mask = this_item_weights != 0.0
                if np.any(nonzero_mask):
                    relevant_items_partition = relevant_items_partition[nonzero_mask]
                    this_item_weights = this_item_weights[nonzero_mask]

                    similarity_builder.add_data_lists(
                        row_list_to_add=np.ones(len(this_item_weights), dtype=np.int32) * item_idx,
                        col_list_to_add=relevant_items_partition,
                        data_list_to_add=this_item_weights)

            processedItems += block_size

            if time.time() - start_time_batch >= 300 or processedItems == self.n_items:
                elapsed_time = time.time() - start_time
                new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                if self.verbose:
                    print(self.RECOMMENDER_NAME + ": Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.0f}".format(
                        processedItems,
                        100.0 * float(processedItems) / self.n_items,
                        new_time_value,
                        new_time_unit,
                        float(processedItems) / elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        self.W_sparse = similarity_builder.get_SparseMatrix()

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')

        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Computation completed")