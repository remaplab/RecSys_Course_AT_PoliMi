#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for the top-K selection of Compute_Similarity_Cython.

When an item had fewer than topK neighbours, the top-K selection could pick buffer positions holding
item ids left over from previously processed items. If such an id was also a neighbour of the current item
it was written twice, and since csr_matrix sums duplicates its similarity was inflated (cosine values > 1).

Requires the Cython extension to be compiled (run_compile_all_cython.py). Run from the repository root with:
    python -m unittest Recommenders/Similarity/Compute_Similarity_Cython_topK_test.py
"""

import unittest

import numpy as np
import scipy.sparse as sps

try:
    from Recommenders.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


def _reference_cosine(data_matrix):
    """Brute-force column-wise cosine similarity with zero diagonal, same epsilon as the implementation."""
    data_matrix = sps.csc_matrix(data_matrix, dtype=np.float64)
    norms = np.sqrt(np.asarray(data_matrix.power(2).sum(axis=0)).ravel())
    W = (data_matrix.T @ data_matrix).toarray()
    np.fill_diagonal(W, 0.0)
    return W / (np.outer(norms, norms) + 1e-6)


@unittest.skipUnless(CYTHON_AVAILABLE, "Compute_Similarity_Cython is not compiled")
class Compute_Similarity_Cython_TopK_Test(unittest.TestCase):

    def test_no_duplicate_entries_minimal(self):
        # User 1 rated items 0 and 2, so cosine(0, 2) = 1.
        # Item 2 has a single neighbour (item 0) but topK = 2, so the selection also looked at an unused buffer
        # position whose (initial) id is 0: item 0 was added twice and W[0, 2] became 2.0
        data_matrix = sps.csr_matrix(np.array([[0, 0, 0],
                                               [1, 0, 1]], dtype=np.float32))

        W = Compute_Similarity_Cython(data_matrix, topK=2, shrink=0, normalize=True, similarity="cosine").compute_similarity()

        self.assertTrue(np.allclose(W.toarray(), _reference_cosine(data_matrix), atol=1e-4),
                        "Similarity does not match reference:\n{}".format(W.toarray()))

    def test_matches_reference_when_topK_exceeds_neighbours(self):
        # Sparse data and large topK: most items have fewer than topK neighbours, which triggered stale-id selection
        data_matrix = sps.random(300, 200, density=0.02, random_state=1, format="csr", dtype=np.float32)
        data_matrix.data[:] = 1.0

        reference = _reference_cosine(data_matrix)
        topK = data_matrix.shape[1] - 1     # Larger than any neighbourhood, so no ties to break

        W = Compute_Similarity_Cython(data_matrix, topK=topK, shrink=0, normalize=True, similarity="cosine").compute_similarity()
        W = sps.csr_matrix(W)

        self.assertLessEqual(W.max(), 1.0 + 1e-4, "Cosine similarity greater than 1, duplicate entries were summed")
        self.assertEqual(W.nnz, np.count_nonzero(reference), "Unexpected number of non-zero similarities")
        self.assertTrue(np.allclose(W.toarray(), reference, atol=1e-4), "Similarity does not match reference")


if __name__ == '__main__':
    unittest.main()
