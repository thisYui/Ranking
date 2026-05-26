import unittest

import numpy as np

from src.loss import pairwise_hinge_loss_and_grad
from src.model import LinearRanker


class PairwiseHingeLossTests(unittest.TestCase):
    def test_gradient_is_averaged_over_all_pairs(self):
        model = LinearRanker(n_features=1, seed=42)
        model.w = np.array([1.0])

        X = np.array([[0.0], [2.0], [10.0]])
        pairs = np.array([[0, 1], [0, 2]])
        labels = np.array([-1, 1])

        loss, grad = pairwise_hinge_loss_and_grad(model, X, pairs, labels, reg=0.0)

        self.assertAlmostEqual(loss, 5.5)
        np.testing.assert_allclose(grad, np.array([5.0]))


if __name__ == "__main__":
    unittest.main()
