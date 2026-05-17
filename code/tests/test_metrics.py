import numpy as np
import unittest

from src.metrics import kendall_tau_distance, pairwise_accuracy, ranking_loss, top_k_overlap
from src.pairs import generate_preference_pairs


class MetricTests(unittest.TestCase):
    def test_pairwise_accuracy_and_loss_for_perfect_order(self):
        true_scores = np.array([3.0, 2.0, 1.0])
        pred_scores = np.array([9.0, 8.0, 7.0])
        pairs, labels = generate_preference_pairs(true_scores)

        self.assertEqual(pairwise_accuracy(pred_scores, pairs, labels), 1.0)
        self.assertEqual(ranking_loss(pred_scores, pairs, labels), 0.0)
        self.assertEqual(kendall_tau_distance(true_scores, pred_scores), 0.0)

    def test_pairwise_accuracy_and_loss_for_reversed_order(self):
        true_scores = np.array([3.0, 2.0, 1.0])
        pred_scores = np.array([1.0, 2.0, 3.0])
        pairs, labels = generate_preference_pairs(true_scores)

        self.assertEqual(pairwise_accuracy(pred_scores, pairs, labels), 0.0)
        self.assertEqual(ranking_loss(pred_scores, pairs, labels), 1.0)
        self.assertEqual(kendall_tau_distance(true_scores, pred_scores), 1.0)

    def test_top_k_overlap(self):
        true_scores = np.array([5.0, 4.0, 1.0, 0.0])
        pred_scores = np.array([0.0, 4.0, 5.0, 1.0])

        self.assertEqual(top_k_overlap(true_scores, pred_scores, k=2), 0.5)


if __name__ == "__main__":
    unittest.main()
