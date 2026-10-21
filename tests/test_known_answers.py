"""Small arithmetic controls for the published clustering audit.

These tests protect known calculations; they do not re-run the research cohort.
"""
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler
from scipy.optimize import linear_sum_assignment


class AuditArithmeticTests(unittest.TestCase):
    def test_cross_weight_transition(self):
        # Rows represent the w=.600 groups; columns represent the w=1.000 groups.
        cells = np.array([[6449, 435], [2931, 185]], dtype=int)
        a = np.repeat([0, 1], cells.sum(axis=1))
        b = np.r_[np.repeat(0, 6449), np.repeat(1, 435),
                  np.repeat(0, 2931), np.repeat(1, 185)]
        changed = cells.sum() - cells[linear_sum_assignment(-cells)].sum()
        self.assertEqual(changed, 3366)
        self.assertAlmostEqual(100 * changed / 10000, 33.66, places=2)
        self.assertAlmostEqual(adjusted_rand_score(a, b), -0.0024380965, places=8)

    def test_label_names_do_not_change_ari(self):
        left = [0, 0, 1, 1, 0, 1]
        right = [1, 1, 0, 0, 1, 0]
        self.assertAlmostEqual(adjusted_rand_score(left, right), 1.0)

    def test_zero_iqr_column_has_unit_scale(self):
        values = np.array([[0.0], [0.0], [0.0], [0.0], [0.0],
                           [0.0], [0.0], [0.0], [0.0], [4.0]])
        scaled = RobustScaler().fit(values)
        self.assertAlmostEqual(float(scaled.scale_[0]), 1.0)
        np.testing.assert_allclose(scaled.transform(values), values)

    def test_dispersion_shares_sum_to_one(self):
        parts = np.array([3.2, 1.8, 0.6])
        shares = parts / parts.sum()
        self.assertAlmostEqual(float(shares.sum()), 1.0)

    def test_readmission_unadjusted_wald_bounds(self):
        # R2 K=2: held-out groups n=57,692 and 3,826.
        p0, p1 = 5052 / 57692, 383 / 3826
        diff = 100 * (p1 - p0)
        se = 100 * ((p0 * (1 - p0) / 57692) +
                    (p1 * (1 - p1) / 3826)) ** 0.5
        self.assertAlmostEqual(diff, 1.25360807988, places=7)
        self.assertAlmostEqual(diff - 1.96 * se, 0.27497978056, places=7)
        self.assertAlmostEqual(diff + 1.96 * se, 2.23223637921, places=7)


if __name__ == "__main__":
    unittest.main()
