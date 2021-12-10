import unittest

import numpy as np

from idvalidator.validator import centroids_swap

class TestSwap(unittest.TestCase):

    def setUp(self):
        
        self._centroids_before = np.array([
            [0., 0.],
            [100., 100.],
            [200., 200.],
            [300., 300.],
            [310., 310.],
        ])

        self._centroids_after = np.array([
            [0., 0.],
            [200., 200.],
            [210., 210.],
            [290., 290.],
            [300.,300.],
        ])

    def test_centroids_swap_can_catch_weird_behavior(self):
        swap = centroids_swap(self._centroids_before, self._centroids_after, body_length_px=15)
        target = np.array([[3,2], [4, 5]])

        # check the swap is the same as target
        diff = swap-target
        status = np.mean(diff == 0).tolist() == 1
        self.assertTrue(status)


if __name__ == '__main__':
    unittest.main()
