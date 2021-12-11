import unittest
from argparse import Namespace

import numpy as np

from idvalidator.validator import centroids_swap
from idvalidator.bin.validator import single_validator

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


class TestValidator(unittest.TestCase):

    session_folder = "tests/static_data/session_000003"

    def test_validator(self):
        args = Namespace(input=self.session_folder, output="test.pkl")
        output = single_validator(args=args)
        print(output)

    def tearDown(self):
        #os.remove("test.pkl")
        pass


if __name__ == '__main__':
    unittest.main()
