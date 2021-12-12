import unittest
from argparse import Namespace

import numpy as np

from idvalidator.validator import centroids_swap
from idvalidator.validator import (
    check_blob_has_identity,
    check_all_identities_are_found,
    check_blobs_annotation_function,
)

from idvalidator.bin.validator import single_validator


class TestSwap(unittest.TestCase):
    def setUp(self):

        self._centroids_before = np.array(
            [
                [0.0, 0.0],
                [100.0, 100.0],
                [200.0, 200.0],
                [300.0, 300.0],
                [310.0, 310.0],
            ]
        )

        self._centroids_after = np.array(
            [
                [0.0, 0.0],
                [200.0, 200.0],
                [210.0, 210.0],
                [290.0, 290.0],
                [300.0, 300.0],
            ]
        )

    def test_centroids_swap_can_catch_weird_behavior(self):
        swap = centroids_swap(
            self._centroids_before, self._centroids_after, body_length_px=15
        )
        target = np.array([[3, 2], [4, 5]])

        # check the swap is the same as target
        diff = swap - target
        status = np.mean(diff == 0).tolist() == 1
        self.assertTrue(status)


class TestValidator(unittest.TestCase):

    session_folder = "tests/static_data/session_000003"

    def test_validator(self):
        args = Namespace(input=self.session_folder, output="test.pkl")
        output = single_validator(args=args)

    def tearDown(self):
        # os.remove("test.pkl")
        pass


class TestFilters(unittest.TestCase):
    def test_check_blob_has_identity_detects_missing_id(self):
        blob = Namespace(final_identities=[])
        self.assertFalse(check_blob_has_identity(blob))
        blob = Namespace(final_identities=[0])
        self.assertFalse(check_blob_has_identity(blob))
        blob = Namespace(final_identities=[None])
        self.assertFalse(check_blob_has_identity(blob))

    def test_check_blob_has_identity_detects_ids(self):
        blob = Namespace(final_identities=[1])
        self.assertTrue(check_blob_has_identity(blob))
        blob = Namespace(final_identities=1)
        self.assertTrue(check_blob_has_identity(blob))

    def test_check_blob_annotation_function(self):

        identities = list(range(1, 7))
        blobs = [Namespace(final_identities=[i]) for i in identities]
        status, blobs = check_blobs_annotation_function(
            blobs, check_blob_has_identity
        )

        self.assertTrue(status)
        self.assertTrue(len(blobs) == 0)

        blobs = [Namespace(final_identities=[i]) for i in identities]
        blobs[1].final_identities[0] = None
        status, blobs = check_blobs_annotation_function(
            blobs, check_blob_has_identity
        )
        self.assertFalse(status)
        self.assertTrue(blobs[0].final_identities[0] is None)

    def test_check_all_identities_are_found(self):
        identities = list(range(1, 7))
        blobs = [Namespace(final_identities=[i]) for i in identities]
        status, ids = check_all_identities_are_found(blobs, identities)
        self.assertTrue(status)
        self.assertTrue(len(ids) == 0)
        blobs[1].final_identities[0] = 0
        status, ids = check_all_identities_are_found(blobs, identities)
        self.assertFalse(status)
        self.assertTrue(ids[0] == identities[1])

    def test_blobs_swap_catches_a_jump(self):
        pass

        # body_length_px = 1
        # blovs_previous = [
        #    Namespace(final_identities=[0], final_centroids=np.array([0, 0])),
        #    Namespace(final_identities=[1], final_centroids=np.array([10, 0])),
        #    Namespace(final_identities=[2], final_centroids=np.array([20, 0])),
        # ]
        # blovs_next = [
        #    Namespace(final_identities=[2], final_centroids=np.array([5, 0])),
        #    Namespace(final_identities=[1], final_centroids=np.array([10, 0])),
        #    Namespace(final_identities=[0], final_centroids=np.array([1, 0])),
        # ]


if __name__ == "__main__":
    unittest.main()
