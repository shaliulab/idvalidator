import argparse
import itertools
from collections import namedtuple

import numpy as np
from scipy.spatial.distance import cdist

from idtrackerai.list_of_blobs import ListOfBlobs
from trajectorytools.trajectories.concatenate import _compute_distance_matrix


Validation = namedtuple("Validation", ["file", "identified", "tracked", "swaps"])


def check_blobs_f(blobs_in_frame, f):
    blobs = [blob for blob in blobs_in_frame if not f(blob)]
    return blobs # empty if all blobs are ok


def check_blob_has_identity(blob):
    return blob.final_identities is not None and len(blob.final_identities) >= 1


def check_all_identities_are_found(blobs_in_frame, identities):
    tracked_identities = list(itertools.chain(*[blob.final_identity for blob in blobs_in_frame]))
    return [i for i in tracked_identities if not i in identities] # empty if all ids are found


def get_centroids(blobs_in_frame):

    blobs_in_frame = sorted(blobs_in_frame, key=lambda blob: blob.final_identity)
    centroids = np.vstack([blob.centroid for blob in blobs_in_frame])
    return centroids


def centroids_swap(centroids_previous, centroids_next, body_length_px):

    number_of_animals = centroids_previous.shape[0]
    jump_size = 1 # units of body size

    distances = _compute_distance_matrix(centroids_previous, centroids_next)
    swap = np.bitwise_and(
        (distances < body_length_px * jump_size),
        np.eye(number_of_animals) == 0
    )

    id_previous, id_next = np.where(swap)
    id_previous += 1
    id_next += 1

    swap_ids = np.stack([id_previous, id_next], axis=1)
    return swap_ids


def blobs_swap(blobs_in_frame_previous, blobs_in_frame_next):

    centroids_previous = get_centroids(blobs_in_frame_previous)
    centroids_next = get_centroids(blobs_in_frame_next)
    swap = centroids_swap(centroids_previous, centroids_next)


def check_blobs(blob_file, number_of_animals):
    list_of_blobs = ListOfBlobs.load(blob_file)
    identities = list(range(number_of_animals))

    blobs = list_of_blobs.blobs_in_video

    frames_fully_identified = [check_blobs_f(blobs_in_frame, check_blob_has_identity) for blobs_in_frame in blobs]
    frames_fully_tracked = [check_all_identities_are_found(blobs_in_frame, identities) for blobs_in_frame in blobs]
    identities_dont_swap = [blobs_swap(blobs[i], blobs[i+1]) for i in range(len(blobs)-1)]

    return Validation(blob_file, frames_fully_identified, frames_fully_tracked, identities_dont_swap)
