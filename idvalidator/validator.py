import os.path
import itertools
from collections import namedtuple
import logging
import traceback

import numpy as np
from scipy.spatial.distance import cdist

from idtrackerai.list_of_blobs import ListOfBlobs
from trajectorytools.trajectories.concatenate import _compute_distance_matrix

logger = logging.getLogger(__name__)


Validation = namedtuple(
    "Validation", ["file", "identified", "tracked", "swaps"]
)



def check_blob_has_identity(blob):
    """
    True if blob has a final identity
    """

    if isinstance(blob.final_identities, list):
        return (
            len(blob.final_identities) > 0
            and blob.final_identities[0] not in  [None, 0]
        )
    else:
        return blob.final_identities is not None and blob.final_identities != 0

def check_blobs_annotation_function(blobs_in_frame, function=check_blob_has_identity):
    """
    True if every blob passes function
    """
    blobs = [blob for blob in blobs_in_frame if not function(blob)]
    if len(blobs) == 0:
        check = True
    else:
        check = False

    return check, blobs


def check_all_identities_are_found(blobs_in_frame, identities):
    """
    True if all queried identities are found in this frame
    """
    tracked_identities = list(
        itertools.chain(*[blob.final_identities for blob in blobs_in_frame])
    )
    # tracked_identities = [*i if isinstance(i, list) else i for i in tracked_identities]
    #blobs = [
    #    blob for blob in blobs_in_frame if blobs_in_frame.final_identities[0] in identities
    #]  # empty if all ids are found

    ids = [id for id in identities if id not in tracked_identities]

    if len(ids) == 0:
        check = True
    else:
        check = False

    return check, ids


def get_identity(final_identities):

    if isinstance(final_identities, list):
        return min(final_identities)
    else:
        return final_identities


def get_centroids(blobs_in_frame):

    # this assumes all blobs have a final_identity
    assert all(
        [blob.final_identities[0] is not None for blob in blobs_in_frame]
    )

    blobs_in_frame = sorted(
        blobs_in_frame, key=lambda blob: get_identity(blob.final_identities)
    )
    centroids = np.vstack([blob.final_centroids for blob in blobs_in_frame])
    return centroids


def centroids_swap(
    centroids_previous, centroids_next, body_length_px, jump_size=1
):
    """
    Checks if any of the centroids in the previous frame is less than
    jump_size bodies away from where a centroid
    with a different id is in the next frame

    Example animal 1 is at x y, and then animal 2 is at x', y' in the next frame
    where x',y' is a point within jump_size bodies from x,y.
    Then these animals could have swaped

    The centroids are sorted by id
    so the diagonal distance should always be least in a healthy frame

    """

    n_animals_previous = centroids_previous.shape[0]
    n_animals_next = centroids_next.shape[0]

    diff = n_animals_previous - n_animals_next
    if diff != 0:
        padding = np.array([[0.0, 0.0]] * np.abs(diff))
        if diff < 0:
            centroids_previous = np.vstack([centroids_previous, padding])
        else:
            centroids_next = np.vstack([centroids_next, padding])

    assert centroids_previous.shape[0] == centroids_next.shape[0]
    n_animals_in_this_pair = centroids_previous.shape[0]

    distances = _compute_distance_matrix(centroids_previous, centroids_next)

    diagonal = np.eye(n_animals_in_this_pair) == 1
    diag_distances = distances[diagonal]

    # the element in the ith row is the distance between the centroids with id i in between frames
    # no animal should have less distance than this
    diag_distances_expanded = np.stack(
        [diag_distances.tolist()] * n_animals_in_this_pair, axis=1
    )

    swap = np.bitwise_and(
        (distances < body_length_px * jump_size),
        distances < diag_distances_expanded,
    )

    id_previous, id_next = np.where(swap)
    id_previous += 1
    id_next += 1

    swap_ids = np.stack([id_previous, id_next], axis=1)
    if len(swap_ids) == 0:
        return True, []
    else:
        return False, swap_ids


def blobs_swap(blobs_in_frame_previous, blobs_in_frame_next, **kwargs):

    centroids_previous = get_centroids(blobs_in_frame_previous)
    centroids_next = get_centroids(blobs_in_frame_next)
    status, swap = centroids_swap(centroids_previous, centroids_next, **kwargs)
    return status, swap


def check_blobs(blob_file, **kwargs):

    list_of_blobs = ListOfBlobs.load(blob_file)
    video_file = os.path.join(
        os.path.dirname(os.path.dirname(blob_file)), "video_object.npy"
    )
    video_object = np.load(video_file, allow_pickle=True).item()
    number_of_animals = video_object.user_defined_parameters[
        "number_of_animals"
    ]
    body_length_px = video_object.median_body_length_full_resolution

    identities = list(range(1, number_of_animals + 1))

    blobs = list_of_blobs.blobs_in_video

    # frames_fully_identified = [
    #     check_blobs_f(blobs_in_frame, check_blob_has_identity)
    #     for blobs_in_frame in blobs
    # ]
    # frames_fully_tracked = [
    #     check_all_identities_are_found(blobs_in_frame, identities)
    #     for blobs_in_frame in blobs
    # ]
    frames_fully_identified = []
    frames_fully_tracked = []
    identities_dont_swap = []
    last_good_frame = None
    for i in range(len(blobs)):

        blobs_in_frame = blobs[i]
        ##############################################################

        # this checks if all animals detected have an id
        # (still may be that not animals are found)
        fully_idd, blobs_with_unknown_id = check_blobs_annotation_function(
            blobs_in_frame, check_blob_has_identity
        )
        if fully_idd:
            frames_fully_identified.append(fully_idd)
        else:
            frames_fully_identified.append(blobs_with_unknown_id)
        ##############################################################

        ##############################################################

        # this checks if all animals are found
        # and all animals have a unique id
        # which should match the identities provided
        fully_tracked, not_tracked_frames = check_all_identities_are_found(
            blobs_in_frame, identities
        )

        if fully_tracked:
            frames_fully_tracked.append(fully_tracked)
        else:
            frames_fully_tracked.append(not_tracked_frames)
        ##############################################################

        if fully_tracked and fully_idd:
            previous_good_frame = last_good_frame
            last_good_frame = i

            if previous_good_frame is not None:

                if i == len(blobs):
                    identities_dont_swap.append(None)
                else:
                    try:
                        status, ids = blobs_swap(
                            blobs[previous_good_frame],
                            blobs[i],
                            body_length_px=body_length_px,
                            **kwargs,
                        )
                    except Exception as error:
                        logger.error(f"Problem with {blob_file} - frame {i} vs {previous_good_frame}")
                        logger.error(error)
                        logger.error(traceback.print_exc())

                        identities_dont_swap.append(
                            None
                        )  # this pair had a nissue
                    else:
                        if status:
                            identities_dont_swap.append(status)
                        else:
                            identities_dont_swap.append(
                                ((previous_good_frame, i), ids)
                            )
        else:
            logger.debug(f"Frame {i} is not fully tracked and identified")
            identities_dont_swap.append(None)  # this pair

    return Validation(
        blob_file,
        frames_fully_identified,
        frames_fully_tracked,
        identities_dont_swap,
    )
