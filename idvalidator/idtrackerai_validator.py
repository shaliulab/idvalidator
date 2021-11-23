import os.path
import shutil

import pythonvideoannotator
from pythonvideoannotator_module_idtrackerai.models.video.objects.idtrackerai_object_io import (
    IdtrackeraiObjectIO,
)


def load_correction_row(row):
    """
    Load a row of a csv file with the structure
    chunk,start_end,idtrackerai,human

    where:
      * chunk is an integer stating to which chunk this correction belongs to
      * start_end is a string stating when does the fragment start and end, separated by ;
      * idtrackerai is a string stating the identity that idtrackrai has given to the fragment
      * human is a string stating the identity the human thinkgs the fragment should have

    fragments can have more than one identity (because of crossings). in order to pass more than one identity,
    separate each with ;

    """
    start_end = row["start_end"].split(";")
    start_end = tuple([int(e) for e in start_end])

    assigned_identities = row["idtrackerai"].strip(";").split(";")
    assigned_identities = [eval(e) for e in assigned_identities]

    human_identities = row["human"].strip(";").split(";")
    human_identities = [eval(e) for e in human_identities]
    return start_end, assigned_identities, human_identities


def find_fragment(idtrackeraiobjectio, start_end, assigned_identities):
    """
    Find the fragment with passed start_end and final_identities
    """
    for fragment in idtrackeraiobjectio.list_of_framents.fragments:
        if (
            fragment.start_end == start_end
            and fragment.assigned_identities == assigned_identities
        ):
            return fragment

    raise Exception(f"No fragment found {start_end}, {assigned_identities}")


def find_blobs(idtrackeraiobjectio, fragment):
    """
    Find the blobs that belong to the passed fragment
    """
    all_blobs = idtrackeraiobjectio.list_of_blobs.blobs_in_video[
        fragment.start_end[0] : fragment.start_end[1]
    ]

    fragment_blobs = []
    for frame_blobs in all_blobs:
        fragment_blobs.extend(
            [
                blob
                for blob in frame_blobs
                if blob.fragment_identifier == fragment.identifier
            ]
        )

    return fragment_blobs


def apply_corrections(idtrackeraiobjectio, corrections):

    for i, row in corrections.iterrows():

        start_end, assigned_identities, human_identities = load_correction_row(
            row
        )
        fragment = find_fragment(
            idtrackeraiobjectio, start_end, assigned_identities
        )

        blobs = find_blobs(idtrackeraiobjectio, fragment)

        if blobs:
            # update identity of fragment
            fragment.user_generated_identities = human_identities

            # update identity of all blobs of the fragment
            for blob in blobs:
                blob._user_generated_identities = human_identities

        else:
            raise Exception(
                f"No blobs found for correction {i}/{corrections.shape[0]}"
            )

    return idtrackeraiobjectio


def get_backup_folder_name(project_path):

    dest_0 = project_path + "-original"
    dest = project_path + "-original"
    i = 0
    while os.path.exists(dest):
        i += 1
        dest = dest_0 + f"_{i}"

    return dest


def validate_single_thread(corrections, session_folder):
    session_corrections = corrections.loc[
        corrections["folder"] == os.path.basename(session_folder.strip("/"))
    ]
    project_path = session_folder
    idtrackeraiobjectio = IdtrackeraiObjectIO()

    # backup original machine-only results
    shutil.copytree(project_path, get_backup_folder_name(project_path))
    # load idtracker.ai results
    idtrackeraiobjectio.load_from_idtrackerai(project_path)
    # apply human corrections
    apply_corrections(idtrackeraiobjectio, session_corrections)
    # save updated results
    idtrackeraiobjectio.save_updated_identities()
