import argparse
import os
import os.path
import pickle

import numpy as np
import joblib

from idvalidator.validator import check_blobs
from idtrackerai.utils.py_utils import (
    pick_blob_collection,
    is_idtrackerai_folder,
)


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str)
    # ap.add_argument("--number-of-animals", type=int, required=True)
    # ap.add_argument("--body-length-px", type=int, required=True)
    return ap


def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        ap.add_argument(
            "--experiment-folder", "--input", dest="input", required=True
        )
        ap.add_argument("--ncores", type=int, default=-2)
        args = ap.parse_args()

    folders = os.listdir(args.input)
    folders = [
        folder
        for folder in folders
        if is_idtrackerai_folder(os.path.join(args.input, folder))
    ]
    folders = sorted(folders)
    files = {
        folder: pick_blob_collection(os.path.join(args.input, folder))
        for folder in folders
    }

    output = joblib.Parallel(n_jobs=args.ncores)(
        joblib.delayed(check_blobs)(blobs_file)
        for blobs_file in files.values()
    )

    validation = {folders[i]: output[i] for i in range(len(output))}

    if args.output is not None:
        with open(args.output, "wb") as fh:
            pickle.dump(validation, fh)

    return validation


def single_validator(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        ap.add_argument(
            "--session-folder", "--input", dest="input", required=True
        )
        args = ap.parse_args()

    blobs_file = pick_blob_collection(args.input)
    assert blobs_file is not None, f"blobs file not found in {args.input}"

    validation = {
        os.path.basename(args.input.rstrip("/")): check_blobs(blobs_file)
    }

    if args.output is not None:
        with open(args.output, "wb") as fh:
            pickle.dump(validation, fh)

    return validation


if __name__ == "__main__":
    main()
