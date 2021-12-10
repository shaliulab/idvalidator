import argparse
import os
import os.path

import numpy as np
import joblib

from idvalidator.validator import check_blobs
from idtrackerai.utils.py_utils import pick_blob_collection, is_idtrackerai_folder

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--experiment-folder", "--input", dest="input", required=True)
    ap.add_argument("--ncores", type=int, default=-2, required=True)
    return ap

def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()



    folders = os.listdir(args.input)
    folders = [folder for folder in folders if is_idtrackerai_folder(os.path.join(args.input, folder))]
    files = {folder: pick_blob_collection(os.path.join(args.input, folder)) for folder in folders}

    output = joblib.Parallel(n_jobs=args.ncores)(
        joblib.delayed(check_blobs)(
            blobs_file
        ) for blobs_file in files.values()
    )
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
