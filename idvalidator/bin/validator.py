import argparse
import os
import os.path
import pickle

import numpy as np
import joblib

from idvalidator.validator import check_blobs
from idtrackerai.utils.py_utils import pick_blob_collection, is_idtrackerai_folder

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--experiment-folder", "--input", dest="input", required=True)
    ap.add_argument("--ncores", type=int, default=-2)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--number-of-animals", type=int, required=True)
    ap.add_argument("--body-length-px", type=int, required=True)
    return ap

def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()



    folders = os.listdir(args.input)
    folders = [folder for folder in folders if is_idtrackerai_folder(os.path.join(args.input, folder))]
    folders = sorted(folders)
    files = {folder: pick_blob_collection(os.path.join(args.input, folder)) for folder in folders}


    #output = check_blobs(list(files.values())[0], number_of_animals=args.number_of_animals, body_length_px=args.body_length_px)

    output = joblib.Parallel(n_jobs=args.ncores)(
        joblib.delayed(check_blobs)(
            blobs_file, number_of_animals=args.number_of_animals, body_length_px=args.body_length_px
        ) for blobs_file in files.values()
    )


    output = {folders[i]: output[i] for i in range(len(output))}

    with open(args.output, "wb") as fh:
        pickle.dump(output, fh)


if __name__ == "__main__":
    main()