import argparse
import os.path
import pandas as pd
import joblib

from idvalidator import validate_single_thread

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-folder", "--input", type=str, required=True, help="Path to single idtrackerai results folder")
    ap.add_argument("--ncores", "-j", type=int, default=1, help="Number of cores to use")
    return ap


def validate(experiment_folder, ncores=1):

    corrections = pd.read_csv(os.path.join(experiment_folder, "corrections.csv"))
    folders = corrections["folder"].unique()

    if ncores == 1 or len(folders) == 1:
        for folder in folders:
            validate_single_thread(corrections, os.path.join(experiment_folder, folder))
    else:
        joblib.Parallel(n_jobs=ncores)(joblib.delayed(validate_single_thread)(corrections, os.path.join(experiment_folder, folder)) for folder in folders)


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    validate(args.experiment_folder, args.ncores)


if __name__ == "__main__":
    main()
