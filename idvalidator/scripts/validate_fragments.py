import argparse

import pandas as pd
from idvalidator import validate_single_thread

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections", type=str, required=True)
    ap.add_argument("--experiment-folder", "--input", type=str, required=True)
    ap.add_argument("--ncores", "-j", type=int, default=1)
    return ap


def validate(corrections_file, experiment_folder, ncores=1):

    corrections = pd.read_csv(corrections_file)
    chunks = corrections["chunk"].unique()

    if ncores == 1:
        for chunk in chunks:
            validate_single_thread(corrections, experiment_folder, chunk)
    else:
        joblib.Parallel(n_jobs=ncores)(joblib.delayed(validate_single_thread)(corrections, experiment_folder, chunk) for chunk in chunks)


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    validate(args.corrections, args.experiment_folder, args.ncores)


if __name__ == "__main__":
    main()
