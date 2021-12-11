import argparse
from argparse import Namespace
import os
import os.path
import subprocess
import logging

import numpy as np
import cv2
import joblib
import imgstore
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.utils.py_utils import get_spaced_colors_util

import idvalidator.bin.validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--experiment-folder", "--input", dest="input", type=str, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--output", required=True, type=str, help="Output folder with .csv and .mp4 files")
    return ap

def load_data(session_folder):

    video_object = np.load(os.path.join(session_folder, "video_object.npy"), allow_pickle=True).item()
    colors = get_spaced_colors_util(
        video_object.user_defined_parameters["number_of_animals"],
        black=True
    )

    blobs_file = os.path.join(session_folder, "preprocessing", "blobs_collection_no_gaps.npy")
    list_of_blobs = ListOfBlobs.load(blobs_file)
    return list_of_blobs, colors

def load_store(experiment_folder, chunk=None):

    if chunk is not None:
        chunk = [chunk]

    store = imgstore.new_for_filename(experiment_folder, chunk_numbers=chunk)
    lowres_store = imgstore.new_for_filename(os.path.join(experiment_folder, "lowres"), chunk_numbers=chunk)
    return store, lowres_store


def compute_step(framerate, ms_res=10):
    step = int((1000/framerate)/ms_res)*ms_res # in units of 10 ms
    return step

def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    session_name = f"session_{str(args.chunk).zfill(6)}"
    session_folder = os.path.join(args.input, session_name)
    experiment_name = os.path.basename(args.input.rstrip("/"))


    validation_pickle = os.path.join(args.output, experiment_name + ".pkl")
    os.makedirs(args.output, exist_ok=True)
    validation_args = Namespace(input=session_folder, output=validation_pickle)
    validation = idvalidator.bin.validator.single_validator(args=validation_args)

    missing=[0 if non_id in [None, True] else len(non_id) for non_id in validation[session_name].identified]
    problem_frames = np.where(np.diff([0] + missing) > 0)[0]

    time_window_length = 5 # seconds

    store, lowres_store = load_store(args.input, args.chunk)

    fps = int(round(store._metadata["framerate"]))

    # windows = [(frame_number - 1 * fps):(frame_number + (time_window_length - 1)*fps)]
    start_frames = problem_frames - 1 * fps # start one second in the past

    start_frames = np.stack([[0,] * len(start_frames), start_frames], axis=1)
    start_frames = start_frames.max(axis=1)
    start_str = ' '.join([str(e) for e in start_frames])
    logger.info(f"Episodes start -> {start_str}")

    step = compute_step(lowres_store._metadata["framerate"], 10)

    list_of_blobs, colors = load_data(session_folder)

    for frame_in_chunk in start_frames:
        make_episode(
            list_of_blobs.blobs_in_video, store, args.chunk, frame_in_chunk, args.output,
            time_window_length=time_window_length, step=step,
            colors=colors
        )


def make_episode(blobs_in_video, store, chunk, frame_in_chunk, output_folder, time_window_length=5000, step=50, colors=None):

    (frame, (framenumber_start, timestamp_start)) = store.get_image(store._index.get_chunk_metadata(chunk)["frame_number"][frame_in_chunk])
    os.makedirs(output_folder, exist_ok=True)
    timestamps = list(range(int(timestamp_start), int(timestamp_start + time_window_length), step))

    print("*****")
    print(timestamps)
    print("*****")

    kwargs={
        "experiment_folder": os.path.dirname(store.full_path),
        "output_folder": output_folder,
        "blobs_in_video": blobs_in_video,
        "colors": colors,
        "chunk": chunk
    }

    _ = joblib.Parallel(n_jobs=-2)(joblib.delayed(check_feed)(timestamp=timestamp, **kwargs) for timestamp in timestamps)


    experiment_name = os.path.basename(os.path.dirname(store.full_path).rstrip("/"))
    session_name = f"session_{str(chunk).zfill(6)}"

    cmd = f"cat {output_folder}/*.png | ffmpeg -y -f image2pipe -i - -framerate 1 -c:v libx264 {experiment_name}_{session_name}_{str(timestamps[0]).zfill(10)}.mp4"

    p = subprocess.Popen(cmd.split(" "))
    status = p.communicate()
    return status


def check_feed(blobs_in_video, timestamp, output_folder, colors=None, stores=None, experiment_folder=None, chunk=None):

    assert not (stores is None and experiment_folder is None)

    dest = os.path.join(output_folder, f"{str(int(timestamp)).zfill(10)}.png")

    #if os.path.exists(dest):
    #    return None

    if stores is None:
        if chunk is not None:
            chunk_numbers = [chunk]
        logger.warning(experiment_folder)
        store = imgstore.new_for_filename(experiment_folder, chunk_numbers=chunk_numbers)
        lowres_store = imgstore.new_for_filename(os.path.join(experiment_folder, "lowres"), chunk_numbers=chunk_numbers)

    else:
        store, lowres_store = stores

    chunk_first_frame = store._get_chunk_metadata(chunk)["frame_number"][0]


    (frame, (fn, _)) = store._get_image_by_time(timestamp)
    (lowres_frame, (fn_lowres, _)) = lowres_store._get_image_by_time(timestamp)

    lowres_frame = cv2.resize(lowres_frame, frame.shape[::-1], cv2.INTER_AREA)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    lowres_frame = cv2.cvtColor(lowres_frame, cv2.COLOR_GRAY2BGR)
    blobs = blobs_in_video[fn-chunk_first_frame]

    for blob in blobs:
        blob.draw(frame, colors_lst=colors, is_selected=False)

    frames = [frame, lowres_frame]

    fig = []

    for i in range(2):
        fig.append(frames[i][:,:,[2,1,0]])

    fig = np.hstack(fig)
    cv2.imwrite(dest, fig)
    return fig

if __name__ == "__main__":
    main()
