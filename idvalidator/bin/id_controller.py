import argparse
from argparse import Namespace
import os
import os.path
import subprocess
import logging
import glob
from tqdm import tqdm

import numpy as np
import cv2
import joblib
import imgstore
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.utils.py_utils import get_spaced_colors_util

import idvalidator.bin.validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_imgstore = logging.getLogger("imgstore")
logger_imgstore.setLevel(logging.WARNING)
logger_lob = logging.getLogger("__main__.list_of_blobs")
logger_lob.setLevel(logging.WARNING)


def pipe_subprocess(cmd):
    cmds = cmd.split("|")
    subprocesses = []

    for i in range(len(cmds)):

        if i == 0:
            stdin = None
        else:
            stdin = subprocesses[i - 1].stdout

        if i == len(cmds):
            stdout = None
        else:
            stdout = subprocess.PIPE

        cmd_i = [e for e in cmds[i].split(" ") if e != ""]
        for j, token in enumerate(cmd_i):
            if "*" in token:
                token = sorted(glob.glob(token))
                cmd_i[j] = token

        cmd_i_processed = []
        for token in cmd_i:
            if isinstance(token, str):
                cmd_i_processed.append(token)
            elif isinstance(token, list):
                cmd_i_processed.extend(token)

        ps_i = subprocess.Popen(cmd_i_processed, stdin=stdin, stdout=stdout)
        subprocesses.append(ps_i)

    out = subprocesses[-1].communicate()

    return out, subprocesses


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--experiment-folder", "--input", dest="input", type=str, required=True
    )
    ap.add_argument("--chunk", type=int, required=False, default=None)
    ap.add_argument(
        "--skip-frame-collection",
        dest="skip_frame_collection",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output folder with .csv and .mp4 files",
    )
    return ap


def load_data(session_folder):

    video_object = np.load(
        os.path.join(session_folder, "video_object.npy"), allow_pickle=True
    ).item()
    colors = get_spaced_colors_util(
        video_object.user_defined_parameters["number_of_animals"], black=True
    )

    blobs_file = os.path.join(
        session_folder, "preprocessing", "blobs_collection_no_gaps.npy"
    )
    list_of_blobs = ListOfBlobs.load(blobs_file)
    return list_of_blobs, colors


def load_store(experiment_folder, chunk=None):

    if chunk is not None:
        chunk = [chunk]

    store = imgstore.new_for_filename(experiment_folder, chunk_numbers=chunk)
    lowres_store = imgstore.new_for_filename(
        os.path.join(experiment_folder, "lowres"), chunk_numbers=chunk
    )
    return store, lowres_store


def compute_step(framerate, ms_res=10):
    step = int((1000 / framerate) / ms_res) * ms_res  # in units of 10 ms
    return step


def validate_session(input, output, chunk, skip_frame_collection=False):

    session_name = f"session_{str(chunk).zfill(6)}"
    session_folder = os.path.join(input, session_name)
    experiment_name = os.path.basename(input.rstrip("/"))
    output = os.path.join(output, session_name + "_validation")

    validation_pickle = os.path.join(output, experiment_name + ".pkl")
    os.makedirs(output, exist_ok=True)
    validation_args = Namespace(input=session_folder, output=validation_pickle)
    validation = idvalidator.bin.validator.single_validator(
        args=validation_args
    )

    missing = [
        0 if non_id in [None, True] else len(non_id)
        for non_id in validation[session_name].identified
    ]
    problem_frames = np.where(np.diff(missing) != 0)[0]

    if len(problem_frames) == 0:
        pass

    else:
        time_window_length = 5000  # seconds

        store, lowres_store = load_store(input, chunk=None)
        fps = int(round(store._metadata["framerate"]))

        # windows = [(frame_number - 1 * fps):(frame_number + (time_window_length - 1)*fps)]
        start_frames = problem_frames - 1 * fps  # start one second in the past

        start_frames = np.stack(
            [
                [
                    0,
                ]
                * len(start_frames),
                start_frames,
            ],
            axis=1,
        )
        start_frames = start_frames.max(axis=1)
        start_str = " ".join([str(e) for e in start_frames])
        logger.info(f"Chunk {session_name}. Episodes start -> {start_str}")

        step = compute_step(lowres_store._metadata["framerate"], 10)
        logger.info(f"Timestamp step is set to {step}")

        list_of_blobs, colors = load_data(session_folder)

        for frame_in_chunk in start_frames:
            make_episode(
                experiment_folder=input,
                blobs_in_video=list_of_blobs.blobs_in_video,
                chunk=chunk,
                frame_in_chunk=frame_in_chunk,
                output_folder=output,
                time_window_length=time_window_length,
                step=step,
                colors=colors,
                skip_frame_collection=skip_frame_collection,
            )

        logger.info(f"Chunk {session_name} DONE")
        return


def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    n_jobs = -2

    kwargs = {
        "input": args.input,
        "output": args.output,
        "skip_frame_collection": args.skip_frame_collection,
    }

    if args.chunk is None:
        store = imgstore.new_for_filename(args.input)
        chunks = list(store._index.chunks)
        # chunks = [3]
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(validate_session)(chunk=chunk, **kwargs)
            for chunk in chunks
        )
    else:
        validate_session(chunk=args.chunk, **kwargs)


def make_episode(
    experiment_folder,
    blobs_in_video,
    chunk,
    frame_in_chunk,
    output_folder,
    time_window_length=5000,
    step=50,
    colors=None,
    n_jobs=1,
    skip_frame_collection=False,  # if true, it skips the frame retrieval
):

    stores = load_store(experiment_folder, chunk=None)
    store, lowres_store = stores

    # precompute the timepoints that will be queried
    timestamp_start = store._index.get_chunk_metadata(chunk)["frame_time"][
        frame_in_chunk
    ]
    timestamps = list(
        range(
            int(timestamp_start),
            int(timestamp_start + time_window_length),
            step,
        )
    )

    # give a name to the episode and make its folder
    episode_name = str(timestamps[0])
    episode_folder = os.path.join(output_folder, episode_name)
    os.makedirs(episode_folder, exist_ok=True)

    kwargs = {
        "experiment_folder": os.path.dirname(store.full_path),
        "output_folder": episode_folder,
        "blobs_in_video": blobs_in_video,
        "colors": colors,
        "chunk": chunk,
    }

    if not skip_frame_collection:
        for i in tqdm(range(len(timestamps))):
            check_feed(timestamp=timestamps[i], stores=stores, **kwargs)

    experiment_name = os.path.basename(
        os.path.dirname(store.full_path).rstrip("/")
    )
    session_name = f"session_{str(chunk).zfill(6)}"

    cmd = f"cat {episode_folder}/*.png | ffmpeg -y -f image2pipe -i - -framerate 1 -c:v libx264 {episode_folder}/{experiment_name}_{session_name}_{episode_name.zfill(10)}.mp4"

    print("****")
    print("CMD:", cmd)
    print("****")

    out, subprocesses = pipe_subprocess(cmd)
    return out


def check_feed(
    blobs_in_video,
    timestamp,
    output_folder,
    stores,
    colors=None,
    experiment_folder=None,
    chunk=None,
):

    assert not (stores is None and experiment_folder is None)

    dest = os.path.join(output_folder, f"{str(int(timestamp)).zfill(10)}.png")

    if os.path.exists(dest):
        return None

    store, lowres_store = stores

    chunk_first_frame = store._get_chunk_metadata(chunk)["frame_number"][0]

    (frame, (fn, t_ms)) = store._get_image_by_time(timestamp)
    (lowres_frame, (fn_lowres, t_ms_lowres)) = lowres_store._get_image_by_time(
        timestamp
    )

    lowres_frame = cv2.resize(lowres_frame, frame.shape[::-1], cv2.INTER_AREA)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    lowres_frame = cv2.cvtColor(lowres_frame, cv2.COLOR_GRAY2BGR)
    blobs = blobs_in_video[fn - chunk_first_frame]

    for blob in blobs:
        blob.draw(frame, colors_lst=colors, is_selected=False)

    frames = [frame, lowres_frame]
    t_ms = [t_ms, t_ms_lowres]

    fig = []

    for i in range(2):
        img = frames[i]
        pos = (int(img.shape[1] * 0.1), int(img.shape[0] * 0.9))

        if len(img.shape) == 2:
            nchannels = 1
        else:
            nchannels = img.shape[2]

        color = (200,) * nchannels
        img = cv2.putText(
            img,
            f"Time: {t_ms[i]}",
            org=pos,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=3,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        fig.append(img[:, :, [2, 1, 0]])

    fig = np.hstack(fig)
    cv2.imwrite(dest, fig)
    return fig


if __name__ == "__main__":
    main()
