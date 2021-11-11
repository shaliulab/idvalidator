import argparse
import os.path
import pythonvideoannotator
from pythonvideoannotator_module_idtrackerai.models.video.objects.idtrackerai_object_io import IdtrackeraiObjectIO


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-folder", "--input", type=str, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    return ap


def list_fragments(experiment_folder, chunk):
    project_path = os.path.join(experiment_folder, f"session_{str(chunk).zfill(6)}")
    idtrackeraiobjectio = IdtrackeraiObjectIO()
    idtrackeraiobjectio.load_from_idtrackerai(project_path)
    rjust=20
    header_list = ["id", "final_identities", "length", "start_end", "identifier"]
    header = [str(d).rjust(rjust) for d in header_list]
    header = " | ".join(header)
    print(header)
    print("-" * rjust*len(header_list) + "-"*(len(header_list)-1))
    
    for i, f in enumerate(idtrackeraiobjectio.list_of_framents.fragments):
        data = [i, f.final_identities, f.centroids.shape[0], f.start_end, f.identifier]
        data = [str(d).rjust(rjust) for d in data]
        data = " | ".join(data)
        print(data)





def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    list_fragments(args.experiment_folder, args.chunk)

if __name__ == "__main__":
    main()
