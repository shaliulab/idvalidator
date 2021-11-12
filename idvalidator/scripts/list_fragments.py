import argparse
import os.path
import pythonvideoannotator
from pythonvideoannotator_module_idtrackerai.models.video.objects.idtrackerai_object_io import IdtrackeraiObjectIO


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--session-folder", "--input", type=str, dest="input", required=True, help="Path to single idtrackerai results folder")
    return ap


def list_fragments(project_path):
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

    list_fragments(args.input)

if __name__ == "__main__":
    main()
