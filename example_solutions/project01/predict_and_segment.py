import argparse
import os

import h5py
from ilastik.experimental.api import from_project_file
from torch_em.transform.raw import normalize
from xarray import DataArray

# TODO make this parameters to argparse
ILP = "/home/pape/Work/data/dl-course-2022/project01/boundary_and_defects.ilp"


def predict_ilastik(input_path, output_path):
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if "ilastik_prediction/boundaries" in f:
                print("Load ilsastik prediction")
                return f["ilastik_prediction/boundaries"][:], f["ilastik_prediction/defects"][:]

    print("Run ilastik prediction")
    print("Load raw data")
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]
    raw = normalize(raw)

    print("Run prediction...")
    ilp = from_project_file(ILP)
    raw = DataArray(raw, dims=("z", "y", "x"))
    pred = ilp.predict(raw)
    print(pred.shape)
    boundaries = pred[..., 1]
    defects = pred[..., 2]

    print("Save prediction")
    with h5py.File(output_path, "a") as f:
        f.create_dataset("ilasti_prediction/boundaries", data=boundaries, compression="gzip")
        f.create_dataset("ilasti_prediction/defects", data=defects, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    predict_ilastik(args.input, args.output)


if __name__ == "__main__":
    main()
