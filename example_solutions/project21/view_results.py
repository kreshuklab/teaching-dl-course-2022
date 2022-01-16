import argparse
import os
from glob import glob

import imageio
import h5py
import napari


def view_result(in_path, out_path, show_pred, show_seg):
    image = imageio.imread(in_path)
    with h5py.File(out_path, "r") as f:
        if show_pred:
            predictions = {f"{name}_prediction": ds[:] for name, ds in f["predictions"].items()}
        else:
            predictions = {}
        if show_seg:
            segmentations = {f"{name}_segmentation": ds[:] for name, ds in f["segmentation"].items()}
        else:
            segmentations = {}
    v = napari.Viewer()
    v.add_image(image)
    for name, pred in predictions.items():
        v.add_image(pred, name=name)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    napari.run()


def view_results(input_folder, output_folder, show_pred, show_seg):
    input_paths = glob(os.path.join(input_folder, "*.tif"))
    input_paths.sort()
    output_paths = glob(os.path.join(output_folder, "*.h5"))
    output_paths.sort()
    assert len(input_paths) == len(output_paths)
    for in_path, out_path in zip(input_paths, output_paths):
        view_result(in_path, out_path, show_pred, show_seg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--predictions", type=int, default=1)
    parser.add_argument("--segmentations", type=int, default=1)
    args = parser.parse_args()
    view_results(args.input_folder, args.output_folder, bool(args.predictions), bool(args.segmentations))


if __name__ == "__main__":
    main()
