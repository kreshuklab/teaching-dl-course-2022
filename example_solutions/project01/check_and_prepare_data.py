import argparse
import os
from glob import glob

import imageio
import h5py
import napari
import numpy as np


def check_data(input_folder):
    raw_path = os.path.join(input_folder, "plex2.tif")
    vol = imageio.volread(raw_path)

    label_paths = glob(os.path.join(input_folder, "plex2_neuron_*.tif"))
    labels = np.zeros(vol.shape, dtype="int16")
    for label_id, lp in enumerate(label_paths, 1):
        mask = imageio.volread(lp) > 0
        assert mask.shape == labels.shape
        labels[mask] = label_id

    v = napari.Viewer()
    v.add_image(vol)
    v.add_labels(labels)
    napari.run()

    return vol, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    raw, labels = check_data(args.input)
    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")


if __name__ == "__main__":
    main()
