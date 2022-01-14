import argparse
import os
from glob import glob

import imageio
import napari
from tqdm import tqdm


# visualize one volume
def visualize_data(input_path):
    assert os.path.exists(input_path)
    label_path = input_path.replace("images", "labels").replace("Rec.tif", "Rec_labels.tif")
    vol = imageio.volread(input_path)
    labels = imageio.volread(label_path)
    assert vol.shape == labels.shape
    viewer = napari.Viewer()
    viewer.add_image(vol)
    viewer.add_labels(labels)
    napari.run()


# check the shape for all volumes
def check_all_volumes(root):
    volumes = glob(os.path.join(root, "images", "*.tif"))
    volumes.sort()
    shapes = []
    for vol_path in tqdm(volumes):
        lab_path = vol_path.replace("images", "labels").replace("Rec.tif", "Rec_labels.tif")
        assert os.path.exists(lab_path), lab_path
        vol_shape = imageio.volread(vol_path).shape
        lab_shape = imageio.volread(lab_path).shape
        if vol_shape != lab_shape:
            print("Volume and label shape don't agree for", vol_path)
            print("Volume shape:", vol_shape)
            print("Label shape:", lab_shape)
        shapes.append(vol_shape)
    shapes = set(shapes)
    print("Unique shapes:")
    print(shapes)


# shapes don't match for volume 24, the rest is fine
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-c", "--check_data", type=int, default=0)
    args = parser.parse_args()
    if args.check_data:
        check_all_volumes(args.input)
    else:
        visualize_data(args.input)


if __name__ == "__main__":
    main()
