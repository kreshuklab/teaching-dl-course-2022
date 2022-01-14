import argparse
import os
from glob import glob

import imageio
import napari


def visualize_data(root):
    image_paths = glob(os.path.join(root, "raw", "*.tif"))
    image_paths.sort()
    semantic_paths = glob(os.path.join(root, "semantic_labels", "*.tif"))
    semantic_paths.sort()
    instance_paths = glob(os.path.join(root, "instance_labels", "*.tif"))
    instance_paths.sort()

    for im, sem, inst in zip(image_paths, semantic_paths, instance_paths):
        image = imageio.imread(im)
        semantic_labels = imageio.imread(sem)
        assert semantic_labels.shape == image.shape
        instance_labels = imageio.imread(inst)
        assert instance_labels.shape == image.shape
        print("Shape:", image.shape)

        viewer = napari.Viewer()
        viewer.add_image(image)
        viewer.add_labels(semantic_labels)
        viewer.add_labels(instance_labels)
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root")
    args = parser.parse_args()
    visualize_data(args.root)


if __name__ == "__main__":
    main()
