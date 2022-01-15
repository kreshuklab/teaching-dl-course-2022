import argparse
import os
from glob import glob
from shutil import copyfile


def create_train_val_split(root, out_folder):
    image_paths = glob(os.path.join(root, "raw", "*.tif"))
    image_paths.sort()
    semantic_paths = glob(os.path.join(root, "semantic_labels", "*.tif"))
    semantic_paths.sort()
    instance_paths = glob(os.path.join(root, "instance_labels", "*.tif"))
    instance_paths.sort()
    assert len(image_paths) == len(semantic_paths) == len(instance_paths)

    for ii, (im, sem, inst) in enumerate(zip(image_paths, semantic_paths, instance_paths)):
        # we use the last image for validation and the others for train
        split = "train" if ii < len(image_paths) - 1 else "val"
        im_out = os.path.join(out_folder, split, "raw")
        sem_out = os.path.join(out_folder, split, "semantic_labels")
        inst_out = os.path.join(out_folder, split, "instance_labels")
        os.makedirs(im_out, exist_ok=True)
        os.makedirs(sem_out, exist_ok=True)
        os.makedirs(inst_out, exist_ok=True)
        copyfile(im, os.path.join(im_out, os.path.split(im)[1]))
        copyfile(sem, os.path.join(sem_out, os.path.split(sem)[1]))
        copyfile(inst, os.path.join(inst_out, os.path.split(inst)[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root")
    parser.add_argument("-o", "--out")
    args = parser.parse_args()
    create_train_val_split(args.root, args.out)


if __name__ == "__main__":
    main()
