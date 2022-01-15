import argparse
import os
from concurrent import futures
from functools import partial
from glob import glob
from shutil import copyfile

import imageio
import h5py
import numpy as np
from skimage.transform import rescale
from tqdm import tqdm


def to_target_shape(data, target_shape):
    if data.shape == target_shape:
        return data
    for dim, (sh, tsh) in enumerate(zip(data.shape, target_shape)):
        if sh == tsh:  # shapes in this dim agree, do nothing
            continue
        elif sh > tsh:  # shape is bigger than the target shape, crop
            crop_dim = tuple(slice(0, tsh) if i == dim else slice(None) for i in range(data.ndim))
            data = data[crop_dim]
        else:  # shape is smaller than the target shape, pad
            padding = [(0, tsh - sh) if i == dim else (0, 0) for i in range(data.ndim)]
            data = np.pad(data, padding)
    assert data.shape == target_shape
    return data


def prepare_volume(im_path, image_folder, label_folder, scale_factor, target_shape):
    label_path = im_path.replace("images", "labels").replace("Rec.tif", "Rec_labels.tif")
    vol = imageio.volread(im_path)
    labels = imageio.volread(label_path)
    # to skip the volume with a different shape
    if vol.shape != labels.shape:
        return
    vol = rescale(vol, scale=scale_factor, preserve_range=True)
    labels = rescale(labels, scale_factor, order=0, preserve_range=True, anti_aliasing=False).astype(labels.dtype)
    # pad / crop so that we have the target shape
    vol = to_target_shape(vol, target_shape)
    labels = to_target_shape(labels, target_shape)
    # write the downscaled data
    im_name = os.path.split(im_path)[1]
    imageio.volwrite(os.path.join(image_folder, im_name), vol)
    lab_name = os.path.split(label_path)[1]
    imageio.volwrite(os.path.join(label_folder, lab_name), labels)


def rescale_data(input_folder, n_workers):
    output_folder = os.path.join(input_folder, "rescaled")
    image_folder = os.path.join(output_folder, "images")
    label_folder = os.path.join(output_folder, "labels")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "images", "*.tif"))

    scale_factor = (1. / 8, 1. / 4, 1. / 4)
    target_shape = (128,) * 3

    if n_workers > 1:  # parallelized version to speed this up a bit
        with futures.ProcessPoolExecutor(n_workers) as pool:
            func = partial(prepare_volume, image_folder=image_folder,
                           label_folder=label_folder, scale_factor=scale_factor,
                           target_shape=target_shape)
            list(tqdm(pool.map(func, image_paths), total=len(image_paths)))
    else:  # normal version
        for im_path in tqdm(image_paths):
            prepare_volume(im_path, image_folder, label_folder, scale_factor, target_shape)


def make_splits(input_folder):
    output_folder = os.path.join(input_folder, "prepared")
    input_folder = os.path.join(input_folder, "rescaled")
    assert os.path.exists(input_folder)
    image_paths = glob(os.path.join(input_folder, "images", "*.tif"))

    # we split: 70% train, 10% val, 20% test
    n_images = len(image_paths)
    last_train_id = int(0.7 * n_images)
    last_val_id = int(0.8 * n_images)

    for im_id, im_path in enumerate(image_paths):
        label_path = im_path.replace("images", "labels").replace("Rec.tif", "Rec_labels.tif")
        assert os.path.exists(label_path)

        if im_id <= last_train_id:
            split = "train"
        elif im_id <= last_val_id:
            split = "val"
        else:
            split = "test"
        os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

        copyfile(im_path, os.path.join(output_folder, split, "images", os.path.split(im_path)[1]))
        copyfile(label_path, os.path.join(output_folder, split, "labels", os.path.split(label_path)[1]))


def make_splits_h5(input_folder):
    output_folder = os.path.join(input_folder, "prepared")
    input_folder = os.path.join(input_folder, "rescaled")
    assert os.path.exists(input_folder)
    image_paths = glob(os.path.join(input_folder, "images", "*.tif"))

    # we split: 70% train, 10% val, 20% test
    n_images = len(image_paths)
    last_train_id = int(0.7 * n_images)
    last_val_id = int(0.8 * n_images)

    for im_id, im_path in enumerate(image_paths):
        label_path = im_path.replace("images", "labels").replace("Rec.tif", "Rec_labels.tif")
        assert os.path.exists(label_path)

        if im_id <= last_train_id:
            split = "train"
        elif im_id <= last_val_id:
            split = "val"
        else:
            split = "test"
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)
        fname = os.path.split(im_path)[1].replace(".tif", ".h5")
        out_path = os.path.join(output_folder, split, fname)
        im = imageio.volread(im_path)
        labels = imageio.volread(label_path)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("image", data=im, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-n", "--n_workers", type=int, default=0)
    parser.add_argument("--to_h5", default=0, type=int)
    args = parser.parse_args()
    # rescale_data(args.input, args.n_workers)
    if bool(args.to_h5):
        make_splits_h5(args.input)
    else:
        make_splits(args.input)


if __name__ == "__main__":
    main()
