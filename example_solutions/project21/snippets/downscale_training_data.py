import os
from glob import glob
import imageio
from skimage.transform import rescale


def rescale_images(input_folder, output_folder):
    for split in ("train", "val"):
        images = glob(os.path.join(input_folder, split, "images", "*.tif"))
        out_split = os.path.join(output_folder, split, "images")
        os.makedirs(out_split, exist_ok=True)
        for im in images:
            data = imageio.imread(im)
            data = rescale(data, 0.5)
            fname = os.path.split(im)[1]
            out_path = os.path.join(out_split, fname)
            imageio.imwrite(out_path, data)


def rescale_labels(input_folder, output_folder):
    for split in ("train", "val"):
        images = glob(os.path.join(input_folder, split, "labels", "*.tif"))
        out_split = os.path.join(output_folder, split, "labels")
        os.makedirs(out_split, exist_ok=True)
        for im in images:
            data = imageio.imread(im)
            data = rescale(data, 0.5, order=0, anti_aliasing=False, preserve_range=True).astype(data.dtype)
            fname = os.path.split(im)[1]
            out_path = os.path.join(out_split, fname)
            imageio.imwrite(out_path, data)


if __name__ == "__main__":
    in_folder = "/g/kreshuk/data/dl-course-2022/project21/for-course/prepared_data_v1"
    out_folder = "/g/kreshuk/data/dl-course-2022/project21/for-course/prepared_data_v2"
    rescale_images(in_folder, out_folder)
    rescale_labels(in_folder, out_folder)
