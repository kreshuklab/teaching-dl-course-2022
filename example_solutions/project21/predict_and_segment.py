import os
import argparse
from glob import glob

import imageio
import h5py
import numpy as np
from torch_em.util import get_trainer
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo
from skimage.transform import rescale
from skimage.measure import label
from skimage.segmentation import watershed

# this export is needed to load the model again
from train_semantic_downscaled import myelin_label_transform, MyelinTransform


def predict(input_path, output_path, output_keys, model, device, downscale_input):
    image = imageio.imread(input_path)
    if downscale_input:
        print("Downscaling model inputs")
        image = rescale(image, 0.5)
    tile_shape = (1024, 1024)
    halo = (64, 64)
    prediction = predict_with_halo(image, model, [device], tile_shape, halo, preprocess=standardize)
    assert len(output_keys) == prediction.shape[0]
    with h5py.File(output_path, "a") as f:
        for pred, out_key in zip(prediction, output_keys):
            ds = f.require_dataset(out_key, shape=pred.shape, compression="gzip", dtype=pred.dtype)
            ds[:] = pred


def predict_semantic(input_folder, output_folder, model_path, downscale_input):
    output_keys = ["predictions/axon", "predictions/tongue", "predictions/myelin"]
    inputs = glob(os.path.join(input_folder, "*.tif"))
    trainer = get_trainer(model_path)
    model, device = trainer.model, trainer.device
    model.eval()
    print("Run semantic prediction")
    for input_path in inputs:
        fname = os.path.split(input_path)[1].replace(".tif", ".h5")
        output_path = os.path.join(output_folder, fname)
        predict(input_path, output_path, output_keys, model, device, downscale_input)


def segment_image(path, upscale):
    with h5py.File(path, "r") as f:
        axon = f["predictions/axon"][:]
        tongue = f["predictions/tongue"][:]
        myelin = f["predictions/myelin"][:]

    # run instance segmentation via watershed, where we use the
    # axon predictions as seeds, the combined predictions as mask
    # and the myelin segmentation as mask
    threshold = 0.5
    seeds = label(axon > threshold)
    mask = (axon + tongue + myelin) > threshold
    instances = watershed(myelin, seeds, mask=mask)

    # remove small objects
    min_axon_size = 1000
    ids, sizes = np.unique(instances, return_counts=True)
    remove_ids = ids[sizes < min_axon_size]
    instances[np.isin(instances, remove_ids)] = 0

    # assing the max semantic label inside the instance mask
    semantic_pred = np.concatenate([axon[None], tongue[None], myelin[None]], axis=0)
    semantic = np.argmax(semantic_pred, axis=0) + 1
    # use the instance segmentation as foreground mask
    semantic[instances == 0] = 0

    if upscale:
        instances = rescale(instances, 2, order=0, anti_aliasing=False, preserve_range=True).astype(instances.dtype)
        semantic = rescale(semantic, 2, order=0, anti_aliasing=False, preserve_range=True).astype(semantic.dtype)

    with h5py.File(path, "a") as f:
        ds = f.require_dataset(
            "segmentation/instances", shape=instances.shape, compression="gzip", dtype=instances.dtype
        )
        ds[:] = instances
        ds = f.require_dataset(
            "segmentation/semantic", shape=semantic.shape, compression="gzip", dtype=semantic.dtype
        )
        ds[:] = semantic


def segment_images(folder, upscale):
    files = glob(os.path.join(folder, "*.h5"))
    print("Run instance segmentation")
    for path in files:
        segment_image(path, upscale)


# alternative post-processing approaches:
# maybe we don't need the semantic predictions here and can just use the myelin prediction as boundary channel
# and use the combined axon, myelin and tongue predictions as foreground channel
# this would however fail if there are two axons touching without myelin
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--model", default="./checkpoints/semantic-model-dice-downscaled")
    parser.add_argument("--downscale", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    predict_semantic(args.input_folder, args.output_folder, args.model, bool(args.downscale))
    segment_images(args.output_folder, bool(args.downscale))


if __name__ == "__main__":
    main()
