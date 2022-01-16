import os
import argparse
from glob import glob

import imageio
import h5py
from torch_em.util import get_trainer
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo

# this export is needed to load the model again
from train_semantic_dice import myelin_label_transform


def predict(input_path, output_path, output_keys, model, device):
    image = imageio.imread(input_path)
    tile_shape = (1024, 1024)
    halo = (64, 64)
    prediction = predict_with_halo(image, model, [device], tile_shape, halo,
                                   preprocess=standardize)
    assert len(output_keys) == prediction.shape[0]
    with h5py.File(output_path, "a") as f:
        for pred, out_key in zip(prediction, output_keys):
            ds = f.require_dataset(out_key, shape=pred.shape, compression="gzip", dtype=pred.dtype)
            ds[:] = pred


def predict_semantic(input_folder, output_folder, model_path):
    output_keys = ["predictions/axon", "predictions/tongue", "predictions/myelin"]
    inputs = glob(os.path.join(input_folder, "*.tif"))
    trainer = get_trainer(model_path)
    model, device = trainer.model, trainer.device
    model.eval()
    print("Run semantic prediction")
    for input_path in inputs:
        fname = os.path.split(input_path)[1].replace(".tif", ".h5")
        output_path = os.path.join(output_folder, fname)
        predict(input_path, output_path, output_keys, model, device)


def predict_boundaries(input_folder, output_folder, model_path):
    output_keys = ["predictions/foreground", "predictions/boundaries"]
    inputs = glob(os.path.join(input_folder, "*.tif"))
    trainer = get_trainer(model_path)
    model, device = trainer.model, trainer.device
    model.eval()
    print("Run boundary prediction")
    for input_path in inputs:
        fname = os.path.split(input_path)[1].replace(".tif", ".h5")
        output_path = os.path.join(output_folder, fname)
        predict(input_path, output_path, output_keys, model, device)


# TODO
def segment(path):
    pass
    # with h5py.File(path, "a") as f:
    #     pass


def segment_instances(folder):
    files = glob(os.path.join(folder, "*.h5"))
    print("Run instance segmentation")
    for path in files:
        segment(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--boundary_model", default="./checkpoints/boundary-model")
    parser.add_argument("--semantic_model", default="./checkpoints/semantic-model-dice")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    predict_semantic(args.input_folder, args.output_folder, args.semantic_model)
    predict_boundaries(args.input_folder, args.output_folder, args.boundary_model)
    segment_instances(args.output_folder)


if __name__ == "__main__":
    main()
