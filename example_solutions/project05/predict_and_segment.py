import os
import argparse
from glob import glob

import h5py
import numpy as np
import torch
from torch_em.util import get_trainer
from torch_em.transform.raw import standardize
from tqdm import tqdm

# this export is needed to load the model again
from train_semantic_dice import microct_label_transform


def predict(input_path, output_path, output_keys, model, device):
    with h5py.File(input_path, "r") as f:
        raw = f["image"][:]
    raw = standardize(raw)
    with torch.no_grad():
        raw = torch.from_numpy(raw[None, None]).to(device)
        prediction = model(raw).cpu().numpy()[0]
    assert len(output_keys) == prediction.shape[0]
    with h5py.File(output_path, "a") as f:
        for pred, out_key in zip(prediction, output_keys):
            ds = f.require_dataset(out_key, shape=pred.shape, compression="gzip", dtype=pred.dtype)
            ds[:] = pred


def predict_semantic(input_folder, output_folder, model_path):
    # TODO I am not sure about the order of semantic labels here
    output_keys = ["predictions/stomach", "predictions/gut", "predictions/left_ovaries", "predictions/right_ovaries"]
    inputs = glob(os.path.join(input_folder, "*.h5"))
    trainer = get_trainer(model_path)
    model, device = trainer.model, trainer.device
    model.eval()
    for input_path in tqdm(inputs, desc="Run semantic prediction"):
        fname = os.path.split(input_path)[1]
        output_path = os.path.join(output_folder, fname)
        predict(input_path, output_path, output_keys, model, device)


def segment(path):
    # again, not quite sure if this is the actual label order
    class_names = ["stomach", "gut", "left_ovaries", "right_ovaries"]
    with h5py.File(path, "r") as f:
        g = f["predictions"]
        prediction = np.concatenate(
            [g[name][:][None] for name in class_names], axis=0
        )
    # add a background channel, which is 1. - the sum of predictions
    bg_channel = np.clip(1. - np.sum(prediction, axis=0), 0, 1)
    prediction = np.concatenate([bg_channel[None], prediction], axis=0)
    semantic_seg = np.argmax(prediction, axis=0)
    with h5py.File(path, "a") as f:
        ds = f.require_dataset(
            "segmentation/semantic", shape=semantic_seg.shape, dtype=semantic_seg.dtype, compression="gzip"
        )
        ds[:] = semantic_seg


def segment_semantic(folder):
    files = glob(os.path.join(folder, "*.h5"))
    for path in tqdm(files, desc="Run semantic segmentation"):
        segment(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--semantic_model", default="./checkpoints/semantic-model-dice")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    predict_semantic(args.input_folder, args.output_folder, args.semantic_model)
    segment_semantic(args.output_folder)


if __name__ == "__main__":
    main()
