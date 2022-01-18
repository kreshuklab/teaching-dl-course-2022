import os
import argparse
from glob import glob

import imageio
import h5py
import elf.segmentation as eseg
import numpy as np
from torch_em.util import get_trainer
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo
from skimage.measure import regionprops, label
from skimage.segmentation import relabel_sequential, watershed
# from skimage.morphology import remove_small_holes

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


def sharpen_predictions(predictions, percentile=95, clip=True):
    """ Sharpen the predictions by dividing with a high percentile.
    """
    predictions /= np.percentile(predictions, percentile)
    if clip:
        predictions = np.clip(predictions, 0.0, 1.0)
    return predictions


def run_multicut(path):
    with h5py.File(path, "r") as f:
        foreground = f["predictions/foreground"][:]
        boundaries = f["predictions/boundaries"][:]
    # percentile based prediction normalization
    foreground = sharpen_predictions(foreground)
    boundaries = sharpen_predictions(boundaries)

    # make superpixels via watershed
    ws, max_id = eseg.distance_transform_watershed(boundaries, threshold=0.25, sigma_seeds=2.0)

    # compute graph represenation and edge features
    rag = eseg.compute_rag(ws, max_id+1, n_threads=4)
    costs = eseg.compute_boundary_mean_and_length(rag, boundaries, n_threads=4)[:, 0]
    costs = eseg.compute_edge_costs(costs)

    # run multicut segmentation
    node_labels = eseg.multicut.multicut_decomposition(rag, costs, n_threads=4)
    if 0 in node_labels:  # avoid having a segment with id 0
        node_labels += 1
    segmentation = eseg.project_node_labels_to_pixels(rag, node_labels, n_threads=4)

    # postprocessing to filter out background
    fg_threshold = 0.5
    props = regionprops(segmentation, foreground)
    bg_ids = [prop.label for prop in props if prop.mean_intensity < fg_threshold]
    postprocessed = segmentation.copy()
    postprocessed[np.isin(segmentation, bg_ids)] = 0
    postprocessed = relabel_sequential(postprocessed)[0]

    with h5py.File(path, "a") as f:
        ds = f.require_dataset(
            "segmentation/multicut", shape=segmentation.shape, compression="gzip", dtype=segmentation.dtype
        )
        ds[:] = segmentation
        ds = f.require_dataset(
            "segmentation/postprocessed", shape=postprocessed.shape, compression="gzip", dtype=postprocessed.dtype
        )
        ds[:] = postprocessed


def run_connected_components(path):
    with h5py.File(path, "r") as f:
        foreground = f["predictions/foreground"][:]
        boundaries = f["predictions/boundaries"][:]
    # percentile based prediction normalization
    foreground = sharpen_predictions(foreground)
    boundaries = sharpen_predictions(boundaries)

    threshold = 0.5
    seeds = np.clip(foreground - boundaries, 0, 1)
    seeds = label(seeds > threshold)

    mask = foreground + boundaries > threshold
    segmentation = watershed(boundaries, seeds, mask=mask)
    # segmentation = remove_small_holes(segmentation, area_threshold=200)

    with h5py.File(path, "a") as f:
        ds = f.require_dataset(
            "segmentation/watershed", shape=segmentation.shape, compression="gzip", dtype=segmentation.dtype
        )
        ds[:] = segmentation


def segment_instances(folder, use_multicut=False):
    files = glob(os.path.join(folder, "*.h5"))
    print("Run instance segmentation")
    for path in files:
        if use_multicut:
            run_multicut(path)
            return "segmentation/watershed"
        else:
            run_connected_components(path)
            return "segmentation/postprocessed"


def semantic_segmentation(path, instance_seg_key):
    # load the isntance segmentation and the semantic predictions
    with h5py.File(path, "r") as f:
        instances = f[instance_seg_key][:]
        pred_keys = ["axon", "tongue", "myelin"]
        semantic_pred = []
        for key in pred_keys:
            semantic_pred.append(f[f"predictions/{key}"][:][None])
        semantic_pred = np.concatenate(semantic_pred, axis=0)

    # assing the max semantic label inside the instance mask
    semantic_seg = np.argmax(semantic_pred, axis=0) + 1
    # use the instance segmentation as foreground mask
    semantic_seg[instances == 0] = 0

    with h5py.File(path, "a") as f:
        ds = f.require_dataset(
            "segmentation/semantic", shape=semantic_seg.shape, dtype=semantic_seg.dtype, compression="gzip"
        )
        ds[:] = semantic_seg


def segment_semantic(folder, instance_seg_key):
    files = glob(os.path.join(folder, "*.h5"))
    print("Run semantic segmentation")
    for path in files:
        semantic_segmentation(path, instance_seg_key)


# alternative post-processing approaches:
# maybe we don't need the semantic predictions here and can just use the myelin prediction as boundary channel
# and use the combined axon, myelin and tongue predictions as foreground channel
# this would however fail if there are two axons touching without myelin
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("--boundary_model", default="./checkpoints/boundary-model")
    parser.add_argument("--semantic_model", default="./checkpoints/semantic-model-dice")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    # predict_semantic(args.input_folder, args.output_folder, args.semantic_model)
    # predict_boundaries(args.input_folder, args.output_folder, args.boundary_model)
    instance_seg_key = segment_instances(args.output_folder)
    segment_semantic(args.output_folder, instance_seg_key)


if __name__ == "__main__":
    main()
