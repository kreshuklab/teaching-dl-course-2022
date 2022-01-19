import os
import imageio
import h5py
import napari
import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed


# TODO rerun with the new network and then evaluate the results and prepare some images


# ignore this, it's just here so that I can test the postprocessing ;)
def predict_bioimageio(image, model_path, output_path):
    import bioimageio.core
    from bioimageio.core.prediction import predict_with_tiling
    from xarray import DataArray

    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if "prediction/axon" in f:
                return f["prediction/axon"][:], f["prediction/tongue"][:], f["prediction/myelin"][:]
    model = bioimageio.core.load_resource_description(model_path)
    tiling = {"tile": {"x": 2048, "y": 2048}, "halo": {"x": 64, "y": 64}}
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model, devices=["cpu"]) as pp:
        input_ = DataArray(image[None, None], dims=("b", "c", "y", "x"))
        pred = predict_with_tiling(pp, input_, tiling=tiling, verbose=True)[0][0].values

    with h5py.File(output_path, "a") as f:
        f.create_dataset("prediction/axon", data=pred[0], compression="gzip")
        f.create_dataset("prediction/tongue", data=pred[1], compression="gzip")
        f.create_dataset("prediction/myelin", data=pred[2], compression="gzip")

    return pred[0], pred[1], pred[2]


# TODO run prediction with the torch network
def predict_torch(image, model_path):
    pass


# TODO fill holes in the instance segmentation or do some other morphological operations to improve the results
# segment instances with a watershed
def instance_segmentation(axon, tongue, myelin, output_path, threshold=0.5, image=None):

    # subtract the axon and tongue predictions form the axon predictions to find seeds
    # (here we use connected regions that very likely belong to the same axon as seeds)
    # seed_mask = np.clip(axon - tongue - myelin, 0.0,  1.0) > threshold
    seed_mask = np.clip(axon + tongue - myelin, 0.0,  1.0) > threshold

    # apply connected components (everything in the seeed mask that's touching will become an object)
    seeds = label(seed_mask, connectivity=1)
    # define the foreground mask (= combined axon, tongue and myelin probabilities)
    mask = (axon + tongue + myelin) > threshold
    # expand the seeds into the foreground mask using a watershed transform with myelin predictions
    # as heightmap
    instances = watershed(myelin, markers=seeds, mask=mask)

    with h5py.File(output_path, "a") as f:
        ds = f.require_dataset(
            "segmentation/instances", shape=instances.shape, dtype=instances.dtype, compression="gzip"
        )
        ds[:] = instances

    # for viewing the results during development
    # v = napari.Viewer()
    # v.add_image(image)
    # v.add_image(axon)
    # v.add_image(tongue)
    # v.add_image(myelin)
    # v.add_labels(seeds)
    # v.add_labels(mask, name="mask")
    # v.add_labels(instances)
    # napari.run()

    return instances


def semantic_segmentation(axon, tongue, myelin, instances, output_path):
    semantic_prediction = np.concatenate([axon[None], tongue[None], myelin[None]], axis=0)
    # assing the max semantic label inside the instance mask
    semantic = np.argmax(semantic_prediction, axis=0) + 1
    # use the instance segmentation as foreground mask
    semantic[instances == 0] = 0

    with h5py.File(output_path, "a") as f:
        ds = f.require_dataset(
            "segmentation/semantic", shape=semantic.shape, dtype=semantic.dtype, compression="gzip"
        )
        ds[:] = semantic

    return semantic


def main():
    # where your model is saved
    model_path = "../modelzoo/semantic-model/AxonSemanticSegmentation.zip"
    # the data to segment
    # data_path = "/home/pape/Work/data/dl-course-2022/project21/prepared/val/raw/E72_21_0003.tif"
    data_path = "/g/kreshuk/data/dl-course-2022/project21/prepared/test/raw/e72_33_0021.tif"
    # where to save the results
    output_path = "./results.h5"
    image = imageio.imread(data_path)
    axon, tongue, myelin = predict_bioimageio(image, model_path, output_path)
    instances = instance_segmentation(axon, tongue, myelin, output_path, image=image)
    semantic = semantic_segmentation(axon, tongue, myelin, instances, output_path)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    v.add_labels(semantic)
    napari.run()


if __name__ == "__main__":
    main()
