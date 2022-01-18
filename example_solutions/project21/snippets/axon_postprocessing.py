import imageio
import napari
import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed


# ignore this, it's just here so that I can test the postprocessing ;)
def predict_bioimageio(image, model_path):
    import bioimageio.core
    from bioimageio.core.prediction import predict_with_tiling
    from xarray import DataArray
    model = bioimageio.core.load_resource_description(model_path)
    tiling = {"tile": {"x": 1024, "y": 1024}, "halo": {"x": 32, "y": 32}}
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = DataArray(image[None, None], dims=("b", "c", "y", "x"))
        pred = predict_with_tiling(pp, input_, tiling=tiling, verbose=True)[0][0].values
    return pred[0], pred[1], pred[2]


# TODO run prediction with the torch network
def predict_torch(image, model_path):
    pass


# TODO fill holes in the instance segmentation
# segment instances with a watershed
def instance_segmentation(axon, tongue, myelin, threshold=0.5):
    # subtract the axon and tongue predictions form the axon predictions to find seeds
    # (here we use connected regions that very likely belong to the same axon as seeds)
    seed_mask = np.clip(axon - tongue - myelin, 0.0,  1.0) > threshold
    # apply connected components (everything in the seeed mask that's touching will become an object)
    seeds = label(seed_mask, connectivity=1)
    # define the foreground mask (= combined axon, tongue and myelin probabilities)
    mask = (axon + tongue + myelin) > threshold
    # expand the seeds into the foreground mask using a watershed transform with myelin predictions
    # as heightmap
    instances = watershed(myelin, markers=seeds, mask=mask)
    return instances


def semantic_segmentation(axon, tongue, myelin, instances):
    semantic_prediction = np.concatenate([axon[None], tongue[None], myelin[None]], axis=0)
    # assing the max semantic label inside the instance mask
    semantic = np.argmax(semantic_prediction, axis=0) + 1
    # use the instance segmentation as foreground mask
    semantic[instances == 0] = 0
    return semantic


if __name__ == "__main__":
    # where your model is saved
    model_path = "../AxonSemanticSegmentation.zip"
    # the data to segment
    data_path = "/home/pape/Work/data/dl-course-2022/project21/prepared/val/raw/E72_21_0003.tif"
    image = imageio.imread(data_path)
    axon, tongue, myelin = predict_bioimageio(image, model_path)
    instances = instance_segmentation(axon, tongue, myelin)
    semantic = semantic_segmentation(axon, tongue, myelin, instances)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    v.add_labels(semantic)
    napari.run()
