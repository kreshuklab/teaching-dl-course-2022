# Example Solution: Project 21

This is an example solution for the myelin segmentation project.

## Visualisation & Preparation

- **data_preparation** Visualize the images and labels of the myelin dataset with [napari](https://napari.org/).
- **data_preparation** Create a train / validation split. Note: once we have more images this needs to be updated and we should also create a test split.

## Training

- **train_semantic_downscaled** Train a network for semantic segmentation of axon, myelin and tongue labels with the dice loss; the images are downscaled by a factor of two for increased field of view.

The training is implemented with [torch_em](https://github.com/constantinpape/torch-em).

## Prediction & Postprocessing

- **predict_and_segment** Run prediction on the test images and postprocess the results to obtain an instance and semantic segmentation.
- **view_results** Visually inspect the results.
- **export_for_modelzoo** Export in bioimage.io modelzoo format to make the model shareable.

The instance segmentation is done via connected components and watershed using the implementations from scikit-image.
For semantic segmentation  the max class is assigned inside of segmented obejcts.
