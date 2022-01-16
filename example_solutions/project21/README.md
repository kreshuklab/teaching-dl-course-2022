# Example Solution: Project 21

This is an example solution for the myelin segmentation project.

## Visualisation & Preparation

- **data_preparation** Visualize the images and labels of the myelin dataset with [napari](https://napari.org/).
- **data_preparation** Create a train / validation split. Note: once we have more images this needs to be updated and we should also create a test split.

Alternative ideas: maybe downsampling the data by a factor of 2 might help for the downstream task to improve the effective field of view.

## Training

- **train_semantic_dice** Train a network for semantic segmentation of axon, myelin and tongue labels with the dice loss.
- **train_boundaries** Train a network to predict foreground and boundary probabilities for the axons.

The training is implemented with [torch_em](https://github.com/constantinpape/torch-em).
Alternative ideas: train the semantic network with cross entropy or try a single network per semantic class.

## Prediction & Postprocessing

- **predict_and_segment** Run prediction on the test images and postprocess the results to obtain an instance and semantic segmentation.
- **view_results** Visually inspect the results.

The instance segmentation is done via multicut segmentation using the [elf](https://github.com/constantinpape/elf) segmentation library.
For semantic segmentation  the max class is assigned inside of segmented obejcts.

Alternative ideas: maybe the boundary predictions are not required and the instance segmentation could be done from the predictions of the semantic network as well, by using the myelin channel as boundary probabilities and the combined axon, myelin and foreground probabilities as foreground channel. But this approach would be inferior if axons touch without discerinible myelin in between.
Adanced: use a joint instance and semantic segmentation approach.
