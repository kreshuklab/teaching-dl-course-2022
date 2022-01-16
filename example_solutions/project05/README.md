# Example Solution: Project 5

This is an example solution for the microCT drosophila semgentation project.

## Visualisation & Preparation

- **data_preparation** Visualize the images and labels of the microCT dataset with [napari](https://napari.org/).
- **data_preparation** Downsample the volumes so that a full volume fits duriong training. Create a train / validation / test split split.

## Training

- **train_semantic_dice** Train a network for semantic segmentation of stomach, gut and left/right ovaries with the dice loss.

The training is implemented with [torch_em](https://github.com/constantinpape/torch-em).
Alternative ideas:
- train the semantic network with cross entropy or try a single network per semantic class.
- merge the ovaries into one class (and then distinguish left/right in post-processing)
- Advanced: add global coordinate as additional input to the network to help with the ovary segmentation task.

## Prediction & Postprocessing

- **predict_and_segment** Run prediction on the test images and postprocess the results to obtain a semantic segmentation.
- **view_results** Visually inspect the results.
- **export_for_modelzoo** Export in bioimage.io modelzoo format to make the model shareable.

For semantic segmentation the max class is assigned.
