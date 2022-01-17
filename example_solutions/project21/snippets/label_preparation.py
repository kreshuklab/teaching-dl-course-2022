import os
from glob import glob
import imageio


# this function loads the labels from tif,
# merges the label "4" (= ignore label) into the background ("0")
# and saves the label again
def process_labels(label_path):
    # load the labels into a numpy array
    labels = imageio.imread(label_path)
    # merge the label "4" into the background
    labels[labels == 4] = 0
    # write the new labes (this overwrites the image at label path)
    imageio.imwrite(label_path, labels)


# process all the label files
def process_all_labels(input_folder):
    # this will select all tif files in the folder "<input_folder>/train/labels"
    train_labels = glob(os.path.join(input_folder, "train", "labels", "*.tif"))
    for label_path in train_labels:
        process_labels(label_path)
    # this will select all tif files in the folder "<input_folder>/val/labels"
    val_labels = glob(os.path.join(input_folder, "val", "labels", "*.tif"))
    for label_path in val_labels:
        process_labels(label_path)


# adapt this path to where you have saved the data
input_folder = os.path.expanduser("~", "data")
process_all_labels(input_folder)
