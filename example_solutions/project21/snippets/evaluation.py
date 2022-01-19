# Here, we quantitatively evaluate the instance segmentation result on the test image
# - instance segmentation with object based intersection over union
# - semantic segmentation with dice for each class

import imageio
import h5py

# these are implementations of the evaluation methods from my own library, which is also
# installed in our BAND environment
from elf.evaluation import matching, dice_score


def evaluate_instances(result, instance_labels):
    with h5py.File(result, "r") as f:
        instances = f["segmentation/instances"][:]
    labels = imageio.imread(instance_labels)
    assert instances.shape == labels.shape
    # average precision of object matching using intersection over union
    score = matching(instances, labels, threshold=0.5)["precision"]
    print("Evaluation results of the instance segmentation:")
    print(score)


def evaluate_semantic(result, semantic_labels):
    with h5py.File(result, "r") as f:
        semantic = f["segmentation/semantic"][:]
    labels = imageio.imread(semantic_labels)
    print("Evaluation results of the semantic segmentation:")
    for class_id, name in enumerate(["axon", "tongue", "myelin"], 1):
        score = dice_score(semantic == class_id, labels == class_id)
        print(name, score)


def main():
    # where the results are saved
    result = "./results.h5"
    # where the instance labels for this image are saved
    instance_labels = "/g/kreshuk/data/dl-course-2022/project21/prepared/test/instance_labels/e72_33_0021_Instances.tif"
    # where the semantic labels are saved
    semantic_labels = "/g/kreshuk/data/dl-course-2022/project21/prepared/test/semantic_labels/Labeled_e72_33_0021.tif"

    evaluate_instances(result, instance_labels)
    evaluate_semantic(result, semantic_labels)


if __name__ == "__main__":
    main()
