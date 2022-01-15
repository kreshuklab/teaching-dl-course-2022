import os
from glob import glob
import torch_em
import numpy as np
from torch_em.model import UNet3d
from torch_em.util import parser_helper


# to one hot encoding
def microct_label_transform(labels):
    # compute the one hot encoding for the 4 target channels
    one_hot = np.zeros((4,) + labels.shape, dtype="float32")
    for chan, label_id in enumerate(range(1, 5)):
        one_hot[chan] = labels == label_id
    return one_hot


def get_loader(args, patch_shape, split):
    data_paths = glob(os.path.join(args.input, split, "*.h5"))
    assert len(data_paths) > 0
    n_samples = 100 if split == "train" else 4
    loader = torch_em.default_segmentation_loader(
        data_paths, "image", data_paths, "labels",
        label_transform=microct_label_transform,
        batch_size=args.batch_size, patch_shape=patch_shape,
        num_workers=8, shuffle=True, is_seg_dataset=True,
        n_samples=n_samples
    )
    return loader


def train_semantic_dice(args):
    # we have four output channels: gut, stomach, left + right ovaries
    n_out = 4
    # could also try a softmax here
    model = UNet3d(in_channels=1, out_channels=n_out, final_activation="Sigmoid")

    # shape of input patches used for training
    patch_shape = [128, 128, 128]

    train_loader = get_loader(args, patch_shape, "train")
    val_loader = get_loader(args, patch_shape, "val")

    loss = torch_em.loss.DiceLoss()
    name = "semantic-model-dice"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        device=args.device,
    )
    trainer.fit(args.n_iterations)


def check(args, n_images=2):
    from torch_em.util.debug import check_loader
    patch_shape = [128, 128, 128]

    print("Check train loader")
    loader = get_loader(args, patch_shape, "train")
    check_loader(loader, n_images)

    print("Check val loader")
    loader = get_loader(args, patch_shape, "val")
    check_loader(loader, n_images)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    if args.check:
        check(args)
    else:
        train_semantic_dice(args)
