import os
import torch_em
from torch_em.model import UNet2d
from torch_em.util import parser_helper


def get_loader(args, patch_shape, split):
    raw_root = os.path.join(args.input, split, "raw")
    labels_root = os.path.join(args.input, split, "instance_labels")
    assert os.path.exists(raw_root), raw_root
    assert os.path.exists(labels_root)
    label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True, ndim=2)
    loader = torch_em.default_segmentation_loader(
        raw_root, "*.tif", labels_root, "*.tif",
        label_transform=label_transform,
        batch_size=args.batch_size, patch_shape=patch_shape,
        num_workers=8, shuffle=True, is_seg_dataset=False,
    )
    return loader


def train_semantic_dice(args):
    # we have two output channels: foreground and boundaries
    n_out = 2
    # could also try a softmax here
    model = UNet2d(in_channels=1, out_channels=n_out, final_activation="Sigmoid")

    # shape of input patches used for training
    patch_shape = [1024, 1024]

    train_loader = get_loader(args, patch_shape, "train")
    val_loader = get_loader(args, patch_shape, "val")

    loss = torch_em.loss.DiceLoss()

    name = "boundary-model"
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
    patch_shape = [1024, 1024]

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
