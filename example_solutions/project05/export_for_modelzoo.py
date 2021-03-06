import h5py
from torch_em.util import export_bioimageio_model, get_default_citations, export_parser_helper

# this export is needed to load the model again
from train_semantic_dice import microct_label_transform


def export_for_modelzoo(checkpoint, input_, output):
    with h5py.File(input_, "r") as f:
        input_data = f["image"][:]
    name = "MicroCTSemanticSegmentation"
    tags = ["unet", "micro-ct", "semantic-segmentation"]

    cite = get_default_citations(model="UNet3d")
    doc = "#MicroCT Segmentation Model"

    export_bioimageio_model(
        checkpoint, output, input_data,
        name=name,
        authors=[{"name": "Your Name"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/kreshuklab/teaching-dl-course-2022",
        cite=cite,
        input_optional_parameters=False,
    )


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_for_modelzoo(args.checkpoint, args.input, args.output)
