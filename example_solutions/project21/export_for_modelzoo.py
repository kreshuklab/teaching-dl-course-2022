import imageio
from skimage.transform import rescale
from torch_em.util import export_bioimageio_model, get_default_citations, export_parser_helper

# this export is needed to load the model again
from train_semantic_dice import myelin_label_transform
from train_semantic_downscaled import MyelinTransform


def export_for_modelzoo(checkpoint, input_, output, downscaled):
    if downscaled:
        print("Exporting downscaled model")
        input_data = imageio.imread(input_)[:1024, :1024]
        input_data = rescale(input_data, 0.5)
    else:
        print("Exporting normal model")
        input_data = imageio.imread(input_)[:512, :512]
    name = "AxonSemanticSegmentation"
    tags = ["unet", "neurons", "semantic-segmentation", "electron-microscopy"]

    cite = get_default_citations(model="UNet2d")
    doc = "#Axon Segmentation Model"

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
    parser.add_argument("-d", "--downscaled", type=int, default=1)
    args = parser.parse_args()
    export_for_modelzoo(args.checkpoint, args.input, args.output, bool(args.downscaled))
