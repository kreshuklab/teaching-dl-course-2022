import imageio
import napari
import numpy as np
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_halo
from torch_em.util import get_trainer
from skimage.transform import rescale
from train_semantic_downscaled import myelin_label_transform, MyelinTransform

im = imageio.imread("/g/kreshuk/data/dl-course-2022/project21/prepared/test/raw/e72_33_0021.tif")
im_small = rescale(im, 0.5)

tile_shape = (1024, 1024)
halo = (64, 64)

trainer = get_trainer("./checkpoints/semantic-model-dice-downscaled", device="cpu")
model, device = trainer.model, trainer.device
model.eval()

prediction = predict_with_halo(im_small, model, ["cpu"], tile_shape, halo, preprocess=standardize)
# some weird artifacts...
# prediction_big = rescale(prediction, (1, 2, 2))
prediction_big = np.zeros((3,) + im.shape, dtype="float32")
for c in range(3):
    prediction_big[c] = rescale(prediction[c], 2)

v = napari.Viewer()
# v.add_image(im_small)
# v.add_image(prediction[0])
# v.add_image(prediction[1])
# v.add_image(prediction[2])
v.add_image(im)
v.add_image(prediction_big[0])
v.add_image(prediction_big[1])
v.add_image(prediction_big[2])
napari.run()
