import numpy as np
import torch


# the image needs to be normalized before this function is called
# NOTE this is not using any overlap, which should be added in practice
def predict_with_tiling(model, image, tile, n_channels=3, device="cuda"):
    with torch.no_grad():
        assert image.ndim == 2
        n_tiles_x = int(np.ceil(image.shape[0] / float(tile[0])))
        n_tiles_y = int(np.ceil(image.shape[1] / float(tile[1])))
        prediction = np.zeros((n_channels,) + image.shape, dtype="float32")
        for tile_x in range(n_tiles_x):
            for tile_y in range(n_tiles_y):

                start_x = tile_x * tile[0]
                start_y = tile_y * tile[1]

                stop_x = start_x + tile[0]
                stop_y = start_y + tile[1]

                print("Predict tile", start_x, ":", stop_x, ";", start_y, ":", stop_y)
                input_ = image[start_x:stop_x, start_y:stop_y]
                # the last tiles might not fully fit, we need to pad in that case
                if input_.shape != tile:
                    to_pad = tuple((0, tsh - sh) for tsh, sh in zip(tile, input_.shape))
                    input_ = np.pad(input_, to_pad)
                    to_crop = tuple(slice(0, sh) for sh in input_.shape)
                else:
                    to_crop = None
                input_ = torch.from_numpy(input_[None, None]).to(device)

                pred = model(input_).cpu().numpy()[0]
                if to_crop is not None:
                    pred = pred[to_crop]
                prediction[:, start_x:stop_x, start_y:stop_y] = pred

    return prediction


# example usage: predict with tiles of size 2048 x 2048
# predict_with_tiling(model, image, tile=(2048, 2048))
