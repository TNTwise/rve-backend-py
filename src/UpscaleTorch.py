import os
import math
import numpy as np
import cv2
import torch as torch
from spandrel import ModelLoader, ImageModelDescriptor

# tiling code permidently borrowed from https://github.com/chaiNNer-org/spandrel/issues/113#issuecomment-1907209731


class UpscalePytorch:
    def __init__(
        self,
        modelPath: str,
        device="cuda",
        tile_pad: int = 10,
        precision: str = "float16",
        width: int = 1920,
        height: int = 1080,
    ):

        self.tile_pad = tile_pad
        self.dtype = self.handlePrecision(precision)
        self.device = device
        self.model = self.loadModel(
            modelPath=modelPath, device=device, dtype=self.handlePrecision(precision)
        )
        self.scale = self.model.scale
        self.width = width
        self.height = height

    def handlePrecision(self, precision):
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    def loadModel(
        self, modelPath: str, dtype: torch.dtype = torch.float32, device: str = "cuda"
    ) -> ImageModelDescriptor:
        model = ModelLoader().load_from_file(modelPath)
        assert isinstance(model, ImageModelDescriptor)
        # get model attributes

        model.to(device=device, dtype=dtype)
        return model

    def bytesToFrame(self, frame):
        return (
            torch.frombuffer(frame, dtype=torch.uint8)
            .reshape(self.height, self.width, 3)
            .to(self.device, non_blocking=True, dtype=self.dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .mul_(1 / 255)
        )

    def loadImage(self, imagePath: str) -> torch.Tensor:
        image = cv2.imread(imagePath)
        imageTensor = (
            torch.from_numpy(image)
            .to(device=self.device, dtype=self.dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .mul_(1 / 255)
        )

        return imageTensor.to(device=self.device, dtype=self.dtype)

    def tensorToNPArray(self, image: torch.Tensor) -> np.array:
        return image.squeeze(0).permute(1, 2, 0).float().mul(255).cpu().numpy()

    @torch.inference_mode()
    def renderImage(self, image: torch.Tensor) -> torch.Tensor:
        upscaledImage = self.model(image)
        return upscaledImage

    def renderToNPArray(self, image: torch.Tensor) -> torch.Tensor:
        return (
            self.model(image)
            .squeeze(0)
            .permute(1, 2, 0)
            .float()
            .mul(255)
            .byte()
            .contiguous()
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def renderImagesInDirectory(self, dir):
        pass

    def getScale(self):
        return self.scale

    def saveImage(self, image: np.array, fullOutputPathLocation):
        cv2.imwrite(fullOutputPathLocation, image)

    @torch.inference_mode()
    def renderTiledImage(
        self,
        image: torch.Tensor,
        tile_size: int = 32,
    ) -> torch.Tensor:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = image.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = image.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = image[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                with torch.no_grad():
                    output_tile = self.renderImage(input_tile)

                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        return output
