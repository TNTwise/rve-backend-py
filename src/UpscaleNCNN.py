import cv2
import numpy as np
import os
import ncnn


class UpscaleNCNNImage:
    def __init__(
        self,
        modelPath: str = "models",
        modelName: str = "",
        vulkan: bool = True,
        tile_pad=10,
    ):
        self.modelPath = modelPath
        self.modelName = modelName
        self.vulkan = vulkan
        self.fullModelPath = os.path.join(self.modelPath, self.modelName)

    def renderImage(self, fullImagePath) -> np.array:
        net = ncnn.Net()

        # Use vulkan compute
        net.opt.use_vulkan_compute = self.vulkan

        # Load model param and bin
        net.load_param(self.fullModelPath + ".param")
        net.load_model(self.fullModelPath + ".bin")

        ex = net.create_extractor()

        # Load image using opencv
        img = cv2.imread(fullImagePath)

        # Convert image to ncnn Mat
        mat_in = ncnn.Mat.from_pixels(
            img, ncnn.Mat.PixelType.PIXEL_BGR, img.shape[1], img.shape[0]
        )

        # Normalize image (required)
        # Note that passing in a normalized numpy array will not work.
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        # Try/except block to catch out-of-memory error
        try:
            # Make sure the input and output names match the param file
            ex.input("data", mat_in)
            ret, mat_out = ex.extract("output")
            out = np.array(mat_out)

            # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
            output = out.transpose(1, 2, 0) * 255

            return output
        except:
            ncnn.destroy_gpu_instance()

    def saveImage(self, imageNPArray, outputPath: str):
        cv2.imwrite(filename=outputPath, img=imageNPArray)
