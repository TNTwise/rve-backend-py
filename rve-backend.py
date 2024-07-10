import torch
import argparse
import os

from src.RenderVideo import Render


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        self.checkArguments()
        self.setDType()
        ffmpegSettings = Render(
            inputFile=self.args.input,
            outputFile=self.args.output,
            interpolateTimes=1,
            upscaleModel=self.args.upscaleModel,
            device="cuda",
            backend=self.args.backend,
            precision="float16" if self.args.half else "float32",
            overwrite=self.args.overwrite,
            crf=self.args.crf
        )

    def setDType(self):
        if self.args.half:
            self.dtype = torch.half
        elif self.args.bfloat16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

    def returnDevice(self):
        if not self.args.cpu:
            return "cuda" if torch.cuda.is_available() else "cpu"

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Upscale any image, with most torch models, using spandrel."
        )

        parser.add_argument(
            "-i",
            "--input",
            default=None,
            help="input video path",
            required=True,
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="output video path or PIPE",
            required=True,
            type=str,
        )
        parser.add_argument(
            "-t",
            "--tilesize",
            help="tile size (default=0)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-l",
            "--overlap",
            help="overlap size on tiled rendering (default=10)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-b",
            "--backend",
            help="backend used to upscale image. (pytorch/ncnn, default=pytorch)",
            default="pytorch",
            type=str,
        )
        parser.add_argument(
            "--upscaleModel",
            help="Direct path to upscaling model, will automatically upscale if model is valid.",
            type=str,
        )
        parser.add_argument(
            "--interpolateModel",
            help="Direct path to interpolation model, will automatically upscale if model is valid.",
            type=str,
        )
        parser.add_argument(
            "-c",
            "--cpu",
            help="use only CPU for upscaling, instead of cuda. default=auto",
            action="store_true",
        )
        parser.add_argument(
            "-f",
            "--format",
            help="output image format (jpg/png/webp, auto=same as input, default=auto)",
        )
        parser.add_argument(
            "--half",
            help="half precision, only works with NVIDIA RTX 20 series and above.",
            action="store_true",
        )
        parser.add_argument(
            "--bfloat16",
            help="like half precision, but more intesive. This can be used with a wider range of models than half.",
            action="store_true",
        )

        parser.add_argument(
            "-e",
            "--export",
            help="Export PyTorch models to ONNX and NCNN. Options: (onnx/ncnn)",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--overwrite",
            help="Overwrite output video if it already exists.",
            action="store_true",
        )
        parser.add_argument(
            "--crf",
            help="Constant rate factor for videos, lower setting means higher quality.",
            default='18'
        )
        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        if os.path.isfile(self.args.output) and not self.args.overwrite:
            raise os.error("Output file already exists!")


if __name__ == "__main__":
    HandleApplication()
