from threading import Thread

from .UpscaleTorch import UpscalePytorch
from .UpscaleNCNN import UpscaleNCNN, getNCNNScale
from .FFmpeg import FFMpegRender
from .InterpolateNCNN import InterpolateRIFENCNN


class Render(FFMpegRender):
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    RenderOptions:
    interpolationMethod
    upscaleModel
    backend (pytorch,ncnn)
    device (cpu,cuda)
    precision (float16,float32)
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        interpolateFactor: int = 1,
        encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        backend="pytorch",
        interpolationMethod=None,
        upscaleModel=None,
        interpolateModel=None,
        device="cuda",
        precision="float16",
    ):
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        self.device = device
        self.precision = precision
        self.upscaleTimes = 1  # if no upscaling, it will default to 1
        self.interpolateFactor = interpolateFactor
        self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through
        self.frame0 = None

        self.getVideoProperties(inputFile)
        if upscaleModel:
            self.setupUpscale()
            self.renderThread = Thread(target=self.renderUpscale)
        if interpolateModel:
            self.setupInterpolate()
            self.renderThread = Thread(target=self.renderInterpolate)

        super().__init__(
            inputFile=inputFile,
            outputFile=outputFile,
            interpolateFactor=interpolateFactor,
            upscaleTimes=self.upscaleTimes,
            encoder=encoder,
            pixelFormat=pixelFormat,
            benchmark=benchmark,
            overwrite=overwrite,
            frameSetupFunction=self.setupRender,
            crf=crf,
        )
        self.ffmpegReadThread = Thread(target=self.readinVideoFrames)
        self.ffmpegWriteThread = Thread(target=self.writeOutVideoFrames)

        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()

    def renderUpscale(self):
        """
        self.setupRender, method that is mapped to the bytesToFrame in each respective backend
        self.upscale, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        """
        for i in range(self.totalFrames - 1):
            frame = self.readQueue.get()
            frame = self.upscale(frame)
            self.writeQueue.put(frame)
        self.writeQueue.put(None)
        print("Done with Upscale")
    def renderInterpolate(self):
        """
        self.setupRender, method that is mapped to the bytesToFrame in each respective backend
        self.interpoate, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        """
        self.frame0 = self.readQueue.get()
        for frameNum in range(self.totalFrames - self.interpolateFactor):
            frame1 = self.readQueue.get()
            for n in range(self.interpolateFactor):
                frame = self.interpolate(
                    self.frame0, frame1, 1 / (self.interpolateFactor - n)
                )
                self.writeQueue.put(frame)
            
            self.frame0 = frame1
        self.writeQueue.put(None)
        print("Done with interpolation")
    def setupUpscale(self):
        """
        This is called to setup an upscaling model if it exists.
        Maps the self.upscaleTimes to the actual scale of the model
        Maps the self.setupRender function that can setup frames to be rendered
        Maps the self.upscale the upscale function in the respective backend.
        """
        if self.backend == "pytorch":
            upscalePytorch = UpscalePytorch(
                self.upscaleModel,
                device=self.device,
                precision=self.precision,
                width=self.width,
                height=self.height,
            )
            self.upscaleTimes = upscalePytorch.getScale()
            self.setupRender = upscalePytorch.bytesToFrame
            self.upscale = upscalePytorch.renderToNPArray

        if self.backend == "ncnn":
            self.upscaleTimes = getNCNNScale(modelPath=self.upscaleModel)
            upscaleNCNN = UpscaleNCNN(
                modelPath=self.upscaleModel,
                num_threads=1,
                scale=self.upscaleTimes,
                gpuid=0,  # might have this be a setting
                width=self.width,
                height=self.height,
            )
            self.setupRender = self.returnFrame
            self.upscale = upscaleNCNN.Upscale

    def setupInterpolate(self):
        if self.backend == "ncnn":
            interpolateRifeNCNN = InterpolateRIFENCNN(
                interpolateModel=self.interpolateModel,
                width=self.width,
                height=self.height,
            )
            self.setupRender = interpolateRifeNCNN.bytesToByteArray
            self.interpolate = interpolateRifeNCNN.process
