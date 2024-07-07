import cv2
import os
import subprocess
import queue
from .Util import currentDirectory
from .UpscaleTorch import UpscalePytorchImage, loadTorchModel
from threading import Thread

class FFMpegRender:
    def __init__(self,
                 inputFile: str,
                 outputFile: str,
                 interpolateTimes: int = 1,
                 upscaleTimes: int = 1,
                 encoder: str = "libx264",
                 pixelFormat: str = "yuv420p",
                 benchmark: bool = False
                 ):
        """
        Generates FFmpeg I/O commands to be used with VideoIO
        Options:
        inputFile: str, The path to the input file.
        outputFile: str, The path to the output file.
        interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
        upscaleTimes: int,
        encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
        pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)
        """
        self.inputFile = inputFile
        self.outputFile = outputFile
        
        # upsacletimes will be set to the scale of the loaded model with spandrel
        self.upscaleTimes = upscaleTimes
        self.interpolateTimes = interpolateTimes
        self.encoder = encoder
        self.pixelFormat = pixelFormat
        self.benchmark = benchmark
        self.benchmark = False
        self.readingDone = False
        self.writeOutPipe = False

        if self.outputFile == "PIPE":
            self.writeOutPipe = True
        
        self.readQueue = queue.Queue(maxsize=50)
        self.writeQueue = queue.Queue(maxsize=50)
        self.getVideoProperties()
        
        

    def getVideoProperties(self):
        cap = cv2.VideoCapture(self.inputFile)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.frameChunkSize = self.width * self.height * 3

    def getFFmpegReadCommand(self):
        command = [
            f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
            "-i",
            f"{self.inputFile}",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-",
        ]
        return command
    
    def getFFmpegWriteCommand(self):
        if not self.outputFile == 'PIPE':
            if not self.benchmark:
                #maybe i can split this so i can just use ffmpeg normally like with vspipe
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-r",
                    f"{self.fps * self.interpolateTimes}",
                    "-i",
                    "-",
                    "-i",
                    f"{self.inputFile}",
                    "-c:v",
                    self.encoder,
                    f"-crf",
                    f'18',
                    "-pix_fmt",
                    self.pixelFormat,
                    "-c:a",
                    "copy",
                    f"{self.outputFile}",
                ]
            else:
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-y",
                    "-v",
                    "warning",
                    "-stats",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-pix_fmt",
                    f"yuv420p",
                    "-r",
                    f"{self.fps * self.interpolateTimes}",
                    "-i",
                    "-",
                    "-benchmark",
                    "-f",
                    "null",
                    "-",
                ]
            return command                 
        '''else:
            
            command = [
            f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
            "-r",
            f"{self.fps * self.interpolateTimes}",
            "-i",
            f"-",
            "-pix_fmt",
            "rgb24", 
            '-f', 
            'rawvideo',
            "-",
        ]'''
           
    def readinVideoFrames(self):
        self.readProcess = subprocess.Popen(
            self.getFFmpegReadCommand(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        
        for i in range(self.totalFrames):
            chunk = self.readProcess.stdout.read(self.frameChunkSize)
            self.readQueue.put(chunk)
            
        
        self.readProcess.stdout.close()
        self.readProcess.terminate()
        self.readingDone = True
        self.readQueue.put(None)
        
    def writeOutVideoFrames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required, 
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4 
        """

        if self.writeOutPipe == False:
            self.writeProcess = subprocess.Popen(
                self.getFFmpegWriteCommand(),
                stdin=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )

            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break
                self.writeProcess.stdin.buffer.write(frame)

            self.writeProcess.stdin.close()
            self.writeProcess.wait()

        else:
            process = subprocess.Popen(['cat'], stdin=subprocess.PIPE)
            while True:
                
                frame = self.writeQueue.get()
                if frame is None:
                    break
                process.stdin.write(frame)

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
        backend
        device
        precision

    """
    def __init__(self,
                 inputFile: str,
                 outputFile: str,
                 interpolateTimes: int = 1,
                 encoder: str = "libx264",
                 pixelFormat: str = "yuv420p",
                 benchmark: bool = False,
                 backend = "pytorch",
                 interpolationMethod = None,
                 upscaleModel = None,
                 device = "cuda",
                 precision = "float16"
                 ):
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.device = device
        self.precision = precision
        self.setupUpscale()
        super().__init__(
            inputFile=inputFile, 
            outputFile=outputFile,
            interpolateTimes=interpolateTimes,
            upscaleTimes=self.upscaleTimes,
            encoder=encoder,
            pixelFormat=pixelFormat,
            benchmark=benchmark,
            )
        self.ffmpegReadThread = Thread(target=self.readinVideoFrames)
        self.ffmpegWriteThread = Thread(target=self.writeOutVideoFrames)
        self.renderThread = Thread(target=self.render)
        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()
    def render(self):
        """
        self.bytesToFrame, method that is mapped to the bytesToFrame in each respective backend
        self.upscale, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        """
        for i in range(self.totalFrames):
            frame = self.readQueue.get()
            frame = self.bytesToFrame(frame, height=self.height, width=self.width)
            if self.upscaleModel:
                frame = self.upscale(frame)
                print("Rendered Frame")
            self.writeQueue.put(frame)
    def setupUpscale(self):
        if self.backend == "pytorch":
            model = loadTorchModel(
                                   self.upscaleModel,
                                   self.precision,
                                   self.device
                                   )
            upscalePytorch = UpscalePytorchImage(
                model,
                device=self.device,
                precision=self.precision
                )
            self.upscaleTimes = upscalePytorch.getScale()
            self.bytesToFrame = upscalePytorch.bytesToFrame
            self.upscale = upscalePytorch.renderToNPArray
    def interpolate(self):
        pass