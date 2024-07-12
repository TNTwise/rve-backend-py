import torch
from .InterpolateArchs.DetectInterpolateArch import loadModel

class InterpolateRifeTorch:
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        device: str = "cuda",
        dtype: str = "float16",
        backend: str = "pytorch",
    ):
        self.interpolateModel = interpolateModelPath
        self.width = width
        self.height = height
        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        state_dict = torch.load(interpolateModelPath)
        
        model = loadModel(state_dict)
        self.flownet = model.getIFnet().to(device=self.device,dtype=self.dtype)
        
    def handlePrecision(self, precision):
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    def process(self):
        pass
