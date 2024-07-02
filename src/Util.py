from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
import os
import warnings
import torch

cwd = os.getcwd()


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check if it's a valid image file
        return True
    except (IOError, SyntaxError):
        return False


def loadModelWithScale(
    modelPath: str, dtype: torch.dtype = torch.float32, device: str = "cuda"
):
    model = ModelLoader().load_from_file(modelPath)
    assert isinstance(model, ImageModelDescriptor)
    # get model attributes
    scale = model.scale

    model.to(device=device, dtype=dtype)
    return model, scale


def loadModel(modelPath: str, dtype: torch.dtype = torch.float32, device: str = "cuda"):
    model = ModelLoader().load_from_file(modelPath)
    assert isinstance(model, ImageModelDescriptor)
    # get model attributes

    model.to(device=device, dtype=dtype)
    return model


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def printAndLog(message: str):
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(cwd, "log.txt"), "a") as f:
        f.write(message + "\n")

def currentDirectory():
    return cwd