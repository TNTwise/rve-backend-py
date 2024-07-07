from PIL import Image
import os
import warnings

cwd = os.getcwd()


def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check if it's a valid image file
        return True
    except (IOError, SyntaxError):
        return False


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
