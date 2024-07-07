import subprocess
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep

command = [
    "python3",
    "rve-backend.py",
    "-i",
    "KimetsuNoYaibaHashiraGeikoHen-OP1.webm",
    "-o",
    "PIPE",
    "-u",
    "2x_AnimeJaNai_V2_SuperUltraCompact_100k.pth",
]
mainProc = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
i = 0
while True:
    try:
        chunk = mainProc.stdout.read(1280 * 4 * 720 * 3)
        chunk = np.frombuffer(chunk, dtype=np.uint8).reshape(720 * 2, 1280 * 2, 3)
        print(f"{i} chunk recieved!")
        i += 1
        cv2.imwrite("img.png", chunk)

    except Exception as e:
        print(f"Chunk not available {e}")
