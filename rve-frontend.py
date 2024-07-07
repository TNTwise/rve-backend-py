import subprocess
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import sleep
import os

from src.Util import currentDirectory
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
command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{1280 * 2}x{720 * 2}",
                    "-r",
                    f"{24 * 1}",
                    "-i",
                    "-",
                    "-i",
                    f"KimetsuNoYaibaHashiraGeikoHen-OP1.webm",
                    "-c:v",
                    "libx264",
                    f"-crf",
                    f"18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    f"out10.mp4",
                    "-y"
                ]
writeProcess = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )

    

while True:
    '''try:
        chunk = mainProc.stdout.read(1280 * 4 * 720 * 3)
        chunk = np.frombuffer(chunk, dtype=np.uint8).reshape(720 * 2, 1280 * 2, 3)
        print(f"{i} chunk recieved!")
        i += 1
        cv2.imwrite("img.png", cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print(f"Chunk not available {e}")'''
    frame = mainProc.stdout.read(1280 * 4 * 720 * 3)
    if frame is None:
        break
    writeProcess.stdin.buffer.write(frame)
writeProcess.stdin.close()
writeProcess.wait()