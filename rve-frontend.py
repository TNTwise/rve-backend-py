import subprocess
from time import sleep, time
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
    "2x_ModernSpanimationV1.pth.ncnn",
    "-b",
    "ncnn",
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
    "-y",
]
writeProcess = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    text=True,
    universal_newlines=True,
)


outputChunk = 1280 * 4 * 720 * 3
startTime = time()
while True:
    frame = mainProc.stdout.read(outputChunk)
    i += 1
    print(i / (time() - startTime))
    print(i)
    if frame is None:
        break
    writeProcess.stdin.buffer.write(frame)
writeProcess.stdin.close()
writeProcess.wait()
