import os

def createDir(path: str):
    os.makedirs(path, exist_ok=True)

def saveFile(base: str, index: int, conf: float, ext: str = "png") -> str:
    return f"{base}_{index}_conf{conf:.2f}.{ext}"