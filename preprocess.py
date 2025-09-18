import cv2
import numpy as np
import os
from PIL import Image
import onnxruntime as ort
from pathlib import Path

def edgeDetect(img: Image.Image) -> Image.Image:
    cvImg = np.array(img)
    gray = cv2.cvtColor(cvImg, cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(th)


onnxModelPath = "weight/u2net.onnx"
ortSession = ort.InferenceSession(onnxModelPath)

def load_image_u2net_from_image(img: Image.Image, target_size=(320, 320)):
    img = img.convert("RGB")
    imgResized = img.resize(target_size)
    imgNp = np.array(imgResized).astype(np.float32) / 255.0
    imgNp = imgNp.transpose(2, 0, 1)
    imgNp = imgNp[np.newaxis, ...]
    return imgNp

def predict_mask_u2net(imgNp, outputSize):
    inputName = ortSession.get_inputs()[0].name
    outputName = ortSession.get_outputs()[0].name
    result = ortSession.run([outputName], {inputName: imgNp})[0]
    mask = result[0][0]
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, outputSize)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # 平滑邊界
    return mask

def apply_mask_u2net_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    imgRgba = img.convert("RGBA")
    maskImg = Image.fromarray(mask).convert("L")
    imgRgba.putalpha(maskImg)
    return imgRgba

def imageMatting(img: Image.Image) -> Image.Image:
    origW, origH = img.size
    imgNp = load_image_u2net_from_image(img)
    mask = predict_mask_u2net(imgNp, (origW, origH))
    resultImg = apply_mask_u2net_image(img, mask)
    return resultImg





