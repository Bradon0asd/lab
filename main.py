import os
from predictorBig import Predict
from ultralytics import YOLO


fineModelDict = {
    "casp": [YOLO("weight/pill.pt"), 0.8],
    "whi_wor": [YOLO("weight/whiteWithText.pt"), 0.25],
    "whi_onwor": [YOLO("weight/whiteNoText.pt"), 0.25], 
    "nowhi_rou": [YOLO("weight/circleNotWhite.pt"), 0.25], 
    "not_rou": [YOLO("weight/unCircle.pt"), 0.8],
}

predictor = Predict("weight/big.pt", fineModelDict)

# 資料夾
sourceDir = "images"

for imgName in os.listdir(sourceDir):
    if not imgName.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    imgPath = os.path.join(sourceDir, imgName)
    predictor.run(imgPath)