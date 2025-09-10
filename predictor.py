from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
from preprocess import edgeDetect, imageMatting
from loader import createDir, saveFile


class Predict:
    def __init__(self, coarseWeight, fineModelDict, outputDir="crops"):
        self.modelCoarse = YOLO(coarseWeight)
        self.modelFineDict = fineModelDict
        self.outputDir = outputDir
        createDir(self.outputDir)

    def run(self, imgPath: str):
        imgName = os.path.basename(imgPath).split(".")[0]

        img_pil = Image.open(imgPath).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        globalIndex = 0
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

        for i, (coarseClass, (fineModel, fineConf)) in enumerate(self.modelFineDict.items()):
            print(f"[INFO] Running fine model for class {coarseClass} ...")
            resultsFine = fineModel(img_pil, conf=fineConf)

            for rFine in resultsFine:
                for f_box in rFine.boxes:
                    x1, y1, x2, y2 = map(int, f_box.xyxy[0])
                    conf_f = f_box.conf.item()
                    cls = rFine.names[int(f_box.cls.item())]

                    fineCrop = img_pil.crop((x1, y1, x2, y2))
                    saveDir = os.path.join(self.outputDir, coarseClass, cls)
                    os.makedirs(saveDir, exist_ok=True)
                    fineOutputPath = os.path.join(saveDir, f"{imgName}_{globalIndex}_{conf_f:.2f}.jpg")
                    fineCrop.save(fineOutputPath)
                    globalIndex += 1


                    color = colors[i % len(colors)]
                    font_scale = 0.8
                    thickness = 2
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 4)
                    label = f"{cls} {conf_f:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
   
                    cv2.rectangle(img_cv, (x1, y1 - text_height - baseline - 2), 
                                            (x1 + text_width, y1), color, -1)

                    cv2.putText(img_cv, label, (x1, y1 - baseline - 1), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

        cv2.imwrite(f"{self.outputDir}/all_{imgName}.jpg", img_cv)

        print(f"[DONE] All fine models processed for {imgName}")