from ultralytics import YOLO
from PIL import Image
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

        coarseResults = self.modelCoarse(img_pil)

        globalIndex = 0
        colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]

        for r in coarseResults:
            for box in r.boxes:
                clsIndex = int(box.cls.item())
                coarseClass = r.names[clsIndex]

                # 如果 coarseClass 沒有對應 fine model，就跳過
                if coarseClass not in self.modelFineDict:
                    continue

                # 取出 coarse 區域
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropPredictImg = cropImg = img_pil.crop((x1, y1, x2, y2))

                # 特殊前處理
                if coarseClass == "whi_word":
                    cropPredictImg = edgeDetect(cropImg)

                # === 跑 fine model ===
                fineModel, fineConf = self.modelFineDict[coarseClass]
                resultsFine = fineModel(cropPredictImg, conf=fineConf)

                for rFine in resultsFine:
                    for f_box in rFine.boxes:
                        f_cls_index = int(f_box.cls.item())
                        fineClass = rFine.names[f_cls_index]
                        conf_f = f_box.conf.item()

                        # fine box 是在 crop 座標系，要轉回原圖座標
                        fx1, fy1, fx2, fy2 = map(int, f_box.xyxy[0])
                        abs_fx1, abs_fy1 = x1 + fx1, y1 + fy1
                        abs_fx2, abs_fy2 = x1 + fx2, y1 + fy2

                        # 儲存 fine crop
                        fineCrop = (cropImg.crop((fx1, fy1, fx2, fy2)))
                        saveDir = os.path.join(self.outputDir, coarseClass, fineClass)
                        createDir(saveDir)

                        fineOutputPath = os.path.join(
                            saveDir,
                            saveFile(imgName, globalIndex, conf_f)
                        )
                        fineCrop.save(fineOutputPath)
                        globalIndex += 1

                        # === 在原圖畫框 (使用 coarse 偏移後的座標) ===
                        color = colors[clsIndex % len(colors)]
                        font_scale = 0.8
                        thickness = 2

                        cv2.rectangle(img_cv, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), color, 4)
                        label = f"{fineClass} {conf_f:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                        y1_text = max(abs_fy1 - text_height - baseline - 2, 0)
                        cv2.rectangle(img_cv, (abs_fx1, y1_text),
                                                (abs_fx1 + text_width, abs_fy1), color, -1)
                        cv2.putText(img_cv, label, (abs_fx1, abs_fy1 - baseline - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

        # === 存下整張有標記的圖 ===
        cv2.imwrite(f"{self.outputDir}/all_{imgName}.jpg", img_cv)
        print(f"[DONE] Coarse-to-Fine processed for {imgName}")
