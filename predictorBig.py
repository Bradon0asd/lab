from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import csv
from shapely.geometry import Polygon
from preprocess import edgeDetect, imageMatting
from loader import createDir, saveFile

labelArray = ['MB50E','sy25','CS40E','D','sp5','PLT','STD006','DXXD/NVR','chiisiro','orange','pink line','23UL','STD007','sanylsc','ookishiro','309hex','16C','CG/hhh']


def yolo_bbox_to_polygon(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def read_poly_label(txt_path, img_w, img_h):
    polys = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polys.append((cls, Polygon(coords)))
    return polys


def compute_iou(pred_poly, gt_poly):
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0)
    if not gt_poly.is_valid:
        gt_poly = gt_poly.buffer(0)

    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0

    inter = pred_poly.intersection(gt_poly).area
    union = pred_poly.union(gt_poly).area
    if union == 0:
        return 0.0
    return inter / union


class Predict:
    def __init__(self, coarseWeight, fineModelDict, labelDir="images", outputDir="crops"):
        self.modelCoarse = YOLO(coarseWeight)
        self.modelFineDict = fineModelDict
        self.outputDir = outputDir
        self.labelDir = labelDir
        createDir(self.outputDir)
        self.csvPath = os.path.join(self.outputDir, "results.csv")
        with open(self.csvPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "pred_class", "gt_class", "iou", "correct"])

    def run(self, imgPath: str):
        imgName = os.path.basename(imgPath).split(".")[0]

        img_pil = Image.open(imgPath).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        coarseResults = self.modelCoarse(img_pil)

        globalIndex = 0

        for r in coarseResults:
            for box in r.boxes:
                clsIndex = int(box.cls.item())
                coarseClass = r.names[clsIndex]

                if coarseClass not in self.modelFineDict:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropPredictImg = cropImg = img_pil.crop((x1, y1, x2, y2))

                if coarseClass == "whi_word":
                    cropPredictImg = edgeDetect(cropImg)

                fineModel, fineConf = self.modelFineDict[coarseClass]
                resultsFine = fineModel(cropPredictImg, conf=fineConf)

                for rFine in resultsFine:
                    for f_box in rFine.boxes:
                        f_cls_index = int(f_box.cls.item())
                        fineClass = rFine.names[f_cls_index]
                        conf_f = f_box.conf.item()

                        fx1, fy1, fx2, fy2 = map(int, f_box.xyxy[0])
                        abs_fx1, abs_fy1 = x1 + fx1, y1 + fy1
                        abs_fx2, abs_fy2 = x1 + fx2, y1 + fy2

                        fineCrop = (cropImg.crop((fx1, fy1, fx2, fy2)))
                        saveDir = os.path.join(self.outputDir, coarseClass, fineClass)
                        createDir(saveDir)

                        fineOutputPath = os.path.join(
                            saveDir,
                            saveFile(imgName, globalIndex, conf_f)
                        )
                        fineCrop.save(fineOutputPath)
                        globalIndex += 1

                        

                        labelPath = os.path.join(self.labelDir, imgName + ".txt")
                        if os.path.exists(labelPath):
                            gt_polys = read_poly_label(labelPath, img_cv.shape[1], img_cv.shape[0])
                            pred_poly = yolo_bbox_to_polygon(abs_fx1, abs_fy1, abs_fx2, abs_fy2)

                            best_iou, best_gt_cls = 0, None
                            
                            
                            for gt_cls, gt_poly in gt_polys:
                                iou = compute_iou(pred_poly, gt_poly)
                                if iou > best_iou:
                                    best_iou, best_gt_cls = iou, gt_cls
                           
                            try:
                                cr_cls_label = labelArray[int(best_gt_cls)]
                            except:
                                cr_cls_label = best_gt_cls
                            
                            correct = True and fineClass == cr_cls_label
                            print(f_cls_index,best_gt_cls)
                            
                            with open(self.csvPath, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([imgName, fineClass, cr_cls_label, best_iou, correct])

                            for gt_cls, gt_poly in gt_polys:
                                pts = np.array(gt_poly.exterior.coords, np.int32)
                                cv2.polylines(img_cv, [pts], True, (0, 255, 0), 2)  # 綠色為 GT
                            label = ""
                            if not correct:
                                color = (0, 0, 255)
            
                                label =  f"{fineClass} {conf_f:.2f} | {cr_cls_label}"
                            else:
                                color = (255, 0, 0)
                                label = f"{fineClass} {conf_f:.2f}"
                            font_scale = 0.8
                            thickness = 2
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 4)
                            
                            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                            
        
                            cv2.rectangle(img_cv, (x1, y1 - text_height - baseline - 2), 
                                                    (x1 + text_width, y1), color, -1)

                            cv2.putText(img_cv, label, (x1, y1 - baseline - 1), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

        cv2.imwrite(f"{self.outputDir}/all_{imgName}.jpg", img_cv)
        print(f"[DONE] Coarse-to-Fine processed & evaluated for {imgName}")
