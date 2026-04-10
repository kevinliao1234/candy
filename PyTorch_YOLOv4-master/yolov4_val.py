import os
import shutil
import random
from pathlib import Path
from collections import Counter
from PIL import Image
from yolo import YOLO


output_dir = Path('sample/normal')   # 測試集輸出資料夾
img_suffix = ['.jpg']           # 圖片副檔名
model_path = r'logs\yolov4_best.pth'  # 模型路徑


# === 載入 YOLOv4 模型 ===
yolo = YOLO(model_path=model_path)

# === 統計預測結果（每張圖片計一個主要類別） ===
class_names = yolo.class_names
priority = ['hole', 'leak', 'white', 'normal']  # 類別優先順序
prediction_counter = Counter()

for img_file in output_dir.iterdir():
    if img_file.suffix.lower() not in img_suffix:
        continue
    image = Image.open(img_file)
    _, labels, _ = yolo.detect_image_raw(image)

    labels_set = set(labels)
    for cls_name in priority:
        if cls_name in labels_set:
            prediction_counter[cls_name] += 1
            break


# === 顯示統計結果 ===
print('\n📊 預測統計結果（以圖片為單位）：')
for cls_name in ['hole', 'normal', 'leak', 'white']:
    count = prediction_counter.get(cls_name, 0)
    print(f"{cls_name} 出現在 {count} 張圖片中")
