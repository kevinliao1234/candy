import os
import shutil
import random
from pathlib import Path
from collections import Counter
import torch
from PIL import Image


# === 設定參數 ===
n = 200

if n >= 184:
    x = 0.6
    y = 0.4
else:
    x = 0.8
    y = 0.2

dataset_dir = Path('obj')
output_dir = Path('test_set')
model_path = r'yolo_model\yolov4_best.pt'
img_suffix = ['.jpg']

# === 建立測試資料夾 ===
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

# === 隨機抽樣工具 ===
def sample_images(label_folder: str, sample_count: int):
    folder = dataset_dir / label_folder
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in img_suffix]
    sampled = random.sample(image_files, min(sample_count, len(image_files)))
    return sampled

# === 抽樣並複製檔案 ===
hole_imgs = sample_images('hole', int(n * x))
normal_imgs = sample_images('normal', int(n * y))
selected_imgs = hole_imgs + normal_imgs

for img_path in selected_imgs:
    label_path = img_path.with_suffix('.txt')
    shutil.copy(img_path, output_dir / img_path.name)
    if label_path.exists():
        shutil.copy(label_path, output_dir / label_path.name)

print(f'✅ 測試集建立完成，共 {len(selected_imgs)} 張圖片。')

# === 載入 YOLOv5 模型 ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.25  # 設定信心閾值

# === 預測並統計類別（以圖片為單位）===
class_names = model.names  # 類別名稱對應
prediction_counter = Counter()
priority = ['hole', 'leak', 'white', 'normal']  # 類別優先順序

for img_path in output_dir.iterdir():
    if img_path.suffix.lower() not in img_suffix:
        continue

    results = model(str(img_path))
    pred_names = set()
    for *box, conf, cls in results.xyxy[0].tolist():
        class_name = class_names[int(cls)]
        pred_names.add(class_name)

    # 計入優先類別（每張圖只算一次）
    for cls_name in priority:
        if cls_name in pred_names:
            prediction_counter[cls_name] += 1
            break

# === 顯示統計結果 ===
print('\n📊 預測統計結果（以圖片為單位）：')
for cls_name in priority:
    count = prediction_counter.get(cls_name, 0)
    print(f"預測為 '{cls_name}' 的圖片數量：{count} 張")
