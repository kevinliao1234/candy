import os
import shutil
import random
import subprocess
from pathlib import Path
from collections import Counter

# === 設定參數 ===
n = 374  # 設定要抽取的總數量

if n <= 184:
    x = 0.6
    y = 0.4
else:
    x = 0.8
    y = 0.2

dataset_dir = Path('obj')
output_dir = Path('test_set')
yolov7_dir = Path(r"C:\Users\rem\Desktop\yolov7")  # <-- 新增 YOLOv7 專案路徑
model_path = r"C:\Users\rem\Desktop\yolov7\runs\v7_model\weights\best.pt" # <-- 你的 YOLOv7 模型路徑
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

# === 執行抽樣 ===
hole_imgs = sample_images('normal', int(n * x))
normal_imgs = sample_images('normal', int(n * y))
selected_imgs = hole_imgs + normal_imgs

# === 複製圖片與標註 ===
for img_path in selected_imgs:
    label_path = img_path.with_suffix('.txt')
    shutil.copy(img_path, output_dir / img_path.name)
    if label_path.exists():
        shutil.copy(label_path, output_dir / label_path.name)

print(f'✅ 測試集建立完成，共 {len(selected_imgs)} 張圖片。')

# --- 執行 YOLOv7 推論（使用 detect.py） ---
detect_cmd = [
    'python', 'detect.py',
    '--weights', model_path,
    '--conf', '0.50',
    '--img-size', '640',
    '--source', str(output_dir),
    '--save-txt',
    '--save-conf',
    '--project', 'runs/test',
    '--name', 'predict_results',
    '--exist-ok'
]

print("\n▶️ 執行 YOLOv7 模型推論...")
subprocess.run(detect_cmd, cwd=str(yolov7_dir))

# --- 統計預測結果（改為統計所有預測框的類別） ---
result_txt_dir = yolov7_dir / 'runs/test/predict_results/labels'
prediction_counter = Counter()

# 類別對應
class_names = {
    0: 'hole',
    1: 'leak',
    2: 'normal',
    3: 'white'
}
priority = ['hole', 'leak', 'normal', 'white']

for txt_file in result_txt_dir.glob('*.txt'):
    best_cls = None
    best_conf = -1.0

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:  # 有 confidence
                class_id = int(parts[0])
                conf = float(parts[5])
                if conf > best_conf:
                    best_conf = conf
                    best_cls = class_id
            elif len(parts) == 5:
                # 沒有 confidence 的情況無法比較，只能直接取第一行
                class_id = int(parts[0])
                best_cls = class_id
                break  # 沒有信心值，直接結束

    if best_cls is not None and best_cls in class_names:
        cls_name = class_names[best_cls]
        prediction_counter[cls_name] += 1


# === 顯示統計結果 ===
print('\n📊 預測統計結果（統計所有預測框的類別）：')
for cls_name in priority:
    count = prediction_counter.get(cls_name, 0)
    print(f"預測為 '{cls_name}' 的預測框數量：{count} 個")
