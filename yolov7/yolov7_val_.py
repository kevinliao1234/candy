import torch
import os
from pathlib import Path
from collections import Counter
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from tqdm import tqdm

# 參數設定
weights = r"F:\yolov7\runs\train\exp\weights\best.pt"  # ✅ 你自己的 YOLOv7 模型路徑
val_img_dir = r"C:\Users\rem\Desktop\val\sample\back_hole\images"  # ✅ 驗證集圖片資料夾
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 載入模型
model = attempt_load(weights, map_location=device)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

# 載入圖片
dataset = LoadImages(val_img_dir, img_size=imgsz)

# 統計類別
all_preds = []

for path, img, im0s, vid_cap in tqdm(dataset):
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        top_pred_idx = det[:, 4].argmax()
        cls_id = int(det[top_pred_idx, 5].item())
        all_preds.append(cls_id)

# 統計結果
count = Counter(all_preds)

print("\n統計每個類別被預測的圖片數量：")
for cls_id in sorted(count):
    name = names[cls_id] if cls_id in names else f"Class {cls_id}"
    print(f"預測為 {name} 的圖片數量: {count[cls_id]}")
