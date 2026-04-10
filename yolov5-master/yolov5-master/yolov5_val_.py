import torch
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import os
from tqdm import tqdm

# 載入 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"C:\Users\rem\Desktop\val\v5_model\weights\best.pt", force_reload=True)

# 資料夾路徑
img_dir = r"sample\back_hole"  # ✅ 驗證集圖片資料夾

# 統計預測的類別
pred_count = Counter()
failed_imgs = []  # 儲存失敗圖片名稱
empty_pred_imgs = []  # 儲存預測為空的圖片

all_img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(all_img_files, desc="預測中"):
    img_path = os.path.join(img_dir, img_name)

    try:
        # 嘗試開啟圖片確認無損壞
        Image.open(img_path).verify()

        # 模型預測
        results = model(img_path)

        # 取得預測結果
        df = results.pandas().xyxy[0]
        if len(df) == 0:
            empty_pred_imgs.append(img_name)
            continue

        # 取最高置信度的結果
        top_pred = df.loc[df['confidence'].idxmax()]
        pred_class = int(top_pred['class'])
        pred_count[pred_class] += 1

    except (UnidentifiedImageError, OSError) as e:
        failed_imgs.append(img_name)
        print(f"❌ 圖片讀取失敗: {img_name} - 錯誤訊息: {e}")
    except Exception as e:
        failed_imgs.append(img_name)
        print(f"❗ 未知錯誤發生於 {img_name}: {e}")

# 顯示預測統計結果
names = model.names  # 類別對應名稱
print("\n🔍 預測結果統計：")
for cls_id in sorted(pred_count.keys()):
    name = names.get(cls_id, f"Class {cls_id}")
    print(f"預測為 {name} 的圖片數量: {pred_count[cls_id]}")

print(f"\n✅ 預測成功圖片數量: {sum(pred_count.values())}")
print(f"🟡 預測為空（沒有辨識到物件）的圖片數量: {len(empty_pred_imgs)}")
print(f"❌ 讀檔失敗的圖片數量: {len(failed_imgs)}")

# 顯示錯誤圖片名稱
if failed_imgs:
    print("\n❌ 以下圖片讀取失敗：")
    for f in failed_imgs:
        print(f)

if empty_pred_imgs:
    print("\n🟡 以下圖片未預測出任何結果：")
    for f in empty_pred_imgs:
        print(f)
