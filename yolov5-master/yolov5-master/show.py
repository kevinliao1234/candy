import matplotlib.pyplot as plt
import pandas as pd

# 手動填上你的結果（或從驗證結果自動讀取）
data = {
    'Class': ['hole', 'leak', 'normal', 'white', 'all'],
    'mAP@0.5': [0.269, 0.394, 0.908, 0.207, 0.444],
    'mAP@0.5:0.95': [0.206, 0.281, 0.449, 0.130, 0.267]
}

df = pd.DataFrame(data)

# 畫出表格
fig, ax = plt.subplots(figsize=(6, 2))  # 可調整圖大小
ax.axis('off')
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center')

# 美化表格
table.scale(1, 1.5)  # 字體縮放
table.auto_set_font_size(False)
table.set_fontsize(10)

# 儲存成圖片
plt.savefig("map_table.png", dpi=300, bbox_inches='tight')
plt.show()
