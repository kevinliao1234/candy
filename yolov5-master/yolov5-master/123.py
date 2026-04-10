import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 建立一個大的 tensor
x = torch.randn((10000, 10000), device=device)
y = torch.randn((10000, 10000), device=device)

# 執行大量計算
start = time.time()
for _ in range(10):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"完成矩陣乘法，耗時: {end - start:.2f} 秒")
