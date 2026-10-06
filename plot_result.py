# plot_result.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tmm import tmm_reflectance
from train_mlp import MLP

# 强制使用CPU，避免没有显卡报错
device = torch.device('cpu')

# 读取数据集
data = np.load("dataset.npz")
lam_grid = data["lam_grid"]
X_test,Y_test = data["X_test"],data["Y_test"]

# 修复浮点数匹配问题，查找620nm下标
target_idx = np.where(np.isclose(lam_grid, 620))[0][0]

# 加载模型，强制cpu加载
model = MLP().to(device)
model.load_state_dict(torch.load("mlp_4000.pt", map_location=torch.device('cpu')))
model.eval()

# ========== 图8：测试集误差最大失败样例 ==========
with torch.no_grad():
    x_test_t = torch.from_numpy(X_test).float().to(device)
    y_pred_all = model(x_test_t).cpu().numpy()

err_per_sample = np.mean((y_pred_all - Y_test)**2, axis=1)
worst_idx = np.argmax(err_per_sample)
print(f"【图8失败样例】样本索引 = {worst_idx}，该样本MSE = {err_per_sample[worst_idx]:.6f}")

plt.figure(figsize=(7,4))
plt.plot(lam_grid, Y_test[worst_idx], label="TMM Ground Truth")
plt.plot(lam_grid, y_pred_all[worst_idx], "--", label="MLP Prediction")
plt.xlabel("Wavelength(nm)")
plt.ylabel("Reflectance")
plt.title("Representative failure case")
plt.legend()
plt.tight_layout()
plt.savefig("fail_case.png", dpi=150)
plt.show()


# ========== 图7：film_design输出第一名候选【⚠这里一定要替换成你自己运行出来的4个数字！！】 ==========
# 运行 film_design.py，复制第一行d1 d2 d3 d4四个数字替换下面
# 示例，不要直接跑！！！！
d_top1 = np.array([121.52,98.33,142.11,66.72])

# MLP预测
x1 = torch.from_numpy(d_top1[None,:]).float().to(device)
with torch.no_grad():
    yp1 = model(x1).cpu().numpy()[0]
# TMM真值
yt1 = tmm_reflectance(d_top1, lam_grid)

plt.figure(figsize=(7,4))
plt.plot(lam_grid, yt1, label="TMM Ground Truth")
plt.plot(lam_grid, yp1, "--", label="MLP Prediction")
plt.axvline(x=620,color="red",linestyle=":",label="$\\lambda_{target}$=620 nm")
plt.xlabel("Wavelength(nm)")
plt.ylabel("Reflectance")
plt.title("Top‑1 candidate spectrum comparison")
plt.legend()
plt.tight_layout()
plt.savefig("fig7_top1.png", dpi=150)
plt.show()

