# train_mlp.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

seed = 270148
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("运行设备 device =", device)

# ========= MLP网络，作业规定结构 4‑128‑128‑64‑41 =========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 41)
        )

    def forward(self, x):
        # x: [batch,4] 输入四层膜厚；输出 [batch,41]反射光谱
        return self.net(x)


def run_exp(n_train_samples, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """
    n_train_samples: 使用训练集前多少条样本训练（500/1000/2000/4000）
    验证、测试集固定不变，不改变！
    return:训练好的模型 + 测试集MSE损失
    """
    x_tr = torch.from_numpy(X_train[:n_train_samples]).float().to(device)
    y_tr = torch.from_numpy(Y_train[:n_train_samples]).float().to(device)

    x_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(Y_val).float().to(device)

    x_te = torch.from_numpy(X_test).float().to(device)
    y_te = torch.from_numpy(Y_test).float().to(device)

    model = MLP().to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epoch_num = 800

    for e in range(epoch_num):
        model.train()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 训练结束，评估测试集
    model.eval()
    with torch.no_grad():
        pred_te = model(x_te)
        test_loss = loss_fn(pred_te, y_te).item()
    return model, test_loss


if __name__ == "__main__":
    data = np.load("dataset.npz")
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val,   Y_val   = data["X_val"],   data["Y_val"]
    X_test,  Y_test  = data["X_test"],  data["Y_test"]
    lam_grid = data["lam_grid"]

    # 任务③：不同训练样本规模实验
    exp_sizes = [500, 1000, 2000, 4000]
    test_err_list = []
    model_4000 = None

    for sz in exp_sizes:
        model, te_err = run_exp(sz, X_train,Y_train,X_val,Y_val,X_test,Y_test)
        test_err_list.append(te_err)
        print(f"训练样本 {sz:4d} | Test MSE = {te_err:.6f}")
        if sz == 4000:
            model_4000 = model
            torch.save(model.state_dict(), "mlp_4000.pt")
            print("已保存4000样本训练模型 mlp_4000.pt")

    # 图1：测试MSE随训练样本数量变化
    plt.figure(figsize=(6,4))
    plt.plot(exp_sizes, test_err_list, "-o")
    plt.xlabel("Number of training samples")
    plt.ylabel("Test MSE Loss")
    plt.title("Effect of training data size")
    plt.tight_layout()
    plt.savefig("data_size_error.png", dpi=150)
    plt.show()

    # 图2：随机取一条测试样本，MLP预测光谱 vs TMM真值光谱对比
    model_4000.eval()
    idx_sample = 10
    x_sample = torch.from_numpy(X_test[idx_sample:idx_sample+1]).float().to(device)
    with torch.no_grad():
        y_pred = model_4000(x_sample).cpu().numpy()[0]
    y_true = Y_test[idx_sample]

    plt.figure(figsize=(7,4))
    plt.plot(lam_grid, y_true, label="TMM Ground Truth")
    plt.plot(lam_grid, y_pred, "--", label="MLP Prediction")
    plt.xlabel("Wavelength(nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.title("Spectrum Prediction Comparison")
    plt.tight_layout()
    plt.savefig("spec_compare.png", dpi=150)
    plt.show()
