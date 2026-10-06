# generate_dataset.py
import numpy as np
import random
from tmm import tmm_reflectance

# ===== 个人参数（来自你的学号2023270148）=====
seed = 270148
random.seed(seed)
np.random.seed(seed)

lam_grid = np.arange(400, 810, 10)   # 400‑800nm，步长10nm，41个点
n_sample_total = 5000                # 总样本5000
d_min, d_max = 40, 180                # 每层厚度范围40‑180nm

if __name__ == "__main__":
    print("正在生成5000组四层膜样本...")
    # 随机生成5000组 [d1,d2,d3,d4]
    d_all = np.random.uniform(low=d_min, high=d_max, size=(n_sample_total, 4))

    R_all = []
    for idx, d in enumerate(d_all):
        R = tmm_reflectance(d, lam_grid)
        R_all.append(R)
        if (idx+1) % 1000 == 0:
            print(f"已完成 {idx+1}/{n_sample_total}")

    R_all = np.array(R_all)

    # 严格按照作业要求固定划分，不随机打乱
    X_train = d_all[:4000]
    Y_train = R_all[:4000]

    X_val   = d_all[4000:4500]
    Y_val   = R_all[4000:4500]

    X_test  = d_all[4500:]
    Y_test  = R_all[4500:]

    # 保存dataset.npz，把训练、验证、测试全部存进去
    np.savez("dataset.npz",
             X_train=X_train,Y_train=Y_train,
             X_val=X_val,Y_val=Y_val,
             X_test=X_test,Y_test=Y_test,
             lam_grid=lam_grid)

    print("数据集保存完成 dataset.npz")
    print(f"训练集 {X_train.shape} | 验证集 {X_val.shape} | 测试集 {X_test.shape}")
