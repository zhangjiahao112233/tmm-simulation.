# film_design.py
import numpy as np
import random
import torch
from tmm import tmm_reflectance
from train_mlp import MLP

# =========个人参数========
seed_design = 270149   # design_seed = seed+1
target_lam = 620        # λtarget=620nm
random.seed(seed_design)
np.random.seed(seed_design)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lam_grid = np.arange(400, 810, 10)
idx_target = np.where(lam_grid == target_lam)[0][0] # 找到620nm在数组里的下标

if __name__ == "__main__":
    # 生成10000全新候选膜系，4层，40‑180nm
    n_candidate = 10000
    d_candidate = np.random.uniform(40, 180, size=(n_candidate, 4))

    # 加载已经训练好的MLP代理模型
    model = MLP().to(device)
    model.load_state_dict(torch.load("mlp_4000.pt", map_location=device))
    model.eval()

    # MLP批量预测10000组光谱
    x_can = torch.from_numpy(d_candidate).float().to(device)
    with torch.no_grad():
        Y_pred = model(x_can).cpu().numpy()

    # 取出每组候选在620nm的MLP预测反射率
    R620_pred = Y_pred[:, idx_target]
    # 从大到小排序，取前20
    sort_idx = np.argsort(-R620_pred)
    topN = 20
    top_d = d_candidate[sort_idx[:topN]]
    top_mlp_R620 = R620_pred[sort_idx[:topN]]

    print("="*65)
    print(f"Top {topN}候选膜系｜目标波长 λtarget = {target_lam} nm")
    print(f"{'No':<3}{'d1':>7}{'d2':>7}{'d3':>7}{'d4':>7}{'MLP_R620':>12}{'TMM_R620':>12}")
    print("-"*65)
    for i, d in enumerate(top_d):
        R_true = tmm_reflectance(d, lam_grid)
        r620_true = R_true[idx_target]
        print(f"{i:<3}{d[0]:7.2f}{d[1]:7.2f}{d[2]:7.2f}{d[3]:7.2f}{top_mlp_R620[i]:12.4f}{r620_true:12.4f}")
    print("="*65)
