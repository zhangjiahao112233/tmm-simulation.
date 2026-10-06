# tmm.py
import numpy as np

def tmm_reflectance(d: np.ndarray, lam: np.ndarray,
                     nH=2.30, nL=1.45, n0=1.0, ns=1.52):
    """
    膜系结构：Air / H / L / H / L / Glass 四层膜
    d: [d1,d2,d3,d4] 四层膜厚度，单位nm
    lam: 波长数组 400‑800nm，步长10nm，共41点
    nH=2.30 高折射率层；nL=1.45低折射率层；n0空气；ns玻璃基底
    return: R:和lam同长度的反射率数组
    """
    d = np.array(d)
    # 四层膜折射率依次 H L H L
    n_list = [nH, nL, nH, nL]
    N_wl = len(lam)
    # 每个波长独立的2×2传输矩阵，初始单位矩阵
    M = np.tile(np.eye(2, dtype=complex), (N_wl, 1, 1))

    for ni, di in zip(n_list, d):
        beta = 2 * np.pi * ni * di / lam
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        Mi = np.zeros((N_wl, 2, 2), dtype=complex)
        Mi[:, 0, 0] = cosb
        Mi[:, 0, 1] = 1j * sinb / ni
        Mi[:, 1, 0] = 1j * ni * sinb
        Mi[:, 1, 1] = cosb
        M = M @ Mi

    m11, m12 = M[:,0,0], M[:,0,1]
    m21, m22 = M[:,1,0], M[:,1,1]
    # 菲涅尔反射系数公式
    r = (n0*(m11 + m12*ns) - (m21 + m22*ns)) / (n0*(m11 + m12*ns) + (m21 + m22*ns))
    R = np.abs(r) ** 2
    return R


if __name__ == "__main__":
    # 简单自测代码
    lam_test = np.arange(400, 810, 10)
    d_test = [100, 100, 100, 100]
    R_test = tmm_reflectance(d_test, lam_test)
    print(f"测试光谱形状 {R_test.shape}，应该是(41,)")
