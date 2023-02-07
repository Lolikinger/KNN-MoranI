import numpy as np

def MoranI(W,X):
    # W：空间权重矩阵
    # X；观测值矩阵
    W = np.array(W)
    X = np.array(X)
    X = X.reshape(1, -1)
    W = W / W.sum(axis=1)  # 权重矩阵归一化
    n = W.shape[0]  # 空间单元数
    Z = X - X.mean()  # 离差阵
    S0 = W.sum()
    S1 = 0
    for i in range(n):
        for j in range(n):
            S1 += 0.5 * (W[i, j] + W[j, i]) ** 2
    S2 = 0
    for i in range(n):
        S2 += (W[i, :].sum() + W[:, i].sum()) ** 2
    # 全局moran指数
    I = np.dot(Z, W)
    I = np.dot(I, Z.T)
    I = n / S0 * I / np.dot(Z, Z.T)
    # 在正太分布假设下的检验数
    EI_N = -1 / (n - 1)
    VARI_N = (n ** 2 * S1 - n * S2 + 3 * S0 ** 2) / (S0 ** 2 * (n ** 2 - 1)) - EI_N ** 2
    ZI_N = (I - EI_N) / (VARI_N ** 0.5)
    # 在随机分布假设下检验数
    EI_R = -1 / (n - 1)
    b2 = 0
    for i in range(n):
        b2 += n * Z[0, i] ** 4
    b2 = b2 / ((Z * Z).sum() ** 2)
    VARI_R = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) - b2 * (
            (n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    VARI_R = VARI_R / (S0 ** 2 * (n - 1) * (n - 2) * (n - 3)) - EI_R ** 2
    ZI_R = (I - EI_R) / (VARI_R ** 0.5)
    # 计算局部moran指数
    Ii = []
    for i in range(n):
        Ii_ = n * Z[0, i]
        # print(f"I{i+1}_:{Ii_}")
        Ii__ = 0
        for j in range(n):
            if i != j:
                Ii__ += W[i, j] * Z[0, j]
        Ii_ = Ii_ * Ii__ / ((Z * Z).sum())
        Ii.append(Ii_)
    Ii = np.array(Ii)
    # 局部检验数
    ZIi = list()
    EIi = Ii.mean()
    VARIi = Ii.var()
    for i in range(n):
        ZIi_ = (Ii[i] - EIi) / (VARIi ** 0.5)
        ZIi.append(ZIi_)
    ZIi = np.array(ZIi)
    return {
        'I': {'value': I[0, 0], 'desc': '全局moran指数'},
        'ZI_N': {'value': ZI_N[0, 0], 'desc': '正太分布假设下检验数'},
        'ZI_R': {'value': ZI_R[0, 0], 'desc': '随机分布假设下检验数'},
        'Ii': {'value': Ii, 'desc': '局部moran指数'},
        'ZIi': {'value': ZIi, 'desc': '局部检验数'},
    }
if __name__ == "__main__":
    w = [
         [0,1,1,0,0],
         [1,0,1,1,0],
         [1,1,0,1,0],
         [0,1,1,0,1],
         [0,0,0,1,0]
        ]
    w = np.array(w)
    x = [
         [8,6,6,3,2]
         ]
    x = np.array(x)
    print(MoranI(w,x)["I"])
    print(MoranI(w, x)["Ii"])

