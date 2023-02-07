import random

import numpy as np
import matplotlib.pyplot as plt


def data_set():
    data = [[94, 98], [99, 29], [73, 14], [52, 10], [33, 37],
            [71, 53], [51, 50], [79, 48], [70, 93], [49, 66]]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return [data, labels]


def data_show(train_data, w):
    # 显示训练数据
    x = []
    y = []
    for i in range(len(train_data[0])):
        x.append(train_data[0][i][0])
        y.append(train_data[0][i][1])
        label_x = train_data[0][i][0]
        label_y = train_data[0][i][1]
        label_type = train_data[1][i][0]
        plt.text(label_x, label_y + 2, f"{label_type}\n({label_x},{label_y})", ha='center', va='bottom')
        color_list = ["r", "b", "g", "c", "y"]
        for j in range(len(w[0])):
            if w[i][j] == 1:
                _x = train_data[0][i][0], train_data[0][j][0]
                _y = train_data[0][i][1], train_data[0][j][1]
                plt.plot(_x, _y, color=color_list[i % 5])

    plt.plot(x, y, "*")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def knn(tran_data, k=3, method=1):
    """
    knn算法得到空间权重矩阵，可选空间位置或属性值作为计算指标
    :param tran_data: 需分类数据，列表
    :param k: 最大邻接数
    :param method: 分类指标，1为一维指标，2为二维指标
    :return:空间权重矩阵
    """
    tran_data = np.array(tran_data)
    # 获取数据组数
    m = tran_data.shape[0]
    # print(tran_data)
    print(m)
    # 初始化距离矩阵
    distance_matrix = [[0 for i in range(m)] for j in range(m)]
    add_distance = distance_matrix
    # 距离矩阵计算
    if method == 2:
        for i in range(m):
            for j in range(m):
                # 对应特征值相减
                add_distance[i][j] = tran_data[i] - tran_data[j]
                # 不同维度特征值相加
                distance_matrix[i][j] = add_distance[i][j][0] ** 2 + add_distance[i][j][1] ** 2
                # 开方操作
                distance_matrix[i][j] = distance_matrix[i][j] ** 0.5
    if method == 1:
        for i in range(m):
            for j in range(m):
                # 对应特征值相减
                distance_matrix[i][j] = abs(tran_data[i] - tran_data[j])
    # 去除自我邻接距离
    for i in range(m):
        distance_matrix[i][i] = np.inf
    # 根据距离升序排序
    tem_index = np.argsort(distance_matrix)
    # 初始化排序结果索引
    index = [0 for i in range(m)]
    # 根据k值切片将每个元素邻接的元素索引提取出来
    for i in range(m):
        index[i] = tem_index[i][0:k]
    # 构建空间权重矩阵
    w = [[0 for i in range(m)] for j in range(m)]
    index = np.array(index)
    w = np.array(w)
    for i in range(m):
        for j in range(k):
            n = index[i][j]
            w[i][n] = 1
    return w


if __name__ == '__main__':
    tran_data, labels = data_set()
    # 一维数据验证
    # tran_data = [random.randint(1, 50) for i in range(11)]
    # for i in tran_data:
    #     print(i)
    w = knn(tran_data, method=2)
    for i in w:
        print(i)
    data_show(data_set(), w)
