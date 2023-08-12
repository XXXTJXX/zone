# 反比距离加权插值算法

import numpy as np

def IDW_InsertOne(x_insert, y_insert, x, y, z):
    '''
    根据原始数据对网格中某一个点进行插值
    :param x_insert: 待插值网格点的x坐标
    :param y_insert: 待插值网格点的y坐标
    :param x: 原始数据的x坐标列表
    :param y: 原始数据的y坐标列表
    :param z: 原始数据的z坐标列表
    :return: 待插值网格点的z坐标
    '''
    d = [] # 距离
    q = [] # 权重
    qsum = 0 # 权重和，分母
    z_sum = 0 # 分子
    z_insert = 0
    count = 0 # 计数器
    for i in range(len(x)):
        # 将待插入点与其他点的距离d转化成权重q
        d.append(np.sqrt((x[i] - x_insert) ** 2 + (y[i] - y_insert) ** 2))
        if d[i] == 0: # 若点之间重合
            # 待插入点的z值等于其他重合点的z值
            z_insert = z[i]
            count += 1
            break
        else:
            q.append((1 / d[i]) ** 2)

    if count == 0: # 若点之间没有重合
        # 利用反距离加权插值公式计算待插入点的z值
        for i in range(len(z)):
            z_sum = z[i] * q[i] + z_sum
        for i in range(len(q)):
            qsum = qsum + q[i]
        z_insert = z_sum / qsum

        return z_insert
    else:
        return z_insert

def IDW_main(x, y, z, d, row, col, vol):
    '''
    网格插值主函数（注意区分原始数据与网格数据！！！）
    :param x: 原始数据归一化后的x坐标列表
    :param y: 原始数据归一化后的x坐标列表
    :param z: 原始数据归一化后的x坐标列表
    :param d: 原始数据的d坐标列表
    :param row: 网格的行数
    :param col: 网格的列数
    :param vol: 立方体的层数
    :return: 输出的X,Y,Z为row*col大小的二维数组，分别代表网格中对应位置点的x,y,z坐标
    '''
    # 因为输入的.txt文件就是网格数据,故只需转成三维数据一一对应即可
    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    D = np.array(d)
    X = X.reshape(row, col, vol)
    Y = Y.reshape(row, col, vol)
    Z = Z.reshape(row, col, vol)
    D = D.reshape(row, col, vol)

    return X, Y, Z, D


def IDW_normalize(x, y, z, d):  # 多加了参数--d = f（x,y,z)
    '''
    坐标归一化
    :param x: 原始数据的x坐标列表
    :param y: 原始数据的y坐标列表
    :param z: 原始数据的z坐标列表
    :return: 返回归一化后的坐标列表
    '''
    # 求出最大、最小值
    x_max = max(x)
    y_max = max(y)
    z_max = max(z)
    x_min = min(x)
    y_min = min(y)
    z_min = min(z)

    # 归一化处理
    for i in range(len(x)):
        x[i] = (x[i] - x_min) / (x_max - x_min)
    for i in range(len(y)):
        y[i] = (y[i] - y_min) / (y_max - y_min)
    for i in range(len(z)):
        z[i] = (z[i] - z_min) / (z_max - z_min)

    return x, y, z, d











