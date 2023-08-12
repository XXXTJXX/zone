def ColorMapping(m_colors):
    '''
    根据设定的控制点个数完成对长度为256的色表的插值
    :param m_colors: 种子点颜色，值形式：[[R,G,B],[R,G,B],...,[R,G,B]]
                     初始默认值为：[[255, 0.0, 255], [0.0, 255, 255],
                     [0.0, 0.0, 255], [0.0, 255, 0.0], [255, 255, 0.0],
                     [255, 125, 0.0], [255, 0.0, 0.0]]
    :return: 返回TVTK色表LookupTable，长度为256
    '''
    # 初始化色表
    LookupTable = []
    delcolor = 256/(len(m_colors) - 1)

    # 在各个控制点的范围内实现0-256的颜色赋值
    for j in range(256):
        for i in range(len(m_colors) - 1):
            if j >= i * delcolor and j <= (i + 1) * delcolor:
                LookupTable.append(insert_one_color(m_colors[i], m_colors[i + 1], (j - i * delcolor) / delcolor))

    return LookupTable


def insert_one_color(color1, color2, D):
    '''
    根据两端点的颜色对其中某一点进行颜色插值（线性插值）
    :param color1: 起始颜色(R,G,B)(0~255)
    :param color2: 终止颜色(R,G,B)(0~255)
    :param D: 插值颜色到起始颜色的距离/终止颜色到起始颜色的距离
    :return: 返回该点插值后的颜色
    '''
    r1 = color1[0]
    g1 = color1[1]
    b1 = color1[2]
    r2 = color2[0]
    g2 = color2[1]
    b2 = color2[2]

    rd = r1 + (r2 - r1) * D
    gd = g1 + (g2 - g1) * D
    bd = b1 + (b2 - b1) * D

    color = [rd, gd, bd]

    return color


