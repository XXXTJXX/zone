import numpy as np

def readPoints(filename):
    # 读取散点数据
    x = []
    y = []
    z = []
    d = []

    with open(filename, "r") as f:
        while True:
            lines = f.readline()
            if not lines:
                break
                pass
            x_tmp, y_tmp, z_tmp, d_tmp = [float(i) for i in lines.split()]
            x.append(x_tmp)
            y.append(y_tmp)
            z.append(z_tmp)
            d.append(d_tmp)
            pass
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        d = np.array(d)
    length = len(x)

    data = np.zeros((length, 4))
    for i in range(0, length - 1):
        data[i][0] = x[i]
        data[i][1] = y[i]
        data[i][2] = z[i]
        data[i][3] = d[i]

    z = 10 * z
    return x, y, z, d, data
