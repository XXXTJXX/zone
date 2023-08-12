# 等值面算法中的数据结构

class Point:
    def __init__(self, xuhao, x, y, z, d):
        self.Xuhao = xuhao  # 索引编号
        self.x = x  # X坐标
        self.y = y  # Y坐标
        self.z = z  # Z坐标
        self.d = d  # 标量值d

    def __eq__(self, other):
        # 重载"=="判断符号
        if (self.x == other.x) and (self.y == other.y):
            return 1
        else:
            return 0


class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other):
        # 重载"=="判断符号
        if (self.p1 == other.p2 and
                self.p2 == other.p1):
            return 1
        elif (self.p1 == other.p1 and
              self.p2 == other.p2):
            return 1
        else:
            return 0

# 三角形面片
class Triangle:
    def __init__(self, point_b, point_e, point_s):
        self.point0 = point_b
        self.point1 = point_e
        self.point2 = point_s
        self.BaseLine = Edge(point_b, point_e)
        self.newLine1 = Edge(point_e, point_s)
        self.newLine2 = Edge(point_s, point_b)

    def __eq__(self, other):
        # 重载
        if(self.BaseLine==other.BaseLine and self.newLine1==other.newLine1 and self.newLine2==other.newLine2):
            return 1
        elif(self.BaseLine==other.BaseLine and self.newLine1==other.newLine2 and self.newLine2==other.newLine1):
            return 1
        elif(self.BaseLine==other.newLine1 and self.newLine1==other.newLine2 and self.newLine2==other.BaseLine):
            return 1
        elif(self.BaseLine==other.newLine1 and self.newLine1==other.BaseLine and self.newLine2==other.newLine2):
            return 1
        elif(self.BaseLine==other.newLine2 and self.newLine1==other.newLine1 and self.newLine2==other.BaseLine):
            return 1
        elif(self.BaseLine==other.newLine2 and self.newLine1==other.BaseLine and self.newLine2==other.newLine1):
            return 1
        else:
            return 0





