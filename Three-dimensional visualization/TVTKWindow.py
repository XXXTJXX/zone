from PyQt5.QtWidgets import QWidget, QVBoxLayout
from traits.api import *
from traitsui.api import *
from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.pyface.scene import Scene
from tvtk.pyface.scene_model import SceneModel
from IDW import *
from contour_mapping import *
from ColorMapping import *
import numpy as np
from mayavi import mlab


class TVTKViewer(HasTraits):
    scene = Instance(SceneModel, ())  # SceneModel表示TVTK的场景模型

    # 建立视图布局
    view = View(
        Item(name='scene',
             editor=SceneEditor(scene_class=Scene),  # 设置mayavi的编辑器，让它能正确显示scene所代表的模型
             resizable=True,
             ),
        resizable=True
    )


class TVTKQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)  # 定义布局
        self.viewer = TVTKViewer()  # 定义TVTK界面对象
        ui = self.viewer.edit_traits(parent=self, kind='subpanel').control  # 使TVTK界面对象可以被调用
        layout.addWidget(ui)  # 将TVTK界面对象放到布局中

        self.x = []
        self.y = []
        self.z = []
        self.d = []
        self.colors = [[255, 0.0, 255], [0.0, 255, 255], [0.0, 0.0, 255], [0.0, 255, 0.0], [255, 255, 0.0],
                       [255, 125, 0.0], [255, 0.0, 0.0]]

        self.row = 50
        self.col = 50
        self.vol = 50
        self.contour_num = 10  # 等值线数
        self.IDW_form = []

    def plot_IDW(self):
        x, y, z, d = IDW_normalize(self.x, self.y, self.z, self.d)
        X, Y, Z, D = IDW_main(x, y, z, d, self.row, self.col, self.vol)

        zmax = max(Z.flatten())
        zmin = min(Z.flatten())

        row = len(Z)
        col = len(Z[0])
        vol = len(Z[0][0])

        points = tvtk.Points()
        points.number_of_points = row * col * vol
        number_points = 0
        for k in range(0, vol):
            for i in range(0, row):
                for j in range(0, col):
                    points.set_point(number_points, X[i][j][k], Y[i][j][k], Z[i][j][k])
                    number_points = number_points + 1

        faces = []
        for k in range(0, vol - 1):
            for i in range(0, row - 1):
                for j in range(0, col - 1):
                    indexTTL = i * col + j + (k + 1) * row * col
                    indexTTR = i * col + j + 1 + (k + 1) * row * col
                    indexBBL = indexTTL + col
                    indexBBR = indexTTR + col
                    indexTL = i * col + j + k * row * col
                    indexTR = i * col + j + 1 + k * row * col
                    indexBL = indexTL + col
                    indexBR = indexTR + col
                    # tmp = [indexTL, indexTTL, indexTTR, indexBBL, indexBBR, indexTR, indexBR, indexBL, indexTL]
                    tmp = [indexTL, indexTTL, indexBBL, indexBL, indexTL, indexTTR, indexBBR, indexBR, indexTR]
                    faces.append(tmp)

        polyData = tvtk.PolyData(points=points, lines=faces)
        z_new = Z.flatten()
        polyData.point_data.scalars = z_new

        polyDataMapper = tvtk.PolyDataMapper()
        polyDataMapper.set_input_data(polyData)
        polyDataMapper.scalar_range = (zmin, zmax)

        colorTable = tvtk.LookupTable()
        colorTable.number_of_table_values = 256

        colors_new = ColorMapping(self.colors)
        for i in range(len(colors_new)):
            colorTable.set_table_value(i, colors_new[i][0] / 255, colors_new[i][1] / 255, colors_new[i][2] / 255)

        polyDataMapper.lookup_table = colorTable

        self.actor_grid = tvtk.Actor(mapper=polyDataMapper)
        self.viewer.scene.add_actor(self.actor_grid)

        # self.actor = tvtk.Actor(mapper=polyDataMapper)
        # self.viewer.scene.add_actor(self.actor)

    def plot_surface(self):
        x, y, z, d = IDW_normalize(self.x, self.y, self.z, self.d)
        X, Y, Z, D = IDW_main(x, y, z, d, self.row, self.col, self.vol)
        zuixiao = np.min(d)
        print(zuixiao)
        obj = mlab.contour3d(D, colormap='hot')

        '''
        系统自带提取等值面
        '''
        obj.contour.maximum_contour = 0.287  # 等值面的上限值为 0.287
        obj.contour.number_of_contours = 3  # 在M 小值到 0.287 之间绘制 2 个等值面
        obj.actor.property.opacity = 0.4  # 0.4

        mlab.show()

    def plot_contour(self):
        x, y, z, d = IDW_normalize(self.x, self.y, self.z, self.d)
        X, Y, Z, D = IDW_main(x, y, z, d, self.row, self.col, self.vol)
        c = isocontour(X, Y, Z, D, self.contour_num, self.row, self.col, self.vol)  # 计算等值面

        zmax = max(self.z)
        zmin = min(self.z)

        Triangles = List2Triangles(c)
        x = []
        y = []
        z = []
        for i in c:
            a = i.point0
            b = i.point1
            c = i.point2
            x.append(a.x)
            y.append(a.y)
            z.append(a.z)
            x.append(b.x)
            y.append(b.y)
            z.append(b.z)
            x.append(c.x)
            y.append(c.y)
            z.append(c.z)
        points_painter = np.vstack([x, y, z]).T
        polyData = tvtk.PolyData(points=points_painter, polys=Triangles)
        # polyData.point_data.scalars = z

        polyDataMapper = tvtk.PolyDataMapper()
        polyDataMapper.set_input_data(polyData)
        polyDataMapper.scalar_range = (zmin, zmax)

        self.actor_contour = tvtk.Actor(mapper=polyDataMapper)
        # self.actor_contour = tvtk.Renderer(background = (1,0,0))
        self.actor_contour.property.color = (1, 0, 0)
        self.viewer.scene.add_actor(self.actor_contour)

    def hide_contour(self):
        self.viewer.scene.remove_actor(self.actor_contour)
        self.viewer.scene.remove_actor(self.actor_contour_)
        del self.actor_contour
        del self.actor_contour_

    def getRowCol(self, row, col, vol, contour):
        self.row = row
        self.col = col
        self.vol = vol
        self.contour_num = contour
        if self.IDW_form == "Grid":
            self.viewer.scene.remove_actor(self.actor_grid)
        elif self.IDW_form == "Surface":
            self.viewer.scene.remove_actor(self.actor_surface)

        if self.IDW_form == "Grid":
            self.plot_IDW()
        elif self.IDW_form == "Surface":
            self.plot_surface()

        if 'actor_contour' in dir(self):
            self.viewer.scene.remove_actor(self.actor_contour)
            self.viewer.scene.remove_actor(self.actor_contour_)
            self.plot_contour()

    def getColors(self, colors_list):
        self.colors = colors_list
        if self.IDW_form == "Grid":
            self.viewer.scene.remove_actor(self.actor_grid)
        elif self.IDW_form == "Surface":
            self.viewer.scene.remove_actor(self.actor_surface)

        if self.IDW_form == "Grid":
            self.plot_IDW()
        elif self.IDW_form == "Surface":
            self.plot_surface()
