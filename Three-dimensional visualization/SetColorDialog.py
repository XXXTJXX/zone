from PyQt5.QtWidgets import QLabel,QHBoxLayout,QPushButton,QColorDialog,QMainWindow,QLineEdit,QDialog,QApplication
from PyQt5.QtGui import QPainter,QCursor,QMouseEvent,QFont,QPaintEvent,QColor,QLinearGradient,QPolygon
from PyQt5.QtCore import QPoint,QRect,pyqtSignal
from ColorMapping import insert_one_color
import sys

class SetColorDialog(QDialog):
    _signal = pyqtSignal(list)
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle("Color Setting")

        self.setGeometry(700, 400, 800, 100)
        self.setFixedSize(800, 100)

        self.ft = QFont()
        self.ft.setPointSize(14)

        self.label0 = QLabel("0",self)
        self.label0.setGeometry(QRect(30, 10, 20, 30))
        self.label0.setFont(self.ft)

        self.label1 = QLabel("1",self)
        self.label1.setGeometry(QRect(760, 10, 20, 30))
        self.label1.setFont(self.ft)

        self.setColor = QPushButton(self)
        self.setColor.setText("确认")
        self.setColor.setGeometry(350,60,80,30)
        self.setColor.clicked.connect(self.SetColorSlot)
        self.ControlNum = 7

        self.m_colors = [[255, 0.0, 255], [0.0, 255, 255], [0.0, 0.0, 255], [0.0, 255, 0.0], [255, 255, 0.0],
                       [255, 125, 0.0], [255, 0.0, 0.0]]

        self.IsAccept = 0



    def paintEvent(self,QPaintEvent):
        pen = QPainter()
        color_num = len(self.m_colors)
        delta = float(1.0/(color_num-1))
        linearGradient = QLinearGradient(50,75,750,75)
        for i in range(color_num):
            linearGradient.setColorAt(delta * i, QColor(self.m_colors[i][0],self.m_colors[i][1],self.m_colors[i][2]))
        pen.begin(self)
        pen.setBrush(linearGradient)
        pen.drawRect(50, 10, 700, 30)
        for i in range(color_num):
            pen.setPen(QColor(self.m_colors[i][0],self.m_colors[i][1],self.m_colors[i][2]))
            pen.setBrush(QColor(self.m_colors[i][0],self.m_colors[i][1],self.m_colors[i][2]))
            triPoints = []
            triPoints.append(QPoint(50 + 700 * delta*i, 40))
            triPoints.append(QPoint(40 + 700 * delta*i, 50))
            triPoints.append(QPoint(60 + 700 * delta*i, 50))
            cursor = QPolygon(triPoints)
            pen.drawPolygon(cursor)
        pen.end()

    def mousePressEvent(self,event):
        color_num = len(self.m_colors)
        delta = float(700/(color_num - 1))
        curPos = event.pos()
        xPos = curPos.x()
        yPos = curPos.y()
        nomxPos = (xPos-50)/delta
        xF = round(nomxPos)
        if (yPos>40)&(yPos<50):
            if ((xF * delta + 50 - 10) < xPos) & ((xF * delta + 50 + 10) > xPos):
                color = QColorDialog.getColor(QColor(self.m_colors[xF][0],self.m_colors[xF][1],self.m_colors[xF][2]), self)
                if (color.isValid()):
                    self.m_colors[xF] = [color.red(),color.green(),color.blue()]
                    self.update()

    def SetColorSlot(self):
        self._signal.emit(self.m_colors)


    def getControlNum(self,ControlNum,IsAccept):
        self.ControlNum = ControlNum
        self.IsAccept = IsAccept
        self.m_colors = self.ColorResampling(self.m_colors,self.ControlNum)
        self.update()

    def ColorResampling(self,m_colors,ControlNum):
        colors_new = []
        colors_length = len(m_colors)
        delta = float(255 / (colors_length - 1))
        for i in range(colors_length - 1):
            color_num1 = int(i * delta)
            color_num2 = int((i + 1) * delta)
            for j in range(color_num1, color_num2):
                D = (j - color_num1) / (color_num2 - color_num1)
                color_insert = insert_one_color(m_colors[i], m_colors[i + 1], D)
                colors_new.append(color_insert)
        colors_new.append(m_colors[-1])

        delta_new = float(255 / (ControlNum - 1))
        colors_result = []
        for i in range(ControlNum):
            ids = int(i * delta_new)
            colors_result.append(colors_new[ids])
        return colors_result







