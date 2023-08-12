from PyQt5.QtWidgets import QLabel,QHBoxLayout,QPushButton,QColorDialog,QMainWindow,QLineEdit,QDialog,QApplication
from PyQt5.QtGui import QPainter,QCursor,QMouseEvent,QFont,QPaintEvent,QColor,QLinearGradient,QPolygon
from PyQt5.QtCore import QPoint,QRect,pyqtSignal
import sys
class SetControlNumDialog(QDialog):
    _signal = pyqtSignal(int,int)
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle("Set Control Num")

        self.label = QLabel()
        self.label.setText("颜色控制点数:")

        self.LineEdit = QLineEdit()
        self.LineEdit.textEdited.connect(self.LineEditSlot)

        self.setControlNum = QPushButton()
        self.setControlNum.setText("确认")
        self.setControlNum.clicked.connect(self.setControlNumSlot)

        self.Hlayout = QHBoxLayout()
        self.Hlayout.addWidget(self.label)
        self.Hlayout.addWidget(self.LineEdit)
        self.Hlayout.addWidget(self.setControlNum)

        self.setLayout(self.Hlayout)
        self.ControlNum = 7
        self.IsAccept = 0


    def LineEditSlot(self):
        if (self.LineEdit.text() != ""):
            self.ControlNum = int(self.LineEdit.text())

    def setControlNumSlot(self):
        self.IsAccept = 1
        self._signal.emit(self.ControlNum,self.IsAccept)
        self.accept()



