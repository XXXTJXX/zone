from PyQt5.QtWidgets import QLineEdit,QVBoxLayout,QDialog,QLabel,QPushButton
from PyQt5.QtCore import pyqtSignal


class SetRowColDialog(QDialog):
    _signal = pyqtSignal(int, int, int)
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.flag_set = 0

        self.Row_Label = QLabel()
        self.Row_Label.setText("规则网格行数:")

        self.Row_LineEdit = QLineEdit()

        self.Col_Label = QLabel()
        self.Col_Label.setText("规则网格列数:")

        self.Col_LineEdit = QLineEdit()

        self.Contour_Label = QLabel()
        self.Contour_Label.setText("等高线级别数:")

        self.Contour_LineEdit = QLineEdit()

        self.setRowCol =  QPushButton()
        self.setRowCol.setText("确认")

        self.Row_LineEdit.textEdited.connect(self.Row_LineEditSlot)
        self.Col_LineEdit.textEdited.connect(self.Col_LineEditSlot)
        self.Contour_LineEdit.textEdited.connect(self.Contour_LineEditSlot)
        self.setRowCol.clicked.connect(self.SetSlot)

        self.Hlayout = QVBoxLayout()
        self.Hlayout.addWidget(self.Row_Label)
        self.Hlayout.addWidget(self.Row_LineEdit)
        self.Hlayout.addWidget(self.Col_Label)
        self.Hlayout.addWidget(self.Col_LineEdit)
        self.Hlayout.addWidget(self.Contour_Label)
        self.Hlayout.addWidget(self.Contour_LineEdit)
        self.Hlayout.addWidget(self.setRowCol)

        self.setLayout(self.Hlayout)
        self.setWindowTitle("Set Row Col Contour")


    def Row_LineEditSlot(self):
        if (self.Row_LineEdit.text() != ""):
            self.row = int(self.Row_LineEdit.text())

    def Col_LineEditSlot(self):
        if (self.Col_LineEdit.text() != ""):
            self.col = int(self.Col_LineEdit.text())

    def Contour_LineEditSlot(self):
        if (self.Contour_LineEdit.text() != ""):
            self.contour = int(self.Contour_LineEdit.text())


    def SetSlot(self):
        if (self.Row_LineEdit.text() != "") and (self.Col_LineEdit.text() != "") and (self.Contour_LineEdit.text() != ""):
            self._signal.emit(self.row, self.col, self.contour)










