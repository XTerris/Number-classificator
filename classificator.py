from PyQt5 import QtWidgets, QtCore
from sys import exit


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание цифр")
        self.resize(600, 400)
        
        self.main = QtWidgets.QWidget(self)
        self.grid = QtWidgets.QGridLayout()
        self.grid.setGeometry(QtCore.QRect(50, 50, 100, 100))
        
        self.load_button = QtWidgets.QPushButton("Загрузить изображение")
        self.recognize_button = QtWidgets.QPushButton("Распознать цифру")
        
        self.grid.addWidget(self.load_button, 0, 0, 2, 5)
        self.grid.addWidget(self.recognize_button, 0, 5, 2, 5)

        self.main.setLayout(self.grid)

    def onClick(self):
        print("Clicked")

app = QtWidgets.QApplication([])
main = MainWindow()
main.show()

exit(app.exec())
