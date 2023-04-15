from PyQt5.QtWidgets import (QWidget, QSlider, QLineEdit, QLabel, QPushButton, QScrollArea,QApplication,
                             QHBoxLayout, QVBoxLayout, QMainWindow)
from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtWidgets, uic
import sys

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pb = QPushButton("Nějaký button")
        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pb)
        self.setLayout(layout)

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()              # The Vertical Box that contains the Horizontal Boxes of  labels and buttons
        self.mlayout = QVBoxLayout()
        self.pb = MyWidget()

        self.vbox.setSpacing(100)
        self.widget.setLayout(self.vbox)
        for i in range(1,50):
            object = QPushButton("TextLabel")
            self.vbox.addWidget(object)

        # self.widget.setLayout(self.vbox)

        #Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.mlayout.addWidget(self.pb)
        self.mlayout.addWidget(self.scroll)
        self.setLayout(self.mlayout)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        return

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()