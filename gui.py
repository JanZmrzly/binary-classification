import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from prettytable import PrettyTable

from neural_network import NeuralNetwork

NET = NeuralNetwork()

class OpenCsv(QtWidgets.QWidget):
    disp_data = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.data_loaded = False
        self.label = QtWidgets.QLabel("Press the button to upload data")
        self.upload_button = QtWidgets.QPushButton("Upload data")
        self.upload_button.clicked.connect(self.upload_data)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

    def upload_data(self):
        self.file_dialog = QtWidgets.QFileDialog(self)
        self.file_dialog.setNameFilter("CSV Files (*.csv)")
        data_path, _ = self.file_dialog.getOpenFileName(self, caption="Open CSV file", filter="CSV Files (*.csv)")
        _,self.data_loaded = NET.load_data(data_path)
        self.disp_data.emit(self.data_loaded)

class DataFramePreview(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel("Uploaded data preview")
        self.table = QtWidgets.QTableWidget()
        # self.table.setMinimumSize(500,200)
        self.table.setMinimumWidth(500)
        self.table.setFixedHeight(250)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(bool)
    def disp_data(self, data_loaded):
        if data_loaded:
            df = NET.df.head()
            self.table.setRowCount(df.shape[0])
            self.table.setColumnCount(df.shape[1])
            self.table.setHorizontalHeaderLabels(df.columns)

            for row in range(df.shape[0]):
                for column in range(df.shape[1]):
                    item = QtWidgets.QTableWidgetItem(str(df.iloc[row, column]))
                    self.table.setItem(row, column, item)
            
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            left = self.table.verticalHeader()
            left.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            left.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

class ModelLayer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.activation_label = QtWidgets.QLabel("Select activation")
        self.neurons_label = QtWidgets.QLabel("Select the number of neurons")
        self.activation = QtWidgets.QComboBox()
        self.activation.addItem("relu")
        self.activation.addItem("sigmoid")

        self.neurons = QtWidgets.QSpinBox()
        self.neurons.setRange(1, 512)

        # Set layout
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.neurons_label)
        left_layout.addWidget(self.neurons)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.activation_label)
        right_layout.addWidget(self.activation)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        self.setLayout(layout)

class ModelConfiguration(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.label = QtWidgets.QLabel("Configurate your neural network model")
        self.add = QtWidgets.QPushButton("Add layer")
        self.add.clicked.connect(self.add_layer)
        self.remove = QtWidgets.QPushButton("Remove layer")
        self.remove.clicked.connect(self.remove_layer)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.add)
        btn_layout.addWidget(self.remove)

        # Set layout
        self.layer_layout = QtWidgets.QVBoxLayout()
        self.set_default_layers()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)
        layout.addLayout(self.layer_layout)
        self.setLayout(layout)
    
    def set_default_layers(self):
        layers = [
            (11, "relu"),
            (512, "relu"),
            (128, "relu"),
            (1, "sigmoid"),
        ]
        for layer in layers:
            neurons, activation = layer
            index = 0
            if activation == "sigmoid":
                index = 1
            l = ModelLayer()
            l.activation.setCurrentIndex(index)
            l.neurons.setValue(neurons)
            self.layer_layout.addWidget(l)
            self.layers.append(l)

    def build_model(self):
        if self.layers != []:
            NET.build_model(self.layers)

    def add_layer(self):
        l = ModelLayer()
        self.layer_layout.addWidget(l)
        self.layers.append(l)
    
    def remove_layer(self):
        pass

class Plot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_data(self, results, items_length):
        items_length = np.array(items_length)
        
        results = [result[1] for result in results]
        result_matrix = np.array(results)*np.array(items_length)
        results = result_matrix.T

        names = list(range(results.shape[1]))
        width = 0.5
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        left = np.zeros(results.shape[1])
        for pattern in results:
            ax.barh(names, pattern, width, left=left)
            left += pattern
        
        ax.set_xlabel('Length')
        ax.set_title("Patterns")
        ax.legend(items_length)
        self.canvas.draw()

class Results(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.result_box = QtWidgets.QTextEdit(self)
        self.result_box.setMinimumSize(300,270)
        self.result_box.setMaximumSize(400,300)
        self.result_box.setReadOnly(True)
        self.result_box.setLineWrapColumnOrWidth(1000) #Here you set the width you want
        self.result_box.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.FixedPixelWidth)
        self.result_box.setPlainText("The results will be displayed here")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.result_box)
        self.setLayout(layout)
        
    def print_results(self, semiproduct_count, results, items_length):
        self.result_box.clear()
        self.result_box.insertPlainText("Best solution found\n\n")
        self.result_box.insertPlainText(f"""{semiproduct_count} pieces of the default length will be needed for production\n\n""")
        self.print_table(results, items_length)

    def print_table(self, result):
        self.result_box.insertPlainText(f"{result}")     
       
class UiWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1000, 600)
        self.setWindowTitle("Binary Classification")

        self.csv_open = OpenCsv()
        self.data_preview = DataFramePreview()
        self.data_preview.setMaximumHeight(300)
        self.csv_open.disp_data.connect(self.data_preview.disp_data)
        self.model_configuration = ModelConfiguration()
        self.calculate_button = QtWidgets.QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate)      
        
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.csv_open)
        left_layout.addWidget(self.data_preview)
        left_layout.addWidget(self.model_configuration)
        left_layout.addWidget(self.calculate_button)
    
        self.plot = Plot()
        self.results = Results()

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.plot)
        right_layout.addWidget(self.results)
      
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        self.setLayout(main_layout)

        icon = QtGui.QIcon("")
        self.setWindowIcon(icon)

    def calculate(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UiWindow()
    window.show()
    sys.exit(app.exec_())