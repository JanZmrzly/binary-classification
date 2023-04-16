import sys
import os
import qdarkgraystyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')
plt.style.use("fivethirtyeight")

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
        layout.setContentsMargins(10,0,0,0)
        layout.addWidget(self.label)
        layout.addWidget(self.upload_button)
        
        self.setFixedHeight(55)
        self.setLayout(layout)

    def upload_data(self):
        self.file_dialog = QtWidgets.QFileDialog(self)
        data_path, _ = self.file_dialog.getOpenFileName(self, caption="Open CSV file", filter="CSV Files (*.csv)")
        _,self.data_loaded = NET.load_data(data_path)
        self.disp_data.emit(self.data_loaded)

class ModelOperations(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.label_load = QtWidgets.QLabel("Load trained model")
        self.load_button = QtWidgets.QPushButton("Load")
        self.label_save = QtWidgets.QLabel("Save trained model")
        self.save_button = QtWidgets.QPushButton("Save")   
        self.load_button.clicked.connect(self.load)
        self.save_button.clicked.connect(self.save)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10,0,0,0)
        layout.addWidget(self.label_load)
        layout.addWidget(self.load_button)
        layout.addWidget(self.label_save)
        layout.addWidget(self.save_button)

        self.setFixedHeight(120)
        self.setLayout(layout)

    def load(self):
        self.file_dialog = QtWidgets.QFileDialog(self)
        path, _ = self.file_dialog.getOpenFileName(self, caption="Open Neural Network Model", filter="Neural Network (*.h5)")
        print(path)
        NET.load_model(path)

    def save(self):
        dir_name = "models"
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.abspath(dir_name)
        # path = f"{os.path.abspath(dir_name)}\\".encode("cp1250")
        NET.save_model(path)

class DataFramePreview(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel("Uploaded data preview")
        self.table = QtWidgets.QTableWidget()
        self.table.setMinimumWidth(500)
        self.table.setFixedHeight(150)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.table)
        
        self.setFixedHeight(170)
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
        self.activation = QtWidgets.QComboBox()
        self.activation.addItem("relu")
        self.activation.addItem("sigmoid")

        self.neurons = QtWidgets.QSpinBox()
        self.neurons.setRange(1, 512)

        # Set layout
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,3,0,0)
        layout.addWidget(self.neurons)
        layout.addWidget(self.activation)
        self.setLayout(layout)

class ModelConfiguration(QtWidgets.QWidget):
    plot_history = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layers = []
        self.label = QtWidgets.QLabel("Configurate your neural network model")
        self.add = QtWidgets.QPushButton("Add layer")
        self.add.clicked.connect(self.add_layer)
        self.remove = QtWidgets.QPushButton("Remove layer")
        self.remove.clicked.connect(self.remove_layer)
        self.train = QtWidgets.QPushButton("Train model")
        self.train.clicked.connect(self.train_model)
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1,1000)
        self.epochs.setToolTip("Number of training epochs")
        self.epochs.setValue(10)

        #Scroll Area Properties
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.add)
        btn_layout.addWidget(self.remove)
        btn_layout.addWidget(self.epochs)
        btn_layout.addWidget(self.train)

        # Set layout
        self.layer_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.layer_layout.setContentsMargins(10,10,0,0)
        self.layer_layout.setSpacing(0)
        self.set_default_layers()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.scroll_area)

        self.setFixedHeight(200)
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
        if self.layers != []:
            l = self.layers.pop()
            self.layer_layout.removeWidget(l)
            l.deleteLater()
    
    def train_model(self):
        epochs = self.epochs.value()
        layers:list[tuple[int,str]] = []
        if self.layers != []:
            for layer in self.layers:
                activation = layer.activation.currentText()
                neurons = layer.neurons.value()
                layers.append((neurons, activation))

            if NET.df is not None:
                NET.build_model(layers)
                NET.compile_model(epochs)
                self.plot_history.emit()
            else:
                pass
                # [TODO]: Need To upload dataset

class Plot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def plot_history(self):
        if NET.model_trained:
            self.figure.clear()
            
            # Plotting Metrics Curve
            plots = [(311,'accuracy'), (312, 'mae'), (313, 'loss')]
            for plot in plots:
                pos, mplt = plot
                metric = NET.history.history[mplt]
                val_metric = NET.history.history[f"val_{mplt}"]
                epochs = range(len(metric))
                plt.subplot(pos)         
                plt.plot(epochs, metric, label=f"Training {mplt}")
                plt.plot(epochs, val_metric, label=f"Validation {mplt}")
                plt.legend(fontsize=6)
                plt.title(f"Training and Validation for {mplt}", fontsize=8)
                plt.tick_params(axis="both", labelsize=6)
                plt.tight_layout(pad=1)
            self.canvas.draw()

class Empty(QtWidgets.QWidget):
       def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

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
        self.model_operations = ModelOperations()
        self.data_preview = DataFramePreview()
        
        self.csv_open.disp_data.connect(self.data_preview.disp_data)
        self.empty = Empty()
        self.model_configuration = ModelConfiguration()
        
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.csv_open)
        left_layout.addWidget(self.model_operations)
        left_layout.addWidget(self.data_preview)
        left_layout.addWidget(self.model_configuration)
        left_layout.addWidget(self.empty)
    
        self.plot = Plot()
        self.model_configuration.plot_history.connect(self.plot.plot_history)
        self.results = Results()

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.plot)
        # right_layout.addWidget(self.results)
      
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        
        self.setLayout(main_layout)

        icon = QtGui.QIcon("")
        self.setWindowIcon(icon)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    window = UiWindow()
    window.show()
    sys.exit(app.exec_())