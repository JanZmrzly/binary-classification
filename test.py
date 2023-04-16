from PyQt5 import QtCore, QtWidgets

class MonitorThread(QtCore.QThread):
    value_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.is_running = True

    def run(self):
        while self.is_running:
            # Sledujte změnu hodnoty
            new_value = self.get_current_value()
            if new_value != self.value:
                self.value = new_value
                # Pokud se hodnota změní, vytvořte signál s novou hodnotou
                self.value_changed.emit(self.value)
            # Počkejte krátkou chvíli, než to znovu provedete
            self.msleep(50)

    def get_current_value(self):
        # Získat aktuální hodnotu, kterou sledujeme (například ze sítě)
        pass

    def stop(self):
        self.is_running = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QtWidgets.QLabel("Hodnota: 0", self)
        self.thread = MonitorThread(self)
        self.thread.value_changed.connect(self.update_label)
        self.thread.start()

    def update_label(self, value):
        self.label.setText(f"Hodnota: {value}")
