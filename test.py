from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

app = QApplication([])

# Vytvoření widgetu a nastavení layoutu
widget = QWidget()
layout = QVBoxLayout(widget)

# Přidání labelů do layoutu
layout.addWidget(QLabel("První řádek"))
layout.addWidget(QLabel("Druhý řádek"))
layout.addWidget(QLabel("Třetí řádek"))

# Nastavení zarovnání widgetů shora
layout.setAlignment(Qt.AlignTop)

widget.show()
app.exec_()