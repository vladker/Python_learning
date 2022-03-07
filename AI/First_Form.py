from PyQt5 import QtWidgets, uic
import sys

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi('AI_Design.ui')
window.show()
sys.exit(app.exec_())
