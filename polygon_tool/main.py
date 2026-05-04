import sys
from PySide6.QtWidgets import QApplication
from main_window import LabelDrawingUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelDrawingUI()
    window.show()
    sys.exit(app.exec())
