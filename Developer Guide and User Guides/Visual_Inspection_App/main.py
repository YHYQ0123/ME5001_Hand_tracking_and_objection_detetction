import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import VisualInspectionApp
import config # Access config if needed (e.g., for app name)

# Optional: Add better exception handling
# def exception_hook(exctype, value, traceback):
#     print("Unhandled exception:", exctype, value, traceback)
#     # Potentially log to file
#     sys.__excepthook__(exctype, value, traceback)
#     sys.exit(1)
# sys.excepthook = exception_hook

def main():
    """Main application entry point."""
    print(f"Starting {config.APP_NAME}...")
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)

    main_window = VisualInspectionApp()
    main_window.show()

    print("Application event loop started.")
    exit_code = app.exec_()
    print(f"Application finished with exit code: {exit_code}")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()