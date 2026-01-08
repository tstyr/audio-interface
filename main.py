"""
UAD Console風オーディオ管理アプリケーション
メインエントリーポイント
"""
import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def check_dependencies():
    """依存関係チェック"""
    try:
        import sounddevice as sd
        import numpy as np
        return True
    except ImportError as e:
        QMessageBox.critical(None, "依存関係エラー", 
                           f"必要なライブラリが見つかりません:\n{e}\n\n"
                           "pip install -r requirements.txt を実行してください。")
        return False


def main():
    """メイン関数"""
    # アプリケーション作成
    app = QApplication(sys.argv)
    app.setApplicationName("UAD Console風オーディオミキサー")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Audio Mixer Pro")
    
    # 高DPI対応
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 依存関係チェック
    if not check_dependencies():
        return 1
    
    try:
        # メインウィンドウ作成・表示
        window = MainWindow()
        window.show()
        
        # イベントループ開始
        return app.exec()
        
    except Exception as e:
        QMessageBox.critical(None, "アプリケーションエラー", 
                           f"アプリケーションの起動に失敗しました:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())