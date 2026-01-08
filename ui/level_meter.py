"""
垂直レベルメーターウィジェット
リアルタイム音量表示とピークホールド機能
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QLinearGradient
import math


class VerticalLevelMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(30)
        self.setMinimumHeight(200)
        
        # レベル値
        self.current_level = 0.0  # 0.0 - 1.0
        self.peak_level = 0.0     # 0.0 - 1.0
        
        # 色設定
        self.bg_color = QColor(26, 26, 26)  # #1a1a1a
        self.border_color = QColor(85, 85, 85)  # #555555
        
        # グラデーション色（下から上へ）
        self.gradient_colors = [
            (0.0, QColor(0, 255, 0)),      # 緑
            (0.7, QColor(255, 255, 0)),    # 黄色
            (0.9, QColor(255, 165, 0)),    # オレンジ
            (1.0, QColor(255, 0, 0))       # 赤
        ]
        
        # dBスケール表示用
        self.db_labels = [
            (1.0, "0"),
            (0.9, "-6"),
            (0.8, "-12"),
            (0.6, "-18"),
            (0.4, "-24"),
            (0.2, "-36"),
            (0.1, "-48"),
            (0.0, "-∞")
        ]
    
    def set_level(self, current: float, peak: float):
        """レベル値を設定（0.0-1.0の範囲）"""
        self.current_level = max(0.0, min(1.0, current))
        self.peak_level = max(0.0, min(1.0, peak))
        self.update()  # 再描画をトリガー
    
    def paintEvent(self, event):
        """カスタム描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        meter_rect = rect.adjusted(5, 10, -15, -10)  # 右側にdBラベル用スペース
        
        # 背景描画
        painter.fillRect(meter_rect, self.bg_color)
        painter.setPen(self.border_color)
        painter.drawRect(meter_rect)
        
        # メーターの高さ
        meter_height = meter_rect.height()
        
        # 現在レベルの描画
        if self.current_level > 0:
            level_height = int(meter_height * self.current_level)
            level_rect = meter_rect.adjusted(1, meter_height - level_height + 1, -1, -1)
            
            # グラデーション作成
            gradient = QLinearGradient(0, level_rect.bottom(), 0, level_rect.top())
            
            for pos, color in self.gradient_colors:
                if pos <= self.current_level:
                    gradient.setColorAt(pos / self.current_level, color)
            
            painter.fillRect(level_rect, gradient)
        
        # ピークレベルの描画（細い線）
        if self.peak_level > 0:
            peak_y = meter_rect.bottom() - int(meter_height * self.peak_level)
            painter.setPen(QColor(255, 255, 255, 200))  # 半透明の白
            painter.drawLine(meter_rect.left() + 1, peak_y, meter_rect.right() - 1, peak_y)
        
        # dBスケール描画
        painter.setPen(QColor(204, 204, 204))  # #cccccc
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        
        for level_pos, db_text in self.db_labels:
            y_pos = meter_rect.bottom() - int(meter_height * level_pos)
            
            # 目盛り線
            painter.drawLine(meter_rect.right() + 1, y_pos, meter_rect.right() + 3, y_pos)
            
            # dBテキスト
            text_rect = painter.fontMetrics().boundingRect(db_text)
            painter.drawText(
                meter_rect.right() + 5,
                y_pos + text_rect.height() // 2,
                db_text
            )


class LevelMeterWidget(QWidget):
    """レベルメーターとラベルを含むウィジェット"""
    
    def __init__(self, title="LEVEL", parent=None):
        super().__init__(parent)
        self.setup_ui(title)
    
    def setup_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # タイトルラベル
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 9px;
                font-weight: bold;
                padding: 2px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # レベルメーター
        self.meter = VerticalLevelMeter()
        layout.addWidget(self.meter)
        
        # 数値表示ラベル
        self.value_label = QLabel("0.0")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-size: 8px;
                font-weight: bold;
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 2px;
                padding: 1px;
            }
        """)
        layout.addWidget(self.value_label)
    
    def set_level(self, current: float, peak: float):
        """レベル値を設定"""
        self.meter.set_level(current, peak)
        
        # dB値に変換して表示
        if current > 0:
            db_value = 20 * math.log10(current)
            db_value = max(db_value, -60.0)  # -60dB以下は-60dBとする
            self.value_label.setText(f"{db_value:.1f}")
        else:
            self.value_label.setText("-∞")


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
    from PySide6.QtCore import QTimer
    import random
    
    app = QApplication(sys.argv)
    
    # テスト用メインウィンドウ
    window = QMainWindow()
    window.setWindowTitle("Level Meter Test")
    window.setStyleSheet("background-color: #2b2b2b;")
    
    central_widget = QWidget()
    layout = QHBoxLayout(central_widget)
    
    # 複数のメーターを作成
    meters = []
    for i in range(4):
        meter = LevelMeterWidget(f"CH {i+1}")
        meters.append(meter)
        layout.addWidget(meter)
    
    window.setCentralWidget(central_widget)
    
    # ランダムレベル更新用タイマー
    def update_levels():
        for meter in meters:
            current = random.random() * 0.8
            peak = current + random.random() * 0.2
            meter.set_level(current, peak)
    
    timer = QTimer()
    timer.timeout.connect(update_levels)
    timer.start(50)  # 20fps
    
    window.show()
    sys.exit(app.exec())