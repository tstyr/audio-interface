"""
UAD Console風カスタムUIコンポーネント
高級感のあるスキューモーフィックデザイン
"""
import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QSlider, QFrame)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QLinearGradient, 
                           QRadialGradient, QConicalGradient, QFont, QFontMetrics,
                           QPainterPath, QPolygonF)
import random


class AnalogKnob(QWidget):
    """UAD風アナログノブウィジェット"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, size=80):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        
        # 値の範囲
        self.min_value = 0.0
        self.max_value = 100.0
        self.current_value = 50.0
        
        # 角度の範囲（-135度から+135度、270度の範囲）
        self.min_angle = -135
        self.max_angle = 135
        
        # マウス操作用
        self.dragging = False
        self.last_mouse_y = 0
        self.sensitivity = 0.5
        
        # 外観設定
        self.knob_size = size - 20
        self.center_x = size // 2
        self.center_y = size // 2
    
    def setValue(self, value):
        """値を設定"""
        value = max(self.min_value, min(self.max_value, value))
        if value != self.current_value:
            self.current_value = value
            self.update()
            self.valueChanged.emit(value)
    
    def getValue(self):
        """現在の値を取得"""
        return self.current_value
    
    def mousePressEvent(self, event):
        """マウス押下時"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_y = event.y()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """マウス移動時"""
        if self.dragging:
            delta_y = self.last_mouse_y - event.y()  # 上に動かすと値が増加
            value_change = delta_y * self.sensitivity
            new_value = self.current_value + value_change
            self.setValue(new_value)
            self.last_mouse_y = event.y()
    
    def mouseReleaseEvent(self, event):
        """マウス離した時"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """マウスホイール"""
        delta = event.angleDelta().y() / 120.0  # 標準的なホイールの1ステップ
        value_change = delta * 2.0
        new_value = self.current_value + value_change
        self.setValue(new_value)
    
    def paintEvent(self, event):
        """カスタム描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        center = QPoint(self.center_x, self.center_y)
        
        # 外側のリング（目盛り）を描画
        self.draw_scale_ring(painter, center)
        
        # ノブ本体を描画
        self.draw_knob_body(painter, center)
        
        # インジケーター（針）を描画
        self.draw_indicator(painter, center)
        
        # 中央のキャップを描画
        self.draw_center_cap(painter, center)
    
    def draw_scale_ring(self, painter, center):
        """目盛りリングを描画"""
        outer_radius = self.knob_size // 2 + 8
        inner_radius = self.knob_size // 2 + 4
        
        # 目盛りの数
        tick_count = 21  # -10から+10まで21個
        
        for i in range(tick_count):
            # 角度計算
            angle_ratio = i / (tick_count - 1)
            angle = self.min_angle + (self.max_angle - self.min_angle) * angle_ratio
            angle_rad = math.radians(angle)
            
            # 目盛りの位置
            x1 = center.x() + inner_radius * math.cos(angle_rad)
            y1 = center.y() + inner_radius * math.sin(angle_rad)
            x2 = center.x() + outer_radius * math.cos(angle_rad)
            y2 = center.y() + outer_radius * math.sin(angle_rad)
            
            # 色設定（オレンジ）
            if i == tick_count // 2:  # 中央の目盛り
                painter.setPen(QPen(QColor(255, 140, 0), 2))
            else:
                painter.setPen(QPen(QColor(255, 140, 0), 1))
            
            painter.drawLine(x1, y1, x2, y2)
    
    def draw_knob_body(self, painter, center):
        """ノブ本体を描画"""
        radius = self.knob_size // 2
        
        # 外側の影
        shadow_gradient = QRadialGradient(center.x() + 2, center.y() + 2, radius + 4)
        shadow_gradient.setColorAt(0, QColor(0, 0, 0, 100))
        shadow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(shadow_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center.x() - radius - 2, center.y() - radius - 2, 
                          (radius + 4) * 2, (radius + 4) * 2)
        
        # メインのノブ（金属質感）
        main_gradient = QRadialGradient(center.x() - radius//3, center.y() - radius//3, radius * 1.5)
        main_gradient.setColorAt(0, QColor(120, 120, 120))  # ハイライト
        main_gradient.setColorAt(0.3, QColor(80, 80, 80))   # 中間
        main_gradient.setColorAt(0.7, QColor(50, 50, 50))   # 暗部
        main_gradient.setColorAt(1, QColor(30, 30, 30))     # 最暗部
        
        painter.setBrush(QBrush(main_gradient))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawEllipse(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        
        # 内側のリング
        inner_radius = radius - 8
        inner_gradient = QRadialGradient(center.x(), center.y(), inner_radius)
        inner_gradient.setColorAt(0, QColor(60, 60, 60))
        inner_gradient.setColorAt(1, QColor(40, 40, 40))
        
        painter.setBrush(QBrush(inner_gradient))
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.drawEllipse(center.x() - inner_radius, center.y() - inner_radius, 
                          inner_radius * 2, inner_radius * 2)
    
    def draw_indicator(self, painter, center):
        """インジケーター（針）を描画"""
        # 現在の値から角度を計算
        value_ratio = (self.current_value - self.min_value) / (self.max_value - self.min_value)
        angle = self.min_angle + (self.max_angle - self.min_angle) * value_ratio
        angle_rad = math.radians(angle)
        
        # 針の長さ
        indicator_length = self.knob_size // 2 - 12
        
        # 針の終点
        end_x = center.x() + indicator_length * math.cos(angle_rad)
        end_y = center.y() + indicator_length * math.sin(angle_rad)
        
        # 針を描画（太めの線）
        painter.setPen(QPen(QColor(255, 140, 0), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(center.x(), center.y(), end_x, end_y)
        
        # 針の先端に小さな円
        painter.setBrush(QBrush(QColor(255, 140, 0)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(end_x - 2, end_y - 2, 4, 4)
    
    def draw_center_cap(self, painter, center):
        """中央のキャップを描画"""
        cap_radius = 6
        
        # キャップのグラデーション
        cap_gradient = QRadialGradient(center.x() - 2, center.y() - 2, cap_radius)
        cap_gradient.setColorAt(0, QColor(100, 100, 100))
        cap_gradient.setColorAt(1, QColor(40, 40, 40))
        
        painter.setBrush(QBrush(cap_gradient))
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.drawEllipse(center.x() - cap_radius, center.y() - cap_radius, 
                          cap_radius * 2, cap_radius * 2)


class LEDMeter(QWidget):
    """LEDスタイルのレベルメーター"""
    
    def __init__(self, parent=None, width=20, height=200):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.current_level = 0.0  # 0.0 - 1.0
        self.peak_level = 0.0
        self.peak_hold_time = 30  # フレーム数
        self.peak_hold_counter = 0
        
        # LEDセグメント設定
        self.segment_count = 20
        self.segment_height = (height - 10) // self.segment_count
        self.segment_width = width - 6
        self.segment_spacing = 1
    
    def setLevel(self, level, peak=None):
        """レベルを設定（0.0-1.0）"""
        self.current_level = max(0.0, min(1.0, level))
        
        if peak is not None:
            if peak > self.peak_level:
                self.peak_level = peak
                self.peak_hold_counter = self.peak_hold_time
            elif self.peak_hold_counter > 0:
                self.peak_hold_counter -= 1
            else:
                self.peak_level = max(self.peak_level - 0.02, self.current_level)
        
        self.update()
    
    def paintEvent(self, event):
        """カスタム描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # 背景
        painter.fillRect(rect, QColor(20, 20, 20))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        # LEDセグメントを描画
        active_segments = int(self.current_level * self.segment_count)
        peak_segment = int(self.peak_level * self.segment_count)
        
        for i in range(self.segment_count):
            segment_y = rect.height() - (i + 1) * (self.segment_height + self.segment_spacing) - 5
            segment_rect = QRect(3, segment_y, self.segment_width, self.segment_height)
            
            # セグメントの色を決定
            segment_ratio = (i + 1) / self.segment_count
            
            if segment_ratio <= 0.6:  # 緑ゾーン（-inf ~ -12dB相当）
                color_on = QColor(0, 255, 0)
                color_off = QColor(0, 60, 0)
            elif segment_ratio <= 0.85:  # 黄ゾーン（-12dB ~ -3dB相当）
                color_on = QColor(255, 255, 0)
                color_off = QColor(60, 60, 0)
            else:  # 赤ゾーン（-3dB ~ 0dB相当）
                color_on = QColor(255, 0, 0)
                color_off = QColor(60, 0, 0)
            
            # セグメントの状態を決定
            if i < active_segments:
                # アクティブなセグメント
                painter.fillRect(segment_rect, color_on)
            elif i == peak_segment - 1 and self.peak_hold_counter > 0:
                # ピークホールド
                painter.fillRect(segment_rect, color_on)
            else:
                # 非アクティブなセグメント
                painter.fillRect(segment_rect, color_off)
            
            # セグメントの境界線
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.drawRect(segment_rect)


class UADChannelStrip(QFrame):
    """UAD風チャンネルストリップ"""
    
    def __init__(self, channel_name="CH 1", parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.setup_ui()
        self.setup_styles()
        
        # テスト用タイマー
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.update_test_levels)
        self.test_timer.start(50)  # 20fps
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(100)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # チャンネル名
        name_label = QLabel(self.channel_name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                background-color: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        layout.addWidget(name_label)
        
        # ゲインノブ
        gain_layout = QVBoxLayout()
        gain_layout.setSpacing(4)
        
        gain_label = QLabel("GAIN")
        gain_label.setAlignment(Qt.AlignCenter)
        gain_label.setStyleSheet("color: #cccccc; font-size: 9px; font-weight: bold;")
        gain_layout.addWidget(gain_label)
        
        self.gain_knob = AnalogKnob(size=60)
        self.gain_knob.setValue(50)
        gain_layout.addWidget(self.gain_knob, alignment=Qt.AlignCenter)
        
        self.gain_value_label = QLabel("50")
        self.gain_value_label.setAlignment(Qt.AlignCenter)
        self.gain_value_label.setStyleSheet("color: #ff8c00; font-size: 8px; font-weight: bold;")
        gain_layout.addWidget(self.gain_value_label)
        
        layout.addLayout(gain_layout)
        
        # レベルメーター
        meter_layout = QVBoxLayout()
        meter_layout.setSpacing(4)
        
        meter_label = QLabel("LEVEL")
        meter_label.setAlignment(Qt.AlignCenter)
        meter_label.setStyleSheet("color: #cccccc; font-size: 9px; font-weight: bold;")
        meter_layout.addWidget(meter_label)
        
        self.level_meter = LEDMeter(width=24, height=150)
        meter_layout.addWidget(self.level_meter, alignment=Qt.AlignCenter)
        
        layout.addLayout(meter_layout)
        
        # フェーダー
        fader_layout = QVBoxLayout()
        fader_layout.setSpacing(4)
        
        fader_label = QLabel("FADER")
        fader_label.setAlignment(Qt.AlignCenter)
        fader_label.setStyleSheet("color: #cccccc; font-size: 9px; font-weight: bold;")
        fader_layout.addWidget(fader_label)
        
        self.fader = QSlider(Qt.Vertical)
        self.fader.setRange(0, 100)
        self.fader.setValue(75)
        self.fader.setFixedHeight(80)
        fader_layout.addWidget(self.fader, alignment=Qt.AlignCenter)
        
        layout.addLayout(fader_layout)
        
        # MUTE/SOLOボタン
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)
        
        self.mute_button = QPushButton("MUTE")
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedHeight(25)
        self.mute_button.setObjectName("muteButton")
        button_layout.addWidget(self.mute_button)
        
        self.solo_button = QPushButton("SOLO")
        self.solo_button.setCheckable(True)
        self.solo_button.setFixedHeight(25)
        self.solo_button.setObjectName("soloButton")
        button_layout.addWidget(self.solo_button)
        
        layout.addLayout(button_layout)
        
        # シグナル接続
        self.gain_knob.valueChanged.connect(self.on_gain_changed)
        self.mute_button.toggled.connect(self.on_mute_toggled)
        self.solo_button.toggled.connect(self.on_solo_toggled)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            UADChannelStrip {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #3d3d3d,
                                           stop: 0.5 #2b2b2b,
                                           stop: 1 #3d3d3d);
                border-top: 1px solid #3d3d3d;
                border-left: 1px solid #3d3d3d;
                border-right: 1px solid #111111;
                border-bottom: 1px solid #111111;
                border-radius: 6px;
            }
            
            QSlider::groove:vertical {
                background: #1a1a1a;
                width: 8px;
                border-radius: 4px;
                border: 1px solid #333333;
            }
            
            QSlider::handle:vertical {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                           stop: 0 #ff8c00,
                                           stop: 1 #cc6600);
                border: 2px solid #444444;
                height: 15px;
                margin: 0 -4px;
                border-radius: 8px;
            }
            
            QSlider::handle:vertical:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                           stop: 0 #ffa500,
                                           stop: 1 #ff8c00);
            }
            
            QSlider::sub-page:vertical {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #ff8c00,
                                           stop: 1 #cc6600);
                border-radius: 4px;
            }
            
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a,
                                           stop: 1 #2a2a2a);
                border: 1px solid #555555;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
                font-size: 9px;
                padding: 4px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5a5a5a,
                                           stop: 1 #3a3a3a);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #2a2a2a,
                                           stop: 1 #4a4a4a);
            }
            
            QPushButton#muteButton:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ff8c00,
                                           stop: 1 #cc6600);
                color: #000000;
            }
            
            QPushButton#soloButton:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ffff00,
                                           stop: 1 #cccc00);
                color: #000000;
            }
        """)
    
    def on_gain_changed(self, value):
        """ゲイン変更時"""
        self.gain_value_label.setText(f"{int(value)}")
        print(f"{self.channel_name} Gain: {value:.1f}")
    
    def on_mute_toggled(self, checked):
        """ミュート切り替え時"""
        print(f"{self.channel_name} Mute: {checked}")
    
    def on_solo_toggled(self, checked):
        """ソロ切り替え時"""
        print(f"{self.channel_name} Solo: {checked}")
    
    def update_test_levels(self):
        """テスト用レベル更新"""
        # ランダムなレベルを生成
        level = random.random() * 0.8
        peak = level + random.random() * 0.2
        self.level_meter.setLevel(level, peak)


class UADMainWindow(QMainWindow):
    """メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """UI構築"""
        self.setWindowTitle("UAD Console風カスタムウィジェット デモ")
        self.setMinimumSize(800, 600)
        
        # 背景色設定
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
        """)
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # レイアウト
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # チャンネルストリップを複数作成
        for i in range(4):
            strip = UADChannelStrip(f"CH {i+1}")
            main_layout.addWidget(strip)
        
        # 右側にスペース
        main_layout.addStretch()
        
        # 説明ラベル
        info_layout = QVBoxLayout()
        
        title_label = QLabel("UAD Console風カスタムウィジェット")
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        info_layout.addWidget(title_label)
        
        info_text = QLabel("""
機能:
• アナログノブ: マウスドラッグで値変更
• LEDメーター: リアルタイム表示
• UAD風フェーダー: 垂直スライダー
• MUTE/SOLOボタン: 発光エフェクト

操作方法:
• ノブ: 上下ドラッグまたはホイール
• ボタン: クリックで切り替え
• フェーダー: ドラッグで調整
        """)
        info_text.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        info_layout.addWidget(info_text)
        info_layout.addStretch()
        
        main_layout.addLayout(info_layout)


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    
    # アプリケーション設定
    app.setApplicationName("UAD Custom Widgets")
    app.setApplicationVersion("1.0")
    
    # メインウィンドウ作成・表示
    window = UADMainWindow()
    window.show()
    
    # イベントループ開始
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())