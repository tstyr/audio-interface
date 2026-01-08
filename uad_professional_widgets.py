"""
UAD Console完全再現：プロフェッショナルウィジェット
スキューモーフィック・デザインの徹底実装
"""
import sys
import math
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QFrame, QSlider)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QTimer
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QLinearGradient, 
                           QRadialGradient, QFont, QFontMetrics, QPainterPath)


class ProfessionalFader(QWidget):
    """UAD風プロフェッショナルフェーダー"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, width=24, height=200):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setMouseTracking(True)
        
        # 値の設定
        self.min_value = 0.0
        self.max_value = 100.0
        self.current_value = 75.0
        
        # フェーダーキャップのサイズ
        self.cap_width = width - 4
        self.cap_height = 20
        self.track_width = 4
        
        # マウス操作用
        self.dragging = False
        self.drag_start_y = 0
        self.drag_start_value = 0
        
        # 有効範囲（上下のマージン）
        self.margin_top = self.cap_height // 2
        self.margin_bottom = self.cap_height // 2
        self.effective_height = height - self.margin_top - self.margin_bottom
    
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
            cap_y = self.get_cap_position()
            cap_rect = QRect(2, cap_y, self.cap_width, self.cap_height)
            
            if cap_rect.contains(event.pos()):
                self.dragging = True
                self.drag_start_y = event.y()
                self.drag_start_value = self.current_value
                self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """マウス移動時"""
        if self.dragging:
            delta_y = event.y() - self.drag_start_y
            # Y座標の変化を値の変化に変換（上に動かすと値が増加）
            value_change = -(delta_y / self.effective_height) * (self.max_value - self.min_value)
            new_value = self.drag_start_value + value_change
            self.setValue(new_value)
    
    def mouseReleaseEvent(self, event):
        """マウス離した時"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """マウスホイール"""
        delta = event.angleDelta().y() / 120.0
        value_change = delta * 2.0
        new_value = self.current_value + value_change
        self.setValue(new_value)
    
    def get_cap_position(self):
        """フェーダーキャップの位置を計算"""
        value_ratio = (self.current_value - self.min_value) / (self.max_value - self.min_value)
        # 値が大きいほど上に位置する
        y_pos = self.margin_top + (1.0 - value_ratio) * self.effective_height
        return int(y_pos - self.cap_height // 2)
    
    def paintEvent(self, event):
        """カスタム描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # フェーダートラック（溝）を描画
        self.draw_track(painter, rect)
        
        # フェーダーキャップを描画
        self.draw_cap(painter, rect)
    
    def draw_track(self, painter, rect):
        """フェーダートラック（溝）を描画"""
        track_x = (rect.width() - self.track_width) // 2
        track_rect = QRect(track_x, self.margin_top, self.track_width, self.effective_height)
        
        # 溝の影（内側の陰影）
        painter.fillRect(track_rect, QColor(15, 15, 15))
        
        # 左側のハイライト
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawLine(track_rect.left(), track_rect.top(), 
                        track_rect.left(), track_rect.bottom())
        
        # 右側の影
        painter.setPen(QPen(QColor(10, 10, 10), 1))
        painter.drawLine(track_rect.right(), track_rect.top(), 
                        track_rect.right(), track_rect.bottom())
        
        # 上下の境界線
        painter.setPen(QPen(QColor(30, 30, 30), 1))
        painter.drawLine(track_rect.left(), track_rect.top(), 
                        track_rect.right(), track_rect.top())
        painter.drawLine(track_rect.left(), track_rect.bottom(), 
                        track_rect.right(), track_rect.bottom())
    
    def draw_cap(self, painter, rect):
        """フェーダーキャップを描画"""
        cap_y = self.get_cap_position()
        cap_rect = QRect(2, cap_y, self.cap_width, self.cap_height)
        
        # キャップの影（下側）
        shadow_rect = cap_rect.adjusted(1, 1, 1, 1)
        painter.fillRect(shadow_rect, QColor(0, 0, 0, 80))
        
        # メインのキャップ（シルバー金属質感）
        cap_gradient = QLinearGradient(0, cap_rect.top(), 0, cap_rect.bottom())
        cap_gradient.setColorAt(0, QColor(160, 160, 160))  # 上部ハイライト
        cap_gradient.setColorAt(0.1, QColor(140, 140, 140))
        cap_gradient.setColorAt(0.5, QColor(120, 120, 120))  # 中央
        cap_gradient.setColorAt(0.9, QColor(100, 100, 100))
        cap_gradient.setColorAt(1, QColor(80, 80, 80))    # 下部影
        
        painter.fillRect(cap_rect, QBrush(cap_gradient))
        
        # キャップの境界線
        painter.setPen(QPen(QColor(180, 180, 180), 1))  # 上部ハイライト
        painter.drawLine(cap_rect.left(), cap_rect.top(), 
                        cap_rect.right(), cap_rect.top())
        
        painter.setPen(QPen(QColor(60, 60, 60), 1))  # 下部影
        painter.drawLine(cap_rect.left(), cap_rect.bottom(), 
                        cap_rect.right(), cap_rect.bottom())
        
        # 左右の境界線
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawLine(cap_rect.left(), cap_rect.top(), 
                        cap_rect.left(), cap_rect.bottom())
        painter.drawLine(cap_rect.right(), cap_rect.top(), 
                        cap_rect.right(), cap_rect.bottom())
        
        # 中央のオレンジインジケーター線
        indicator_y = cap_rect.center().y()
        indicator_left = cap_rect.left() + 3
        indicator_right = cap_rect.right() - 3
        
        painter.setPen(QPen(QColor(255, 140, 0), 2))
        painter.drawLine(indicator_left, indicator_y, indicator_right, indicator_y)
        
        # インジケーター線の光沢効果
        painter.setPen(QPen(QColor(255, 180, 60), 1))
        painter.drawLine(indicator_left, indicator_y - 1, indicator_right, indicator_y - 1)


class ProfessionalKnob(QWidget):
    """UAD風プロフェッショナルノブ"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, size=50):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        
        # 値の範囲
        self.min_value = 0.0
        self.max_value = 100.0
        self.current_value = 50.0
        
        # 角度の範囲（-135度から+135度）
        self.min_angle = -135
        self.max_angle = 135
        
        # マウス操作用
        self.dragging = False
        self.last_mouse_y = 0
        self.sensitivity = 0.3
        
        # 外観設定
        self.knob_radius = (size - 8) // 2
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
            delta_y = self.last_mouse_y - event.y()
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
        delta = event.angleDelta().y() / 120.0
        value_change = delta * 2.0
        new_value = self.current_value + value_change
        self.setValue(new_value)
    
    def paintEvent(self, event):
        """カスタム描画"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center = QPoint(self.center_x, self.center_y)
        
        # ノブの影を描画
        self.draw_shadow(painter, center)
        
        # ノブ本体を描画
        self.draw_body(painter, center)
        
        # 指がかりの溝を描画
        self.draw_grooves(painter, center)
        
        # ポインター線を描画
        self.draw_pointer(painter, center)
    
    def draw_shadow(self, painter, center):
        """ノブの影を描画"""
        shadow_gradient = QRadialGradient(center.x() + 2, center.y() + 2, self.knob_radius + 3)
        shadow_gradient.setColorAt(0, QColor(0, 0, 0, 100))
        shadow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(shadow_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center.x() - self.knob_radius - 1, center.y() - self.knob_radius - 1,
                          (self.knob_radius + 3) * 2, (self.knob_radius + 3) * 2)
    
    def draw_body(self, painter, center):
        """ノブ本体を描画"""
        # メインのノブ（暗いグレー）
        main_gradient = QRadialGradient(center.x() - self.knob_radius//3, 
                                       center.y() - self.knob_radius//3, 
                                       self.knob_radius * 1.2)
        main_gradient.setColorAt(0, QColor(70, 70, 70))   # 中央ハイライト
        main_gradient.setColorAt(0.7, QColor(45, 45, 45)) # 中間
        main_gradient.setColorAt(0.9, QColor(25, 25, 25)) # 外周
        main_gradient.setColorAt(1, QColor(15, 15, 15))   # 最外周
        
        painter.setBrush(QBrush(main_gradient))
        painter.setPen(QPen(QColor(10, 10, 10), 1))
        painter.drawEllipse(center.x() - self.knob_radius, center.y() - self.knob_radius,
                          self.knob_radius * 2, self.knob_radius * 2)
        
        # 外周の光沢エッジ
        edge_gradient = QRadialGradient(center.x(), center.y(), self.knob_radius)
        edge_gradient.setColorAt(0.85, QColor(0, 0, 0, 0))
        edge_gradient.setColorAt(0.95, QColor(120, 120, 120, 150))
        edge_gradient.setColorAt(1, QColor(80, 80, 80, 200))
        
        painter.setBrush(QBrush(edge_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center.x() - self.knob_radius, center.y() - self.knob_radius,
                          self.knob_radius * 2, self.knob_radius * 2)
    
    def draw_grooves(self, painter, center):
        """指がかりの溝を描画"""
        groove_count = 24
        groove_radius = self.knob_radius - 6
        
        for i in range(groove_count):
            angle = (360.0 / groove_count) * i
            angle_rad = math.radians(angle)
            
            # 溝の位置
            x = center.x() + groove_radius * math.cos(angle_rad)
            y = center.y() + groove_radius * math.sin(angle_rad)
            
            # 溝の描画（小さな暗い点）
            painter.setBrush(QBrush(QColor(15, 15, 15)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 1, y - 1, 2, 2)
    
    def draw_pointer(self, painter, center):
        """ポインター線を描画"""
        # 現在の値から角度を計算
        value_ratio = (self.current_value - self.min_value) / (self.max_value - self.min_value)
        angle = self.min_angle + (self.max_angle - self.min_angle) * value_ratio
        angle_rad = math.radians(angle)
        
        # ポインターの長さ
        pointer_length = self.knob_radius - 8
        
        # ポインターの終点
        end_x = center.x() + pointer_length * math.cos(angle_rad)
        end_y = center.y() + pointer_length * math.sin(angle_rad)
        
        # ポインター線を描画
        painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(center.x(), center.y(), end_x, end_y)
        
        # ポインターの先端に小さな円
        painter.setBrush(QBrush(QColor(255, 140, 0)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(end_x - 2, end_y - 2, 4, 4)


class ProfessionalLEDMeter(QWidget):
    """UAD風プロフェッショナルLEDメーター（dB目盛り付き）"""
    
    def __init__(self, parent=None, width=30, height=200):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.current_level = 0.0  # 0.0 - 1.0
        self.peak_level = 0.0
        self.peak_hold_time = 30
        self.peak_hold_counter = 0
        
        # LEDセグメント設定
        self.segment_count = 24
        self.meter_width = 12
        self.meter_x = 8
        self.segment_height = (height - 20) // self.segment_count
        self.segment_spacing = 1
        
        # dB目盛り設定
        self.db_marks = [
            (1.0, "0"),
            (0.9, "-6"),
            (0.75, "-12"),
            (0.5, "-18"),
            (0.25, "-24"),
            (0.1, "-36"),
            (0.05, "-48"),
            (0.0, "-∞")
        ]
    
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
        
        # メーター背景（彫り込み効果）
        meter_rect = QRect(self.meter_x, 10, self.meter_width, rect.height() - 20)
        
        # 背景の影
        painter.fillRect(meter_rect, QColor(10, 10, 10))
        
        # 彫り込み効果の境界線
        painter.setPen(QPen(QColor(5, 5, 5), 1))
        painter.drawRect(meter_rect.adjusted(-1, -1, 0, 0))
        
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(meter_rect.adjusted(0, 0, 1, 1))
        
        # LEDセグメントを描画
        self.draw_segments(painter, meter_rect)
        
        # dB目盛りを描画
        self.draw_db_scale(painter, rect)
    
    def draw_segments(self, painter, meter_rect):
        """LEDセグメントを描画"""
        active_segments = int(self.current_level * self.segment_count)
        peak_segment = int(self.peak_level * self.segment_count)
        
        for i in range(self.segment_count):
            segment_y = meter_rect.bottom() - (i + 1) * (self.segment_height + self.segment_spacing)
            segment_rect = QRect(meter_rect.left() + 1, segment_y, 
                               meter_rect.width() - 2, self.segment_height)
            
            # セグメントの色を決定
            segment_ratio = (i + 1) / self.segment_count
            
            if segment_ratio <= 0.6:  # 緑ゾーン
                color_on = QColor(0, 255, 0)
                color_off = QColor(0, 40, 0)
            elif segment_ratio <= 0.85:  # 黄ゾーン
                color_on = QColor(255, 255, 0)
                color_off = QColor(40, 40, 0)
            else:  # 赤ゾーン
                color_on = QColor(255, 0, 0)
                color_off = QColor(40, 0, 0)
            
            # セグメントの状態を決定
            if i < active_segments:
                painter.fillRect(segment_rect, color_on)
            elif i == peak_segment - 1 and self.peak_hold_counter > 0:
                painter.fillRect(segment_rect, color_on)
            else:
                painter.fillRect(segment_rect, color_off)
            
            # セグメント間の境界線
            if i < self.segment_count - 1:
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                painter.drawLine(segment_rect.left(), segment_rect.bottom() + 1,
                               segment_rect.right(), segment_rect.bottom() + 1)
    
    def draw_db_scale(self, painter, rect):
        """dB目盛りを描画"""
        painter.setPen(QColor(180, 180, 180))
        font = QFont("Segoe UI", 7)
        painter.setFont(font)
        
        meter_height = rect.height() - 20
        
        for level_pos, db_text in self.db_marks:
            y_pos = 10 + (1.0 - level_pos) * meter_height
            
            # 目盛り線
            tick_x = self.meter_x + self.meter_width + 2
            painter.drawLine(tick_x, y_pos, tick_x + 3, y_pos)
            
            # dBテキスト
            text_rect = painter.fontMetrics().boundingRect(db_text)
            painter.drawText(tick_x + 5, y_pos + text_rect.height() // 2 - 1, db_text)


def test_professional_widgets():
    """プロフェッショナルウィジェットのテスト"""
    app = QApplication(sys.argv)
    
    # テスト用ウィンドウ
    window = QWidget()
    window.setWindowTitle("UAD Professional Widgets Test")
    window.setStyleSheet("background-color: #2b2b2b;")
    window.resize(400, 300)
    
    layout = QHBoxLayout(window)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(20)
    
    # フェーダーテスト
    fader_layout = QVBoxLayout()
    fader_label = QLabel("FADER")
    fader_label.setStyleSheet("color: white; font-size: 9px; font-weight: bold;")
    fader_label.setAlignment(Qt.AlignCenter)
    fader_layout.addWidget(fader_label)
    
    fader = ProfessionalFader(width=24, height=150)
    fader_layout.addWidget(fader, alignment=Qt.AlignCenter)
    
    layout.addLayout(fader_layout)
    
    # ノブテスト
    knob_layout = QVBoxLayout()
    knob_label = QLabel("GAIN")
    knob_label.setStyleSheet("color: white; font-size: 9px; font-weight: bold;")
    knob_label.setAlignment(Qt.AlignCenter)
    knob_layout.addWidget(knob_label)
    
    knob = ProfessionalKnob(size=60)
    knob_layout.addWidget(knob, alignment=Qt.AlignCenter)
    
    layout.addLayout(knob_layout)
    
    # メーターテスト
    meter_layout = QVBoxLayout()
    meter_label = QLabel("LEVEL")
    meter_label.setStyleSheet("color: white; font-size: 9px; font-weight: bold;")
    meter_label.setAlignment(Qt.AlignCenter)
    meter_layout.addWidget(meter_label)
    
    meter = ProfessionalLEDMeter(width=50, height=150)
    meter_layout.addWidget(meter, alignment=Qt.AlignCenter)
    
    layout.addLayout(meter_layout)
    
    # テスト用タイマー
    import random
    def update_meter():
        level = random.random() * 0.8
        peak = level + random.random() * 0.2
        meter.setLevel(level, peak)
    
    timer = QTimer()
    timer.timeout.connect(update_meter)
    timer.start(50)
    
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(test_professional_widgets())