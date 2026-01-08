"""
UAD Console Professional - 完全再現版
商用ソフトウェア・クオリティの実装
"""
import sys
import math
import threading
import time as time_module
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QComboBox, QSlider, QMenu,
    QFileDialog, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QRect, QPoint
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath, QFontMetrics
)

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ============================================================================
# グローバルスタイル定義
# ============================================================================
DARK_BG = "#1a1a1a"
PANEL_BG = "#2b2b2b"
SURFACE = "#3a3a3a"
BORDER = "#4a4a4a"
ACCENT = "#ff8c00"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#cccccc"
METER_GREEN = "#00ff00"
METER_YELLOW = "#ffff00"
METER_RED = "#ff0000"


GLOBAL_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 8pt;
}}

QFrame {{
    background-color: transparent;
}}

QLabel {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
}}

QScrollArea {{
    background-color: {DARK_BG};
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: {DARK_BG};
}}

QScrollBar:horizontal {{
    background: {SURFACE};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background: {ACCENT};
    border-radius: 5px;
    min-width: 20px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QComboBox {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
    border: 1px solid #555;
    border-radius: 3px;
    color: {TEXT_PRIMARY};
    padding: 2px 6px;
    font-weight: bold;
    font-size: 8pt;
}}

QComboBox::drop-down {{
    border: none;
    width: 12px;
}}

QComboBox::down-arrow {{
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {TEXT_PRIMARY};
}}

QComboBox QAbstractItemView {{
    background-color: {PANEL_BG};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT};
}}
"""


# ============================================================================
# カスタムウィジェット: プロフェッショナルLEDメーター
# ============================================================================
class ProLEDMeter(QWidget):
    """高精度LEDメーター（-60dB〜0dB、ノイズゲート付き）"""
    
    def __init__(self, parent=None, width=14, height=180, horizontal=False):
        super().__init__(parent)
        self.horizontal = horizontal
        if horizontal:
            self.setFixedSize(height, width)
        else:
            self.setFixedSize(width, height)
        
        self.level = 0.0
        self.peak = 0.0
        self.peak_hold_frames = 0
        self.peak_hold_max = 30
        self.noise_gate_db = -90.0
        self.segment_count = 24
    
    def setLevel(self, level_db: float, peak_db: float = None):
        """レベル設定（dB値）"""
        # ノイズゲート
        if level_db < self.noise_gate_db:
            level_db = -100.0
        
        # 正規化（-60dB〜0dB → 0.0〜1.0）
        self.level = max(0.0, min(1.0, (level_db + 60.0) / 60.0))
        
        if peak_db is not None:
            if peak_db < self.noise_gate_db:
                peak_db = -100.0
            new_peak = max(0.0, min(1.0, (peak_db + 60.0) / 60.0))
            
            if new_peak > self.peak:
                self.peak = new_peak
                self.peak_hold_frames = self.peak_hold_max
            elif self.peak_hold_frames > 0:
                self.peak_hold_frames -= 1
            else:
                self.peak = max(self.peak - 0.02, self.level)
        
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # 背景
        painter.fillRect(rect, QColor(15, 15, 15))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        # セグメント描画
        if self.horizontal:
            self._draw_horizontal(painter, rect)
        else:
            self._draw_vertical(painter, rect)
    
    def _draw_vertical(self, painter, rect):
        seg_height = (rect.height() - 4) / self.segment_count
        active_segs = int(self.level * self.segment_count)
        peak_seg = int(self.peak * self.segment_count)
        
        for i in range(self.segment_count):
            y = rect.height() - 3 - (i + 1) * seg_height
            seg_rect = QRect(2, int(y), rect.width() - 4, int(seg_height - 1))
            
            ratio = (i + 1) / self.segment_count
            if ratio <= 0.6:
                color_on, color_off = QColor(0, 255, 0), QColor(0, 40, 0)
            elif ratio <= 0.85:
                color_on, color_off = QColor(255, 255, 0), QColor(40, 40, 0)
            else:
                color_on, color_off = QColor(255, 0, 0), QColor(40, 0, 0)
            
            if i < active_segs:
                painter.fillRect(seg_rect, color_on)
            elif i == peak_seg - 1 and self.peak_hold_frames > 0:
                painter.fillRect(seg_rect, color_on)
            else:
                painter.fillRect(seg_rect, color_off)
    
    def _draw_horizontal(self, painter, rect):
        seg_width = (rect.width() - 4) / self.segment_count
        active_segs = int(self.level * self.segment_count)
        peak_seg = int(self.peak * self.segment_count)
        
        for i in range(self.segment_count):
            x = 2 + i * seg_width
            seg_rect = QRect(int(x), 2, int(seg_width - 1), rect.height() - 4)
            
            ratio = (i + 1) / self.segment_count
            if ratio <= 0.6:
                color_on, color_off = QColor(0, 255, 0), QColor(0, 40, 0)
            elif ratio <= 0.85:
                color_on, color_off = QColor(255, 255, 0), QColor(40, 40, 0)
            else:
                color_on, color_off = QColor(255, 0, 0), QColor(40, 0, 0)
            
            if i < active_segs:
                painter.fillRect(seg_rect, color_on)
            elif i == peak_seg - 1 and self.peak_hold_frames > 0:
                painter.fillRect(seg_rect, color_on)
            else:
                painter.fillRect(seg_rect, color_off)


# ============================================================================
# カスタムウィジェット: プロフェッショナルフェーダー
# ============================================================================
class ProFader(QWidget):
    """UAD風プロフェッショナルフェーダー"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, width=30, height=180):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setMouseTracking(True)
        
        self.value = 0.75
        self.dragging = False
        self.cap_height = 24
        self.cap_width = width - 6
        self.margin = self.cap_height // 2
    
    def setValue(self, val):
        self.value = max(0.0, min(1.0, val))
        self.update()
        self.valueChanged.emit(self.value)
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = True
            self._update_from_mouse(e.position().y())
    
    def mouseMoveEvent(self, e):
        if self.dragging:
            self._update_from_mouse(e.position().y())
    
    def mouseReleaseEvent(self, e):
        self.dragging = False
    
    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 1200.0
        self.setValue(self.value + delta)
    
    def _update_from_mouse(self, y):
        effective = self.height() - 2 * self.margin
        val = 1.0 - (y - self.margin) / effective
        self.setValue(val)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # トラック（溝）
        track_x = (w - 6) // 2
        track_rect = QRect(track_x, self.margin, 6, h - 2 * self.margin)
        
        grad = QLinearGradient(track_rect.left(), 0, track_rect.right(), 0)
        grad.setColorAt(0, QColor(10, 10, 10))
        grad.setColorAt(0.5, QColor(30, 30, 30))
        grad.setColorAt(1, QColor(10, 10, 10))
        painter.fillRect(track_rect, grad)
        
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(track_rect)
        
        # dBスケール
        painter.setPen(QColor(100, 100, 100))
        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        
        db_marks = [(1.0, "+12"), (0.85, "0"), (0.7, "-6"), (0.5, "-12"), (0.25, "-24"), (0.0, "-∞")]
        for pos, text in db_marks:
            y = self.margin + (1.0 - pos) * (h - 2 * self.margin)
            painter.drawLine(2, int(y), 6, int(y))
            painter.drawText(w - 18, int(y) + 3, text)
        
        # フェーダーキャップ
        cap_y = self.margin + (1.0 - self.value) * (h - 2 * self.margin) - self.cap_height // 2
        cap_rect = QRect(3, int(cap_y), self.cap_width, self.cap_height)
        
        # キャップのグラデーション（シルバー金属質感）
        cap_grad = QLinearGradient(0, cap_rect.top(), 0, cap_rect.bottom())
        cap_grad.setColorAt(0, QColor(180, 180, 180))
        cap_grad.setColorAt(0.15, QColor(140, 140, 140))
        cap_grad.setColorAt(0.5, QColor(120, 120, 120))
        cap_grad.setColorAt(0.85, QColor(100, 100, 100))
        cap_grad.setColorAt(1, QColor(70, 70, 70))
        
        painter.fillRect(cap_rect, cap_grad)
        
        # キャップの境界線
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawLine(cap_rect.left(), cap_rect.top(), cap_rect.right(), cap_rect.top())
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawLine(cap_rect.left(), cap_rect.bottom(), cap_rect.right(), cap_rect.bottom())
        
        # 中央のオレンジインジケーター
        ind_y = cap_rect.center().y()
        painter.setPen(QPen(QColor(255, 140, 0), 2))
        painter.drawLine(cap_rect.left() + 4, ind_y, cap_rect.right() - 4, ind_y)


# ============================================================================
# カスタムウィジェット: プロフェッショナルノブ
# ============================================================================
class ProKnob(QWidget):
    """UAD風プロフェッショナルノブ"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, size=32):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.value = 0.5
        self.dragging = False
        self.last_y = 0
    
    def setValue(self, val):
        self.value = max(0.0, min(1.0, val))
        self.update()
        self.valueChanged.emit(self.value)
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = True
            self.last_y = e.position().y()
    
    def mouseMoveEvent(self, e):
        if self.dragging:
            delta = (self.last_y - e.position().y()) * 0.005
            self.setValue(self.value + delta)
            self.last_y = e.position().y()
    
    def mouseReleaseEvent(self, e):
        self.dragging = False
    
    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 1200.0
        self.setValue(self.value + delta)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        size = min(self.width(), self.height())
        center = QPoint(self.width() // 2, self.height() // 2)
        radius = (size - 4) // 2
        
        # 影
        shadow_grad = QRadialGradient(center.x() + 1, center.y() + 1, radius + 2)
        shadow_grad.setColorAt(0, QColor(0, 0, 0, 80))
        shadow_grad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(shadow_grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, radius + 2, radius + 2)
        
        # ノブ本体
        knob_grad = QRadialGradient(center.x() - radius//3, center.y() - radius//3, radius * 1.2)
        knob_grad.setColorAt(0, QColor(70, 70, 70))
        knob_grad.setColorAt(0.7, QColor(40, 40, 40))
        knob_grad.setColorAt(1, QColor(20, 20, 20))
        
        painter.setBrush(knob_grad)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawEllipse(center, radius, radius)
        
        # ポインター
        angle = -135 + self.value * 270
        angle_rad = math.radians(angle)
        ptr_len = radius - 4
        end_x = center.x() + ptr_len * math.cos(angle_rad)
        end_y = center.y() + ptr_len * math.sin(angle_rad)
        
        painter.setPen(QPen(QColor(255, 140, 0), 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(center.x(), center.y(), int(end_x), int(end_y))


# ============================================================================
# カスタムウィジェット: プラグインスロット
# ============================================================================
class PluginSlot(QFrame):
    """INSERTプラグインスロット"""
    
    clicked = Signal()
    
    def __init__(self, slot_name="", parent=None):
        super().__init__(parent)
        self.slot_name = slot_name
        self.plugin_name = ""
        self.enabled = False
        self.setFixedHeight(18)
        self.setCursor(Qt.PointingHandCursor)
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            PluginSlot {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border: 1px solid #4a4a4a;
                border-radius: 2px;
            }}
            PluginSlot:hover {{
                border: 1px solid {ACCENT};
            }}
        """)
    
    def setPlugin(self, name: str):
        self.plugin_name = name
        self.enabled = bool(name)
        self.update()
    
    def mousePressEvent(self, e):
        self.clicked.emit()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect().adjusted(2, 2, -2, -2)
        
        # プラグイン名表示
        if self.plugin_name:
            painter.setPen(QColor(255, 140, 0) if self.enabled else QColor(150, 150, 150))
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            
            text = self.plugin_name[:15] + "..." if len(self.plugin_name) > 15 else self.plugin_name
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        else:
            painter.setPen(QColor(80, 80, 80))
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignCenter, "- empty -")


# ============================================================================
# カスタムウィジェット: トグルボタン
# ============================================================================
class ToggleButton(QPushButton):
    """発光トグルボタン"""
    
    def __init__(self, text, color_on=ACCENT, color_off="#4a4a4a", parent=None):
        super().__init__(text, parent)
        self.color_on = color_on
        self.color_off = color_off
        self.setCheckable(True)
        self.setFixedHeight(22)
        self._update_style()
        self.toggled.connect(self._update_style)
    
    def _update_style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {self.color_on}, stop:1 #cc6600);
                    border: 1px solid #666;
                    border-radius: 3px;
                    color: #000;
                    font-weight: bold;
                    font-size: 8pt;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5a5a5a, stop:1 #3a3a3a);
                    border: 1px solid #555;
                    border-radius: 3px;
                    color: #fff;
                    font-weight: bold;
                    font-size: 8pt;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6a6a6a, stop:1 #4a4a4a);
                }}
            """)


# ============================================================================
# チャンネルストリップ（完全版）
# ============================================================================
class ChannelStrip(QFrame):
    """UAD風完全チャンネルストリップ"""
    
    removeRequested = Signal()
    
    def __init__(self, channel_name="CH 1", device_info=None, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.device_info = device_info
        self.audio_processor = None
        
        self.setFixedWidth(75)
        self.setMinimumHeight(520)
        self._setup_ui()
        self._setup_style()
        
        # レベル更新タイマー
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self._update_levels)
        self.level_timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            ChannelStrip {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a3a3a, stop:0.5 {PANEL_BG}, stop:1 #3a3a3a);
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin: 1px;
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)
        
        # チャンネル名
        self.name_label = QLabel(self.channel_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
            border: 1px solid #555;
            border-radius: 3px;
            color: {TEXT_PRIMARY};
            font-weight: bold;
            font-size: 9pt;
            padding: 2px;
        """)
        layout.addWidget(self.name_label)
        
        # UNISONスロット
        unison_label = QLabel("UNISON")
        unison_label.setAlignment(Qt.AlignCenter)
        unison_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(unison_label)
        
        self.unison_slot = PluginSlot("UNISON")
        self.unison_slot.clicked.connect(lambda: self._show_plugin_menu(self.unison_slot, "UNISON"))
        layout.addWidget(self.unison_slot)
        
        # INSERTSセクション
        inserts_label = QLabel("INSERTS")
        inserts_label.setAlignment(Qt.AlignCenter)
        inserts_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(inserts_label)
        
        self.insert_slots = []
        for i in range(5):
            slot = PluginSlot(f"INSERT {i+1}")
            slot.clicked.connect(lambda s=slot: self._show_plugin_menu(s, "INSERT"))
            self.insert_slots.append(slot)
            layout.addWidget(slot)
        
        # SENDSセクション
        sends_label = QLabel("SENDS")
        sends_label.setAlignment(Qt.AlignCenter)
        sends_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(sends_label)
        
        sends_layout = QGridLayout()
        sends_layout.setSpacing(2)
        self.send_knobs = []
        for i in range(4):
            knob = ProKnob(size=20)
            knob.setValue(0.0)
            self.send_knobs.append(knob)
            sends_layout.addWidget(knob, 0, i)
            
            lbl = QLabel(f"A{i+1}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 6pt;")
            sends_layout.addWidget(lbl, 1, i)
        
        layout.addLayout(sends_layout)
        
        # OUTPUT MONITOR
        output_label = QLabel("OUTPUT")
        output_label.setAlignment(Qt.AlignCenter)
        output_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(output_label)
        
        self.output_combo = QComboBox()
        self.output_combo.addItems(["MONITOR", "AUX 1", "AUX 2", "CUE 1", "CUE 2"])
        self.output_combo.setFixedHeight(20)
        layout.addWidget(self.output_combo)
        
        # メーターとフェーダー
        meter_fader_layout = QHBoxLayout()
        meter_fader_layout.setSpacing(2)
        
        self.meter = ProLEDMeter(width=10, height=140)
        meter_fader_layout.addWidget(self.meter)
        
        self.fader = ProFader(width=28, height=140)
        self.fader.setValue(0.75)
        meter_fader_layout.addWidget(self.fader)
        
        layout.addLayout(meter_fader_layout)
        
        # PANノブ
        pan_layout = QHBoxLayout()
        pan_layout.setSpacing(2)
        
        pan_label = QLabel("PAN")
        pan_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt;")
        pan_layout.addWidget(pan_label)
        
        self.pan_knob = ProKnob(size=24)
        self.pan_knob.setValue(0.5)
        pan_layout.addWidget(self.pan_knob)
        
        layout.addLayout(pan_layout)
        
        # SOLO/MUTEボタン
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)
        
        self.solo_btn = ToggleButton("S", "#ffff00")
        self.solo_btn.setFixedSize(28, 20)
        btn_layout.addWidget(self.solo_btn)
        
        self.mute_btn = ToggleButton("M", "#ff4444")
        self.mute_btn.setFixedSize(28, 20)
        btn_layout.addWidget(self.mute_btn)
        
        layout.addLayout(btn_layout)
    
    def _show_plugin_menu(self, slot, slot_type):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {PANEL_BG};
                color: {TEXT_PRIMARY};
                border: 1px solid #555;
            }}
            QMenu::item:selected {{
                background-color: {ACCENT};
            }}
        """)
        
        plugins = ["Neve 1073", "SSL E Channel", "API Vision", "1176LN", "LA-2A", "Pultec EQ"]
        for plugin in plugins:
            action = menu.addAction(plugin)
            action.triggered.connect(lambda checked, p=plugin, s=slot: s.setPlugin(p))
        
        menu.addSeparator()
        clear_action = menu.addAction("Clear")
        clear_action.triggered.connect(lambda: slot.setPlugin(""))
        
        menu.exec(self.mapToGlobal(slot.pos()))
    
    def _update_levels(self):
        # デモ用ランダムレベル（実際はオーディオプロセッサから取得）
        import random
        if not self.mute_btn.isChecked():
            level = random.random() * 0.6 - 60 + 40  # -60〜-20dB
            peak = level + random.random() * 5
        else:
            level, peak = -100, -100
        
        self.meter.setLevel(level, peak)


# ============================================================================
# トップツールバー
# ============================================================================
class TopToolbar(QFrame):
    """上部コントロールバー"""
    
    newSession = Signal()
    openSession = Signal()
    saveSession = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            TopToolbar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:0.5 #3a3a3a, stop:1 #2a2a2a);
                border-bottom: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # SESSION
        session_frame = QFrame()
        session_layout = QHBoxLayout(session_frame)
        session_layout.setContentsMargins(4, 0, 4, 0)
        session_layout.setSpacing(4)
        
        session_label = QLabel("SESSION")
        session_label.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 8pt;")
        session_layout.addWidget(session_label)
        
        for text, signal in [("NEW", self.newSession), ("OPEN", self.openSession), ("SAVE", self.saveSession)]:
            btn = QPushButton(text)
            btn.setFixedSize(45, 24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a5a5a, stop:1 #3a3a3a);
                    border: 1px solid #666;
                    border-radius: 3px;
                    color: {TEXT_PRIMARY};
                    font-weight: bold;
                    font-size: 8pt;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6a6a6a, stop:1 #4a4a4a);
                }}
            """)
            btn.clicked.connect(signal.emit)
            session_layout.addWidget(btn)
        
        layout.addWidget(session_frame)
        
        # SYSTEM STATUS
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 0, 8, 0)
        status_layout.setSpacing(12)
        
        # RATE
        rate_layout = QVBoxLayout()
        rate_layout.setSpacing(0)
        rate_label = QLabel("RATE")
        rate_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        rate_label.setAlignment(Qt.AlignCenter)
        rate_layout.addWidget(rate_label)
        
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["44.1kHz", "48kHz", "88.2kHz", "96kHz", "192kHz"])
        self.rate_combo.setFixedSize(70, 22)
        rate_layout.addWidget(self.rate_combo)
        status_layout.addLayout(rate_layout)
        
        # CLOCK
        clock_layout = QVBoxLayout()
        clock_layout.setSpacing(0)
        clock_label = QLabel("CLOCK")
        clock_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        clock_label.setAlignment(Qt.AlignCenter)
        clock_layout.addWidget(clock_label)
        
        self.clock_combo = QComboBox()
        self.clock_combo.addItems(["INTERNAL", "WORD", "S/PDIF", "ADAT"])
        self.clock_combo.setFixedSize(80, 22)
        clock_layout.addWidget(self.clock_combo)
        status_layout.addLayout(clock_layout)
        
        # DSP METER
        dsp_layout = QVBoxLayout()
        dsp_layout.setSpacing(0)
        dsp_label = QLabel("DSP")
        dsp_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        dsp_label.setAlignment(Qt.AlignCenter)
        dsp_layout.addWidget(dsp_label)
        
        self.dsp_meter = ProLEDMeter(width=100, height=12, horizontal=True)
        dsp_layout.addWidget(self.dsp_meter)
        status_layout.addLayout(dsp_layout)
        
        layout.addWidget(status_frame, 1)
        
        # TITLE
        title_label = QLabel("Console: UAD Pro")
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 10pt; font-weight: bold;")
        layout.addWidget(title_label)


# ============================================================================
# マスターセクション
# ============================================================================
class MasterSection(QFrame):
    """マスター・モニターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self._setup_ui()
        self._setup_style()
        
        # レベル更新タイマー
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self._update_levels)
        self.level_timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            MasterSection {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {DARK_BG}, stop:1 {PANEL_BG});
                border-left: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        # INSERTSセクション
        inserts_label = QLabel("INSERTS")
        inserts_label.setAlignment(Qt.AlignCenter)
        inserts_label.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(inserts_label)
        
        self.master_inserts = []
        for i in range(2):
            slot = PluginSlot(f"MASTER {i+1}")
            self.master_inserts.append(slot)
            layout.addWidget(slot)
        
        # UAD INSERTS REC/MON
        rec_mon_layout = QHBoxLayout()
        rec_mon_layout.setSpacing(2)
        
        self.rec_btn = ToggleButton("REC", ACCENT)
        self.rec_btn.setChecked(True)
        rec_mon_layout.addWidget(self.rec_btn)
        
        self.mon_btn = ToggleButton("MON", "#888888")
        rec_mon_layout.addWidget(self.mon_btn)
        
        layout.addLayout(rec_mon_layout)
        
        # AUXES
        auxes_label = QLabel("AUXES")
        auxes_label.setAlignment(Qt.AlignCenter)
        auxes_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: bold; font-size: 8pt;")
        layout.addWidget(auxes_label)
        
        # MONITORS
        monitors_label = QLabel("MONITORS")
        monitors_label.setAlignment(Qt.AlignCenter)
        monitors_label.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(monitors_label)
        
        # CONTROL ROOM
        cr_label = QLabel("CONTROL ROOM")
        cr_label.setAlignment(Qt.AlignCenter)
        cr_label.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
            border: 1px solid #555;
            border-radius: 3px;
            color: {TEXT_PRIMARY};
            font-weight: bold;
            font-size: 8pt;
            padding: 4px;
        """)
        layout.addWidget(cr_label)
        
        # CUE
        cue_label = QLabel("CUE")
        cue_label.setAlignment(Qt.AlignCenter)
        cue_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: bold; font-size: 8pt;")
        layout.addWidget(cue_label)
        
        # MONO / MUTE
        mono_mute_layout = QHBoxLayout()
        mono_mute_layout.setSpacing(2)
        
        self.mono_btn = ToggleButton("MONO", "#888888")
        mono_mute_layout.addWidget(self.mono_btn)
        
        self.mute_btn = ToggleButton("MUTE", "#ff4444")
        mono_mute_layout.addWidget(self.mute_btn)
        
        layout.addLayout(mono_mute_layout)
        
        # ステレオメーター
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(4)
        
        l_label = QLabel("L")
        l_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: bold;")
        l_label.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(l_label)
        
        self.left_meter = ProLEDMeter(width=18, height=180)
        meter_layout.addWidget(self.left_meter)
        
        self.right_meter = ProLEDMeter(width=18, height=180)
        meter_layout.addWidget(self.right_meter)
        
        r_label = QLabel("R")
        r_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: bold;")
        r_label.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(r_label)
        
        layout.addLayout(meter_layout)
        
        # マスターフェーダー
        master_label = QLabel("MASTER")
        master_label.setAlignment(Qt.AlignCenter)
        master_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(master_label)
        
        fader_layout = QHBoxLayout()
        fader_layout.addStretch()
        self.master_fader = ProFader(width=35, height=100)
        self.master_fader.setValue(0.85)
        fader_layout.addWidget(self.master_fader)
        fader_layout.addStretch()
        layout.addLayout(fader_layout)
        
        # SESSIONS
        sessions_label = QLabel("SESSIONS")
        sessions_label.setAlignment(Qt.AlignCenter)
        sessions_label.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
            border: 1px solid {ACCENT};
            border-radius: 3px;
            color: {ACCENT};
            font-weight: bold;
            font-size: 8pt;
            padding: 4px;
        """)
        layout.addWidget(sessions_label)
    
    def _update_levels(self):
        import random
        if not self.mute_btn.isChecked():
            level = random.random() * 0.5 - 60 + 35
            peak = level + random.random() * 5
        else:
            level, peak = -100, -100
        
        self.left_meter.setLevel(level, peak)
        self.right_meter.setLevel(level + random.random() * 3 - 1.5, peak)


# ============================================================================
# 左サイドバー
# ============================================================================
class LeftSidebar(QFrame):
    """左サイドバー - ナビゲーション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(70)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            LeftSidebar {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {PANEL_BG}, stop:1 {DARK_BG});
                border-right: 1px solid #4a4a4a;
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(4)
        
        # MENU
        menu_btn = QPushButton("MENU")
        menu_btn.setFixedHeight(24)
        menu_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a5a5a, stop:1 #3a3a3a);
                border: 1px solid #666;
                border-radius: 3px;
                color: {TEXT_PRIMARY};
                font-weight: bold;
                font-size: 8pt;
            }}
        """)
        layout.addWidget(menu_btn)
        
        # ナビゲーションボタン
        nav_buttons = ["OVERVIEW", "INPUTS", "INSERTS", "SENDS"]
        for text in nav_buttons:
            btn = ToggleButton(text, ACCENT)
            btn.setFixedHeight(28)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # PARAMETERS
        params_label = QLabel("PARAMETERS")
        params_label.setAlignment(Qt.AlignCenter)
        params_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(params_label)
        
        default_btn = QPushButton("DEFAULT")
        default_btn.setFixedHeight(24)
        default_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
                border: 1px solid #555;
                border-radius: 3px;
                color: {TEXT_PRIMARY};
                font-size: 7pt;
            }}
        """)
        layout.addWidget(default_btn)
        
        layout.addStretch()
        
        # CLEAR
        clear_label = QLabel("CLEAR")
        clear_label.setAlignment(Qt.AlignCenter)
        clear_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(clear_label)
        
        clear_layout = QHBoxLayout()
        clear_layout.setSpacing(2)
        
        clips_btn = ToggleButton("CLIPS", "#ff4444")
        clips_btn.setFixedSize(30, 20)
        clear_layout.addWidget(clips_btn)
        
        solo_btn = ToggleButton("SOLO", "#ffff00")
        solo_btn.setFixedSize(30, 20)
        clear_layout.addWidget(solo_btn)
        
        layout.addLayout(clear_layout)
        
        # SETTINGS
        settings_btn = QPushButton("SETTINGS")
        settings_btn.setFixedHeight(24)
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #2a2a2a);
                border: 1px solid #555;
                border-radius: 3px;
                color: {TEXT_PRIMARY};
                font-size: 7pt;
            }}
        """)
        layout.addWidget(settings_btn)


# ============================================================================
# チャンネルストリップエリア
# ============================================================================
class ChannelArea(QFrame):
    """チャンネルストリップエリア"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = []
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            ChannelArea {{
                background-color: {DARK_BG};
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # スクロールエリア
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"background-color: {DARK_BG}; border: none;")
        
        # コンテナ
        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {DARK_BG};")
        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setContentsMargins(4, 4, 4, 4)
        self.container_layout.setSpacing(2)
        self.container_layout.addStretch()
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
    
    def add_channel(self, name: str, device_info: dict = None):
        strip = ChannelStrip(name, device_info)
        strip.removeRequested.connect(lambda s=strip: self.remove_channel(s))
        
        self.container_layout.insertWidget(len(self.channels), strip)
        self.channels.append(strip)
        return strip
    
    def remove_channel(self, strip):
        if strip in self.channels:
            self.channels.remove(strip)
            strip.setParent(None)
            strip.deleteLater()


# ============================================================================
# ボトムバー
# ============================================================================
class BottomBar(QFrame):
    """下部ステータスバー"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            BottomBar {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2a2a2a);
                border-top: 1px solid #4a4a4a;
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(16)
        
        # TEMPO
        tempo_layout = QHBoxLayout()
        tempo_layout.setSpacing(4)
        tempo_label = QLabel("TEMPO")
        tempo_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: bold;")
        tempo_layout.addWidget(tempo_label)
        
        tempo_value = QLabel("120.00")
        tempo_value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 8pt; font-weight: bold;")
        tempo_layout.addWidget(tempo_value)
        layout.addLayout(tempo_layout)
        
        # RATE
        rate_layout = QHBoxLayout()
        rate_layout.setSpacing(4)
        rate_label = QLabel("RATE")
        rate_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: bold;")
        rate_layout.addWidget(rate_label)
        
        rate_value = QLabel("48kHz")
        rate_value.setStyleSheet(f"color: {ACCENT}; font-size: 8pt; font-weight: bold;")
        rate_layout.addWidget(rate_value)
        layout.addLayout(rate_layout)
        
        # CLOCK
        clock_layout = QHBoxLayout()
        clock_layout.setSpacing(4)
        clock_label = QLabel("CLOCK")
        clock_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: bold;")
        clock_layout.addWidget(clock_label)
        
        clock_value = QLabel("INTERNAL")
        clock_value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 8pt; font-weight: bold;")
        clock_layout.addWidget(clock_value)
        layout.addLayout(clock_layout)
        
        layout.addStretch()
        
        # DSP
        dsp_layout = QHBoxLayout()
        dsp_layout.setSpacing(4)
        dsp_label = QLabel("DSP")
        dsp_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 8pt; font-weight: bold;")
        dsp_layout.addWidget(dsp_label)
        
        self.dsp_meter = ProLEDMeter(width=80, height=10, horizontal=True)
        dsp_layout.addWidget(self.dsp_meter)
        
        dsp_percent = QLabel("0%")
        dsp_percent.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 8pt;")
        dsp_layout.addWidget(dsp_percent)
        layout.addLayout(dsp_layout)


# ============================================================================
# メインウィンドウ
# ============================================================================
class UADConsoleProWindow(QMainWindow):
    """UAD Console Professional メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Console: UAD Pro")
        self.setMinimumSize(1200, 700)
        self._setup_ui()
        
        # 初期チャンネル追加
        for i in range(8):
            names = ["Kick", "Snare", "Hat", "Tom Low", "Tom High", "Crush", "OH L/R", "Bass"]
            self.channel_area.add_channel(names[i] if i < len(names) else f"CH {i+1}")
    
    def _setup_ui(self):
        # グローバルスタイル適用
        self.setStyleSheet(GLOBAL_STYLESHEET)
        
        # 中央ウィジェット
        central = QWidget()
        central.setStyleSheet(f"background-color: {DARK_BG};")
        self.setCentralWidget(central)
        
        # メインレイアウト
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # トップツールバー
        self.toolbar = TopToolbar()
        self.toolbar.newSession.connect(self._new_session)
        self.toolbar.openSession.connect(self._open_session)
        self.toolbar.saveSession.connect(self._save_session)
        main_layout.addWidget(self.toolbar)
        
        # メインエリア
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # 左サイドバー
        self.left_sidebar = LeftSidebar()
        content_layout.addWidget(self.left_sidebar)
        
        # チャンネルエリア
        self.channel_area = ChannelArea()
        content_layout.addWidget(self.channel_area, 1)
        
        # マスターセクション
        self.master_section = MasterSection()
        content_layout.addWidget(self.master_section)
        
        main_layout.addLayout(content_layout, 1)
        
        # ボトムバー
        self.bottom_bar = BottomBar()
        main_layout.addWidget(self.bottom_bar)
    
    def _new_session(self):
        # 全チャンネルクリア
        for ch in self.channel_area.channels[:]:
            self.channel_area.remove_channel(ch)
        
        # デフォルトチャンネル追加
        for i in range(4):
            self.channel_area.add_channel(f"CH {i+1}")
    
    def _open_session(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "UAD Session (*.json)"
        )
        if file_path:
            QMessageBox.information(self, "Open", f"Session loaded: {file_path}")
    
    def _save_session(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "UAD Session (*.json)"
        )
        if file_path:
            QMessageBox.information(self, "Save", f"Session saved: {file_path}")


# ============================================================================
# メイン
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 8))
    
    window = UADConsoleProWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())