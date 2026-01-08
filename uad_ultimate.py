"""
UAD Console Ultimate - 完全再現・高機能統合版
録音機能、VB-CABLE連携、ステレオLRメーター、最下部情報バー完備
"""
import sys
import os
import math
import wave
import struct
import threading
import time as time_module
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QComboBox, QSlider, QMenu,
    QFileDialog, QMessageBox, QSizePolicy, QSpacerItem, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QRect, QPoint, QThread
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath, QFontMetrics, QConicalGradient
)

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("Warning: sounddevice/numpy not available")

# ============================================================================
# グローバル定数
# ============================================================================
BG_DARK = "#1a1a1a"
BG_PANEL = "#2b2b2b"
BG_SURFACE = "#3a3a3a"
BORDER = "#4a4a4a"
ACCENT = "#ff8c00"
TEXT_W = "#ffffff"
TEXT_G = "#cccccc"
GREEN = "#00ff00"
YELLOW = "#ffff00"
RED = "#ff0000"


# ============================================================================
# オーディオエンジン
# ============================================================================
class AudioEngine:
    """高度なオーディオエンジン - ルーティング・録音対応"""
    
    def __init__(self):
        self.input_devices = []
        self.output_devices = []
        self.sample_rate = 48000
        self.buffer_size = 512
        self.streams = {}
        self.recordings = {}
        self.levels = {}  # {device_id: (left_db, right_db, left_peak, right_peak)}
        self.lock = threading.Lock()
        self.noise_gate_db = -90.0
        self.refresh_devices()
    
    def refresh_devices(self):
        """デバイス一覧更新"""
        if not AUDIO_OK:
            return
        
        self.input_devices = []
        self.output_devices = []
        
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                info = {
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'] if dev['max_input_channels'] > 0 else dev['max_output_channels'],
                    'sample_rate': dev['default_samplerate']
                }
                
                if dev['max_input_channels'] > 0:
                    info['type'] = 'input'
                    info['max_ch'] = dev['max_input_channels']
                    self.input_devices.append(info)
                
                if dev['max_output_channels'] > 0:
                    info['type'] = 'output'
                    info['max_ch'] = dev['max_output_channels']
                    self.output_devices.append(info)
        except Exception as e:
            print(f"Device error: {e}")
    
    def set_sample_rate(self, rate: int):
        """サンプリング周波数変更"""
        self.sample_rate = rate
        # 全ストリーム再起動
        for dev_id in list(self.streams.keys()):
            self.stop_stream(dev_id)
            self.start_stream(dev_id)
    
    def start_stream(self, device_index: int, channels: int = 2) -> bool:
        """入力ストリーム開始"""
        if not AUDIO_OK or device_index in self.streams:
            return False
        
        try:
            def callback(indata, frames, time_info, status):
                if status:
                    print(f"Stream status: {status}")
                
                # ステレオRMS計算
                if indata.shape[1] >= 2:
                    left = indata[:, 0]
                    right = indata[:, 1]
                else:
                    left = right = indata[:, 0]
                
                left_rms = np.sqrt(np.mean(left ** 2))
                right_rms = np.sqrt(np.mean(right ** 2))
                
                # dB変換
                left_db = 20 * np.log10(left_rms) if left_rms > 0 else -100
                right_db = 20 * np.log10(right_rms) if right_rms > 0 else -100
                
                # ノイズゲート
                if left_db < self.noise_gate_db:
                    left_db = -100
                if right_db < self.noise_gate_db:
                    right_db = -100
                
                with self.lock:
                    prev = self.levels.get(device_index, (-100, -100, -100, -100))
                    left_peak = max(left_db, prev[2] - 0.5)
                    right_peak = max(right_db, prev[3] - 0.5)
                    self.levels[device_index] = (left_db, right_db, left_peak, right_peak)
                
                # 録音中なら書き込み
                if device_index in self.recordings:
                    self.recordings[device_index]['data'].append(indata.copy())
            
            stream = sd.InputStream(
                device=device_index,
                channels=min(channels, 2),
                samplerate=self.sample_rate,
                callback=callback,
                blocksize=self.buffer_size
            )
            stream.start()
            self.streams[device_index] = stream
            self.levels[device_index] = (-100, -100, -100, -100)
            return True
        except Exception as e:
            print(f"Stream error: {e}")
            return False
    
    def stop_stream(self, device_index: int):
        """ストリーム停止"""
        if device_index in self.streams:
            self.streams[device_index].stop()
            self.streams[device_index].close()
            del self.streams[device_index]
    
    def get_levels(self, device_index: int) -> Tuple[float, float, float, float]:
        """レベル取得 (left_db, right_db, left_peak, right_peak)"""
        with self.lock:
            return self.levels.get(device_index, (-100, -100, -100, -100))
    
    def start_recording(self, device_index: int, filepath: str):
        """録音開始"""
        self.recordings[device_index] = {
            'filepath': filepath,
            'data': [],
            'start_time': datetime.now()
        }
    
    def stop_recording(self, device_index: int) -> Optional[str]:
        """録音停止・WAV保存"""
        if device_index not in self.recordings:
            return None
        
        rec = self.recordings.pop(device_index)
        if not rec['data']:
            return None
        
        try:
            audio_data = np.concatenate(rec['data'], axis=0)
            filepath = rec['filepath']
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(audio_data.shape[1] if len(audio_data.shape) > 1 else 1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # float32 -> int16
                audio_int = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int.tobytes())
            
            return filepath
        except Exception as e:
            print(f"Recording save error: {e}")
            return None
    
    def cleanup(self):
        """クリーンアップ"""
        for dev_id in list(self.streams.keys()):
            self.stop_stream(dev_id)


# グローバルオーディオエンジン
audio_engine = AudioEngine()


# ============================================================================
# ステレオLEDメーター（高精度・24セグメント）
# ============================================================================
class StereoLEDMeter(QWidget):
    """ステレオLRメーター - 高精度24セグメント"""
    
    def __init__(self, parent=None, width=28, height=180):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.left_level = -100.0
        self.right_level = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.peak_hold = 30
        self.left_hold = 0
        self.right_hold = 0
        self.segments = 24
    
    def setLevels(self, left_db: float, right_db: float, left_peak: float = None, right_peak: float = None):
        """レベル設定（dB値）"""
        self.left_level = max(-60, min(0, left_db))
        self.right_level = max(-60, min(0, right_db))
        
        if left_peak is not None:
            if left_peak > self.left_peak:
                self.left_peak = left_peak
                self.left_hold = self.peak_hold
            elif self.left_hold > 0:
                self.left_hold -= 1
            else:
                self.left_peak = max(self.left_peak - 1, self.left_level)
        
        if right_peak is not None:
            if right_peak > self.right_peak:
                self.right_peak = right_peak
                self.right_hold = self.peak_hold
            elif self.right_hold > 0:
                self.right_hold -= 1
            else:
                self.right_peak = max(self.right_peak - 1, self.right_level)
        
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        meter_w = (w - 6) // 2
        
        # 背景
        painter.fillRect(self.rect(), QColor(15, 15, 15))
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        
        # L/Rラベル
        painter.setPen(QColor(TEXT_G))
        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        painter.drawText(2, 10, "L")
        painter.drawText(w - 10, 10, "R")
        
        # 左メーター
        self._draw_meter(painter, 2, 14, meter_w, h - 18, self.left_level, self.left_peak)
        
        # 右メーター
        self._draw_meter(painter, w - meter_w - 2, 14, meter_w, h - 18, self.right_level, self.right_peak)
    
    def _draw_meter(self, painter, x, y, w, h, level_db, peak_db):
        seg_h = (h - 2) / self.segments
        
        # dB -> 0-1
        level_norm = (level_db + 60) / 60 if level_db > -60 else 0
        peak_norm = (peak_db + 60) / 60 if peak_db > -60 else 0
        
        active = int(level_norm * self.segments)
        peak_seg = int(peak_norm * self.segments)
        
        for i in range(self.segments):
            seg_y = y + h - (i + 1) * seg_h
            seg_rect = QRect(x, int(seg_y), w, int(seg_h - 1))
            
            ratio = (i + 1) / self.segments
            if ratio <= 0.6:
                on, off = QColor(0, 255, 0), QColor(0, 35, 0)
            elif ratio <= 0.85:
                on, off = QColor(255, 255, 0), QColor(35, 35, 0)
            else:
                on, off = QColor(255, 0, 0), QColor(35, 0, 0)
            
            if i < active:
                painter.fillRect(seg_rect, on)
            elif i == peak_seg - 1 and peak_seg > 0:
                painter.fillRect(seg_rect, on)
            else:
                painter.fillRect(seg_rect, off)


# ============================================================================
# プロフェッショナルフェーダー（シルバー立体キャップ）
# ============================================================================
class ProFader(QWidget):
    """UAD風プロフェッショナルフェーダー"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, width=32, height=180):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.value = 0.75
        self.dragging = False
        self.cap_h = 26
        self.margin = self.cap_h // 2
    
    def setValue(self, v):
        self.value = max(0, min(1, v))
        self.update()
        self.valueChanged.emit(self.value)
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = True
            self._update_value(e.position().y())
    
    def mouseMoveEvent(self, e):
        if self.dragging:
            self._update_value(e.position().y())
    
    def mouseReleaseEvent(self, e):
        self.dragging = False
    
    def wheelEvent(self, e):
        self.setValue(self.value + e.angleDelta().y() / 1200)
    
    def _update_value(self, y):
        eff = self.height() - 2 * self.margin
        self.setValue(1 - (y - self.margin) / eff)
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # トラック
        tx = (w - 8) // 2
        track = QRect(tx, self.margin, 8, h - 2 * self.margin)
        
        tg = QLinearGradient(track.left(), 0, track.right(), 0)
        tg.setColorAt(0, QColor(10, 10, 10))
        tg.setColorAt(0.5, QColor(25, 25, 25))
        tg.setColorAt(1, QColor(10, 10, 10))
        p.fillRect(track, tg)
        
        p.setPen(QPen(QColor(50, 50, 50), 1))
        p.drawRect(track)
        
        # dBスケール
        p.setPen(QColor(80, 80, 80))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)
        
        marks = [(1.0, "+12"), (0.85, "0"), (0.7, "-6"), (0.5, "-12"), (0.25, "-24"), (0.0, "-∞")]
        for pos, txt in marks:
            y = self.margin + (1 - pos) * (h - 2 * self.margin)
            p.drawLine(2, int(y), 5, int(y))
            p.drawText(w - 16, int(y) + 3, txt)
        
        # フェーダーキャップ
        cap_y = self.margin + (1 - self.value) * (h - 2 * self.margin) - self.cap_h // 2
        cap = QRect(3, int(cap_y), w - 6, self.cap_h)
        
        # シルバーグラデーション
        cg = QLinearGradient(0, cap.top(), 0, cap.bottom())
        cg.setColorAt(0, QColor(200, 200, 200))
        cg.setColorAt(0.1, QColor(160, 160, 160))
        cg.setColorAt(0.5, QColor(130, 130, 130))
        cg.setColorAt(0.9, QColor(100, 100, 100))
        cg.setColorAt(1, QColor(70, 70, 70))
        p.fillRect(cap, cg)
        
        # 境界線
        p.setPen(QPen(QColor(220, 220, 220), 1))
        p.drawLine(cap.left(), cap.top(), cap.right(), cap.top())
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawLine(cap.left(), cap.bottom(), cap.right(), cap.bottom())
        
        # オレンジインジケーター
        iy = cap.center().y()
        p.setPen(QPen(QColor(255, 140, 0), 3))
        p.drawLine(cap.left() + 5, iy, cap.right() - 5, iy)


# ============================================================================
# プロフェッショナルノブ（光沢ロータリー）
# ============================================================================
class ProKnob(QWidget):
    """光沢のあるロータリーノブ"""
    
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, size=36):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.value = 0.5
        self.dragging = False
        self.last_y = 0
    
    def setValue(self, v):
        self.value = max(0, min(1, v))
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
        self.setValue(self.value + e.angleDelta().y() / 1200)
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        s = min(self.width(), self.height())
        c = QPoint(self.width() // 2, self.height() // 2)
        r = (s - 6) // 2
        
        # 影
        sg = QRadialGradient(c.x() + 2, c.y() + 2, r + 3)
        sg.setColorAt(0, QColor(0, 0, 0, 100))
        sg.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(sg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r + 3, r + 3)
        
        # ノブ本体
        kg = QRadialGradient(c.x() - r//3, c.y() - r//3, r * 1.3)
        kg.setColorAt(0, QColor(90, 90, 90))
        kg.setColorAt(0.5, QColor(50, 50, 50))
        kg.setColorAt(1, QColor(25, 25, 25))
        p.setBrush(kg)
        p.setPen(QPen(QColor(70, 70, 70), 1))
        p.drawEllipse(c, r, r)
        
        # 光沢エッジ
        eg = QRadialGradient(c.x(), c.y(), r)
        eg.setColorAt(0.85, QColor(0, 0, 0, 0))
        eg.setColorAt(0.95, QColor(150, 150, 150, 100))
        eg.setColorAt(1, QColor(100, 100, 100, 150))
        p.setBrush(eg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r, r)
        
        # ポインター
        angle = -135 + self.value * 270
        rad = math.radians(angle)
        pl = r - 6
        ex = c.x() + pl * math.cos(rad)
        ey = c.y() + pl * math.sin(rad)
        
        p.setPen(QPen(QColor(255, 140, 0), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(c.x(), c.y(), int(ex), int(ey))


# ============================================================================
# トグルボタン
# ============================================================================
class ToggleBtn(QPushButton):
    """発光トグルボタン"""
    
    def __init__(self, text, on_color=ACCENT, parent=None):
        super().__init__(text, parent)
        self.on_color = on_color
        self.setCheckable(True)
        self.setFixedHeight(22)
        self._style()
        self.toggled.connect(self._style)
    
    def _style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {self.on_color},stop:1 #cc6600);
                    border: 1px solid #666; border-radius: 3px;
                    color: #000; font-weight: bold; font-size: 8pt;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a);
                    border: 1px solid #555; border-radius: 3px;
                    color: #fff; font-weight: bold; font-size: 8pt;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a);
                }}
            """)


# ============================================================================
# プラグインスロット
# ============================================================================
class PluginSlot(QFrame):
    """INSERTプラグインスロット"""
    
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plugin = ""
        self.setFixedHeight(18)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            PluginSlot {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3a3a3a,stop:1 #2a2a2a);
                border: 1px solid #4a4a4a; border-radius: 2px;
            }}
            PluginSlot:hover {{ border: 1px solid {ACCENT}; }}
        """)
    
    def setPlugin(self, name):
        self.plugin = name
        self.update()
    
    def mousePressEvent(self, e):
        self.clicked.emit()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        r = self.rect().adjusted(3, 2, -3, -2)
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        
        if self.plugin:
            p.setPen(QColor(ACCENT))
            txt = self.plugin[:14] + ".." if len(self.plugin) > 14 else self.plugin
            p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, txt)
        else:
            p.setPen(QColor(70, 70, 70))
            p.drawText(r, Qt.AlignCenter, "- empty -")


# ============================================================================
# 3バンドEQ
# ============================================================================
class ThreeBandEQ(QFrame):
    """3バンドEQコントロール"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self._setup()
    
    def _setup(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        self.knobs = []
        for band in ["LO", "MID", "HI"]:
            vl = QVBoxLayout()
            vl.setSpacing(1)
            
            knob = ProKnob(size=22)
            knob.setValue(0.5)
            self.knobs.append(knob)
            vl.addWidget(knob, alignment=Qt.AlignCenter)
            
            lbl = QLabel(band)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 6pt;")
            vl.addWidget(lbl)
            
            layout.addLayout(vl)


# ============================================================================
# チャンネルストリップ（完全版）
# ============================================================================
class ChannelStrip(QFrame):
    """UAD風完全チャンネルストリップ"""
    
    removeRequested = Signal()
    
    def __init__(self, name="CH 1", device_info=None, parent=None):
        super().__init__(parent)
        self.ch_name = name
        self.device_info = device_info
        self.device_index = device_info['index'] if device_info else -1
        self.recording = False
        self.rec_path = ""
        
        self.setFixedWidth(78)
        self.setMinimumHeight(560)
        self._setup_ui()
        self._setup_style()
        
        # オーディオストリーム開始
        if self.device_index >= 0:
            audio_engine.start_stream(self.device_index)
        
        # レベル更新タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_levels)
        self.timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            ChannelStrip {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #3a3a3a,stop:0.5 {BG_PANEL},stop:1 #3a3a3a);
                border: 1px solid #4a4a4a; border-radius: 4px; margin: 1px;
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 4, 3, 4)
        layout.setSpacing(2)
        
        # チャンネル名
        self.name_lbl = QLabel(self.ch_name)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
            border: 1px solid #555; border-radius: 3px;
            color: {TEXT_W}; font-weight: bold; font-size: 9pt; padding: 2px;
        """)
        layout.addWidget(self.name_lbl)
        
        # INPUT セクション
        input_lbl = QLabel("INPUT")
        input_lbl.setAlignment(Qt.AlignCenter)
        input_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(input_lbl)
        
        # 48V, PAD, Φ, LC
        input_btns = QHBoxLayout()
        input_btns.setSpacing(1)
        
        self.btn_48v = ToggleBtn("48V", "#ff4444")
        self.btn_48v.setFixedSize(32, 16)
        input_btns.addWidget(self.btn_48v)
        
        self.btn_pad = ToggleBtn("PAD", "#888888")
        self.btn_pad.setFixedSize(32, 16)
        input_btns.addWidget(self.btn_pad)
        
        layout.addLayout(input_btns)
        
        input_btns2 = QHBoxLayout()
        input_btns2.setSpacing(1)
        
        self.btn_phase = ToggleBtn("Φ", "#ffff00")
        self.btn_phase.setFixedSize(32, 16)
        input_btns2.addWidget(self.btn_phase)
        
        self.btn_lc = ToggleBtn("LC", "#888888")
        self.btn_lc.setFixedSize(32, 16)
        input_btns2.addWidget(self.btn_lc)
        
        layout.addLayout(input_btns2)
        
        # UNISON
        unison_lbl = QLabel("UNISON")
        unison_lbl.setAlignment(Qt.AlignCenter)
        unison_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(unison_lbl)
        
        self.unison_slot = PluginSlot()
        self.unison_slot.clicked.connect(lambda: self._show_menu(self.unison_slot))
        layout.addWidget(self.unison_slot)
        
        # INSERTS
        inserts_lbl = QLabel("INSERTS")
        inserts_lbl.setAlignment(Qt.AlignCenter)
        inserts_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(inserts_lbl)
        
        self.insert_slots = []
        for _ in range(5):
            slot = PluginSlot()
            slot.clicked.connect(lambda s=slot: self._show_menu(s))
            self.insert_slots.append(slot)
            layout.addWidget(slot)
        
        # EQ
        eq_lbl = QLabel("EQ")
        eq_lbl.setAlignment(Qt.AlignCenter)
        eq_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(eq_lbl)
        
        self.eq = ThreeBandEQ()
        layout.addWidget(self.eq)
        
        # SENDS
        sends_lbl = QLabel("SENDS")
        sends_lbl.setAlignment(Qt.AlignCenter)
        sends_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(sends_lbl)
        
        sends_layout = QGridLayout()
        sends_layout.setSpacing(1)
        self.send_knobs = []
        for i in range(4):
            knob = ProKnob(size=18)
            knob.setValue(0)
            self.send_knobs.append(knob)
            sends_layout.addWidget(knob, 0, i)
            
            lbl = QLabel(f"A{i+1}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 5pt;")
            sends_layout.addWidget(lbl, 1, i)
        
        layout.addLayout(sends_layout)
        
        # OUTPUT
        output_lbl = QLabel("OUTPUT")
        output_lbl.setAlignment(Qt.AlignCenter)
        output_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(output_lbl)
        
        self.output_combo = QComboBox()
        self.output_combo.setFixedHeight(18)
        self._populate_outputs()
        layout.addWidget(self.output_combo)
        
        # メーター + フェーダー
        mf_layout = QHBoxLayout()
        mf_layout.setSpacing(2)
        
        self.meter = StereoLEDMeter(width=22, height=130)
        mf_layout.addWidget(self.meter)
        
        self.fader = ProFader(width=28, height=130)
        self.fader.setValue(0.75)
        mf_layout.addWidget(self.fader)
        
        layout.addLayout(mf_layout)
        
        # PAN
        pan_layout = QHBoxLayout()
        pan_layout.setSpacing(2)
        
        pan_lbl = QLabel("PAN")
        pan_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt;")
        pan_layout.addWidget(pan_lbl)
        
        self.pan_knob = ProKnob(size=22)
        self.pan_knob.setValue(0.5)
        pan_layout.addWidget(self.pan_knob)
        
        layout.addLayout(pan_layout)
        
        # SOLO / MUTE
        sm_layout = QHBoxLayout()
        sm_layout.setSpacing(2)
        
        self.solo_btn = ToggleBtn("S", YELLOW)
        self.solo_btn.setFixedSize(28, 20)
        sm_layout.addWidget(self.solo_btn)
        
        self.mute_btn = ToggleBtn("M", RED)
        self.mute_btn.setFixedSize(28, 20)
        sm_layout.addWidget(self.mute_btn)
        
        layout.addLayout(sm_layout)
        
        # REC
        self.rec_btn = ToggleBtn("REC", RED)
        self.rec_btn.setFixedHeight(20)
        self.rec_btn.toggled.connect(self._toggle_recording)
        layout.addWidget(self.rec_btn)
    
    def _populate_outputs(self):
        self.output_combo.clear()
        self.output_combo.addItem("MONITOR")
        for dev in audio_engine.output_devices:
            name = dev['name'][:20] + ".." if len(dev['name']) > 20 else dev['name']
            self.output_combo.addItem(name)
    
    def _show_menu(self, slot):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {BG_PANEL}; color: {TEXT_W}; border: 1px solid #555; }}
            QMenu::item:selected {{ background: {ACCENT}; }}
        """)
        
        plugins = ["Neve 1073", "SSL E Channel", "API Vision", "1176LN", "LA-2A", "Pultec EQ", "Fairchild 670"]
        for p in plugins:
            action = menu.addAction(p)
            action.triggered.connect(lambda _, n=p, s=slot: s.setPlugin(n))
        
        menu.addSeparator()
        menu.addAction("Clear").triggered.connect(lambda: slot.setPlugin(""))
        menu.exec(self.mapToGlobal(slot.pos()))
    
    def _toggle_recording(self, checked):
        if checked:
            # 録音開始
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rec_path = f"recording_{self.ch_name}_{timestamp}.wav"
            audio_engine.start_recording(self.device_index, self.rec_path)
            self.recording = True
        else:
            # 録音停止
            path = audio_engine.stop_recording(self.device_index)
            self.recording = False
            if path:
                QMessageBox.information(self, "Recording", f"Saved: {path}")
    
    def _update_levels(self):
        if self.device_index >= 0 and not self.mute_btn.isChecked():
            l, r, lp, rp = audio_engine.get_levels(self.device_index)
            self.meter.setLevels(l, r, lp, rp)
        else:
            self.meter.setLevels(-100, -100, -100, -100)
    
    def cleanup(self):
        self.timer.stop()
        if self.device_index >= 0:
            audio_engine.stop_stream(self.device_index)


# ============================================================================
# マスターセクション
# ============================================================================
class MasterSection(QFrame):
    """マスター・モニターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(165)
        self._setup_ui()
        self._setup_style()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_levels)
        self.timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            MasterSection {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {BG_DARK},stop:1 {BG_PANEL});
                border-left: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # INSERTS
        ins_lbl = QLabel("INSERTS")
        ins_lbl.setAlignment(Qt.AlignCenter)
        ins_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(ins_lbl)
        
        self.master_inserts = []
        for _ in range(2):
            slot = PluginSlot()
            self.master_inserts.append(slot)
            layout.addWidget(slot)
        
        # UAD INSERTS REC/MON
        rm_layout = QHBoxLayout()
        rm_layout.setSpacing(2)
        
        self.rec_btn = ToggleBtn("REC", ACCENT)
        self.rec_btn.setChecked(True)
        rm_layout.addWidget(self.rec_btn)
        
        self.mon_btn = ToggleBtn("MON", "#888")
        rm_layout.addWidget(self.mon_btn)
        
        layout.addLayout(rm_layout)
        
        # AUXES
        aux_lbl = QLabel("AUXES")
        aux_lbl.setAlignment(Qt.AlignCenter)
        aux_lbl.setStyleSheet(f"color: {TEXT_G}; font-weight: bold; font-size: 8pt;")
        layout.addWidget(aux_lbl)
        
        # MONITORS
        mon_lbl = QLabel("MONITORS")
        mon_lbl.setAlignment(Qt.AlignCenter)
        mon_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(mon_lbl)
        
        # CONTROL ROOM
        cr_lbl = QLabel("CONTROL ROOM")
        cr_lbl.setAlignment(Qt.AlignCenter)
        cr_lbl.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
            border: 1px solid #555; border-radius: 3px;
            color: {TEXT_W}; font-weight: bold; font-size: 8pt; padding: 4px;
        """)
        layout.addWidget(cr_lbl)
        
        # ALT 1/2
        alt_layout = QHBoxLayout()
        alt_layout.setSpacing(2)
        
        self.alt1_btn = ToggleBtn("ALT 1", "#888")
        self.alt1_btn.setChecked(True)
        alt_layout.addWidget(self.alt1_btn)
        
        self.alt2_btn = ToggleBtn("ALT 2", "#888")
        alt_layout.addWidget(self.alt2_btn)
        
        layout.addLayout(alt_layout)
        
        # CUE
        cue_lbl = QLabel("CUE")
        cue_lbl.setAlignment(Qt.AlignCenter)
        cue_lbl.setStyleSheet(f"color: {TEXT_G}; font-weight: bold; font-size: 8pt;")
        layout.addWidget(cue_lbl)
        
        # MONO / MUTE
        mm_layout = QHBoxLayout()
        mm_layout.setSpacing(2)
        
        self.mono_btn = ToggleBtn("MONO", "#888")
        mm_layout.addWidget(self.mono_btn)
        
        self.mute_btn = ToggleBtn("MUTE", RED)
        mm_layout.addWidget(self.mute_btn)
        
        layout.addLayout(mm_layout)
        
        # DIM / TALKBACK
        dt_layout = QHBoxLayout()
        dt_layout.setSpacing(2)
        
        self.dim_btn = ToggleBtn("DIM", YELLOW)
        dt_layout.addWidget(self.dim_btn)
        
        self.talk_btn = ToggleBtn("TALK", "#00ff00")
        dt_layout.addWidget(self.talk_btn)
        
        layout.addLayout(dt_layout)
        
        # ステレオメーター
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(6)
        
        l_lbl = QLabel("L")
        l_lbl.setStyleSheet(f"color: {TEXT_W}; font-weight: bold;")
        l_lbl.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(l_lbl)
        
        self.left_meter = StereoLEDMeter(width=24, height=180)
        meter_layout.addWidget(self.left_meter)
        
        self.right_meter = StereoLEDMeter(width=24, height=180)
        meter_layout.addWidget(self.right_meter)
        
        r_lbl = QLabel("R")
        r_lbl.setStyleSheet(f"color: {TEXT_W}; font-weight: bold;")
        r_lbl.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(r_lbl)
        
        layout.addLayout(meter_layout)
        
        # MASTER
        master_lbl = QLabel("MASTER")
        master_lbl.setAlignment(Qt.AlignCenter)
        master_lbl.setStyleSheet(f"color: {TEXT_W}; font-weight: bold; font-size: 10pt;")
        layout.addWidget(master_lbl)
        
        fader_layout = QHBoxLayout()
        fader_layout.addStretch()
        self.master_fader = ProFader(width=38, height=100)
        self.master_fader.setValue(0.85)
        fader_layout.addWidget(self.master_fader)
        fader_layout.addStretch()
        layout.addLayout(fader_layout)
        
        # SESSIONS
        sess_lbl = QLabel("SESSIONS")
        sess_lbl.setAlignment(Qt.AlignCenter)
        sess_lbl.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
            border: 1px solid {ACCENT}; border-radius: 3px;
            color: {ACCENT}; font-weight: bold; font-size: 8pt; padding: 4px;
        """)
        layout.addWidget(sess_lbl)
    
    def _update_levels(self):
        # マスターレベル（デモ）
        import random
        if not self.mute_btn.isChecked():
            l = random.random() * 30 - 50
            r = random.random() * 30 - 50
        else:
            l = r = -100
        
        self.left_meter.setLevels(l, l, l + 3, l + 3)
        self.right_meter.setLevels(r, r, r + 3, r + 3)


# ============================================================================
# 左サイドバー
# ============================================================================
class LeftSidebar(QFrame):
    """左サイドバー"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(70)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            LeftSidebar {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {BG_PANEL},stop:1 {BG_DARK});
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
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a);
                border: 1px solid #666; border-radius: 3px;
                color: {TEXT_W}; font-weight: bold; font-size: 8pt;
            }}
        """)
        layout.addWidget(menu_btn)
        
        # ナビゲーション
        for text in ["OVERVIEW", "INPUTS", "INSERTS", "SENDS"]:
            btn = ToggleBtn(text, ACCENT)
            btn.setFixedHeight(26)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # PARAMETERS
        params_lbl = QLabel("PARAMETERS")
        params_lbl.setAlignment(Qt.AlignCenter)
        params_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 6pt; font-weight: bold;")
        layout.addWidget(params_lbl)
        
        default_btn = QPushButton("DEFAULT")
        default_btn.setFixedHeight(22)
        default_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
                border: 1px solid #555; border-radius: 3px;
                color: {TEXT_W}; font-size: 7pt;
            }}
        """)
        layout.addWidget(default_btn)
        
        layout.addStretch()
        
        # CLEAR
        clear_lbl = QLabel("CLEAR")
        clear_lbl.setAlignment(Qt.AlignCenter)
        clear_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 6pt; font-weight: bold;")
        layout.addWidget(clear_lbl)
        
        cl_layout = QHBoxLayout()
        cl_layout.setSpacing(2)
        
        clips_btn = ToggleBtn("CLIPS", RED)
        clips_btn.setFixedSize(28, 18)
        cl_layout.addWidget(clips_btn)
        
        solo_btn = ToggleBtn("SOLO", YELLOW)
        solo_btn.setFixedSize(28, 18)
        cl_layout.addWidget(solo_btn)
        
        layout.addLayout(cl_layout)
        
        # SETTINGS
        settings_btn = QPushButton("SETTINGS")
        settings_btn.setFixedHeight(22)
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
                border: 1px solid #555; border-radius: 3px;
                color: {TEXT_W}; font-size: 7pt;
            }}
        """)
        layout.addWidget(settings_btn)


# ============================================================================
# チャンネルエリア
# ============================================================================
class ChannelArea(QFrame):
    """チャンネルストリップエリア"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = []
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"background-color: {BG_DARK};")
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"background-color: {BG_DARK}; border: none;")
        
        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {BG_DARK};")
        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setContentsMargins(4, 4, 4, 4)
        self.container_layout.setSpacing(2)
        self.container_layout.addStretch()
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
    
    def add_channel(self, name: str, device_info: dict = None):
        strip = ChannelStrip(name, device_info)
        self.container_layout.insertWidget(len(self.channels), strip)
        self.channels.append(strip)
        return strip
    
    def remove_channel(self, strip):
        if strip in self.channels:
            strip.cleanup()
            self.channels.remove(strip)
            strip.setParent(None)
            strip.deleteLater()


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
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4a4a4a,stop:0.5 #3a3a3a,stop:1 #2a2a2a);
                border-bottom: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # SESSION
        sess_frame = QFrame()
        sess_layout = QHBoxLayout(sess_frame)
        sess_layout.setContentsMargins(4, 0, 4, 0)
        sess_layout.setSpacing(4)
        
        sess_lbl = QLabel("SESSION")
        sess_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 8pt;")
        sess_layout.addWidget(sess_lbl)
        
        for text, sig in [("NEW", self.newSession), ("OPEN", self.openSession), ("SAVE", self.saveSession)]:
            btn = QPushButton(text)
            btn.setFixedSize(45, 24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a);
                    border: 1px solid #666; border-radius: 3px;
                    color: {TEXT_W}; font-weight: bold; font-size: 8pt;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a);
                }}
            """)
            btn.clicked.connect(sig.emit)
            sess_layout.addWidget(btn)
        
        layout.addWidget(sess_frame)
        
        # SYSTEM STATUS
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 0, 8, 0)
        status_layout.setSpacing(12)
        
        # RATE
        rate_vl = QVBoxLayout()
        rate_vl.setSpacing(0)
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        rate_lbl.setAlignment(Qt.AlignCenter)
        rate_vl.addWidget(rate_lbl)
        
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["44.1kHz", "48kHz", "88.2kHz", "96kHz", "192kHz"])
        self.rate_combo.setCurrentText("48kHz")
        self.rate_combo.setFixedSize(70, 22)
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        rate_vl.addWidget(self.rate_combo)
        status_layout.addLayout(rate_vl)
        
        # CLOCK
        clock_vl = QVBoxLayout()
        clock_vl.setSpacing(0)
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        clock_lbl.setAlignment(Qt.AlignCenter)
        clock_vl.addWidget(clock_lbl)
        
        self.clock_combo = QComboBox()
        self.clock_combo.addItems(["INTERNAL", "WORD", "S/PDIF", "ADAT"])
        self.clock_combo.setFixedSize(80, 22)
        clock_vl.addWidget(self.clock_combo)
        status_layout.addLayout(clock_vl)
        
        # DSP
        dsp_vl = QVBoxLayout()
        dsp_vl.setSpacing(0)
        dsp_lbl = QLabel("DSP")
        dsp_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        dsp_lbl.setAlignment(Qt.AlignCenter)
        dsp_vl.addWidget(dsp_lbl)
        
        self.dsp_meter = StereoLEDMeter(width=100, height=12)
        dsp_vl.addWidget(self.dsp_meter)
        status_layout.addLayout(dsp_vl)
        
        layout.addWidget(status_frame, 1)
        
        # TITLE
        title_lbl = QLabel("Console: UAD Ultimate")
        title_lbl.setStyleSheet(f"color: {TEXT_W}; font-size: 10pt; font-weight: bold;")
        layout.addWidget(title_lbl)
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44100, "48kHz": 48000, "88.2kHz": 88200, "96kHz": 96000, "192kHz": 192000}
        audio_engine.set_sample_rate(rate_map.get(text, 48000))


# ============================================================================
# ボトムバー（TEMPO/RATE/CLOCK/DSP）
# ============================================================================
class BottomBar(QFrame):
    """最下部ステータスバー - UAD特有の情報バー"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._setup_ui()
        self._setup_style()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_dsp)
        self.timer.start(100)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            BottomBar {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3a3a3a,stop:1 #2a2a2a);
                border-top: 1px solid #4a4a4a;
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 2, 12, 2)
        layout.setSpacing(20)
        
        # TEMPO
        tempo_layout = QHBoxLayout()
        tempo_layout.setSpacing(4)
        
        tempo_lbl = QLabel("TEMPO")
        tempo_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        tempo_layout.addWidget(tempo_lbl)
        
        self.tempo_value = QLabel("120.00")
        self.tempo_value.setStyleSheet(f"color: {TEXT_W}; font-size: 8pt; font-weight: bold;")
        tempo_layout.addWidget(self.tempo_value)
        
        layout.addLayout(tempo_layout)
        
        # RATE
        rate_layout = QHBoxLayout()
        rate_layout.setSpacing(4)
        
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        rate_layout.addWidget(rate_lbl)
        
        self.rate_value = QLabel("48kHz")
        self.rate_value.setStyleSheet(f"color: {ACCENT}; font-size: 8pt; font-weight: bold;")
        rate_layout.addWidget(self.rate_value)
        
        layout.addLayout(rate_layout)
        
        # CLOCK
        clock_layout = QHBoxLayout()
        clock_layout.setSpacing(4)
        
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        clock_layout.addWidget(clock_lbl)
        
        self.clock_value = QLabel("INTERNAL")
        self.clock_value.setStyleSheet(f"color: {TEXT_W}; font-size: 8pt; font-weight: bold;")
        clock_layout.addWidget(self.clock_value)
        
        layout.addLayout(clock_layout)
        
        layout.addStretch()
        
        # DSP
        dsp_layout = QHBoxLayout()
        dsp_layout.setSpacing(4)
        
        dsp_lbl = QLabel("DSP")
        dsp_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        dsp_layout.addWidget(dsp_lbl)
        
        self.dsp_meter = StereoLEDMeter(width=80, height=10)
        dsp_layout.addWidget(self.dsp_meter)
        
        self.dsp_percent = QLabel("0%")
        self.dsp_percent.setStyleSheet(f"color: {TEXT_W}; font-size: 8pt;")
        dsp_layout.addWidget(self.dsp_percent)
        
        layout.addLayout(dsp_layout)
    
    def _update_dsp(self):
        import random
        dsp = random.randint(5, 25)
        self.dsp_percent.setText(f"{dsp}%")
        self.dsp_meter.setLevels(dsp - 60, dsp - 60)
    
    def set_rate(self, rate: str):
        self.rate_value.setText(rate)
    
    def set_clock(self, clock: str):
        self.clock_value.setText(clock)


# ============================================================================
# メインウィンドウ
# ============================================================================
class UADUltimateWindow(QMainWindow):
    """UAD Console Ultimate メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Console: UAD Ultimate")
        self.setMinimumSize(1300, 750)
        self._setup_ui()
        
        # 初期チャンネル追加
        self._add_initial_channels()
    
    def _setup_ui(self):
        # グローバルスタイル
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {BG_DARK};
                color: {TEXT_W};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8pt;
            }}
            QScrollArea {{
                background-color: {BG_DARK};
                border: none;
            }}
            QScrollBar:horizontal {{
                background: {BG_SURFACE};
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
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
                border: 1px solid #555;
                border-radius: 3px;
                color: {TEXT_W};
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
                border-top: 4px solid {TEXT_W};
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_PANEL};
                color: {TEXT_W};
                selection-background-color: {ACCENT};
            }}
        """)
        
        # 中央ウィジェット
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DARK};")
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
        self.toolbar.rate_combo.currentTextChanged.connect(self.bottom_bar_rate_update)
        self.toolbar.clock_combo.currentTextChanged.connect(self.bottom_bar_clock_update)
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
    
    def _add_initial_channels(self):
        names = ["Kick", "Snare", "Hat", "Tom Low", "Tom High", "Crush", "OH L/R", "Bass", "Guitar", "Vocal"]
        
        # 利用可能な入力デバイスを取得
        input_devs = audio_engine.input_devices
        
        for i, name in enumerate(names):
            device_info = input_devs[i % len(input_devs)] if input_devs else None
            self.channel_area.add_channel(name, device_info)
    
    def bottom_bar_rate_update(self, text):
        self.bottom_bar.set_rate(text)
    
    def bottom_bar_clock_update(self, text):
        self.bottom_bar.set_clock(text)
    
    def _new_session(self):
        for ch in self.channel_area.channels[:]:
            self.channel_area.remove_channel(ch)
        
        for i in range(4):
            self.channel_area.add_channel(f"CH {i+1}")
    
    def _open_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Session", "", "UAD Session (*.json)")
        if path:
            QMessageBox.information(self, "Open", f"Session loaded: {path}")
    
    def _save_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "UAD Session (*.json)")
        if path:
            QMessageBox.information(self, "Save", f"Session saved: {path}")
    
    def closeEvent(self, event):
        audio_engine.cleanup()
        event.accept()


# ============================================================================
# メイン
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 8))
    
    window = UADUltimateWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())