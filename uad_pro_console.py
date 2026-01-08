"""
UAD Pro Console - オーディオインターフェース特化型
5バンド・パラメトリックEQ（グラフ操作）、VB-CABLE連携、Discord対応
"""
import sys
import os
import math
import wave
import threading
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QComboBox, QMenu, QDialog,
    QFileDialog, QMessageBox, QSizePolicy, QLineEdit, QGroupBox,
    QListWidget, QListWidgetItem, QDialogButtonBox, QTabWidget, QSlider,
    QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint, QPointF, QRectF
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath, QMouseEvent, QPalette
)

try:
    import sounddevice as sd
    import numpy as np
    from scipy import signal as scipy_signal
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("Warning: sounddevice/numpy/scipy not available")

# ============================================================================
# グローバル定数
# ============================================================================
BG_DARK = "#1a1a1a"
BG_PANEL = "#2b2b2b"
BG_SURFACE = "#3a3a3a"
BORDER = "#4a4a4a"
ACCENT = "#ff8c00"
TEXT_W = "#ffffff"
TEXT_G = "#aaaaaa"
GREEN = "#00ff00"
YELLOW = "#ffff00"
RED = "#ff0000"


# ============================================================================
# 5バンド・パラメトリックEQプロセッサ
# ============================================================================
class ParametricEQBand:
    """単一EQバンド - Biquadフィルタ"""
    
    def __init__(self, freq=1000, gain_db=0, q=1.0, sample_rate=48000, band_type='peak'):
        self.freq = freq
        self.gain_db = gain_db
        self.q = q
        self.sample_rate = sample_rate
        self.band_type = band_type  # 'lowshelf', 'peak', 'highshelf'
        self.enabled = True
        
        # Biquad係数
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        
        # フィルタ状態
        self.z = np.zeros((2, 2))  # ステレオ用
        
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Biquad係数を計算"""
        if not AUDIO_OK:
            return
        
        A = 10 ** (self.gain_db / 40)
        w0 = 2 * np.pi * self.freq / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * self.q)
        
        if self.band_type == 'lowshelf':
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        elif self.band_type == 'highshelf':
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        else:  # peak
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0, a1/a0, a2/a0])
    
    def set_params(self, freq=None, gain_db=None, q=None):
        """パラメータ更新"""
        if freq is not None:
            self.freq = max(20, min(20000, freq))
        if gain_db is not None:
            self.gain_db = max(-18, min(18, gain_db))
        if q is not None:
            self.q = max(0.1, min(10, q))
        self._update_coefficients()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """オーディオ処理"""
        if not self.enabled or self.gain_db == 0 or not AUDIO_OK:
            return audio
        
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1] if audio.ndim > 1 else 1):
            x = audio[:, ch] if audio.ndim > 1 else audio
            y = np.zeros_like(x)
            
            for n in range(len(x)):
                y[n] = (self.b[0] * x[n] + self.b[1] * self.z[0, ch] + self.b[2] * self.z[1, ch]
                        - self.a[1] * self.z[0, ch] - self.a[2] * self.z[1, ch])
                self.z[1, ch] = self.z[0, ch]
                self.z[0, ch] = x[n]
            
            if audio.ndim > 1:
                output[:, ch] = y
            else:
                output = y
        
        return output
    
    def get_frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        """周波数レスポンス取得"""
        if not AUDIO_OK:
            return np.ones_like(freqs)
        
        w = 2 * np.pi * freqs / self.sample_rate
        z = np.exp(1j * w)
        
        num = self.b[0] + self.b[1] * z**(-1) + self.b[2] * z**(-2)
        den = self.a[0] + self.a[1] * z**(-1) + self.a[2] * z**(-2)
        
        return np.abs(num / den)


class FiveBandEQ:
    """5バンド・パラメトリックEQ"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.enabled = True
        
        # 5バンド初期設定
        self.bands = [
            ParametricEQBand(80, 0, 0.7, sample_rate, 'lowshelf'),    # Low Shelf
            ParametricEQBand(300, 0, 1.0, sample_rate, 'peak'),       # Low Mid
            ParametricEQBand(1000, 0, 1.0, sample_rate, 'peak'),      # Mid
            ParametricEQBand(3000, 0, 1.0, sample_rate, 'peak'),      # High Mid
            ParametricEQBand(8000, 0, 0.7, sample_rate, 'highshelf'), # High Shelf
        ]
    
    def set_sample_rate(self, rate):
        self.sample_rate = rate
        for band in self.bands:
            band.sample_rate = rate
            band._update_coefficients()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """全バンド処理"""
        if not self.enabled or not AUDIO_OK:
            return audio
        
        for band in self.bands:
            audio = band.process(audio)
        
        return np.clip(audio, -1.0, 1.0)
    
    def get_combined_response(self, freqs: np.ndarray) -> np.ndarray:
        """合成周波数レスポンス"""
        response = np.ones_like(freqs)
        for band in self.bands:
            if band.enabled:
                response *= band.get_frequency_response(freqs)
        return response


# ============================================================================
# グラフィカルEQウィンドウ
# ============================================================================
class EQGraphWidget(QWidget):
    """EQグラフ - ドラッグ操作可能"""
    
    bandChanged = Signal(int, float, float, float)  # band_idx, freq, gain, q
    
    def __init__(self, eq: FiveBandEQ, parent=None):
        super().__init__(parent)
        self.eq = eq
        self.setMinimumSize(600, 300)
        self.setMouseTracking(True)
        
        self.dragging_band = -1
        self.hover_band = -1
        
        # グラフ領域
        self.margin = {'left': 50, 'right': 20, 'top': 20, 'bottom': 40}
        
        # 周波数軸（対数）
        self.freq_min = 20
        self.freq_max = 20000
        
        # ゲイン軸
        self.gain_min = -18
        self.gain_max = 18
        
        # バンドカラー
        self.band_colors = [
            QColor(255, 100, 100),  # Red
            QColor(255, 200, 100),  # Orange
            QColor(100, 255, 100),  # Green
            QColor(100, 200, 255),  # Cyan
            QColor(200, 100, 255),  # Purple
        ]
    
    def freq_to_x(self, freq):
        """周波数→X座標"""
        w = self.width() - self.margin['left'] - self.margin['right']
        log_min = np.log10(self.freq_min)
        log_max = np.log10(self.freq_max)
        log_freq = np.log10(max(self.freq_min, min(self.freq_max, freq)))
        return self.margin['left'] + w * (log_freq - log_min) / (log_max - log_min)
    
    def x_to_freq(self, x):
        """X座標→周波数"""
        w = self.width() - self.margin['left'] - self.margin['right']
        log_min = np.log10(self.freq_min)
        log_max = np.log10(self.freq_max)
        ratio = (x - self.margin['left']) / w
        log_freq = log_min + ratio * (log_max - log_min)
        return 10 ** log_freq
    
    def gain_to_y(self, gain):
        """ゲイン→Y座標"""
        h = self.height() - self.margin['top'] - self.margin['bottom']
        ratio = (gain - self.gain_min) / (self.gain_max - self.gain_min)
        return self.margin['top'] + h * (1 - ratio)
    
    def y_to_gain(self, y):
        """Y座標→ゲイン"""
        h = self.height() - self.margin['top'] - self.margin['bottom']
        ratio = 1 - (y - self.margin['top']) / h
        return self.gain_min + ratio * (self.gain_max - self.gain_min)
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 背景
        p.fillRect(self.rect(), QColor(20, 20, 25))
        
        # グラフ領域
        graph_rect = QRect(
            self.margin['left'], self.margin['top'],
            w - self.margin['left'] - self.margin['right'],
            h - self.margin['top'] - self.margin['bottom']
        )
        
        # グリッド背景
        grid_grad = QLinearGradient(0, graph_rect.top(), 0, graph_rect.bottom())
        grid_grad.setColorAt(0, QColor(30, 30, 35))
        grid_grad.setColorAt(0.5, QColor(25, 25, 30))
        grid_grad.setColorAt(1, QColor(30, 30, 35))
        p.fillRect(graph_rect, grid_grad)
        
        # グリッド線
        p.setPen(QPen(QColor(50, 50, 55), 1))
        
        # 周波数グリッド
        freq_marks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for freq in freq_marks:
            x = self.freq_to_x(freq)
            p.drawLine(int(x), graph_rect.top(), int(x), graph_rect.bottom())
            
            # ラベル
            p.setPen(QColor(100, 100, 100))
            label = f"{freq//1000}k" if freq >= 1000 else str(freq)
            p.drawText(int(x) - 15, h - 10, label)
            p.setPen(QPen(QColor(50, 50, 55), 1))
        
        # ゲイングリッド
        for gain in range(-18, 19, 6):
            y = self.gain_to_y(gain)
            if gain == 0:
                p.setPen(QPen(QColor(80, 80, 80), 1))
            else:
                p.setPen(QPen(QColor(50, 50, 55), 1))
            p.drawLine(graph_rect.left(), int(y), graph_rect.right(), int(y))
            
            # ラベル
            p.setPen(QColor(100, 100, 100))
            p.drawText(5, int(y) + 4, f"{gain:+d}")
        
        # 周波数レスポンス曲線
        if AUDIO_OK:
            freqs = np.logspace(np.log10(20), np.log10(20000), 500)
            response = self.eq.get_combined_response(freqs)
            response_db = 20 * np.log10(response + 1e-10)
            
            # 曲線パス
            path = QPainterPath()
            for i, (f, db) in enumerate(zip(freqs, response_db)):
                x = self.freq_to_x(f)
                y = self.gain_to_y(db)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # 塗りつぶし
            fill_path = QPainterPath(path)
            fill_path.lineTo(self.freq_to_x(20000), self.gain_to_y(0))
            fill_path.lineTo(self.freq_to_x(20), self.gain_to_y(0))
            fill_path.closeSubpath()
            
            fill_grad = QLinearGradient(0, graph_rect.top(), 0, graph_rect.bottom())
            fill_grad.setColorAt(0, QColor(255, 140, 0, 60))
            fill_grad.setColorAt(0.5, QColor(255, 140, 0, 30))
            fill_grad.setColorAt(1, QColor(255, 140, 0, 60))
            p.fillPath(fill_path, fill_grad)
            
            # 曲線
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawPath(path)
        
        # バンドポイント
        for i, band in enumerate(self.eq.bands):
            x = self.freq_to_x(band.freq)
            y = self.gain_to_y(band.gain_db)
            
            color = self.band_colors[i]
            
            # ホバー/ドラッグ時は大きく
            radius = 10 if (i == self.hover_band or i == self.dragging_band) else 7
            
            # 影
            p.setBrush(QColor(0, 0, 0, 100))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x + 2, y + 2), radius, radius)
            
            # ポイント
            point_grad = QRadialGradient(x - 2, y - 2, radius * 1.5)
            point_grad.setColorAt(0, color.lighter(150))
            point_grad.setColorAt(0.5, color)
            point_grad.setColorAt(1, color.darker(150))
            p.setBrush(point_grad)
            p.setPen(QPen(color.lighter(120), 2))
            p.drawEllipse(QPointF(x, y), radius, radius)
            
            # バンド番号
            p.setPen(QColor(0, 0, 0))
            font = p.font()
            font.setBold(True)
            font.setPointSize(8)
            p.setFont(font)
            p.drawText(int(x) - 4, int(y) + 4, str(i + 1))
        
        # 枠
        p.setPen(QPen(QColor(ACCENT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(graph_rect)
    
    def _get_band_at(self, pos) -> int:
        """座標にあるバンドを取得"""
        for i, band in enumerate(self.eq.bands):
            x = self.freq_to_x(band.freq)
            y = self.gain_to_y(band.gain_db)
            dist = ((pos.x() - x) ** 2 + (pos.y() - y) ** 2) ** 0.5
            if dist < 15:
                return i
        return -1
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging_band = self._get_band_at(event.position())
    
    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        
        if self.dragging_band >= 0:
            # ドラッグ中
            freq = self.x_to_freq(pos.x())
            gain = self.y_to_gain(pos.y())
            
            band = self.eq.bands[self.dragging_band]
            band.set_params(freq=freq, gain_db=gain)
            
            self.bandChanged.emit(self.dragging_band, band.freq, band.gain_db, band.q)
            self.update()
        else:
            # ホバー
            new_hover = self._get_band_at(pos)
            if new_hover != self.hover_band:
                self.hover_band = new_hover
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging_band = -1
    
    def wheelEvent(self, event):
        """ホイールでQ値調整"""
        band_idx = self._get_band_at(event.position())
        if band_idx >= 0:
            band = self.eq.bands[band_idx]
            delta = event.angleDelta().y() / 1200
            new_q = band.q + delta
            band.set_params(q=new_q)
            self.bandChanged.emit(band_idx, band.freq, band.gain_db, band.q)
            self.update()


class EQWindow(QDialog):
    """グラフィカルEQウィンドウ"""
    
    def __init__(self, eq: FiveBandEQ, channel_name: str, parent=None):
        super().__init__(parent)
        self.eq = eq
        self.setWindowTitle(f"Parametric EQ - {channel_name}")
        self.setMinimumSize(700, 450)
        self.resize(800, 500)
        
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_DARK};
                color: {TEXT_W};
            }}
            QGroupBox {{
                color: {ACCENT};
                font-weight: bold;
                border: 1px solid {BORDER};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
            }}
            QLabel {{
                color: {TEXT_G};
            }}
            QDoubleSpinBox, QSpinBox {{
                background: {BG_PANEL};
                border: 1px solid {BORDER};
                border-radius: 3px;
                color: {TEXT_W};
                padding: 2px;
            }}
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a);
                border: 1px solid #666;
                border-radius: 4px;
                color: {TEXT_W};
                font-weight: bold;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a);
            }}
            QPushButton:checked {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600);
                color: #000;
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # グラフ
        self.graph = EQGraphWidget(self.eq)
        self.graph.bandChanged.connect(self._on_band_changed)
        layout.addWidget(self.graph, 1)
        
        # バンドコントロール
        bands_layout = QHBoxLayout()
        bands_layout.setSpacing(8)
        
        self.band_controls = []
        band_names = ["LOW", "LO-MID", "MID", "HI-MID", "HIGH"]
        
        for i, (band, name) in enumerate(zip(self.eq.bands, band_names)):
            group = QGroupBox(f"Band {i+1}: {name}")
            group_layout = QGridLayout(group)
            group_layout.setSpacing(4)
            
            # 周波数
            freq_lbl = QLabel("Freq:")
            group_layout.addWidget(freq_lbl, 0, 0)
            
            freq_spin = QSpinBox()
            freq_spin.setRange(20, 20000)
            freq_spin.setValue(int(band.freq))
            freq_spin.setSuffix(" Hz")
            freq_spin.valueChanged.connect(lambda v, idx=i: self._on_freq_changed(idx, v))
            group_layout.addWidget(freq_spin, 0, 1)
            
            # ゲイン
            gain_lbl = QLabel("Gain:")
            group_layout.addWidget(gain_lbl, 1, 0)
            
            gain_spin = QDoubleSpinBox()
            gain_spin.setRange(-18, 18)
            gain_spin.setValue(band.gain_db)
            gain_spin.setSuffix(" dB")
            gain_spin.setSingleStep(0.5)
            gain_spin.valueChanged.connect(lambda v, idx=i: self._on_gain_changed(idx, v))
            group_layout.addWidget(gain_spin, 1, 1)
            
            # Q値
            q_lbl = QLabel("Q:")
            group_layout.addWidget(q_lbl, 2, 0)
            
            q_spin = QDoubleSpinBox()
            q_spin.setRange(0.1, 10)
            q_spin.setValue(band.q)
            q_spin.setSingleStep(0.1)
            q_spin.valueChanged.connect(lambda v, idx=i: self._on_q_changed(idx, v))
            group_layout.addWidget(q_spin, 2, 1)
            
            self.band_controls.append({
                'freq': freq_spin,
                'gain': gain_spin,
                'q': q_spin
            })
            
            bands_layout.addWidget(group)
        
        layout.addLayout(bands_layout)
        
        # ボタン
        btn_layout = QHBoxLayout()
        
        self.enable_btn = QPushButton("EQ ENABLED")
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(self.eq.enabled)
        self.enable_btn.toggled.connect(self._on_enable_toggled)
        btn_layout.addWidget(self.enable_btn)
        
        reset_btn = QPushButton("RESET")
        reset_btn.clicked.connect(self._reset_eq)
        btn_layout.addWidget(reset_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_band_changed(self, idx, freq, gain, q):
        """グラフからの変更"""
        ctrl = self.band_controls[idx]
        ctrl['freq'].blockSignals(True)
        ctrl['gain'].blockSignals(True)
        ctrl['q'].blockSignals(True)
        
        ctrl['freq'].setValue(int(freq))
        ctrl['gain'].setValue(gain)
        ctrl['q'].setValue(q)
        
        ctrl['freq'].blockSignals(False)
        ctrl['gain'].blockSignals(False)
        ctrl['q'].blockSignals(False)
    
    def _on_freq_changed(self, idx, value):
        self.eq.bands[idx].set_params(freq=value)
        self.graph.update()
    
    def _on_gain_changed(self, idx, value):
        self.eq.bands[idx].set_params(gain_db=value)
        self.graph.update()
    
    def _on_q_changed(self, idx, value):
        self.eq.bands[idx].set_params(q=value)
        self.graph.update()
    
    def _on_enable_toggled(self, checked):
        self.eq.enabled = checked
    
    def _reset_eq(self):
        """EQリセット"""
        default_freqs = [80, 300, 1000, 3000, 8000]
        for i, band in enumerate(self.eq.bands):
            band.set_params(freq=default_freqs[i], gain_db=0, q=1.0)
            ctrl = self.band_controls[i]
            ctrl['freq'].setValue(default_freqs[i])
            ctrl['gain'].setValue(0)
            ctrl['q'].setValue(1.0)
        self.graph.update()


# ============================================================================
# オーディオエンジン
# ============================================================================
class AudioEngine:
    """オーディオエンジン - ルーティング・録音・EQ"""
    
    def __init__(self):
        self.input_devices = []
        self.output_devices = []
        self.sample_rate = 48000
        self.buffer_size = 512
        self.channels = {}
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
                    'hostapi': sd.query_hostapis(dev['hostapi'])['name'],
                    'sample_rate': dev['default_samplerate']
                }
                
                if dev['max_input_channels'] > 0:
                    info['type'] = 'input'
                    info['max_ch'] = dev['max_input_channels']
                    self.input_devices.append(info.copy())
                
                if dev['max_output_channels'] > 0:
                    info['type'] = 'output'
                    info['max_ch'] = dev['max_output_channels']
                    self.output_devices.append(info.copy())
        except Exception as e:
            print(f"Device error: {e}")
    
    def set_sample_rate(self, rate: int):
        """サンプリング周波数変更"""
        self.sample_rate = rate
        for ch in self.channels.values():
            ch.eq.set_sample_rate(rate)
            if ch.stream:
                ch.restart()
    
    def create_channel(self, ch_id: str) -> 'ChannelProcessor':
        ch = ChannelProcessor(ch_id, self.sample_rate, self.noise_gate_db)
        self.channels[ch_id] = ch
        return ch
    
    def remove_channel(self, ch_id: str):
        if ch_id in self.channels:
            self.channels[ch_id].stop()
            del self.channels[ch_id]
    
    def cleanup(self):
        for ch_id in list(self.channels.keys()):
            self.remove_channel(ch_id)


class ChannelProcessor:
    """チャンネルプロセッサ"""
    
    def __init__(self, ch_id: str, sample_rate: int, noise_gate_db: float):
        self.ch_id = ch_id
        self.sample_rate = sample_rate
        self.noise_gate_db = noise_gate_db
        
        self.input_device = None
        self.output_device = None
        self.stream = None
        
        # パラメータ
        self.gain = 1.0
        self.pan = 0.5
        self.mute = False
        self.solo = False
        self.phantom = False
        self.pad = False
        self.phase_inv = False
        self.low_cut = False
        
        # 5バンドEQ
        self.eq = FiveBandEQ(sample_rate)
        
        # レベル
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.peak_hold = 30
        self.left_hold_count = 0
        self.right_hold_count = 0
        
        # 録音
        self.recording = False
        self.rec_data = []
        self.rec_path = ""
        
        self.lock = threading.Lock()
    
    def start(self, input_dev: int, output_dev: int = None) -> bool:
        if not AUDIO_OK:
            return False
        
        self.input_device = input_dev
        self.output_device = output_dev
        
        try:
            if output_dev is not None:
                self.stream = sd.Stream(
                    device=(input_dev, output_dev),
                    channels=2,
                    samplerate=self.sample_rate,
                    blocksize=512,
                    callback=self._stream_callback
                )
            else:
                self.stream = sd.InputStream(
                    device=input_dev,
                    channels=2,
                    samplerate=self.sample_rate,
                    blocksize=512,
                    callback=self._input_callback
                )
            
            self.stream.start()
            return True
        except Exception as e:
            print(f"Stream error [{self.ch_id}]: {e}")
            return False
    
    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.recording:
            self.stop_recording()
    
    def restart(self):
        """ストリーム再起動"""
        if self.input_device is not None:
            self.stop()
            self.start(self.input_device, self.output_device)
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        
        # ステレオ化
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])
        
        # 位相反転
        if self.phase_inv:
            audio = -audio
        
        # PAD (-20dB)
        if self.pad:
            audio = audio * 0.1
        
        # ローカット (80Hz HPF)
        if self.low_cut and AUDIO_OK:
            b, a = scipy_signal.butter(2, 80, btype='highpass', fs=self.sample_rate)
            audio = scipy_signal.lfilter(b, a, audio, axis=0)
        
        # ゲイン
        audio = audio * self.gain
        
        # EQ
        audio = self.eq.process(audio)
        
        # パン
        left_gain = np.sqrt(1 - self.pan)
        right_gain = np.sqrt(self.pan)
        audio[:, 0] *= left_gain * 1.414
        audio[:, 1] *= right_gain * 1.414
        
        # ミュート
        if self.mute:
            audio = audio * 0
        
        return np.clip(audio, -1.0, 1.0)
    
    def _update_levels(self, audio: np.ndarray):
        if audio.size == 0:
            return
        
        if audio.ndim == 1:
            left = right = audio
        else:
            left = audio[:, 0]
            right = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
        
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))
        
        left_db = 20 * np.log10(left_rms) if left_rms > 0 else -100
        right_db = 20 * np.log10(right_rms) if right_rms > 0 else -100
        
        # ノイズゲート
        if left_db < self.noise_gate_db:
            left_db = -100
        if right_db < self.noise_gate_db:
            right_db = -100
        
        with self.lock:
            self.left_db = left_db
            self.right_db = right_db
            
            # ピークホールド
            if left_db > self.left_peak:
                self.left_peak = left_db
                self.left_hold_count = self.peak_hold
            elif self.left_hold_count > 0:
                self.left_hold_count -= 1
            else:
                self.left_peak = max(self.left_peak - 1.5, left_db)
            
            if right_db > self.right_peak:
                self.right_peak = right_db
                self.right_hold_count = self.peak_hold
            elif self.right_hold_count > 0:
                self.right_hold_count -= 1
            else:
                self.right_peak = max(self.right_peak - 1.5, right_db)
    
    def _stream_callback(self, indata, outdata, frames, time_info, status):
        processed = self._process_audio(indata.copy())
        outdata[:] = processed
        self._update_levels(processed)
        
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def _input_callback(self, indata, frames, time_info, status):
        processed = self._process_audio(indata.copy())
        self._update_levels(processed)
        
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def get_levels(self) -> Tuple[float, float, float, float]:
        with self.lock:
            return self.left_db, self.right_db, self.left_peak, self.right_peak
    
    def start_recording(self, filepath: str):
        self.rec_path = filepath
        self.rec_data = []
        self.recording = True
    
    def stop_recording(self) -> Optional[str]:
        self.recording = False
        
        if not self.rec_data:
            return None
        
        try:
            audio = np.concatenate(self.rec_data, axis=0)
            
            with wave.open(self.rec_path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                audio_int = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int.tobytes())
            
            self.rec_data = []
            return self.rec_path
        except Exception as e:
            print(f"Recording error: {e}")
            return None


# グローバルエンジン
audio_engine = AudioEngine()


# ============================================================================
# UIウィジェット
# ============================================================================
class StereoMeter(QWidget):
    """ステレオLRメーター"""
    
    def __init__(self, width=36, height=200, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.segments = 24
    
    def setLevels(self, left_db, right_db, left_peak=None, right_peak=None):
        self.left_db = max(-60, min(3, left_db)) if left_db > -100 else -100
        self.right_db = max(-60, min(3, right_db)) if right_db > -100 else -100
        if left_peak is not None:
            self.left_peak = max(-60, min(3, left_peak)) if left_peak > -100 else -100
        if right_peak is not None:
            self.right_peak = max(-60, min(3, right_peak)) if right_peak > -100 else -100
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 背景
        p.fillRect(self.rect(), QColor(15, 15, 15))
        
        meter_w = (w - 10) // 2
        
        # L/Rラベル
        p.setPen(QColor(TEXT_G))
        font = p.font()
        font.setPointSize(7)
        font.setBold(True)
        p.setFont(font)
        p.drawText(4, 12, "L")
        p.drawText(w - 12, 12, "R")
        
        # 左メーター
        self._draw_meter(p, 3, 16, meter_w, h - 20, self.left_db, self.left_peak)
        
        # 右メーター
        self._draw_meter(p, meter_w + 7, 16, meter_w, h - 20, self.right_db, self.right_peak)
        
        # dB目盛り
        p.setPen(QColor(80, 80, 80))
        font.setPointSize(5)
        p.setFont(font)
        
        marks = [(1.0, "0"), (0.85, "-6"), (0.7, "-12"), (0.5, "-24"), (0.25, "-36"), (0.0, "-∞")]
        for pos, txt in marks:
            y = 16 + (1 - pos) * (h - 20)
            p.drawText(meter_w + 1, int(y) + 3, txt)
    
    def _draw_meter(self, p, x, y, w, h, level_db, peak_db):
        seg_h = (h - 2) / self.segments
        
        level_norm = (level_db + 60) / 63 if level_db > -100 else 0
        peak_norm = (peak_db + 60) / 63 if peak_db > -100 else 0
        
        active = int(level_norm * self.segments)
        peak_seg = int(peak_norm * self.segments)
        
        for i in range(self.segments):
            seg_y = y + h - (i + 1) * seg_h
            seg_rect = QRect(x, int(seg_y), w, int(seg_h - 1))
            
            ratio = (i + 1) / self.segments
            if ratio <= 0.6:
                on_color = QColor(0, 220, 0)
                off_color = QColor(0, 35, 0)
            elif ratio <= 0.85:
                on_color = QColor(220, 220, 0)
                off_color = QColor(35, 35, 0)
            else:
                on_color = QColor(220, 0, 0)
                off_color = QColor(35, 0, 0)
            
            if i < active:
                # グラデーション
                grad = QLinearGradient(seg_rect.left(), 0, seg_rect.right(), 0)
                grad.setColorAt(0, on_color.darker(120))
                grad.setColorAt(0.5, on_color)
                grad.setColorAt(1, on_color.darker(120))
                p.fillRect(seg_rect, grad)
            elif i == peak_seg - 1 and peak_seg > 0:
                p.fillRect(seg_rect, on_color)
            else:
                p.fillRect(seg_rect, off_color)


class ProFader(QWidget):
    """プロフェッショナルフェーダー"""
    
    valueChanged = Signal(float)
    
    def __init__(self, width=36, height=200, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.value = 0.75
        self.dragging = False
        self.cap_h = 28
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
        tx = (w - 10) // 2
        track = QRect(tx, self.margin, 10, h - 2 * self.margin)
        
        tg = QLinearGradient(track.left(), 0, track.right(), 0)
        tg.setColorAt(0, QColor(5, 5, 5))
        tg.setColorAt(0.3, QColor(20, 20, 20))
        tg.setColorAt(0.7, QColor(20, 20, 20))
        tg.setColorAt(1, QColor(5, 5, 5))
        p.fillRect(track, tg)
        
        # dBスケール
        p.setPen(QColor(70, 70, 70))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)
        
        marks = [(1.0, "+12"), (0.85, "0"), (0.65, "-12"), (0.4, "-24"), (0.15, "-48"), (0.0, "-∞")]
        for pos, txt in marks:
            y = self.margin + (1 - pos) * (h - 2 * self.margin)
            p.drawLine(2, int(y), 6, int(y))
            p.drawText(w - 20, int(y) + 3, txt)
        
        # フェーダーキャップ
        cap_y = self.margin + (1 - self.value) * (h - 2 * self.margin) - self.cap_h // 2
        cap = QRect(2, int(cap_y), w - 4, self.cap_h)
        
        # シルバーグラデーション
        cg = QLinearGradient(0, cap.top(), 0, cap.bottom())
        cg.setColorAt(0, QColor(210, 210, 210))
        cg.setColorAt(0.1, QColor(180, 180, 180))
        cg.setColorAt(0.4, QColor(140, 140, 140))
        cg.setColorAt(0.6, QColor(120, 120, 120))
        cg.setColorAt(0.9, QColor(90, 90, 90))
        cg.setColorAt(1, QColor(60, 60, 60))
        
        p.setBrush(cg)
        p.setPen(QPen(QColor(50, 50, 50), 1))
        p.drawRoundedRect(cap, 3, 3)
        
        # ハイライト
        p.setPen(QPen(QColor(255, 255, 255, 80), 1))
        p.drawLine(cap.left() + 2, cap.top() + 2, cap.right() - 2, cap.top() + 2)
        
        # オレンジインジケーター
        iy = cap.center().y()
        p.setPen(QPen(QColor(255, 140, 0), 3))
        p.drawLine(cap.left() + 6, iy, cap.right() - 6, iy)


class ProKnob(QWidget):
    """プロフェッショナルノブ"""
    
    valueChanged = Signal(float)
    
    def __init__(self, size=40, parent=None):
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
        sg = QRadialGradient(c.x() + 2, c.y() + 2, r + 4)
        sg.setColorAt(0, QColor(0, 0, 0, 120))
        sg.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(sg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r + 4, r + 4)
        
        # ノブ本体
        kg = QRadialGradient(c.x() - r//3, c.y() - r//3, r * 1.4)
        kg.setColorAt(0, QColor(100, 100, 100))
        kg.setColorAt(0.5, QColor(55, 55, 55))
        kg.setColorAt(1, QColor(30, 30, 30))
        p.setBrush(kg)
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawEllipse(c, r, r)
        
        # 光沢エッジ
        eg = QRadialGradient(c.x(), c.y(), r)
        eg.setColorAt(0.8, QColor(0, 0, 0, 0))
        eg.setColorAt(0.95, QColor(180, 180, 180, 80))
        eg.setColorAt(1, QColor(120, 120, 120, 120))
        p.setBrush(eg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r, r)
        
        # ポインター
        angle = -135 + self.value * 270
        rad = math.radians(angle)
        pl = r - 5
        ex = c.x() + pl * math.cos(rad)
        ey = c.y() + pl * math.sin(rad)
        
        p.setPen(QPen(QColor(255, 140, 0), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(c.x(), c.y(), int(ex), int(ey))


class ToggleButton(QPushButton):
    """発光トグルボタン"""
    
    def __init__(self, text, on_color=ACCENT, parent=None):
        super().__init__(text, parent)
        self.on_color = on_color
        self.setCheckable(True)
        self.setFixedHeight(22)
        self._update_style()
        self.toggled.connect(self._update_style)
    
    def _update_style(self):
        if self.isChecked():
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {self.on_color},stop:1 #aa5500);
                    border: 1px solid #888; border-radius: 3px;
                    color: #000; font-weight: bold; font-size: 8pt;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #555,stop:1 #333);
                    border: 1px solid #444; border-radius: 3px;
                    color: #ccc; font-weight: bold; font-size: 8pt;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #666,stop:1 #444);
                }}
            """)


class EQSlot(QFrame):
    """EQスロット - クリックでEQウィンドウ表示"""
    
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.setCursor(Qt.PointingHandCursor)
        self.active = False
        self._update_style()
    
    def _update_style(self):
        if self.active:
            self.setStyleSheet(f"""
                EQSlot {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #aa5500);
                    border: 1px solid #666; border-radius: 3px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                EQSlot {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #3a3a3a,stop:1 #2a2a2a);
                    border: 1px solid #4a4a4a; border-radius: 3px;
                }}
                EQSlot:hover {{ border: 1px solid {ACCENT}; }}
            """)
    
    def setActive(self, active):
        self.active = active
        self._update_style()
    
    def mousePressEvent(self, e):
        self.clicked.emit()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        font = p.font()
        font.setPointSize(7)
        font.setBold(True)
        p.setFont(font)
        
        if self.active:
            p.setPen(QColor(0, 0, 0))
        else:
            p.setPen(QColor(ACCENT))
        
        p.drawText(self.rect(), Qt.AlignCenter, "5-BAND EQ")


# ============================================================================
# チャンネルストリップ
# ============================================================================
class ChannelStrip(QFrame):
    """UAD風チャンネルストリップ"""
    
    removeRequested = Signal()
    
    def __init__(self, ch_id: str, name: str, input_dev: dict = None, output_dev: dict = None, parent=None):
        super().__init__(parent)
        self.ch_id = ch_id
        self.ch_name = name
        self.input_dev = input_dev
        self.output_dev = output_dev
        self.processor = None
        self.eq_window = None
        
        self.setFixedWidth(85)
        self.setMinimumHeight(620)
        self._setup_ui()
        self._setup_style()
        self._setup_audio()
        
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
    
    def _setup_audio(self):
        self.processor = audio_engine.create_channel(self.ch_id)
        
        if self.input_dev:
            input_idx = self.input_dev['index']
            output_idx = self.output_dev['index'] if self.output_dev else None
            self.processor.start(input_idx, output_idx)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 5, 4, 5)
        layout.setSpacing(3)
        
        # チャンネル名
        self.name_lbl = QLabel(self.ch_name)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
            border: 1px solid #555; border-radius: 3px;
            color: {TEXT_W}; font-weight: bold; font-size: 9pt; padding: 3px;
        """)
        layout.addWidget(self.name_lbl)
        
        # INPUT セクション
        input_lbl = QLabel("INPUT")
        input_lbl.setAlignment(Qt.AlignCenter)
        input_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(input_lbl)
        
        # 48V / PAD
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        
        self.btn_48v = ToggleButton("48V", RED)
        self.btn_48v.setFixedSize(34, 18)
        self.btn_48v.toggled.connect(lambda c: setattr(self.processor, 'phantom', c) if self.processor else None)
        row1.addWidget(self.btn_48v)
        
        self.btn_pad = ToggleButton("PAD", "#888")
        self.btn_pad.setFixedSize(34, 18)
        self.btn_pad.toggled.connect(lambda c: setattr(self.processor, 'pad', c) if self.processor else None)
        row1.addWidget(self.btn_pad)
        
        layout.addLayout(row1)
        
        # Φ / LC
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        
        self.btn_phase = ToggleButton("Φ", YELLOW)
        self.btn_phase.setFixedSize(34, 18)
        self.btn_phase.toggled.connect(lambda c: setattr(self.processor, 'phase_inv', c) if self.processor else None)
        row2.addWidget(self.btn_phase)
        
        self.btn_lc = ToggleButton("LC", "#888")
        self.btn_lc.setFixedSize(34, 18)
        self.btn_lc.toggled.connect(lambda c: setattr(self.processor, 'low_cut', c) if self.processor else None)
        row2.addWidget(self.btn_lc)
        
        layout.addLayout(row2)
        
        # GAIN
        gain_lbl = QLabel("GAIN")
        gain_lbl.setAlignment(Qt.AlignCenter)
        gain_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(gain_lbl)
        
        gain_row = QHBoxLayout()
        gain_row.addStretch()
        self.gain_knob = ProKnob(size=36)
        self.gain_knob.setValue(0.5)
        self.gain_knob.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self.gain_knob)
        gain_row.addStretch()
        layout.addLayout(gain_row)
        
        # EQ スロット
        eq_lbl = QLabel("EQ")
        eq_lbl.setAlignment(Qt.AlignCenter)
        eq_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(eq_lbl)
        
        self.eq_slot = EQSlot()
        self.eq_slot.clicked.connect(self._open_eq_window)
        layout.addWidget(self.eq_slot)
        
        # OUTPUT
        output_lbl = QLabel("OUTPUT")
        output_lbl.setAlignment(Qt.AlignCenter)
        output_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(output_lbl)
        
        self.output_combo = QComboBox()
        self.output_combo.setFixedHeight(20)
        self._populate_outputs()
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        layout.addWidget(self.output_combo)
        
        # メーター + フェーダー
        mf_layout = QHBoxLayout()
        mf_layout.setSpacing(3)
        
        self.meter = StereoMeter(width=28, height=160)
        mf_layout.addWidget(self.meter)
        
        self.fader = ProFader(width=32, height=160)
        self.fader.setValue(0.75)
        self.fader.valueChanged.connect(self._on_fader_changed)
        mf_layout.addWidget(self.fader)
        
        layout.addLayout(mf_layout)
        
        # PAN
        pan_row = QHBoxLayout()
        pan_row.setSpacing(2)
        
        pan_lbl = QLabel("PAN")
        pan_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt;")
        pan_row.addWidget(pan_lbl)
        
        self.pan_knob = ProKnob(size=26)
        self.pan_knob.setValue(0.5)
        self.pan_knob.valueChanged.connect(self._on_pan_changed)
        pan_row.addWidget(self.pan_knob)
        
        layout.addLayout(pan_row)
        
        # SOLO / MUTE
        sm_row = QHBoxLayout()
        sm_row.setSpacing(2)
        
        self.solo_btn = ToggleButton("S", YELLOW)
        self.solo_btn.setFixedSize(32, 22)
        self.solo_btn.toggled.connect(lambda c: setattr(self.processor, 'solo', c) if self.processor else None)
        sm_row.addWidget(self.solo_btn)
        
        self.mute_btn = ToggleButton("M", RED)
        self.mute_btn.setFixedSize(32, 22)
        self.mute_btn.toggled.connect(lambda c: setattr(self.processor, 'mute', c) if self.processor else None)
        sm_row.addWidget(self.mute_btn)
        
        layout.addLayout(sm_row)
        
        # REC
        self.rec_btn = ToggleButton("● REC", RED)
        self.rec_btn.setFixedHeight(24)
        self.rec_btn.toggled.connect(self._on_rec_toggled)
        layout.addWidget(self.rec_btn)
    
    def _populate_outputs(self):
        self.output_combo.clear()
        self.output_combo.addItem("MONITOR", None)
        for dev in audio_engine.output_devices:
            name = dev['name'][:16] + ".." if len(dev['name']) > 16 else dev['name']
            self.output_combo.addItem(name, dev)
    
    def _open_eq_window(self):
        if self.processor:
            self.eq_slot.setActive(True)
            self.eq_window = EQWindow(self.processor.eq, self.ch_name, self)
            self.eq_window.finished.connect(lambda: self.eq_slot.setActive(self.processor.eq.enabled))
            self.eq_window.show()
    
    def _on_gain_changed(self, v):
        if self.processor:
            self.processor.gain = v * 4  # 0-4x
    
    def _on_fader_changed(self, v):
        # フェーダーは追加ゲイン
        pass
    
    def _on_pan_changed(self, v):
        if self.processor:
            self.processor.pan = v
    
    def _on_output_changed(self, idx):
        dev = self.output_combo.currentData()
        if self.processor and self.input_dev:
            self.processor.stop()
            output_idx = dev['index'] if dev else None
            self.processor.start(self.input_dev['index'], output_idx)
    
    def _on_rec_toggled(self, checked):
        if not self.processor:
            return
        
        if checked:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rec_{self.ch_name}_{timestamp}.wav"
            self.processor.start_recording(filepath)
        else:
            path = self.processor.stop_recording()
            if path:
                QMessageBox.information(self, "Recording", f"保存完了: {path}")
    
    def _update_levels(self):
        if self.processor:
            l, r, lp, rp = self.processor.get_levels()
            self.meter.setLevels(l, r, lp, rp)
        else:
            self.meter.setLevels(-100, -100)
    
    def cleanup(self):
        self.timer.stop()
        if self.eq_window:
            self.eq_window.close()
        audio_engine.remove_channel(self.ch_id)


# ============================================================================
# マスターセクション
# ============================================================================
class MasterSection(QFrame):
    """マスター・モニターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(180)
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        
        # MONITOR SOURCE
        mon_lbl = QLabel("MONITOR SOURCE")
        mon_lbl.setAlignment(Qt.AlignCenter)
        mon_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(mon_lbl)
        
        # MAIN / ALT
        mon_row = QHBoxLayout()
        mon_row.setSpacing(3)
        
        self.main_btn = ToggleButton("MAIN", GREEN)
        self.main_btn.setChecked(True)
        mon_row.addWidget(self.main_btn)
        
        self.alt1_btn = ToggleButton("ALT 1", "#888")
        mon_row.addWidget(self.alt1_btn)
        
        self.alt2_btn = ToggleButton("ALT 2", "#888")
        mon_row.addWidget(self.alt2_btn)
        
        layout.addLayout(mon_row)
        
        # PHONES
        phones_lbl = QLabel("PHONES")
        phones_lbl.setAlignment(Qt.AlignCenter)
        phones_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(phones_lbl)
        
        phones_row = QHBoxLayout()
        phones_row.addStretch()
        self.phones_knob = ProKnob(size=36)
        self.phones_knob.setValue(0.7)
        phones_row.addWidget(self.phones_knob)
        phones_row.addStretch()
        layout.addLayout(phones_row)
        
        # CONTROL
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(3)
        
        self.mono_btn = ToggleButton("MONO", "#888")
        ctrl_row.addWidget(self.mono_btn)
        
        self.dim_btn = ToggleButton("DIM", YELLOW)
        ctrl_row.addWidget(self.dim_btn)
        
        layout.addLayout(ctrl_row)
        
        ctrl_row2 = QHBoxLayout()
        ctrl_row2.setSpacing(3)
        
        self.mute_btn = ToggleButton("MUTE", RED)
        ctrl_row2.addWidget(self.mute_btn)
        
        self.talk_btn = ToggleButton("TALK", GREEN)
        ctrl_row2.addWidget(self.talk_btn)
        
        layout.addLayout(ctrl_row2)
        
        # ステレオメーター
        meter_lbl = QLabel("OUTPUT LEVEL")
        meter_lbl.setAlignment(Qt.AlignCenter)
        meter_lbl.setStyleSheet(f"color: {TEXT_W}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(meter_lbl)
        
        meter_row = QHBoxLayout()
        meter_row.addStretch()
        self.meter = StereoMeter(width=60, height=200)
        meter_row.addWidget(self.meter)
        meter_row.addStretch()
        layout.addLayout(meter_row)
        
        # MASTER
        master_lbl = QLabel("MASTER")
        master_lbl.setAlignment(Qt.AlignCenter)
        master_lbl.setStyleSheet(f"color: {TEXT_W}; font-weight: bold; font-size: 10pt;")
        layout.addWidget(master_lbl)
        
        fader_row = QHBoxLayout()
        fader_row.addStretch()
        self.master_fader = ProFader(width=44, height=120)
        self.master_fader.setValue(0.85)
        fader_row.addWidget(self.master_fader)
        fader_row.addStretch()
        layout.addLayout(fader_row)
        
        layout.addStretch()
    
    def _update_levels(self):
        total_l = -100
        total_r = -100
        
        for ch in audio_engine.channels.values():
            if not ch.mute:
                l, r, _, _ = ch.get_levels()
                if l > total_l:
                    total_l = l
                if r > total_r:
                    total_r = r
        
        if self.mute_btn.isChecked():
            total_l = total_r = -100
        
        self.meter.setLevels(total_l, total_r, total_l + 3, total_r + 3)


# ============================================================================
# I/O設定ダイアログ
# ============================================================================
class IOSettingsDialog(QDialog):
    """I/O設定ダイアログ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("I/O SETTINGS")
        self.setMinimumSize(550, 450)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_DARK};
                color: {TEXT_W};
            }}
            QGroupBox {{
                color: {ACCENT};
                font-weight: bold;
                border: 1px solid {BORDER};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QListWidget {{
                background-color: {BG_PANEL};
                color: {TEXT_W};
                border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {BG_SURFACE};
            }}
            QListWidget::item:selected {{
                background-color: {ACCENT};
                color: #000;
            }}
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a);
                border: 1px solid #666;
                border-radius: 4px;
                color: {TEXT_W};
                font-weight: bold;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a);
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        tabs = QTabWidget()
        
        # 入力タブ
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        
        input_info = QLabel("マイク、ライン入力などの入力デバイスを選択:")
        input_info.setStyleSheet(f"color: {TEXT_G};")
        input_layout.addWidget(input_info)
        
        self.input_list = QListWidget()
        for dev in audio_engine.input_devices:
            item = QListWidgetItem(f"[{dev['hostapi']}] {dev['name']}")
            item.setData(Qt.UserRole, dev)
            self.input_list.addItem(item)
        input_layout.addWidget(self.input_list)
        
        tabs.addTab(input_tab, "INPUT DEVICES")
        
        # 出力タブ
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        output_info = QLabel("出力先デバイスを選択 (VB-CABLE, スピーカー等):")
        output_info.setStyleSheet(f"color: {TEXT_G};")
        output_layout.addWidget(output_info)
        
        self.output_list = QListWidget()
        for dev in audio_engine.output_devices:
            item = QListWidgetItem(f"[{dev['hostapi']}] {dev['name']}")
            item.setData(Qt.UserRole, dev)
            self.output_list.addItem(item)
        output_layout.addWidget(self.output_list)
        
        # VB-CABLE説明
        vb_info = QLabel(
            "💡 Discord連携:\n"
            "   1. 出力で「VB-CABLE Input」を選択\n"
            "   2. Discordの入力デバイスで「VB-CABLE Output」を選択\n"
            "   → EQ/ゲイン調整後の音声がDiscordで使用可能"
        )
        vb_info.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 10px; background: {BG_PANEL}; border-radius: 4px;")
        output_layout.addWidget(vb_info)
        
        tabs.addTab(output_tab, "OUTPUT DEVICES")
        
        layout.addWidget(tabs)
        
        # ボタン
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("ADD CHANNEL")
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600);
                color: #000;
            }}
        """)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def get_selection(self):
        input_dev = None
        output_dev = None
        
        if self.input_list.currentItem():
            input_dev = self.input_list.currentItem().data(Qt.UserRole)
        
        if self.output_list.currentItem():
            output_dev = self.output_list.currentItem().data(Qt.UserRole)
        
        return input_dev, output_dev


# ============================================================================
# ツールバー・ボトムバー
# ============================================================================
class TopToolbar(QFrame):
    """上部ツールバー"""
    
    newSession = Signal()
    openSession = Signal()
    saveSession = Signal()
    ioSettings = Signal()
    addChannel = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
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
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)
        
        # SESSION
        sess_lbl = QLabel("SESSION")
        sess_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(sess_lbl)
        
        for text, sig in [("NEW", self.newSession), ("OPEN", self.openSession), ("SAVE", self.saveSession)]:
            btn = QPushButton(text)
            btn.setFixedSize(50, 28)
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
            layout.addWidget(btn)
        
        layout.addSpacing(20)
        
        # I/O SETTINGS
        io_btn = QPushButton("I/O SETTINGS")
        io_btn.setFixedSize(100, 28)
        io_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600);
                border: 1px solid #666; border-radius: 3px;
                color: #000; font-weight: bold; font-size: 9pt;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffaa00,stop:1 {ACCENT});
            }}
        """)
        io_btn.clicked.connect(self.ioSettings.emit)
        layout.addWidget(io_btn)
        
        # ADD CHANNEL
        add_btn = QPushButton("+ ADD CH")
        add_btn.setFixedSize(80, 28)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00aa00,stop:1 #006600);
                border: 1px solid #666; border-radius: 3px;
                color: #fff; font-weight: bold; font-size: 9pt;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00cc00,stop:1 #00aa00);
            }}
        """)
        add_btn.clicked.connect(self.addChannel.emit)
        layout.addWidget(add_btn)
        
        layout.addStretch()
        
        # RATE
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(rate_lbl)
        
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["44.1kHz", "48kHz", "96kHz", "192kHz"])
        self.rate_combo.setCurrentText("48kHz")
        self.rate_combo.setFixedSize(80, 24)
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        layout.addWidget(self.rate_combo)
        
        # CLOCK
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(clock_lbl)
        
        self.clock_combo = QComboBox()
        self.clock_combo.addItems(["INTERNAL", "WORD", "S/PDIF"])
        self.clock_combo.setFixedSize(90, 24)
        layout.addWidget(self.clock_combo)
        
        # TITLE
        title_lbl = QLabel("UAD Pro Console")
        title_lbl.setStyleSheet(f"color: {TEXT_W}; font-size: 12pt; font-weight: bold;")
        layout.addWidget(title_lbl)
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44100, "48kHz": 48000, "96kHz": 96000, "192kHz": 192000}
        audio_engine.set_sample_rate(rate_map.get(text, 48000))


class BottomBar(QFrame):
    """下部ステータスバー"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._setup_ui()
        self._setup_style()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(1000)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            BottomBar {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2a2a2a,stop:0.5 #1a1a1a,stop:1 #0a0a0a);
                border-top: 1px solid #4a4a4a;
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 2, 15, 2)
        layout.setSpacing(30)
        
        # RATE
        rate_row = QHBoxLayout()
        rate_row.setSpacing(5)
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        rate_row.addWidget(rate_lbl)
        
        self.rate_val = QLabel("48.0 kHz")
        self.rate_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        rate_row.addWidget(self.rate_val)
        layout.addLayout(rate_row)
        
        # CLOCK
        clock_row = QHBoxLayout()
        clock_row.setSpacing(5)
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        clock_row.addWidget(clock_lbl)
        
        self.clock_val = QLabel("INTERNAL")
        self.clock_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        clock_row.addWidget(self.clock_val)
        layout.addLayout(clock_row)
        
        # DSP/CPU
        dsp_row = QHBoxLayout()
        dsp_row.setSpacing(5)
        dsp_lbl = QLabel("CPU")
        dsp_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        dsp_row.addWidget(dsp_lbl)
        
        self.dsp_val = QLabel("8%")
        self.dsp_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        dsp_row.addWidget(self.dsp_val)
        layout.addLayout(dsp_row)
        
        # CHANNELS
        ch_row = QHBoxLayout()
        ch_row.setSpacing(5)
        ch_lbl = QLabel("CHANNELS")
        ch_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        ch_row.addWidget(ch_lbl)
        
        self.ch_val = QLabel("0")
        self.ch_val.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; font-weight: bold;")
        ch_row.addWidget(self.ch_val)
        layout.addLayout(ch_row)
        
        layout.addStretch()
        
        # TIME
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 9pt;")
        layout.addWidget(self.time_lbl)
        
        # VERSION
        ver_lbl = QLabel("UAD Pro Console v1.0")
        ver_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt;")
        layout.addWidget(ver_lbl)
        
        self._update()
    
    def _update(self):
        self.time_lbl.setText(datetime.now().strftime("%H:%M:%S"))
        self.ch_val.setText(str(len(audio_engine.channels)))
    
    def set_rate(self, rate_khz):
        self.rate_val.setText(f"{rate_khz:.1f} kHz")
    
    def set_clock(self, source):
        self.clock_val.setText(source)


# ============================================================================
# チャンネルエリア
# ============================================================================
class ChannelArea(QFrame):
    """チャンネルストリップエリア"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = []
        self.ch_counter = 0
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"background-color: {BG_DARK};")
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"background-color: {BG_DARK}; border: none;")
        
        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {BG_DARK};")
        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(3)
        self.container_layout.addStretch()
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
    
    def add_channel(self, name, input_dev=None, output_dev=None):
        self.ch_counter += 1
        ch_id = f"ch_{self.ch_counter}"
        
        strip = ChannelStrip(ch_id, name, input_dev, output_dev)
        self.container_layout.insertWidget(len(self.channels), strip)
        self.channels.append(strip)
        return strip
    
    def remove_channel(self, strip):
        if strip in self.channels:
            strip.cleanup()
            self.channels.remove(strip)
            strip.deleteLater()


# ============================================================================
# メインウィンドウ
# ============================================================================
class UADProConsoleWindow(QMainWindow):
    """UAD Pro Console メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAD Pro Console - Audio Interface Manager")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)
        
        self._setup_ui()
        self._setup_style()
        self._connect_signals()
        self._add_default_channel()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_DARK};
            }}
            QComboBox {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
                border: 1px solid #555;
                border-radius: 3px;
                color: {TEXT_W};
                font-size: 8pt;
                padding: 2px 6px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {TEXT_W};
            }}
            QComboBox QAbstractItemView {{
                background: {BG_PANEL};
                color: {TEXT_W};
                selection-background-color: {ACCENT};
                selection-color: #000;
            }}
            QScrollBar:horizontal {{
                background: {BG_DARK};
                height: 12px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a);
                border-radius: 5px;
                min-width: 40px;
                margin: 1px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
        """)
    
    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DARK};")
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ツールバー
        self.toolbar = TopToolbar()
        main_layout.addWidget(self.toolbar)
        
        # メインコンテンツ
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # チャンネルエリア
        self.channel_area = ChannelArea()
        content_layout.addWidget(self.channel_area, 1)
        
        # マスターセクション
        self.master = MasterSection()
        content_layout.addWidget(self.master)
        
        main_layout.addLayout(content_layout, 1)
        
        # ボトムバー
        self.bottom_bar = BottomBar()
        main_layout.addWidget(self.bottom_bar)
    
    def _connect_signals(self):
        self.toolbar.newSession.connect(self._new_session)
        self.toolbar.openSession.connect(self._open_session)
        self.toolbar.saveSession.connect(self._save_session)
        self.toolbar.ioSettings.connect(self._show_io_settings)
        self.toolbar.addChannel.connect(self._show_io_settings)
        
        self.toolbar.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        self.toolbar.clock_combo.currentTextChanged.connect(self._on_clock_changed)
    
    def _add_default_channel(self):
        if audio_engine.input_devices:
            dev = audio_engine.input_devices[0]
            name = dev['name'][:12] if len(dev['name']) > 12 else dev['name']
            self.channel_area.add_channel(name, dev, None)
    
    def _new_session(self):
        reply = QMessageBox.question(
            self, "New Session",
            "現在のセッションを破棄して新規作成しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for strip in list(self.channel_area.channels):
                self.channel_area.remove_channel(strip)
            self._add_default_channel()
    
    def _open_session(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "JSON Files (*.json)"
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for strip in list(self.channel_area.channels):
                    self.channel_area.remove_channel(strip)
                
                for ch_data in data.get('channels', []):
                    strip = self.channel_area.add_channel(ch_data['name'], None, None)
                    strip.fader.setValue(ch_data.get('fader', 0.75))
                    strip.pan_knob.setValue(ch_data.get('pan', 0.5))
                    strip.gain_knob.setValue(ch_data.get('gain', 0.5))
                    strip.mute_btn.setChecked(ch_data.get('mute', False))
                    strip.solo_btn.setChecked(ch_data.get('solo', False))
                    
                    # EQ復元
                    if 'eq_bands' in ch_data and strip.processor:
                        for i, band_data in enumerate(ch_data['eq_bands']):
                            if i < len(strip.processor.eq.bands):
                                strip.processor.eq.bands[i].set_params(
                                    freq=band_data.get('freq'),
                                    gain_db=band_data.get('gain'),
                                    q=band_data.get('q')
                                )
                
                QMessageBox.information(self, "Session", f"読み込み完了: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"読み込みエラー: {e}")
    
    def _save_session(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.json", "JSON Files (*.json)"
        )
        if filepath:
            try:
                data = {
                    'version': '1.0',
                    'sample_rate': audio_engine.sample_rate,
                    'channels': []
                }
                
                for strip in self.channel_area.channels:
                    ch_data = {
                        'name': strip.ch_name,
                        'fader': strip.fader.value,
                        'pan': strip.pan_knob.value,
                        'gain': strip.gain_knob.value,
                        'mute': strip.mute_btn.isChecked(),
                        'solo': strip.solo_btn.isChecked(),
                        'eq_bands': []
                    }
                    
                    if strip.processor:
                        for band in strip.processor.eq.bands:
                            ch_data['eq_bands'].append({
                                'freq': band.freq,
                                'gain': band.gain_db,
                                'q': band.q
                            })
                    
                    data['channels'].append(ch_data)
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, "Session", f"保存完了: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"保存エラー: {e}")
    
    def _show_io_settings(self):
        dialog = IOSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            input_dev, output_dev = dialog.get_selection()
            if input_dev:
                name = input_dev['name'][:12] if len(input_dev['name']) > 12 else input_dev['name']
                self.channel_area.add_channel(name, input_dev, output_dev)
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44.1, "48kHz": 48.0, "96kHz": 96.0, "192kHz": 192.0}
        self.bottom_bar.set_rate(rate_map.get(text, 48.0))
    
    def _on_clock_changed(self, text):
        self.bottom_bar.set_clock(text)
    
    def closeEvent(self, event):
        for strip in self.channel_area.channels:
            strip.cleanup()
        audio_engine.cleanup()
        event.accept()


# ============================================================================
# メイン関数
# ============================================================================
def main():
    app = QApplication(sys.argv)
    
    # フォント
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # ダークパレット
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(TEXT_W))
    palette.setColor(QPalette.Base, QColor(BG_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(BG_SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT_W))
    palette.setColor(QPalette.Button, QColor(BG_PANEL))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_W))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#000000"))
    app.setPalette(palette)
    
    window = UADProConsoleWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
