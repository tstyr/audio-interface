"""
UAD Pro Console - Professional Audio Interface Manager
GitHub配布用・完全最終版

Features:
- 5-Band Parametric EQ with graphical curve editing
- Clip indicators with peak hold
- Large stereo L/R meters with dB scale
- VB-CABLE routing for Discord integration
- Dual monitor output (headphones + speakers)
- Session save/load (JSON)
- Recording to WAV
- Noise gate (-90dB threshold)

Requirements: pip install PySide6 sounddevice numpy scipy
"""
import sys
import os
import math
import wave
import threading
import json
from datetime import datetime
from typing import Optional, Tuple
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QComboBox, QDialog,
    QFileDialog, QMessageBox, QTabWidget, QListWidget, QListWidgetItem,
    QSpinBox, QDoubleSpinBox, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint, QPointF
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPainterPath, QPalette
)

try:
    import sounddevice as sd
    import numpy as np
    from scipy import signal as scipy_signal
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Install with: pip install sounddevice numpy scipy")

# ============================================================================
# Color Constants
# ============================================================================
BG_DARK = "#1a1a1a"
BG_PANEL = "#2b2b2b"
BG_SURFACE = "#3a3a3a"
BORDER = "#4a4a4a"
ACCENT = "#ff8c00"
TEXT_WHITE = "#ffffff"
TEXT_GRAY = "#aaaaaa"
GREEN = "#00ff00"
YELLOW = "#ffff00"
RED = "#ff0000"


# ============================================================================
# 5-Band Parametric EQ
# ============================================================================
class EQBand:
    """Single EQ band using Biquad filter"""
    
    def __init__(self, freq=1000, gain_db=0, q=1.0, sample_rate=48000, band_type='peak'):
        self.freq = freq
        self.gain_db = gain_db
        self.q = q
        self.sample_rate = sample_rate
        self.band_type = band_type
        self.enabled = True
        self.b = np.array([1.0, 0.0, 0.0]) if AUDIO_AVAILABLE else [1, 0, 0]
        self.a = np.array([1.0, 0.0, 0.0]) if AUDIO_AVAILABLE else [1, 0, 0]
        self.z = np.zeros((2, 2)) if AUDIO_AVAILABLE else [[0, 0], [0, 0]]
        self._update_coefficients()
    
    def _update_coefficients(self):
        if not AUDIO_AVAILABLE:
            return
        
        A = 10 ** (self.gain_db / 40)
        w0 = 2 * np.pi * self.freq / self.sample_rate
        cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
        alpha = sin_w0 / (2 * self.q)
        
        if self.band_type == 'lowshelf':
            sq = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + sq)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - sq)
            a0 = (A + 1) + (A - 1) * cos_w0 + sq
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - sq
        elif self.band_type == 'highshelf':
            sq = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + sq)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - sq)
            a0 = (A + 1) - (A - 1) * cos_w0 + sq
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - sq
        else:
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0, a1/a0, a2/a0])
    
    def set_params(self, freq=None, gain_db=None, q=None):
        if freq is not None:
            self.freq = max(20, min(20000, freq))
        if gain_db is not None:
            self.gain_db = max(-18, min(18, gain_db))
        if q is not None:
            self.q = max(0.1, min(10, q))
        self._update_coefficients()
    
    def get_response(self, freqs):
        if not AUDIO_AVAILABLE:
            return np.ones_like(freqs)
        w = 2 * np.pi * freqs / self.sample_rate
        z = np.exp(1j * w)
        num = self.b[0] + self.b[1] * z**(-1) + self.b[2] * z**(-2)
        den = self.a[0] + self.a[1] * z**(-1) + self.a[2] * z**(-2)
        return np.abs(num / den)


class FiveBandEQ:
    """5-Band Parametric EQ"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.enabled = True
        self.bands = [
            EQBand(80, 0, 0.7, sample_rate, 'lowshelf'),
            EQBand(300, 0, 1.0, sample_rate, 'peak'),
            EQBand(1000, 0, 1.0, sample_rate, 'peak'),
            EQBand(3000, 0, 1.0, sample_rate, 'peak'),
            EQBand(8000, 0, 0.7, sample_rate, 'highshelf'),
        ]
    
    def set_sample_rate(self, rate):
        self.sample_rate = rate
        for band in self.bands:
            band.sample_rate = rate
            band._update_coefficients()
    
    def process(self, audio):
        if not self.enabled or not AUDIO_AVAILABLE:
            return audio
        
        for band in self.bands:
            if band.enabled and band.gain_db != 0:
                for ch in range(audio.shape[1] if audio.ndim > 1 else 1):
                    x = audio[:, ch] if audio.ndim > 1 else audio
                    y = scipy_signal.lfilter(band.b, band.a, x)
                    if audio.ndim > 1:
                        audio[:, ch] = y
                    else:
                        audio = y
        
        return np.clip(audio, -1.0, 1.0)
    
    def get_combined_response(self, freqs):
        response = np.ones_like(freqs)
        for band in self.bands:
            if band.enabled:
                response *= band.get_response(freqs)
        return response


# ============================================================================
# EQ Graph Widget
# ============================================================================
class EQGraphWidget(QWidget):
    """Draggable EQ frequency response graph"""
    
    bandChanged = Signal(int, float, float, float)
    
    def __init__(self, eq, parent=None):
        super().__init__(parent)
        self.eq = eq
        self.setMinimumSize(650, 320)
        self.setMouseTracking(True)
        self.dragging_band = -1
        self.hover_band = -1
        self.margin = {'left': 55, 'right': 25, 'top': 25, 'bottom': 45}
        self.band_colors = [
            QColor(255, 100, 100), QColor(255, 180, 80), QColor(100, 255, 100),
            QColor(100, 200, 255), QColor(200, 100, 255)
        ]
    
    def freq_to_x(self, freq):
        w = self.width() - self.margin['left'] - self.margin['right']
        log_ratio = (np.log10(max(20, min(20000, freq))) - np.log10(20)) / (np.log10(20000) - np.log10(20))
        return self.margin['left'] + w * log_ratio
    
    def x_to_freq(self, x):
        w = self.width() - self.margin['left'] - self.margin['right']
        ratio = (x - self.margin['left']) / w
        return 10 ** (np.log10(20) + ratio * (np.log10(20000) - np.log10(20)))
    
    def gain_to_y(self, gain):
        h = self.height() - self.margin['top'] - self.margin['bottom']
        return self.margin['top'] + h * (1 - (gain + 18) / 36)
    
    def y_to_gain(self, y):
        h = self.height() - self.margin['top'] - self.margin['bottom']
        return (1 - (y - self.margin['top']) / h) * 36 - 18
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        # Background
        p.fillRect(self.rect(), QColor(20, 20, 25))
        
        graph_rect = QRect(
            self.margin['left'], self.margin['top'],
            self.width() - self.margin['left'] - self.margin['right'],
            self.height() - self.margin['top'] - self.margin['bottom']
        )
        
        # Grid background
        grad = QLinearGradient(0, graph_rect.top(), 0, graph_rect.bottom())
        grad.setColorAt(0, QColor(35, 35, 40))
        grad.setColorAt(0.5, QColor(28, 28, 33))
        grad.setColorAt(1, QColor(35, 35, 40))
        p.fillRect(graph_rect, grad)
        
        # Grid lines
        p.setPen(QPen(QColor(55, 55, 60), 1))
        
        # Frequency grid
        for freq in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
            x = self.freq_to_x(freq)
            p.drawLine(int(x), graph_rect.top(), int(x), graph_rect.bottom())
            p.setPen(QColor(110, 110, 110))
            label = f"{freq//1000}k" if freq >= 1000 else str(freq)
            p.drawText(int(x) - 12, self.height() - 8, label)
            p.setPen(QPen(QColor(55, 55, 60), 1))
        
        # Gain grid
        for gain in range(-18, 19, 6):
            y = self.gain_to_y(gain)
            pen_color = QColor(90, 90, 90) if gain == 0 else QColor(55, 55, 60)
            p.setPen(QPen(pen_color, 1 if gain != 0 else 2))
            p.drawLine(graph_rect.left(), int(y), graph_rect.right(), int(y))
            p.setPen(QColor(110, 110, 110))
            p.drawText(5, int(y) + 4, f"{gain:+d}dB")
        
        # Frequency response curve
        if AUDIO_AVAILABLE:
            freqs = np.logspace(np.log10(20), np.log10(20000), 400)
            response = self.eq.get_combined_response(freqs)
            response_db = 20 * np.log10(response + 1e-10)
            
            path = QPainterPath()
            for i, (f, db) in enumerate(zip(freqs, response_db)):
                x, y = self.freq_to_x(f), self.gain_to_y(db)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            # Fill
            fill_path = QPainterPath(path)
            fill_path.lineTo(self.freq_to_x(20000), self.gain_to_y(0))
            fill_path.lineTo(self.freq_to_x(20), self.gain_to_y(0))
            fill_path.closeSubpath()
            
            fill_grad = QLinearGradient(0, graph_rect.top(), 0, graph_rect.bottom())
            fill_grad.setColorAt(0, QColor(255, 140, 0, 70))
            fill_grad.setColorAt(0.5, QColor(255, 140, 0, 35))
            fill_grad.setColorAt(1, QColor(255, 140, 0, 70))
            p.fillPath(fill_path, fill_grad)
            
            p.setPen(QPen(QColor(ACCENT), 2.5))
            p.drawPath(path)
        
        # Band control points
        for i, band in enumerate(self.eq.bands):
            x, y = self.freq_to_x(band.freq), self.gain_to_y(band.gain_db)
            color = self.band_colors[i]
            radius = 12 if (i == self.hover_band or i == self.dragging_band) else 9
            
            # Shadow
            p.setBrush(QColor(0, 0, 0, 120))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x + 2, y + 2), radius, radius)
            
            # Point
            grad = QRadialGradient(x - 3, y - 3, radius * 1.5)
            grad.setColorAt(0, color.lighter(160))
            grad.setColorAt(0.5, color)
            grad.setColorAt(1, color.darker(140))
            p.setBrush(grad)
            p.setPen(QPen(color.lighter(130), 2))
            p.drawEllipse(QPointF(x, y), radius, radius)
            
            # Band number
            p.setPen(QColor(0, 0, 0))
            font = p.font()
            font.setBold(True)
            font.setPointSize(9)
            p.setFont(font)
            p.drawText(int(x) - 4, int(y) + 4, str(i + 1))
        
        # Border
        p.setPen(QPen(QColor(ACCENT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(graph_rect)
    
    def _get_band_at(self, pos):
        for i, band in enumerate(self.eq.bands):
            x, y = self.freq_to_x(band.freq), self.gain_to_y(band.gain_db)
            if ((pos.x() - x) ** 2 + (pos.y() - y) ** 2) ** 0.5 < 18:
                return i
        return -1
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging_band = self._get_band_at(e.position())
    
    def mouseMoveEvent(self, e):
        if self.dragging_band >= 0:
            band = self.eq.bands[self.dragging_band]
            band.set_params(freq=self.x_to_freq(e.position().x()), gain_db=self.y_to_gain(e.position().y()))
            self.bandChanged.emit(self.dragging_band, band.freq, band.gain_db, band.q)
            self.update()
        else:
            new_hover = self._get_band_at(e.position())
            if new_hover != self.hover_band:
                self.hover_band = new_hover
                self.update()
    
    def mouseReleaseEvent(self, e):
        self.dragging_band = -1
    
    def wheelEvent(self, e):
        idx = self._get_band_at(e.position())
        if idx >= 0:
            band = self.eq.bands[idx]
            band.set_params(q=band.q + e.angleDelta().y() / 1200)
            self.bandChanged.emit(idx, band.freq, band.gain_db, band.q)
            self.update()


class EQWindow(QDialog):
    """EQ Editor Window"""
    
    def __init__(self, eq, channel_name, parent=None):
        super().__init__(parent)
        self.eq = eq
        self.setWindowTitle(f"5-Band Parametric EQ - {channel_name}")
        self.setMinimumSize(750, 500)
        self.resize(850, 550)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {BG_DARK}; color: {TEXT_WHITE}; }}
            QGroupBox {{ color: {ACCENT}; font-weight: bold; border: 1px solid {BORDER}; border-radius: 4px; margin-top: 10px; padding-top: 10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}
            QLabel {{ color: {TEXT_GRAY}; }}
            QDoubleSpinBox, QSpinBox {{ background: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 3px; color: {TEXT_WHITE}; padding: 3px; }}
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a); border: 1px solid #666; border-radius: 4px; color: {TEXT_WHITE}; font-weight: bold; padding: 8px 14px; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a); }}
            QPushButton:checked {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600); color: #000; }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.graph = EQGraphWidget(self.eq)
        self.graph.bandChanged.connect(self._on_band_changed)
        layout.addWidget(self.graph, 1)
        
        # Band controls
        bands_layout = QHBoxLayout()
        bands_layout.setSpacing(8)
        
        self.band_controls = []
        band_names = ["LOW SHELF", "LOW MID", "MID", "HIGH MID", "HIGH SHELF"]
        
        for i, (band, name) in enumerate(zip(self.eq.bands, band_names)):
            group = QGroupBox(f"{i+1}: {name}")
            gl = QGridLayout(group)
            gl.setSpacing(4)
            
            freq_spin = QSpinBox()
            freq_spin.setRange(20, 20000)
            freq_spin.setValue(int(band.freq))
            freq_spin.setSuffix(" Hz")
            freq_spin.valueChanged.connect(lambda v, idx=i: self._on_spin_changed(idx, 'freq', v))
            gl.addWidget(QLabel("Freq:"), 0, 0)
            gl.addWidget(freq_spin, 0, 1)
            
            gain_spin = QDoubleSpinBox()
            gain_spin.setRange(-18, 18)
            gain_spin.setValue(band.gain_db)
            gain_spin.setSuffix(" dB")
            gain_spin.setSingleStep(0.5)
            gain_spin.valueChanged.connect(lambda v, idx=i: self._on_spin_changed(idx, 'gain', v))
            gl.addWidget(QLabel("Gain:"), 1, 0)
            gl.addWidget(gain_spin, 1, 1)
            
            q_spin = QDoubleSpinBox()
            q_spin.setRange(0.1, 10)
            q_spin.setValue(band.q)
            q_spin.setSingleStep(0.1)
            q_spin.valueChanged.connect(lambda v, idx=i: self._on_spin_changed(idx, 'q', v))
            gl.addWidget(QLabel("Q:"), 2, 0)
            gl.addWidget(q_spin, 2, 1)
            
            self.band_controls.append({'freq': freq_spin, 'gain': gain_spin, 'q': q_spin})
            bands_layout.addWidget(group)
        
        layout.addLayout(bands_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.enable_btn = QPushButton("EQ ENABLED")
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(self.eq.enabled)
        self.enable_btn.toggled.connect(lambda c: setattr(self.eq, 'enabled', c))
        btn_layout.addWidget(self.enable_btn)
        
        reset_btn = QPushButton("RESET ALL")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_band_changed(self, idx, freq, gain, q):
        ctrl = self.band_controls[idx]
        for spin in ctrl.values():
            spin.blockSignals(True)
        ctrl['freq'].setValue(int(freq))
        ctrl['gain'].setValue(gain)
        ctrl['q'].setValue(q)
        for spin in ctrl.values():
            spin.blockSignals(False)
    
    def _on_spin_changed(self, idx, param, value):
        band = self.eq.bands[idx]
        if param == 'freq':
            band.set_params(freq=value)
        elif param == 'gain':
            band.set_params(gain_db=value)
        else:
            band.set_params(q=value)
        self.graph.update()
    
    def _reset(self):
        defaults = [(80, 0, 0.7), (300, 0, 1.0), (1000, 0, 1.0), (3000, 0, 1.0), (8000, 0, 0.7)]
        for i, (freq, gain, q) in enumerate(defaults):
            self.eq.bands[i].set_params(freq=freq, gain_db=gain, q=q)
            self.band_controls[i]['freq'].setValue(freq)
            self.band_controls[i]['gain'].setValue(gain)
            self.band_controls[i]['q'].setValue(q)
        self.graph.update()


# ============================================================================
# Audio Engine
# ============================================================================
class AudioEngine:
    """Core audio processing engine"""
    
    def __init__(self):
        self.input_devices = []
        self.output_devices = []
        self.sample_rate = 48000
        self.channels = {}
        self.monitor_device = None
        self.noise_gate_db = -90.0
        self.refresh_devices()
    
    def refresh_devices(self):
        if not AUDIO_AVAILABLE:
            return
        self.input_devices = []
        self.output_devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                info = {'index': i, 'name': dev['name'], 'hostapi': sd.query_hostapis(dev['hostapi'])['name']}
                if dev['max_input_channels'] > 0:
                    self.input_devices.append({**info, 'type': 'input', 'max_ch': dev['max_input_channels']})
                if dev['max_output_channels'] > 0:
                    self.output_devices.append({**info, 'type': 'output', 'max_ch': dev['max_output_channels']})
        except Exception as e:
            print(f"Device enumeration error: {e}")
    
    def set_sample_rate(self, rate):
        self.sample_rate = rate
        for ch in self.channels.values():
            ch.eq.set_sample_rate(rate)
            ch.restart()
    
    def create_channel(self, ch_id):
        ch = ChannelProcessor(ch_id, self.sample_rate, self.noise_gate_db)
        self.channels[ch_id] = ch
        return ch
    
    def remove_channel(self, ch_id):
        if ch_id in self.channels:
            self.channels[ch_id].stop()
            del self.channels[ch_id]
    
    def cleanup(self):
        for ch_id in list(self.channels.keys()):
            self.remove_channel(ch_id)


class ChannelProcessor:
    """Per-channel audio processor"""
    
    def __init__(self, ch_id, sample_rate, noise_gate_db):
        self.ch_id = ch_id
        self.sample_rate = sample_rate
        self.noise_gate_db = noise_gate_db
        
        self.input_device = None
        self.output_device = None
        self.monitor_device = None
        self.stream = None
        self.monitor_stream = None
        
        # Parameters
        self.gain = 1.0
        self.fader = 1.0
        self.pan = 0.5
        self.mute = False
        self.solo = False
        self.phantom = False
        self.pad = False
        self.phase_inv = False
        self.low_cut = False
        
        # EQ
        self.eq = FiveBandEQ(sample_rate)
        
        # Levels with clip detection
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.left_clip = False
        self.right_clip = False
        self.clip_hold_frames = 0
        self.clip_hold_max = 90  # ~3 seconds at 30fps
        
        # Recording
        self.recording = False
        self.rec_data = []
        self.rec_path = ""
        
        self.lock = threading.Lock()
        self.processed_buffer = None
    
    def start(self, input_dev, output_dev=None, monitor_dev=None):
        if not AUDIO_AVAILABLE:
            return False
        
        self.input_device = input_dev
        self.output_device = output_dev
        self.monitor_device = monitor_dev
        
        try:
            if output_dev is not None:
                self.stream = sd.Stream(
                    device=(input_dev, output_dev), channels=2,
                    samplerate=self.sample_rate, blocksize=512,
                    callback=self._stream_callback
                )
            else:
                self.stream = sd.InputStream(
                    device=input_dev, channels=2,
                    samplerate=self.sample_rate, blocksize=512,
                    callback=self._input_callback
                )
            self.stream.start()
            
            # Monitor output (dual routing)
            if monitor_dev is not None and monitor_dev != output_dev:
                self.monitor_stream = sd.OutputStream(
                    device=monitor_dev, channels=2,
                    samplerate=self.sample_rate, blocksize=512,
                    callback=self._monitor_callback
                )
                self.monitor_stream.start()
            
            return True
        except Exception as e:
            print(f"Stream error [{self.ch_id}]: {e}")
            return False
    
    def stop(self):
        for stream in [self.stream, self.monitor_stream]:
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except:
                    pass
        self.stream = None
        self.monitor_stream = None
        if self.recording:
            self.stop_recording()
    
    def restart(self):
        if self.input_device is not None:
            self.stop()
            self.start(self.input_device, self.output_device, self.monitor_device)
    
    def _process_audio(self, audio):
        if audio.size == 0:
            return audio
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])
        
        # Phase invert
        if self.phase_inv:
            audio = -audio
        
        # PAD (-20dB)
        if self.pad:
            audio = audio * 0.1
        
        # Low cut (80Hz HPF)
        if self.low_cut and AUDIO_AVAILABLE:
            b, a = scipy_signal.butter(2, 80, btype='highpass', fs=self.sample_rate)
            audio = scipy_signal.lfilter(b, a, audio, axis=0)
        
        # Gain + Fader
        audio = audio * self.gain * self.fader
        
        # EQ
        audio = self.eq.process(audio)
        
        # Pan (constant power)
        left_gain = np.cos(self.pan * np.pi / 2)
        right_gain = np.sin(self.pan * np.pi / 2)
        audio[:, 0] *= left_gain
        audio[:, 1] *= right_gain
        
        # Mute
        if self.mute:
            audio = audio * 0
        
        return np.clip(audio, -1.0, 1.0)
    
    def _update_levels(self, audio):
        if audio.size == 0:
            return
        
        left = audio[:, 0] if audio.ndim > 1 else audio
        right = audio[:, 1] if audio.ndim > 1 and audio.shape[1] > 1 else left
        
        # Peak detection (not RMS) for accurate clip detection
        left_peak_lin = np.max(np.abs(left))
        right_peak_lin = np.max(np.abs(right))
        
        # RMS for meter display
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))
        
        left_db = 20 * np.log10(left_rms) if left_rms > 0 else -100
        right_db = 20 * np.log10(right_rms) if right_rms > 0 else -100
        
        # Noise gate
        if left_db < self.noise_gate_db:
            left_db = -100
        if right_db < self.noise_gate_db:
            right_db = -100
        
        with self.lock:
            self.left_db = left_db
            self.right_db = right_db
            
            # Clip detection: if peak >= 1.0 (0dB), set clip flag
            if left_peak_lin >= 1.0:
                self.left_clip = True
                self.clip_hold_frames = self.clip_hold_max
            if right_peak_lin >= 1.0:
                self.right_clip = True
                self.clip_hold_frames = self.clip_hold_max
            
            # Clip hold timer
            if self.clip_hold_frames > 0:
                self.clip_hold_frames -= 1
            else:
                self.left_clip = False
                self.right_clip = False
            
            # Peak hold
            if left_db > self.left_peak:
                self.left_peak = left_db
            else:
                self.left_peak = max(self.left_peak - 1.5, left_db)
            
            if right_db > self.right_peak:
                self.right_peak = right_db
            else:
                self.right_peak = max(self.right_peak - 1.5, right_db)
    
    def _stream_callback(self, indata, outdata, frames, time_info, status):
        processed = self._process_audio(indata.copy())
        outdata[:] = processed
        self.processed_buffer = processed.copy()
        self._update_levels(processed)
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def _input_callback(self, indata, frames, time_info, status):
        processed = self._process_audio(indata.copy())
        self.processed_buffer = processed.copy()
        self._update_levels(processed)
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def _monitor_callback(self, outdata, frames, time_info, status):
        if self.processed_buffer is not None and len(self.processed_buffer) >= frames:
            outdata[:] = self.processed_buffer[:frames]
        else:
            outdata.fill(0)
    
    def get_levels(self):
        with self.lock:
            return self.left_db, self.right_db, self.left_peak, self.right_peak, self.left_clip, self.right_clip
    
    def reset_clip(self):
        with self.lock:
            self.left_clip = False
            self.right_clip = False
            self.clip_hold_frames = 0
    
    def start_recording(self, filepath):
        self.rec_path = filepath
        self.rec_data = []
        self.recording = True
    
    def stop_recording(self):
        self.recording = False
        if not self.rec_data:
            return None
        try:
            audio = np.concatenate(self.rec_data, axis=0)
            with wave.open(self.rec_path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            self.rec_data = []
            return self.rec_path
        except Exception as e:
            print(f"Recording save error: {e}")
            return None


# Global engine
audio_engine = AudioEngine()


# ============================================================================
# Large Stereo Meter with Clip Indicators
# ============================================================================
class LargeStereoMeter(QWidget):
    """Large L/R meter with clip indicators"""
    
    clipClicked = Signal()
    
    def __init__(self, width=100, height=280, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setMouseTracking(True)
        
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.left_clip = False
        self.right_clip = False
        self.segments = 32
    
    def setLevels(self, left_db, right_db, left_peak=None, right_peak=None, left_clip=False, right_clip=False):
        self.left_db = max(-60, min(6, left_db)) if left_db > -100 else -100
        self.right_db = max(-60, min(6, right_db)) if right_db > -100 else -100
        if left_peak is not None:
            self.left_peak = max(-60, min(6, left_peak)) if left_peak > -100 else -100
        if right_peak is not None:
            self.right_peak = max(-60, min(6, right_peak)) if right_peak > -100 else -100
        self.left_clip = left_clip
        self.right_clip = right_clip
        self.update()
    
    def mousePressEvent(self, e):
        # Click to reset clip indicators
        self.clipClicked.emit()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        bg_grad = QLinearGradient(0, 0, w, 0)
        bg_grad.setColorAt(0, QColor(15, 15, 18))
        bg_grad.setColorAt(0.5, QColor(20, 20, 25))
        bg_grad.setColorAt(1, QColor(15, 15, 18))
        p.fillRect(self.rect(), bg_grad)
        
        # Border
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        
        meter_w = (w - 30) // 2
        clip_h = 18
        meter_top = clip_h + 8
        meter_h = h - meter_top - 25
        
        # L/R Labels
        p.setPen(QColor(TEXT_WHITE))
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(8, h - 6, "L")
        p.drawText(w - 18, h - 6, "R")
        
        # Clip indicators (top)
        for i, (clip, label) in enumerate([(self.left_clip, "CLIP"), (self.right_clip, "CLIP")]):
            x = 5 + i * (meter_w + 10)
            clip_rect = QRect(x, 3, meter_w, clip_h)
            
            if clip:
                # Bright red glow when clipping
                glow_grad = QRadialGradient(clip_rect.center().x(), clip_rect.center().y(), meter_w)
                glow_grad.setColorAt(0, QColor(255, 0, 0))
                glow_grad.setColorAt(0.5, QColor(200, 0, 0))
                glow_grad.setColorAt(1, QColor(150, 0, 0))
                p.fillRect(clip_rect, glow_grad)
                p.setPen(QColor(255, 255, 255))
            else:
                p.fillRect(clip_rect, QColor(40, 0, 0))
                p.setPen(QColor(80, 40, 40))
            
            font.setPointSize(7)
            p.setFont(font)
            p.drawText(clip_rect, Qt.AlignCenter, label)
        
        # Left meter
        self._draw_meter(p, 5, meter_top, meter_w, meter_h, self.left_db, self.left_peak)
        
        # Right meter
        self._draw_meter(p, meter_w + 15, meter_top, meter_w, meter_h, self.right_db, self.right_peak)
        
        # dB scale (center)
        p.setPen(QColor(100, 100, 100))
        font.setPointSize(6)
        p.setFont(font)
        
        scale_x = meter_w + 7
        marks = [(1.0, "+6"), (0.91, "0"), (0.82, "-6"), (0.7, "-12"), (0.55, "-24"), (0.35, "-36"), (0.15, "-48"), (0.0, "-∞")]
        for pos, txt in marks:
            y = meter_top + (1 - pos) * meter_h
            p.drawText(scale_x - 2, int(y) + 3, txt)
    
    def _draw_meter(self, p, x, y, w, h, level_db, peak_db):
        seg_h = (h - 4) / self.segments
        
        # Convert dB to normalized (0-1), with +6dB headroom
        level_norm = (level_db + 60) / 66 if level_db > -100 else 0
        peak_norm = (peak_db + 60) / 66 if peak_db > -100 else 0
        
        active = int(level_norm * self.segments)
        peak_seg = int(peak_norm * self.segments)
        
        for i in range(self.segments):
            seg_y = y + h - (i + 1) * seg_h - 2
            seg_rect = QRect(x, int(seg_y), w, int(seg_h - 2))
            
            ratio = (i + 1) / self.segments
            
            # Color zones
            if ratio <= 0.55:  # Green zone (-60 to -24dB)
                on_color = QColor(0, 230, 0)
                off_color = QColor(0, 35, 0)
            elif ratio <= 0.82:  # Yellow zone (-24 to -6dB)
                on_color = QColor(230, 230, 0)
                off_color = QColor(35, 35, 0)
            elif ratio <= 0.91:  # Orange zone (-6 to 0dB)
                on_color = QColor(255, 140, 0)
                off_color = QColor(40, 20, 0)
            else:  # Red zone (0 to +6dB)
                on_color = QColor(255, 0, 0)
                off_color = QColor(40, 0, 0)
            
            if i < active:
                # Gradient for 3D effect
                grad = QLinearGradient(seg_rect.left(), 0, seg_rect.right(), 0)
                grad.setColorAt(0, on_color.darker(130))
                grad.setColorAt(0.3, on_color)
                grad.setColorAt(0.7, on_color)
                grad.setColorAt(1, on_color.darker(130))
                p.fillRect(seg_rect, grad)
            elif i == peak_seg - 1 and peak_seg > 0:
                p.fillRect(seg_rect, on_color)
            else:
                p.fillRect(seg_rect, off_color)


# ============================================================================
# Standard Stereo Meter (for channels)
# ============================================================================
class StereoMeter(QWidget):
    """Channel stereo meter"""
    
    clipClicked = Signal()
    
    def __init__(self, width=36, height=180, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.left_clip = False
        self.right_clip = False
        self.segments = 24
    
    def setLevels(self, left_db, right_db, left_peak=None, right_peak=None, left_clip=False, right_clip=False):
        self.left_db = max(-60, min(6, left_db)) if left_db > -100 else -100
        self.right_db = max(-60, min(6, right_db)) if right_db > -100 else -100
        if left_peak is not None:
            self.left_peak = max(-60, min(6, left_peak)) if left_peak > -100 else -100
        if right_peak is not None:
            self.right_peak = max(-60, min(6, right_peak)) if right_peak > -100 else -100
        self.left_clip = left_clip
        self.right_clip = right_clip
        self.update()
    
    def mousePressEvent(self, e):
        self.clipClicked.emit()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(15, 15, 15))
        
        meter_w = (w - 8) // 2
        clip_h = 10
        meter_top = clip_h + 4
        meter_h = h - meter_top - 16
        
        # L/R labels
        p.setPen(QColor(TEXT_GRAY))
        font = p.font()
        font.setPointSize(6)
        font.setBold(True)
        p.setFont(font)
        p.drawText(3, h - 3, "L")
        p.drawText(w - 10, h - 3, "R")
        
        # Clip indicators
        for i, clip in enumerate([self.left_clip, self.right_clip]):
            x = 2 + i * (meter_w + 4)
            clip_rect = QRect(x, 2, meter_w, clip_h)
            if clip:
                p.fillRect(clip_rect, QColor(255, 0, 0))
            else:
                p.fillRect(clip_rect, QColor(40, 0, 0))
        
        # Meters
        self._draw_meter(p, 2, meter_top, meter_w, meter_h, self.left_db, self.left_peak)
        self._draw_meter(p, meter_w + 6, meter_top, meter_w, meter_h, self.right_db, self.right_peak)
    
    def _draw_meter(self, p, x, y, w, h, level_db, peak_db):
        seg_h = (h - 2) / self.segments
        level_norm = (level_db + 60) / 66 if level_db > -100 else 0
        peak_norm = (peak_db + 60) / 66 if peak_db > -100 else 0
        active = int(level_norm * self.segments)
        peak_seg = int(peak_norm * self.segments)
        
        for i in range(self.segments):
            seg_y = y + h - (i + 1) * seg_h
            seg_rect = QRect(x, int(seg_y), w, int(seg_h - 1))
            ratio = (i + 1) / self.segments
            
            if ratio <= 0.55:
                on, off = QColor(0, 220, 0), QColor(0, 30, 0)
            elif ratio <= 0.85:
                on, off = QColor(220, 220, 0), QColor(30, 30, 0)
            else:
                on, off = QColor(220, 0, 0), QColor(30, 0, 0)
            
            if i < active:
                p.fillRect(seg_rect, on)
            elif i == peak_seg - 1 and peak_seg > 0:
                p.fillRect(seg_rect, on)
            else:
                p.fillRect(seg_rect, off)


# ============================================================================
# UI Widgets
# ============================================================================
class ProFader(QWidget):
    """Professional fader with silver cap"""
    
    valueChanged = Signal(float)
    
    def __init__(self, width=38, height=200, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.value = 0.75
        self.dragging = False
        self.cap_h = 30
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
        
        # Track
        tx = (w - 12) // 2
        track = QRect(tx, self.margin, 12, h - 2 * self.margin)
        
        tg = QLinearGradient(track.left(), 0, track.right(), 0)
        tg.setColorAt(0, QColor(5, 5, 5))
        tg.setColorAt(0.3, QColor(25, 25, 25))
        tg.setColorAt(0.7, QColor(25, 25, 25))
        tg.setColorAt(1, QColor(5, 5, 5))
        p.fillRect(track, tg)
        
        # dB scale
        p.setPen(QColor(80, 80, 80))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)
        
        marks = [(1.0, "+12"), (0.85, "0"), (0.65, "-12"), (0.4, "-24"), (0.15, "-48"), (0.0, "-∞")]
        for pos, txt in marks:
            y = self.margin + (1 - pos) * (h - 2 * self.margin)
            p.drawLine(2, int(y), 7, int(y))
            p.drawText(w - 22, int(y) + 3, txt)
        
        # Fader cap
        cap_y = self.margin + (1 - self.value) * (h - 2 * self.margin) - self.cap_h // 2
        cap = QRect(2, int(cap_y), w - 4, self.cap_h)
        
        # Silver gradient
        cg = QLinearGradient(0, cap.top(), 0, cap.bottom())
        cg.setColorAt(0, QColor(220, 220, 220))
        cg.setColorAt(0.1, QColor(190, 190, 190))
        cg.setColorAt(0.4, QColor(150, 150, 150))
        cg.setColorAt(0.6, QColor(130, 130, 130))
        cg.setColorAt(0.9, QColor(100, 100, 100))
        cg.setColorAt(1, QColor(70, 70, 70))
        
        p.setBrush(cg)
        p.setPen(QPen(QColor(50, 50, 50), 1))
        p.drawRoundedRect(cap, 4, 4)
        
        # Highlight
        p.setPen(QPen(QColor(255, 255, 255, 100), 1))
        p.drawLine(cap.left() + 3, cap.top() + 2, cap.right() - 3, cap.top() + 2)
        
        # Orange indicator
        iy = cap.center().y()
        p.setPen(QPen(QColor(255, 140, 0), 4))
        p.drawLine(cap.left() + 7, iy, cap.right() - 7, iy)


class ProKnob(QWidget):
    """Professional rotary knob"""
    
    valueChanged = Signal(float)
    
    def __init__(self, size=42, parent=None):
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
        
        # Shadow
        sg = QRadialGradient(c.x() + 2, c.y() + 2, r + 5)
        sg.setColorAt(0, QColor(0, 0, 0, 140))
        sg.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(sg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r + 5, r + 5)
        
        # Knob body
        kg = QRadialGradient(c.x() - r//3, c.y() - r//3, r * 1.4)
        kg.setColorAt(0, QColor(110, 110, 110))
        kg.setColorAt(0.5, QColor(60, 60, 60))
        kg.setColorAt(1, QColor(35, 35, 35))
        p.setBrush(kg)
        p.setPen(QPen(QColor(90, 90, 90), 1))
        p.drawEllipse(c, r, r)
        
        # Glossy edge
        eg = QRadialGradient(c.x(), c.y(), r)
        eg.setColorAt(0.8, QColor(0, 0, 0, 0))
        eg.setColorAt(0.95, QColor(200, 200, 200, 90))
        eg.setColorAt(1, QColor(140, 140, 140, 130))
        p.setBrush(eg)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r, r)
        
        # Pointer
        angle = -135 + self.value * 270
        rad = math.radians(angle)
        pl = r - 6
        ex = c.x() + pl * math.cos(rad)
        ey = c.y() + pl * math.sin(rad)
        
        p.setPen(QPen(QColor(255, 140, 0), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(c.x(), c.y(), int(ex), int(ey))


class ToggleButton(QPushButton):
    """Illuminated toggle button"""
    
    def __init__(self, text, on_color=ACCENT, parent=None):
        super().__init__(text, parent)
        self.on_color = on_color
        self.setCheckable(True)
        self.setFixedHeight(24)
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
    """EQ slot button"""
    
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setCursor(Qt.PointingHandCursor)
        self.active = False
        self._update_style()
    
    def _update_style(self):
        color = ACCENT if self.active else "#3a3a3a"
        border = "#666" if self.active else "#4a4a4a"
        self.setStyleSheet(f"""
            EQSlot {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {color},stop:1 #2a2a2a);
                border: 1px solid {border}; border-radius: 3px;
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
        p.setPen(QColor(0, 0, 0) if self.active else QColor(ACCENT))
        p.drawText(self.rect(), Qt.AlignCenter, "5-BAND EQ")


# ============================================================================
# Channel Strip
# ============================================================================
class ChannelStrip(QFrame):
    """Complete channel strip"""
    
    def __init__(self, ch_id, name, input_dev=None, output_dev=None, parent=None):
        super().__init__(parent)
        self.ch_id = ch_id
        self.ch_name = name
        self.input_dev = input_dev
        self.output_dev = output_dev
        self.processor = None
        self.eq_window = None
        
        self.setFixedWidth(90)
        self.setMinimumHeight(650)
        self._setup_ui()
        self._setup_style()
        self._setup_audio()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_levels)
        self.timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            ChannelStrip {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3a3a3a,stop:0.5 {BG_PANEL},stop:1 #3a3a3a);
                border: 1px solid #4a4a4a; border-radius: 4px; margin: 1px;
            }}
        """)
    
    def _setup_audio(self):
        self.processor = audio_engine.create_channel(self.ch_id)
        if self.input_dev:
            output_idx = self.output_dev['index'] if self.output_dev else None
            self.processor.start(self.input_dev['index'], output_idx)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(3)
        
        # Channel name
        self.name_lbl = QLabel(self.ch_name)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
            border: 1px solid #555; border-radius: 3px;
            color: {TEXT_WHITE}; font-weight: bold; font-size: 9pt; padding: 4px;
        """)
        layout.addWidget(self.name_lbl)
        
        # INPUT section
        input_lbl = QLabel("INPUT")
        input_lbl.setAlignment(Qt.AlignCenter)
        input_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(input_lbl)
        
        # 48V / PAD
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        self.btn_48v = ToggleButton("48V", RED)
        self.btn_48v.setFixedSize(36, 18)
        self.btn_48v.toggled.connect(lambda c: setattr(self.processor, 'phantom', c) if self.processor else None)
        row1.addWidget(self.btn_48v)
        self.btn_pad = ToggleButton("PAD", "#888")
        self.btn_pad.setFixedSize(36, 18)
        self.btn_pad.toggled.connect(lambda c: setattr(self.processor, 'pad', c) if self.processor else None)
        row1.addWidget(self.btn_pad)
        layout.addLayout(row1)
        
        # Φ / LC
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        self.btn_phase = ToggleButton("Φ", YELLOW)
        self.btn_phase.setFixedSize(36, 18)
        self.btn_phase.toggled.connect(lambda c: setattr(self.processor, 'phase_inv', c) if self.processor else None)
        row2.addWidget(self.btn_phase)
        self.btn_lc = ToggleButton("LC", "#888")
        self.btn_lc.setFixedSize(36, 18)
        self.btn_lc.toggled.connect(lambda c: setattr(self.processor, 'low_cut', c) if self.processor else None)
        row2.addWidget(self.btn_lc)
        layout.addLayout(row2)
        
        # GAIN
        gain_lbl = QLabel("GAIN")
        gain_lbl.setAlignment(Qt.AlignCenter)
        gain_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(gain_lbl)
        
        gain_row = QHBoxLayout()
        gain_row.addStretch()
        self.gain_knob = ProKnob(size=38)
        self.gain_knob.setValue(0.5)
        self.gain_knob.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self.gain_knob)
        gain_row.addStretch()
        layout.addLayout(gain_row)
        
        # EQ slot
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
        output_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(output_lbl)
        
        self.output_combo = QComboBox()
        self.output_combo.setFixedHeight(22)
        self._populate_outputs()
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        layout.addWidget(self.output_combo)
        
        # Meter + Fader
        mf_layout = QHBoxLayout()
        mf_layout.setSpacing(3)
        
        self.meter = StereoMeter(width=30, height=170)
        self.meter.clipClicked.connect(self._reset_clip)
        mf_layout.addWidget(self.meter)
        
        self.fader = ProFader(width=34, height=170)
        self.fader.setValue(0.75)
        self.fader.valueChanged.connect(self._on_fader_changed)
        mf_layout.addWidget(self.fader)
        
        layout.addLayout(mf_layout)
        
        # PAN
        pan_row = QHBoxLayout()
        pan_row.setSpacing(3)
        pan_lbl = QLabel("PAN")
        pan_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 7pt;")
        pan_row.addWidget(pan_lbl)
        self.pan_knob = ProKnob(size=28)
        self.pan_knob.setValue(0.5)
        self.pan_knob.valueChanged.connect(self._on_pan_changed)
        pan_row.addWidget(self.pan_knob)
        layout.addLayout(pan_row)
        
        # SOLO / MUTE
        sm_row = QHBoxLayout()
        sm_row.setSpacing(2)
        self.solo_btn = ToggleButton("S", YELLOW)
        self.solo_btn.setFixedSize(34, 24)
        self.solo_btn.toggled.connect(lambda c: setattr(self.processor, 'solo', c) if self.processor else None)
        sm_row.addWidget(self.solo_btn)
        self.mute_btn = ToggleButton("M", RED)
        self.mute_btn.setFixedSize(34, 24)
        self.mute_btn.toggled.connect(lambda c: setattr(self.processor, 'mute', c) if self.processor else None)
        sm_row.addWidget(self.mute_btn)
        layout.addLayout(sm_row)
        
        # REC
        self.rec_btn = ToggleButton("● REC", RED)
        self.rec_btn.setFixedHeight(26)
        self.rec_btn.toggled.connect(self._on_rec_toggled)
        layout.addWidget(self.rec_btn)
    
    def _populate_outputs(self):
        self.output_combo.clear()
        self.output_combo.addItem("MONITOR", None)
        for dev in audio_engine.output_devices:
            name = dev['name'][:14] + ".." if len(dev['name']) > 14 else dev['name']
            self.output_combo.addItem(name, dev)
    
    def _open_eq_window(self):
        if self.processor:
            self.eq_slot.setActive(True)
            self.eq_window = EQWindow(self.processor.eq, self.ch_name, self)
            self.eq_window.finished.connect(lambda: self.eq_slot.setActive(self.processor.eq.enabled))
            self.eq_window.show()
    
    def _on_gain_changed(self, v):
        if self.processor:
            self.processor.gain = v * 4
    
    def _on_fader_changed(self, v):
        if self.processor:
            self.processor.fader = v * 1.5  # 0 to 1.5 (+3.5dB max)
    
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
            self.processor.start_recording(f"rec_{self.ch_name}_{timestamp}.wav")
        else:
            path = self.processor.stop_recording()
            if path:
                QMessageBox.information(self, "Recording", f"Saved: {path}")
    
    def _reset_clip(self):
        if self.processor:
            self.processor.reset_clip()
    
    def _update_levels(self):
        if self.processor:
            l, r, lp, rp, lc, rc = self.processor.get_levels()
            self.meter.setLevels(l, r, lp, rp, lc, rc)
        else:
            self.meter.setLevels(-100, -100)
    
    def cleanup(self):
        self.timer.stop()
        if self.eq_window:
            self.eq_window.close()
        audio_engine.remove_channel(self.ch_id)


# ============================================================================
# Master Section
# ============================================================================
class MasterSection(QFrame):
    """Master output section with large meters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self._setup_ui()
        self._setup_style()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_levels)
        self.timer.start(33)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            MasterSection {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {BG_DARK},stop:1 {BG_PANEL});
                border-left: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        
        # MONITOR OUTPUT
        mon_lbl = QLabel("MONITOR OUTPUT")
        mon_lbl.setAlignment(Qt.AlignCenter)
        mon_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 10pt;")
        layout.addWidget(mon_lbl)
        
        # Monitor device selector
        self.monitor_combo = QComboBox()
        self.monitor_combo.setFixedHeight(26)
        self._populate_monitors()
        layout.addWidget(self.monitor_combo)
        
        # ALT switches
        alt_row = QHBoxLayout()
        alt_row.setSpacing(3)
        self.main_btn = ToggleButton("MAIN", GREEN)
        self.main_btn.setChecked(True)
        alt_row.addWidget(self.main_btn)
        self.alt1_btn = ToggleButton("ALT 1", "#888")
        alt_row.addWidget(self.alt1_btn)
        self.alt2_btn = ToggleButton("ALT 2", "#888")
        alt_row.addWidget(self.alt2_btn)
        layout.addLayout(alt_row)
        
        # PHONES
        phones_lbl = QLabel("PHONES")
        phones_lbl.setAlignment(Qt.AlignCenter)
        phones_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(phones_lbl)
        
        phones_row = QHBoxLayout()
        phones_row.addStretch()
        self.phones_knob = ProKnob(size=40)
        self.phones_knob.setValue(0.7)
        phones_row.addWidget(self.phones_knob)
        phones_row.addStretch()
        layout.addLayout(phones_row)
        
        # Control buttons
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
        
        # Large stereo meter
        meter_lbl = QLabel("OUTPUT LEVEL")
        meter_lbl.setAlignment(Qt.AlignCenter)
        meter_lbl.setStyleSheet(f"color: {TEXT_WHITE}; font-size: 9pt; font-weight: bold;")
        layout.addWidget(meter_lbl)
        
        meter_row = QHBoxLayout()
        meter_row.addStretch()
        self.meter = LargeStereoMeter(width=120, height=280)
        self.meter.clipClicked.connect(self._reset_all_clips)
        meter_row.addWidget(self.meter)
        meter_row.addStretch()
        layout.addLayout(meter_row)
        
        # MASTER fader
        master_lbl = QLabel("MASTER")
        master_lbl.setAlignment(Qt.AlignCenter)
        master_lbl.setStyleSheet(f"color: {TEXT_WHITE}; font-weight: bold; font-size: 11pt;")
        layout.addWidget(master_lbl)
        
        fader_row = QHBoxLayout()
        fader_row.addStretch()
        self.master_fader = ProFader(width=50, height=130)
        self.master_fader.setValue(0.85)
        fader_row.addWidget(self.master_fader)
        fader_row.addStretch()
        layout.addLayout(fader_row)
        
        layout.addStretch()
    
    def _populate_monitors(self):
        self.monitor_combo.clear()
        self.monitor_combo.addItem("Default Output", None)
        for dev in audio_engine.output_devices:
            name = dev['name'][:20] + ".." if len(dev['name']) > 20 else dev['name']
            self.monitor_combo.addItem(name, dev)
    
    def _reset_all_clips(self):
        for ch in audio_engine.channels.values():
            ch.reset_clip()
    
    def _update_levels(self):
        total_l = -100
        total_r = -100
        any_clip_l = False
        any_clip_r = False
        
        for ch in audio_engine.channels.values():
            if not ch.mute:
                l, r, lp, rp, lc, rc = ch.get_levels()
                if l > total_l:
                    total_l = l
                if r > total_r:
                    total_r = r
                any_clip_l = any_clip_l or lc
                any_clip_r = any_clip_r or rc
        
        if self.mute_btn.isChecked():
            total_l = total_r = -100
        
        self.meter.setLevels(total_l, total_r, total_l + 3, total_r + 3, any_clip_l, any_clip_r)


# ============================================================================
# I/O Settings Dialog
# ============================================================================
class IOSettingsDialog(QDialog):
    """I/O device selection dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("I/O SETTINGS")
        self.setMinimumSize(600, 500)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {BG_DARK}; color: {TEXT_WHITE}; }}
            QGroupBox {{ color: {ACCENT}; font-weight: bold; border: 1px solid {BORDER}; border-radius: 4px; margin-top: 10px; padding-top: 10px; }}
            QListWidget {{ background-color: {BG_PANEL}; color: {TEXT_WHITE}; border: 1px solid {BORDER}; border-radius: 4px; }}
            QListWidget::item {{ padding: 10px; border-bottom: 1px solid {BG_SURFACE}; }}
            QListWidget::item:selected {{ background-color: {ACCENT}; color: #000; }}
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a); border: 1px solid #666; border-radius: 4px; color: {TEXT_WHITE}; font-weight: bold; padding: 10px 18px; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a); }}
        """)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        tabs = QTabWidget()
        
        # Input tab
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.addWidget(QLabel("Select input device (Microphone, Line-in, etc.):"))
        self.input_list = QListWidget()
        for dev in audio_engine.input_devices:
            item = QListWidgetItem(f"[{dev['hostapi']}] {dev['name']}")
            item.setData(Qt.UserRole, dev)
            self.input_list.addItem(item)
        input_layout.addWidget(self.input_list)
        tabs.addTab(input_tab, "INPUT DEVICES")
        
        # Output tab
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.addWidget(QLabel("Select output device (VB-CABLE, Speaker, etc.):"))
        self.output_list = QListWidget()
        for dev in audio_engine.output_devices:
            item = QListWidgetItem(f"[{dev['hostapi']}] {dev['name']}")
            item.setData(Qt.UserRole, dev)
            self.output_list.addItem(item)
        output_layout.addWidget(self.output_list)
        
        # VB-CABLE info
        vb_info = QLabel(
            "💡 Discord Integration:\n"
            "   1. Select 'VB-CABLE Input' as output\n"
            "   2. In Discord, set input device to 'VB-CABLE Output'\n"
            "   → Your processed audio will be sent to Discord!"
        )
        vb_info.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 12px; background: {BG_PANEL}; border-radius: 4px;")
        output_layout.addWidget(vb_info)
        tabs.addTab(output_tab, "OUTPUT DEVICES")
        
        layout.addWidget(tabs)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("ADD CHANNEL")
        ok_btn.setStyleSheet(f"QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600); color: #000; }}")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def get_selection(self):
        input_dev = self.input_list.currentItem().data(Qt.UserRole) if self.input_list.currentItem() else None
        output_dev = self.output_list.currentItem().data(Qt.UserRole) if self.output_list.currentItem() else None
        return input_dev, output_dev


# ============================================================================
# Toolbar and Bottom Bar
# ============================================================================
class TopToolbar(QFrame):
    """Top toolbar"""
    
    newSession = Signal()
    openSession = Signal()
    saveSession = Signal()
    ioSettings = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self._setup_ui()
        self._setup_style()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            TopToolbar {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:0.5 #3a3a3a,stop:1 #2a2a2a);
                border-bottom: 2px solid {ACCENT};
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)
        
        # SESSION
        sess_lbl = QLabel("SESSION")
        sess_lbl.setStyleSheet(f"color: {ACCENT}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(sess_lbl)
        
        for text, sig in [("NEW", self.newSession), ("OPEN", self.openSession), ("SAVE", self.saveSession)]:
            btn = QPushButton(text)
            btn.setFixedSize(55, 30)
            btn.setStyleSheet(f"""
                QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #5a5a5a,stop:1 #3a3a3a); border: 1px solid #666; border-radius: 3px; color: {TEXT_WHITE}; font-weight: bold; font-size: 9pt; }}
                QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a); }}
            """)
            btn.clicked.connect(sig.emit)
            layout.addWidget(btn)
        
        layout.addSpacing(25)
        
        # I/O SETTINGS
        io_btn = QPushButton("I/O SETTINGS")
        io_btn.setFixedSize(110, 30)
        io_btn.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600); border: 1px solid #666; border-radius: 3px; color: #000; font-weight: bold; font-size: 9pt; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffaa00,stop:1 {ACCENT}); }}
        """)
        io_btn.clicked.connect(self.ioSettings.emit)
        layout.addWidget(io_btn)
        
        # ADD CHANNEL
        add_btn = QPushButton("+ ADD CH")
        add_btn.setFixedSize(85, 30)
        add_btn.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00aa00,stop:1 #006600); border: 1px solid #666; border-radius: 3px; color: #fff; font-weight: bold; font-size: 9pt; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00cc00,stop:1 #00aa00); }}
        """)
        add_btn.clicked.connect(self.ioSettings.emit)
        layout.addWidget(add_btn)
        
        layout.addStretch()
        
        # RATE
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(rate_lbl)
        
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["44.1kHz", "48kHz", "96kHz", "192kHz"])
        self.rate_combo.setCurrentText("48kHz")
        self.rate_combo.setFixedSize(85, 26)
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        layout.addWidget(self.rate_combo)
        
        # CLOCK
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 8pt; font-weight: bold;")
        layout.addWidget(clock_lbl)
        
        self.clock_combo = QComboBox()
        self.clock_combo.addItems(["INTERNAL", "WORD", "S/PDIF"])
        self.clock_combo.setFixedSize(95, 26)
        layout.addWidget(self.clock_combo)
        
        # Title
        title_lbl = QLabel("UAD Pro Console")
        title_lbl.setStyleSheet(f"color: {TEXT_WHITE}; font-size: 13pt; font-weight: bold;")
        layout.addWidget(title_lbl)
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44100, "48kHz": 48000, "96kHz": 96000, "192kHz": 192000}
        audio_engine.set_sample_rate(rate_map.get(text, 48000))


class BottomBar(QFrame):
    """Bottom status bar"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._setup_ui()
        self._setup_style()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(1000)
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            BottomBar {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2a2a2a,stop:0.5 #1a1a1a,stop:1 #0a0a0a);
                border-top: 1px solid #4a4a4a;
            }}
        """)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 2, 18, 2)
        layout.setSpacing(35)
        
        # RATE
        self._add_status(layout, "RATE", "48.0 kHz", GREEN)
        
        # CLOCK
        self._add_status(layout, "CLOCK", "INTERNAL", GREEN)
        
        # CPU
        self._add_status(layout, "CPU", "5%", GREEN)
        
        # CHANNELS
        self._add_status(layout, "CHANNELS", "0", ACCENT)
        
        # DEVICES
        self._add_status(layout, "DEVICES", f"{len(audio_engine.input_devices)}I / {len(audio_engine.output_devices)}O", TEXT_GRAY)
        
        layout.addStretch()
        
        # Time
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 9pt;")
        layout.addWidget(self.time_lbl)
        
        # Version
        ver_lbl = QLabel("UAD Pro Console v1.0")
        ver_lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 8pt;")
        layout.addWidget(ver_lbl)
        
        self._update()
    
    def _add_status(self, layout, label, value, color):
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_GRAY}; font-size: 8pt; font-weight: bold;")
        row.addWidget(lbl)
        val = QLabel(value)
        val.setStyleSheet(f"color: {color}; font-size: 9pt; font-weight: bold;")
        val.setObjectName(f"status_{label.lower()}")
        row.addWidget(val)
        layout.addLayout(row)
    
    def _update(self):
        self.time_lbl.setText(datetime.now().strftime("%H:%M:%S"))
        ch_lbl = self.findChild(QLabel, "status_channels")
        if ch_lbl:
            ch_lbl.setText(str(len(audio_engine.channels)))
    
    def set_rate(self, rate_khz):
        lbl = self.findChild(QLabel, "status_rate")
        if lbl:
            lbl.setText(f"{rate_khz:.1f} kHz")
    
    def set_clock(self, source):
        lbl = self.findChild(QLabel, "status_clock")
        if lbl:
            lbl.setText(source)


# ============================================================================
# Channel Area
# ============================================================================
class ChannelArea(QFrame):
    """Scrollable channel strip area"""
    
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
        self.container_layout.setContentsMargins(6, 6, 6, 6)
        self.container_layout.setSpacing(4)
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
# Main Window
# ============================================================================
class UADProConsoleWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAD Pro Console - Professional Audio Interface Manager")
        self.setMinimumSize(1250, 800)
        self.resize(1500, 900)
        
        self._setup_ui()
        self._setup_style()
        self._connect_signals()
        self._add_default_channel()
    
    def _setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {BG_DARK}; }}
            QComboBox {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #4a4a4a,stop:1 #2a2a2a);
                border: 1px solid #555; border-radius: 3px;
                color: {TEXT_WHITE}; font-size: 9pt; padding: 3px 8px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid {TEXT_WHITE}; }}
            QComboBox QAbstractItemView {{ background: {BG_PANEL}; color: {TEXT_WHITE}; selection-background-color: {ACCENT}; selection-color: #000; }}
            QScrollBar:horizontal {{ background: {BG_DARK}; height: 14px; border: none; }}
            QScrollBar::handle:horizontal {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #6a6a6a,stop:1 #4a4a4a); border-radius: 6px; min-width: 50px; margin: 2px; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
    
    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DARK};")
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Toolbar
        self.toolbar = TopToolbar()
        main_layout.addWidget(self.toolbar)
        
        # Main content
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Channel area
        self.channel_area = ChannelArea()
        content_layout.addWidget(self.channel_area, 1)
        
        # Master section
        self.master = MasterSection()
        content_layout.addWidget(self.master)
        
        main_layout.addLayout(content_layout, 1)
        
        # Bottom bar
        self.bottom_bar = BottomBar()
        main_layout.addWidget(self.bottom_bar)
    
    def _connect_signals(self):
        self.toolbar.newSession.connect(self._new_session)
        self.toolbar.openSession.connect(self._open_session)
        self.toolbar.saveSession.connect(self._save_session)
        self.toolbar.ioSettings.connect(self._show_io_settings)
        self.toolbar.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        self.toolbar.clock_combo.currentTextChanged.connect(self._on_clock_changed)
    
    def _add_default_channel(self):
        if audio_engine.input_devices:
            dev = audio_engine.input_devices[0]
            name = dev['name'][:12] if len(dev['name']) > 12 else dev['name']
            self.channel_area.add_channel(name, dev, None)
    
    def _new_session(self):
        if QMessageBox.question(self, "New Session", "Discard current session and create new?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            for strip in list(self.channel_area.channels):
                self.channel_area.remove_channel(strip)
            self._add_default_channel()
    
    def _open_session(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Session", "", "JSON Files (*.json)")
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
                    
                    if 'eq_bands' in ch_data and strip.processor:
                        for i, bd in enumerate(ch_data['eq_bands']):
                            if i < len(strip.processor.eq.bands):
                                strip.processor.eq.bands[i].set_params(freq=bd.get('freq'), gain_db=bd.get('gain'), q=bd.get('q'))
                
                QMessageBox.information(self, "Session", f"Loaded: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Load error: {e}")
    
    def _save_session(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Session", "session.json", "JSON Files (*.json)")
        if filepath:
            try:
                data = {'version': '1.0', 'sample_rate': audio_engine.sample_rate, 'channels': []}
                
                for strip in self.channel_area.channels:
                    ch_data = {
                        'name': strip.ch_name,
                        'fader': strip.fader.value,
                        'pan': strip.pan_knob.value,
                        'gain': strip.gain_knob.value,
                        'mute': strip.mute_btn.isChecked(),
                        'solo': strip.solo_btn.isChecked(),
                        'eq_bands': [{'freq': b.freq, 'gain': b.gain_db, 'q': b.q} for b in strip.processor.eq.bands] if strip.processor else []
                    }
                    data['channels'].append(ch_data)
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, "Session", f"Saved: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Save error: {e}")
    
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
# Main Entry Point
# ============================================================================
def main():
    app = QApplication(sys.argv)
    
    # Font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(TEXT_WHITE))
    palette.setColor(QPalette.Base, QColor(BG_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(BG_SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT_WHITE))
    palette.setColor(QPalette.Button, QColor(BG_PANEL))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_WHITE))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#000000"))
    app.setPalette(palette)
    
    window = UADProConsoleWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
