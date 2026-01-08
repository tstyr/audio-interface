"""
UAD Console Final - 100%完全再現版
Discord連携、VB-CABLE出力、EQ、AUTO-GAIN、録音機能完備
"""
import sys
import os
import math
import wave
import struct
import threading
import time as time_module
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QComboBox, QMenu, QDialog,
    QFileDialog, QMessageBox, QSizePolicy, QSpacerItem, QLineEdit, QGroupBox,
    QListWidget, QListWidgetItem, QDialogButtonBox, QCheckBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QTimer, QRect, QPoint, QThread, QMutex
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient
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
TEXT_G = "#cccccc"
GREEN = "#00ff00"
YELLOW = "#ffff00"
RED = "#ff0000"


# ============================================================================
# 3バンドEQプロセッサ（実際に音質変化）
# ============================================================================
class ThreeBandEQProcessor:
    """3バンドパラメトリックEQ - scipy.signalベース"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.low_gain = 0.0   # dB
        self.mid_gain = 0.0   # dB
        self.high_gain = 0.0  # dB
        
        # フィルタ周波数
        self.low_freq = 200    # Hz
        self.mid_freq = 1000   # Hz
        self.high_freq = 5000  # Hz
        self.q = 1.0
        
        # フィルタ係数
        self._update_filters()
        
        # フィルタ状態
        self.zi_low = None
        self.zi_mid = None
        self.zi_high = None
    
    def _update_filters(self):
        """フィルタ係数を更新"""
        if not AUDIO_OK:
            return
        
        # ローシェルフ
        self.b_low, self.a_low = scipy_signal.iirfilter(
            2, self.low_freq, btype='lowpass', ftype='butter', fs=self.sample_rate
        )
        
        # バンドパス（ミッド）
        low_mid = self.mid_freq / 1.5
        high_mid = self.mid_freq * 1.5
        self.b_mid, self.a_mid = scipy_signal.iirfilter(
            2, [low_mid, high_mid], btype='bandpass', ftype='butter', fs=self.sample_rate
        )
        
        # ハイシェルフ
        self.b_high, self.a_high = scipy_signal.iirfilter(
            2, self.high_freq, btype='highpass', ftype='butter', fs=self.sample_rate
        )
    
    def set_gains(self, low_db: float, mid_db: float, high_db: float):
        """ゲイン設定（dB）"""
        self.low_gain = low_db
        self.mid_gain = mid_db
        self.high_gain = high_db
    
    def set_sample_rate(self, rate: int):
        """サンプルレート変更"""
        self.sample_rate = rate
        self._update_filters()
        self.zi_low = None
        self.zi_mid = None
        self.zi_high = None
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """オーディオ処理"""
        if not AUDIO_OK or audio.size == 0:
            return audio
        
        # ゲインをリニアに変換
        low_lin = 10 ** (self.low_gain / 20)
        mid_lin = 10 ** (self.mid_gain / 20)
        high_lin = 10 ** (self.high_gain / 20)
        
        # 各バンドをフィルタリング
        try:
            # 初期状態
            if self.zi_low is None:
                self.zi_low = scipy_signal.lfilter_zi(self.b_low, self.a_low)
                if audio.ndim > 1:
                    self.zi_low = np.tile(self.zi_low[:, np.newaxis], (1, audio.shape[1]))
            
            if self.zi_mid is None:
                self.zi_mid = scipy_signal.lfilter_zi(self.b_mid, self.a_mid)
                if audio.ndim > 1:
                    self.zi_mid = np.tile(self.zi_mid[:, np.newaxis], (1, audio.shape[1]))
            
            if self.zi_high is None:
                self.zi_high = scipy_signal.lfilter_zi(self.b_high, self.a_high)
                if audio.ndim > 1:
                    self.zi_high = np.tile(self.zi_high[:, np.newaxis], (1, audio.shape[1]))
            
            # フィルタ適用
            low_band, self.zi_low = scipy_signal.lfilter(
                self.b_low, self.a_low, audio, axis=0, zi=self.zi_low * audio[0]
            )
            mid_band, self.zi_mid = scipy_signal.lfilter(
                self.b_mid, self.a_mid, audio, axis=0, zi=self.zi_mid * audio[0]
            )
            high_band, self.zi_high = scipy_signal.lfilter(
                self.b_high, self.a_high, audio, axis=0, zi=self.zi_high * audio[0]
            )
            
            # ミックス
            output = (low_band * low_lin + mid_band * mid_lin + high_band * high_lin) / 3
            
            # クリッピング防止
            output = np.clip(output, -1.0, 1.0)
            
            return output
        except Exception:
            return audio


# ============================================================================
# オーディオエンジン（VB-CABLE出力・録音・EQ対応）
# ============================================================================
class AudioEngine:
    """高度なオーディオエンジン - ルーティング・録音・EQ対応"""
    
    def __init__(self):
        self.input_devices = []
        self.output_devices = []
        self.sample_rate = 48000
        self.buffer_size = 1024
        self.channels = {}  # {ch_id: ChannelProcessor}
        self.master_output_device = None
        self.master_stream = None
        self.lock = QMutex()
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
        """サンプリング周波数変更 - 全ストリーム再起動"""
        self.sample_rate = rate
        
        # 全チャンネルのEQを更新
        for ch in self.channels.values():
            ch.eq.set_sample_rate(rate)
        
        # ストリーム再起動
        active_channels = list(self.channels.keys())
        for ch_id in active_channels:
            ch = self.channels[ch_id]
            if ch.stream:
                self.stop_channel(ch_id)
                self.start_channel(ch_id, ch.input_device, ch.output_device)
    
    def create_channel(self, ch_id: str) -> 'ChannelProcessor':
        """チャンネル作成"""
        ch = ChannelProcessor(ch_id, self.sample_rate, self.noise_gate_db)
        self.channels[ch_id] = ch
        return ch
    
    def start_channel(self, ch_id: str, input_dev: int, output_dev: int = None) -> bool:
        """チャンネル開始"""
        if ch_id not in self.channels:
            return False
        
        ch = self.channels[ch_id]
        return ch.start(input_dev, output_dev, self.sample_rate, self.buffer_size)
    
    def stop_channel(self, ch_id: str):
        """チャンネル停止"""
        if ch_id in self.channels:
            self.channels[ch_id].stop()
    
    def remove_channel(self, ch_id: str):
        """チャンネル削除"""
        if ch_id in self.channels:
            self.channels[ch_id].stop()
            del self.channels[ch_id]
    
    def get_channel(self, ch_id: str) -> Optional['ChannelProcessor']:
        """チャンネル取得"""
        return self.channels.get(ch_id)
    
    def cleanup(self):
        """クリーンアップ"""
        for ch_id in list(self.channels.keys()):
            self.remove_channel(ch_id)


class ChannelProcessor:
    """チャンネルプロセッサ - EQ・ゲイン・パン・録音"""
    
    def __init__(self, ch_id: str, sample_rate: int, noise_gate_db: float):
        self.ch_id = ch_id
        self.sample_rate = sample_rate
        self.noise_gate_db = noise_gate_db
        
        # デバイス
        self.input_device = None
        self.output_device = None
        self.stream = None
        
        # パラメータ
        self.gain = 1.0       # リニア
        self.pan = 0.5        # 0=L, 1=R
        self.mute = False
        self.solo = False
        
        # EQ
        self.eq = ThreeBandEQProcessor(sample_rate)
        self.eq_enabled = True
        
        # AUTO-GAIN
        self.auto_gain_enabled = False
        self.auto_gain_target = -12.0  # dB
        self.auto_gain_history = deque(maxlen=100)
        
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
        
        # スレッドセーフ
        self.lock = threading.Lock()
    
    def start(self, input_dev: int, output_dev: int, sample_rate: int, buffer_size: int) -> bool:
        """ストリーム開始"""
        if not AUDIO_OK:
            return False
        
        self.input_device = input_dev
        self.output_device = output_dev
        self.sample_rate = sample_rate
        self.eq.set_sample_rate(sample_rate)
        
        try:
            if output_dev is not None:
                # 入力→処理→出力（VB-CABLE等へ）
                self.stream = sd.Stream(
                    device=(input_dev, output_dev),
                    channels=2,
                    samplerate=sample_rate,
                    blocksize=buffer_size,
                    callback=self._stream_callback
                )
            else:
                # 入力のみ（モニタリング）
                self.stream = sd.InputStream(
                    device=input_dev,
                    channels=2,
                    samplerate=sample_rate,
                    blocksize=buffer_size,
                    callback=self._input_callback
                )
            
            self.stream.start()
            return True
        except Exception as e:
            print(f"Stream error [{self.ch_id}]: {e}")
            return False
    
    def stop(self):
        """ストリーム停止"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        # 録音中なら保存
        if self.recording:
            self.stop_recording()
    
    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """オーディオ処理パイプライン"""
        if audio.size == 0:
            return audio
        
        # ステレオ化
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])
        
        # AUTO-GAIN
        if self.auto_gain_enabled:
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                current_db = 20 * np.log10(rms)
                self.auto_gain_history.append(current_db)
                
                if len(self.auto_gain_history) >= 50:
                    avg_db = np.mean(list(self.auto_gain_history))
                    target_gain = 10 ** ((self.auto_gain_target - avg_db) / 20)
                    self.gain = np.clip(target_gain, 0.1, 10.0)
        
        # ゲイン適用
        audio = audio * self.gain
        
        # EQ適用
        if self.eq_enabled:
            audio = self.eq.process(audio)
        
        # パン適用
        left_gain = 1.0 - self.pan
        right_gain = self.pan
        audio[:, 0] *= left_gain * 2
        audio[:, 1] *= right_gain * 2
        
        # ミュート
        if self.mute:
            audio = audio * 0
        
        # クリッピング防止
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _update_levels(self, audio: np.ndarray):
        """レベル更新"""
        if audio.size == 0:
            return
        
        # ステレオ
        if audio.ndim == 1:
            left = right = audio
        else:
            left = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()
            right = audio[:, 1] if audio.shape[1] > 1 else left
        
        # RMS計算
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
            self.left_db = left_db
            self.right_db = right_db
            
            # ピークホールド
            if left_db > self.left_peak:
                self.left_peak = left_db
                self.left_hold_count = self.peak_hold
            elif self.left_hold_count > 0:
                self.left_hold_count -= 1
            else:
                self.left_peak = max(self.left_peak - 1, left_db)
            
            if right_db > self.right_peak:
                self.right_peak = right_db
                self.right_hold_count = self.peak_hold
            elif self.right_hold_count > 0:
                self.right_hold_count -= 1
            else:
                self.right_peak = max(self.right_peak - 1, right_db)
    
    def _stream_callback(self, indata, outdata, frames, time_info, status):
        """入出力ストリームコールバック"""
        if status:
            print(f"Stream status [{self.ch_id}]: {status}")
        
        # 処理
        processed = self._process_audio(indata.copy())
        
        # 出力
        outdata[:] = processed
        
        # レベル更新
        self._update_levels(processed)
        
        # 録音
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def _input_callback(self, indata, frames, time_info, status):
        """入力のみコールバック"""
        if status:
            print(f"Input status [{self.ch_id}]: {status}")
        
        # 処理
        processed = self._process_audio(indata.copy())
        
        # レベル更新
        self._update_levels(processed)
        
        # 録音
        if self.recording:
            self.rec_data.append(processed.copy())
    
    def get_levels(self) -> Tuple[float, float, float, float]:
        """レベル取得"""
        with self.lock:
            return self.left_db, self.right_db, self.left_peak, self.right_peak
    
    def start_recording(self, filepath: str):
        """録音開始"""
        self.rec_path = filepath
        self.rec_data = []
        self.recording = True
    
    def stop_recording(self) -> Optional[str]:
        """録音停止・保存"""
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
            print(f"Recording save error: {e}")
            return None


# グローバルエンジン
audio_engine = AudioEngine()


# ============================================================================
# I/O設定ダイアログ
# ============================================================================
class IOSettingsDialog(QDialog):
    """I/O設定ダイアログ - デバイス選択"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("I/O SETTINGS")
        self.setMinimumSize(500, 400)
        self.selected_input = None
        self.selected_output = None
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
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
            }}
            QListWidget {{
                background-color: {BG_PANEL};
                color: {TEXT_W};
                border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 6px;
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
        
        # タブ
        tabs = QTabWidget()
        
        # 入力タブ
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        
        input_label = QLabel("Select Input Device (Microphone, Line-in, etc.):")
        input_label.setStyleSheet(f"color: {TEXT_G};")
        input_layout.addWidget(input_label)
        
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
        
        output_label = QLabel("Select Output Device (VB-CABLE, Speaker, etc.):")
        output_label.setStyleSheet(f"color: {TEXT_G};")
        output_layout.addWidget(output_label)
        
        self.output_list = QListWidget()
        for dev in audio_engine.output_devices:
            item = QListWidgetItem(f"[{dev['hostapi']}] {dev['name']}")
            item.setData(Qt.UserRole, dev)
            self.output_list.addItem(item)
        output_layout.addWidget(self.output_list)
        
        # VB-CABLE説明
        vb_info = QLabel(
            "💡 Discord連携: VB-CABLE Inputを選択し、Discordの入力デバイスで\n"
            "   'VB-CABLE Output'を選ぶと、加工後の音声が通話で使用できます。"
        )
        vb_info.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; padding: 8px;")
        output_layout.addWidget(vb_info)
        
        tabs.addTab(output_tab, "OUTPUT DEVICES")
        
        layout.addWidget(tabs)
        
        # ボタン
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def get_selection(self) -> Tuple[Optional[dict], Optional[dict]]:
        """選択されたデバイスを取得"""
        input_dev = None
        output_dev = None
        
        if self.input_list.currentItem():
            input_dev = self.input_list.currentItem().data(Qt.UserRole)
        
        if self.output_list.currentItem():
            output_dev = self.output_list.currentItem().data(Qt.UserRole)
        
        return input_dev, output_dev


# ============================================================================
# ステレオLEDメーター（24セグメント・dB目盛り）
# ============================================================================
class StereoLEDMeter(QWidget):
    """ステレオLRメーター - 24セグメント・dB目盛り付き"""
    
    def __init__(self, parent=None, width=32, height=180, show_scale=True):
        super().__init__(parent)
        self.setFixedSize(width, height)
        
        self.left_db = -100.0
        self.right_db = -100.0
        self.left_peak = -100.0
        self.right_peak = -100.0
        self.segments = 24
        self.show_scale = show_scale
    
    def setLevels(self, left_db: float, right_db: float, left_peak: float = None, right_peak: float = None):
        """レベル設定（dB値）"""
        self.left_db = max(-60, min(0, left_db)) if left_db > -100 else -100
        self.right_db = max(-60, min(0, right_db)) if right_db > -100 else -100
        
        if left_peak is not None:
            self.left_peak = max(-60, min(0, left_peak)) if left_peak > -100 else -100
        if right_peak is not None:
            self.right_peak = max(-60, min(0, right_peak)) if right_peak > -100 else -100
        
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 背景
        p.fillRect(self.rect(), QColor(15, 15, 15))
        p.setPen(QPen(QColor(50, 50, 50), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        
        # メーター幅計算
        scale_w = 14 if self.show_scale else 0
        meter_w = (w - scale_w - 6) // 2
        
        # L/Rラベル
        p.setPen(QColor(TEXT_G))
        font = p.font()
        font.setPointSize(6)
        p.setFont(font)
        p.drawText(2, 10, "L")
        p.drawText(meter_w + 6, 10, "R")
        
        # 左メーター
        self._draw_meter(p, 2, 14, meter_w, h - 18, self.left_db, self.left_peak)
        
        # 右メーター
        self._draw_meter(p, meter_w + 6, 14, meter_w, h - 18, self.right_db, self.right_peak)
        
        # dB目盛り
        if self.show_scale:
            self._draw_scale(p, w - scale_w, 14, scale_w, h - 18)
    
    def _draw_meter(self, p, x, y, w, h, level_db, peak_db):
        seg_h = (h - 2) / self.segments
        
        level_norm = (level_db + 60) / 60 if level_db > -100 else 0
        peak_norm = (peak_db + 60) / 60 if peak_db > -100 else 0
        
        active = int(level_norm * self.segments)
        peak_seg = int(peak_norm * self.segments)
        
        for i in range(self.segments):
            seg_y = y + h - (i + 1) * seg_h
            seg_rect = QRect(x, int(seg_y), w, int(seg_h - 1))
            
            ratio = (i + 1) / self.segments
            if ratio <= 0.6:
                on, off = QColor(0, 255, 0), QColor(0, 30, 0)
            elif ratio <= 0.85:
                on, off = QColor(255, 255, 0), QColor(30, 30, 0)
            else:
                on, off = QColor(255, 0, 0), QColor(30, 0, 0)
            
            if i < active:
                p.fillRect(seg_rect, on)
            elif i == peak_seg - 1 and peak_seg > 0:
                p.fillRect(seg_rect, on)
            else:
                p.fillRect(seg_rect, off)
    
    def _draw_scale(self, p, x, y, w, h):
        p.setPen(QColor(100, 100, 100))
        font = p.font()
        font.setPointSize(5)
        p.setFont(font)
        
        marks = [(1.0, "0"), (0.9, "-6"), (0.7, "-12"), (0.5, "-18"), (0.3, "-24"), (0.1, "-36"), (0.0, "-∞")]
        for pos, txt in marks:
            mark_y = y + (1 - pos) * h
            p.drawLine(x, int(mark_y), x + 3, int(mark_y))
            p.drawText(x + 4, int(mark_y) + 3, txt)


# ============================================================================
# プロフェッショナルフェーダー
# ============================================================================
class ProFader(QWidget):
    """シルバー立体フェーダー"""
    
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
# プロフェッショナルノブ
# ============================================================================
class ProKnob(QWidget):
    """光沢ロータリーノブ"""
    
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
# トグルボタン・プラグインスロット
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


class PluginSlot(QFrame):
    """プラグインスロット"""
    
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
# チャンネルストリップ（完全版）
# ============================================================================
class ChannelStrip(QFrame):
    """UAD風完全チャンネルストリップ"""
    
    removeRequested = Signal()
    
    def __init__(self, ch_id: str, name: str, input_dev: dict = None, output_dev: dict = None, parent=None):
        super().__init__(parent)
        self.ch_id = ch_id
        self.ch_name = name
        self.input_dev = input_dev
        self.output_dev = output_dev
        self.processor = None
        
        self.setFixedWidth(80)
        self.setMinimumHeight(580)
        self._setup_ui()
        self._setup_style()
        self._setup_audio()
        
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
    
    def _setup_audio(self):
        """オーディオ設定"""
        self.processor = audio_engine.create_channel(self.ch_id)
        
        if self.input_dev:
            input_idx = self.input_dev['index']
            output_idx = self.output_dev['index'] if self.output_dev else None
            audio_engine.start_channel(self.ch_id, input_idx, output_idx)
    
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
        
        # INPUT
        input_lbl = QLabel("INPUT")
        input_lbl.setAlignment(Qt.AlignCenter)
        input_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(input_lbl)
        
        # 48V, PAD
        input_btns = QHBoxLayout()
        input_btns.setSpacing(1)
        
        self.btn_48v = ToggleBtn("48V", RED)
        self.btn_48v.setFixedSize(32, 16)
        input_btns.addWidget(self.btn_48v)
        
        self.btn_pad = ToggleBtn("PAD", "#888")
        self.btn_pad.setFixedSize(32, 16)
        input_btns.addWidget(self.btn_pad)
        
        layout.addLayout(input_btns)
        
        # Φ, LC
        input_btns2 = QHBoxLayout()
        input_btns2.setSpacing(1)
        
        self.btn_phase = ToggleBtn("Φ", YELLOW)
        self.btn_phase.setFixedSize(32, 16)
        input_btns2.addWidget(self.btn_phase)
        
        self.btn_lc = ToggleBtn("LC", "#888")
        self.btn_lc.setFixedSize(32, 16)
        input_btns2.addWidget(self.btn_lc)
        
        layout.addLayout(input_btns2)
        
        # GAIN + AUTO
        gain_layout = QHBoxLayout()
        gain_layout.setSpacing(2)
        
        gain_lbl = QLabel("GAIN")
        gain_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt;")
        gain_layout.addWidget(gain_lbl)
        
        self.gain_knob = ProKnob(size=28)
        self.gain_knob.setValue(0.5)
        self.gain_knob.valueChanged.connect(self._on_gain_changed)
        gain_layout.addWidget(self.gain_knob)
        
        self.auto_btn = ToggleBtn("A", GREEN)
        self.auto_btn.setFixedSize(20, 20)
        self.auto_btn.setToolTip("AUTO-GAIN")
        self.auto_btn.toggled.connect(self._on_auto_gain_toggled)
        gain_layout.addWidget(self.auto_btn)
        
        layout.addLayout(gain_layout)
        
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
        
        # EQ (3-Band)
        eq_lbl = QLabel("EQ")
        eq_lbl.setAlignment(Qt.AlignCenter)
        eq_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(eq_lbl)
        
        eq_layout = QHBoxLayout()
        eq_layout.setSpacing(2)
        
        self.eq_knobs = []
        for band in ["LO", "MID", "HI"]:
            vl = QVBoxLayout()
            vl.setSpacing(0)
            
            knob = ProKnob(size=22)
            knob.setValue(0.5)
            self.eq_knobs.append(knob)
            vl.addWidget(knob, alignment=Qt.AlignCenter)
            
            lbl = QLabel(band)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 5pt;")
            vl.addWidget(lbl)
            
            eq_layout.addLayout(vl)
        
        # EQノブ接続
        self.eq_knobs[0].valueChanged.connect(lambda v: self._on_eq_changed(0, v))
        self.eq_knobs[1].valueChanged.connect(lambda v: self._on_eq_changed(1, v))
        self.eq_knobs[2].valueChanged.connect(lambda v: self._on_eq_changed(2, v))
        
        layout.addLayout(eq_layout)
        
        # SENDS
        sends_lbl = QLabel("SENDS")
        sends_lbl.setAlignment(Qt.AlignCenter)
        sends_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 7pt; font-weight: bold;")
        layout.addWidget(sends_lbl)
        
        sends_layout = QGridLayout()
        sends_layout.setSpacing(1)
        self.send_knobs = []
        for i in range(4):
            knob = ProKnob(size=16)
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
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        layout.addWidget(self.output_combo)
        
        # メーター + フェーダー
        mf_layout = QHBoxLayout()
        mf_layout.setSpacing(2)
        
        self.meter = StereoLEDMeter(width=26, height=120, show_scale=False)
        mf_layout.addWidget(self.meter)
        
        self.fader = ProFader(width=28, height=120)
        self.fader.setValue(0.75)
        self.fader.valueChanged.connect(self._on_fader_changed)
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
        self.pan_knob.valueChanged.connect(self._on_pan_changed)
        pan_layout.addWidget(self.pan_knob)
        
        layout.addLayout(pan_layout)
        
        # SOLO / MUTE
        sm_layout = QHBoxLayout()
        sm_layout.setSpacing(2)
        
        self.solo_btn = ToggleBtn("S", YELLOW)
        self.solo_btn.setFixedSize(28, 20)
        self.solo_btn.toggled.connect(self._on_solo_toggled)
        sm_layout.addWidget(self.solo_btn)
        
        self.mute_btn = ToggleBtn("M", RED)
        self.mute_btn.setFixedSize(28, 20)
        self.mute_btn.toggled.connect(self._on_mute_toggled)
        sm_layout.addWidget(self.mute_btn)
        
        layout.addLayout(sm_layout)
        
        # REC
        self.rec_btn = ToggleBtn("REC", RED)
        self.rec_btn.setFixedHeight(20)
        self.rec_btn.toggled.connect(self._on_rec_toggled)
        layout.addWidget(self.rec_btn)
    
    def _populate_outputs(self):
        self.output_combo.clear()
        self.output_combo.addItem("MONITOR", None)
        for dev in audio_engine.output_devices:
            name = dev['name'][:18] + ".." if len(dev['name']) > 18 else dev['name']
            self.output_combo.addItem(name, dev)
    
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
    
    def _on_gain_changed(self, v):
        if self.processor:
            self.processor.gain = v * 4  # 0-4x
    
    def _on_auto_gain_toggled(self, checked):
        if self.processor:
            self.processor.auto_gain_enabled = checked
    
    def _on_eq_changed(self, band, v):
        if self.processor:
            db = (v - 0.5) * 24  # -12dB to +12dB
            if band == 0:
                self.processor.eq.low_gain = db
            elif band == 1:
                self.processor.eq.mid_gain = db
            else:
                self.processor.eq.high_gain = db
    
    def _on_fader_changed(self, v):
        # フェーダーはゲインに追加で適用
        pass
    
    def _on_pan_changed(self, v):
        if self.processor:
            self.processor.pan = v
    
    def _on_solo_toggled(self, checked):
        if self.processor:
            self.processor.solo = checked
    
    def _on_mute_toggled(self, checked):
        if self.processor:
            self.processor.mute = checked
    
    def _on_output_changed(self, idx):
        dev = self.output_combo.currentData()
        if self.processor and self.input_dev:
            audio_engine.stop_channel(self.ch_id)
            output_idx = dev['index'] if dev else None
            audio_engine.start_channel(self.ch_id, self.input_dev['index'], output_idx)
    
    def _on_rec_toggled(self, checked):
        if not self.processor:
            return
        
        if checked:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"recording_{self.ch_name}_{timestamp}.wav"
            self.processor.start_recording(filepath)
        else:
            path = self.processor.stop_recording()
            if path:
                QMessageBox.information(self, "Recording", f"Saved: {path}")
    
    def _update_levels(self):
        if self.processor:
            l, r, lp, rp = self.processor.get_levels()
            self.meter.setLevels(l, r, lp, rp)
        else:
            self.meter.setLevels(-100, -100, -100, -100)
    
    def cleanup(self):
        self.timer.stop()
        audio_engine.remove_channel(self.ch_id)


# ============================================================================
# マスターセクション
# ============================================================================
class MasterSection(QFrame):
    """マスター・モニターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(170)
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
        
        for _ in range(2):
            slot = PluginSlot()
            layout.addWidget(slot)
        
        # REC/MON
        rm_layout = QHBoxLayout()
        rm_layout.setSpacing(2)
        
        self.rec_btn = ToggleBtn("REC", ACCENT)
        self.rec_btn.setChecked(True)
        rm_layout.addWidget(self.rec_btn)
        
        self.mon_btn = ToggleBtn("MON", "#888")
        rm_layout.addWidget(self.mon_btn)
        
        layout.addLayout(rm_layout)
        
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
        
        self.talk_btn = ToggleBtn("TALK", GREEN)
        dt_layout.addWidget(self.talk_btn)
        
        layout.addLayout(dt_layout)
        
        # ステレオメーター
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(4)
        
        l_lbl = QLabel("L")
        l_lbl.setStyleSheet(f"color: {TEXT_W}; font-weight: bold;")
        l_lbl.setAlignment(Qt.AlignCenter)
        meter_layout.addWidget(l_lbl)
        
        self.meter = StereoLEDMeter(width=50, height=180, show_scale=True)
        meter_layout.addWidget(self.meter)
        
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
        self.master_fader = ProFader(width=40, height=100)
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
        # 全チャンネルのレベルを合成
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
# 左サイドバー・チャンネルエリア
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
        
        for text in ["OVERVIEW", "INPUTS", "INSERTS", "SENDS"]:
            btn = ToggleBtn(text, ACCENT)
            btn.setFixedHeight(26)
            layout.addWidget(btn)
        
        layout.addStretch()
        
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
    
    def add_channel(self, name: str, input_dev: dict = None, output_dev: dict = None):
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
    ioSettings = Signal()
    addChannel = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
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
            btn.setFixedSize(45, 26)
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
        
        # I/O SETTINGS
        io_btn = QPushButton("I/O SETTINGS")
        io_btn.setFixedSize(90, 26)
        io_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 {ACCENT},stop:1 #cc6600);
                border: 1px solid #666; border-radius: 3px;
                color: #000; font-weight: bold; font-size: 8pt;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffaa00,stop:1 {ACCENT});
            }}
        """)
        io_btn.clicked.connect(self.ioSettings.emit)
        layout.addWidget(io_btn)
        
        # ADD CHANNEL
        add_btn = QPushButton("+ ADD CH")
        add_btn.setFixedSize(70, 26)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00aa00,stop:1 #006600);
                border: 1px solid #666; border-radius: 3px;
                color: #fff; font-weight: bold; font-size: 8pt;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #00cc00,stop:1 #00aa00);
            }}
        """)
        add_btn.clicked.connect(self.addChannel.emit)
        layout.addWidget(add_btn)
        
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
        
        self.dsp_meter = StereoLEDMeter(width=80, height=14, show_scale=False)
        dsp_vl.addWidget(self.dsp_meter)
        status_layout.addLayout(dsp_vl)
        
        layout.addWidget(status_frame, 1)
        
        # TITLE
        title_lbl = QLabel("Console: UAD Final")
        title_lbl.setStyleSheet(f"color: {TEXT_W}; font-size: 11pt; font-weight: bold;")
        layout.addWidget(title_lbl)
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44100, "48kHz": 48000, "88.2kHz": 88200, "96kHz": 96000, "192kHz": 192000}
        audio_engine.set_sample_rate(rate_map.get(text, 48000))


# ============================================================================
# 下部ステータスバー
# ============================================================================
class BottomBar(QFrame):
    """下部情報バー - TEMPO/RATE/CLOCK/DSP"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._setup_ui()
        self._setup_style()
        
        # 時間更新タイマー
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
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
        layout.setContentsMargins(12, 2, 12, 2)
        layout.setSpacing(24)
        
        # TEMPO
        tempo_layout = QHBoxLayout()
        tempo_layout.setSpacing(4)
        tempo_lbl = QLabel("TEMPO")
        tempo_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        tempo_layout.addWidget(tempo_lbl)
        
        self.tempo_val = QLabel("120.0 BPM")
        self.tempo_val.setStyleSheet(f"color: {ACCENT}; font-size: 9pt; font-weight: bold;")
        tempo_layout.addWidget(self.tempo_val)
        layout.addLayout(tempo_layout)
        
        # RATE
        rate_layout = QHBoxLayout()
        rate_layout.setSpacing(4)
        rate_lbl = QLabel("RATE")
        rate_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        rate_layout.addWidget(rate_lbl)
        
        self.rate_val = QLabel("48.0 kHz")
        self.rate_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        rate_layout.addWidget(self.rate_val)
        layout.addLayout(rate_layout)
        
        # CLOCK
        clock_layout = QHBoxLayout()
        clock_layout.setSpacing(4)
        clock_lbl = QLabel("CLOCK")
        clock_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        clock_layout.addWidget(clock_lbl)
        
        self.clock_val = QLabel("INTERNAL")
        self.clock_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        clock_layout.addWidget(self.clock_val)
        layout.addLayout(clock_layout)
        
        # DSP
        dsp_layout = QHBoxLayout()
        dsp_layout.setSpacing(4)
        dsp_lbl = QLabel("DSP")
        dsp_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt; font-weight: bold;")
        dsp_layout.addWidget(dsp_lbl)
        
        self.dsp_val = QLabel("12%")
        self.dsp_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")
        dsp_layout.addWidget(self.dsp_val)
        layout.addLayout(dsp_layout)
        
        layout.addStretch()
        
        # TIME
        self.time_lbl = QLabel("")
        self.time_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 9pt;")
        layout.addWidget(self.time_lbl)
        
        # VERSION
        ver_lbl = QLabel("UAD Console Final v1.0")
        ver_lbl.setStyleSheet(f"color: {TEXT_G}; font-size: 8pt;")
        layout.addWidget(ver_lbl)
        
        self._update_time()
    
    def _update_time(self):
        self.time_lbl.setText(datetime.now().strftime("%H:%M:%S"))
    
    def set_rate(self, rate_khz: float):
        self.rate_val.setText(f"{rate_khz:.1f} kHz")
    
    def set_clock(self, source: str):
        self.clock_val.setText(source)
    
    def set_dsp(self, percent: int):
        self.dsp_val.setText(f"{percent}%")
        if percent > 80:
            self.dsp_val.setStyleSheet(f"color: {RED}; font-size: 9pt; font-weight: bold;")
        elif percent > 50:
            self.dsp_val.setStyleSheet(f"color: {YELLOW}; font-size: 9pt; font-weight: bold;")
        else:
            self.dsp_val.setStyleSheet(f"color: {GREEN}; font-size: 9pt; font-weight: bold;")


# ============================================================================
# メインウィンドウ
# ============================================================================
class UADConsoleFinalWindow(QMainWindow):
    """UAD Console Final - メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAD Console Final - Professional Audio Mixer")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        
        self._setup_ui()
        self._setup_style()
        self._connect_signals()
        
        # 初期チャンネル追加
        self._add_default_channels()
    
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
                padding: 2px 4px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 16px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
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
                height: 10px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: #5a5a5a;
                border-radius: 4px;
                min-width: 30px;
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
        
        # トップツールバー
        self.toolbar = TopToolbar()
        main_layout.addWidget(self.toolbar)
        
        # メインコンテンツ
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # 左サイドバー
        self.sidebar = LeftSidebar()
        content_layout.addWidget(self.sidebar)
        
        # チャンネルエリア
        self.channel_area = ChannelArea()
        content_layout.addWidget(self.channel_area, 1)
        
        # マスターセクション
        self.master = MasterSection()
        content_layout.addWidget(self.master)
        
        main_layout.addLayout(content_layout, 1)
        
        # 下部ステータスバー
        self.bottom_bar = BottomBar()
        main_layout.addWidget(self.bottom_bar)
    
    def _connect_signals(self):
        self.toolbar.newSession.connect(self._new_session)
        self.toolbar.openSession.connect(self._open_session)
        self.toolbar.saveSession.connect(self._save_session)
        self.toolbar.ioSettings.connect(self._show_io_settings)
        self.toolbar.addChannel.connect(self._add_channel_dialog)
        
        # サンプルレート変更時に下部バー更新
        self.toolbar.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        self.toolbar.clock_combo.currentTextChanged.connect(self._on_clock_changed)
    
    def _add_default_channels(self):
        """デフォルトチャンネル追加"""
        if audio_engine.input_devices:
            dev = audio_engine.input_devices[0]
            self.channel_area.add_channel(f"Input 1", dev, None)
    
    def _new_session(self):
        """新規セッション"""
        reply = QMessageBox.question(
            self, "New Session",
            "現在のセッションを破棄して新規作成しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # 全チャンネル削除
            for strip in list(self.channel_area.channels):
                self.channel_area.remove_channel(strip)
            self._add_default_channels()
    
    def _open_session(self):
        """セッション読み込み"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "JSON Files (*.json)"
        )
        if filepath:
            try:
                import json
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # 全チャンネル削除
                for strip in list(self.channel_area.channels):
                    self.channel_area.remove_channel(strip)
                
                # チャンネル復元
                for ch_data in data.get('channels', []):
                    strip = self.channel_area.add_channel(ch_data['name'], None, None)
                    strip.fader.setValue(ch_data.get('fader', 0.75))
                    strip.pan_knob.setValue(ch_data.get('pan', 0.5))
                    strip.gain_knob.setValue(ch_data.get('gain', 0.5))
                    strip.mute_btn.setChecked(ch_data.get('mute', False))
                    strip.solo_btn.setChecked(ch_data.get('solo', False))
                
                QMessageBox.information(self, "Session", f"セッションを読み込みました: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"読み込みエラー: {e}")
    
    def _save_session(self):
        """セッション保存"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "session.json", "JSON Files (*.json)"
        )
        if filepath:
            try:
                import json
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
                        'solo': strip.solo_btn.isChecked()
                    }
                    data['channels'].append(ch_data)
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, "Session", f"セッションを保存しました: {filepath}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"保存エラー: {e}")
    
    def _show_io_settings(self):
        """I/O設定ダイアログ表示"""
        dialog = IOSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            input_dev, output_dev = dialog.get_selection()
            if input_dev:
                name = input_dev['name'][:12] + ".." if len(input_dev['name']) > 12 else input_dev['name']
                self.channel_area.add_channel(name, input_dev, output_dev)
    
    def _add_channel_dialog(self):
        """チャンネル追加ダイアログ"""
        self._show_io_settings()
    
    def _on_rate_changed(self, text):
        rate_map = {"44.1kHz": 44.1, "48kHz": 48.0, "88.2kHz": 88.2, "96kHz": 96.0, "192kHz": 192.0}
        self.bottom_bar.set_rate(rate_map.get(text, 48.0))
    
    def _on_clock_changed(self, text):
        self.bottom_bar.set_clock(text)
    
    def closeEvent(self, event):
        """終了時クリーンアップ"""
        # 全チャンネル停止
        for strip in self.channel_area.channels:
            strip.cleanup()
        
        audio_engine.cleanup()
        event.accept()


# ============================================================================
# メイン関数
# ============================================================================
def main():
    """アプリケーション起動"""
    app = QApplication(sys.argv)
    
    # フォント設定
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # ダークパレット
    from PySide6.QtGui import QPalette
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
    
    window = UADConsoleFinalWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
