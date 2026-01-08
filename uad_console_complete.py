"""
UAD Console完全再現：統合オーディオ管理アプリケーション
プロ仕様のスキューモーフィック・デザインとリアルタイムオーディオ処理
"""
import sys
import math
import threading
import time
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                               QListWidget, QListWidgetItem, QSplitter, QGroupBox,
                               QButtonGroup, QRadioButton, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import QFont, QColor, QPalette

# プロフェッショナルウィジェットをインポート
from uad_professional_widgets import ProfessionalFader, ProfessionalKnob, ProfessionalLEDMeter

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio features disabled.")


class AudioProcessor(QObject):
    """オーディオ処理クラス（別スレッド実行）"""
    
    levelUpdated = Signal(float, float)  # current_level, peak_level
    
    def __init__(self, device_index: int, channels: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.device_index = device_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_running = False
        self.stream = None
        
        # レベル計算用
        self.current_level = 0.0
        self.peak_level = 0.0
        self.peak_hold_time = 1.0
        self.last_peak_time = 0.0
        
        # スレッドセーフティ用
        self.lock = threading.Lock()
    
    def audio_callback(self, indata, frames, time_info, status):
        """オーディオストリームコールバック"""
        if status:
            print(f"Audio status: {status}")
        
        if indata.size > 0:
            # ステレオの場合は平均を取る
            if self.channels > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # RMSレベル計算
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # dBに変換して正規化
            if rms > 0:
                db_level = 20 * np.log10(rms)
                db_level = max(db_level, -60.0)
                normalized_level = (db_level + 60.0) / 60.0
            else:
                normalized_level = 0.0
            
            with self.lock:
                self.current_level = normalized_level
                
                # ピークホールド処理
                import time as time_module
                current_time = time_module.time()
                if normalized_level > self.peak_level:
                    self.peak_level = normalized_level
                    self.last_peak_time = current_time
                elif current_time - self.last_peak_time > self.peak_hold_time:
                    self.peak_level = max(self.peak_level - 0.01, normalized_level)
                
                # シグナル発信
                self.levelUpdated.emit(self.current_level, self.peak_level)
    
    def start_monitoring(self) -> bool:
        """モニタリング開始"""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            if self.is_running:
                return True
            
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=1024
            )
            
            self.stream.start()
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Audio stream error: {e}")
            return False
    
    def stop_monitoring(self):
        """モニタリング停止"""
        if self.stream and self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
    
    def get_levels(self) -> tuple[float, float]:
        """現在のレベルを取得"""
        with self.lock:
            return self.current_level, self.peak_level


class DeviceManager:
    """オーディオデバイス管理"""
    
    def __init__(self):
        self.devices = []
        self.input_devices = []
        self.output_devices = []
        self.refresh_devices()
    
    def refresh_devices(self):
        """デバイス一覧を更新"""
        if not AUDIO_AVAILABLE:
            return
        
        try:
            self.devices = sd.query_devices()
            self.input_devices = []
            self.output_devices = []
            
            for i, device in enumerate(self.devices):
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'] if device['max_input_channels'] > 0 else device['max_output_channels'],
                    'sample_rate': device['default_samplerate'],
                    'hostapi': device['hostapi']
                }
                
                if device['max_input_channels'] > 0:
                    device_info['type'] = 'input'
                    device_info['max_channels'] = device['max_input_channels']
                    self.input_devices.append(device_info)
                
                if device['max_output_channels'] > 0:
                    device_info['type'] = 'output'
                    device_info['max_channels'] = device['max_output_channels']
                    self.output_devices.append(device_info)
                    
        except Exception as e:
            print(f"Device enumeration error: {e}")
    
    def get_input_devices(self) -> List[Dict[str, Any]]:
        return self.input_devices
    
    def get_output_devices(self) -> List[Dict[str, Any]]:
        return self.output_devices


class ProfessionalChannelStrip(QFrame):
    """UAD風プロフェッショナルチャンネルストリップ"""
    
    # シグナル定義
    gainChanged = Signal(float)
    panChanged = Signal(float)
    faderChanged = Signal(float)
    muteToggled = Signal(bool)
    soloToggled = Signal(bool)
    removeRequested = Signal()
    
    def __init__(self, channel_name="CH 1", device_info=None, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.device_info = device_info
        self.audio_processor = None
        
        # 状態
        self.is_muted = False
        self.is_solo = False
        self.gain_value = 50.0
        self.pan_value = 50.0  # 0-100 (50=Center)
        self.fader_value = 75.0
        
        self.setup_ui()
        self.setup_styles()
        
        if device_info and device_info['type'] == 'input':
            self.setup_audio_processor()
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(85)
        self.setMinimumHeight(450)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)
        
        # チャンネル名とデバイス情報
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        
        # チャンネル名
        self.name_label = QLabel(self.channel_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 10px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a, stop: 1 #2a2a2a);
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        header_layout.addWidget(self.name_label)
        
        # デバイス情報
        if self.device_info:
            device_text = f"{self.device_info['name'][:12]}..."
            device_label = QLabel(device_text)
            device_label.setAlignment(Qt.AlignCenter)
            device_label.setStyleSheet("""
                QLabel {
                    color: #cccccc;
                    font-size: 7px;
                    padding: 1px;
                }
            """)
            header_layout.addWidget(device_label)
        
        layout.addLayout(header_layout)
        
        # ゲインノブ
        gain_layout = QVBoxLayout()
        gain_layout.setSpacing(3)
        
        gain_label = QLabel("GAIN")
        gain_label.setAlignment(Qt.AlignCenter)
        gain_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        gain_layout.addWidget(gain_label)
        
        self.gain_knob = ProfessionalKnob(size=45)
        self.gain_knob.setValue(self.gain_value)
        self.gain_knob.valueChanged.connect(self.on_gain_changed)
        gain_layout.addWidget(self.gain_knob, alignment=Qt.AlignCenter)
        
        self.gain_value_label = QLabel("50")
        self.gain_value_label.setAlignment(Qt.AlignCenter)
        self.gain_value_label.setStyleSheet("color: #ff8c00; font-size: 7px; font-weight: bold;")
        gain_layout.addWidget(self.gain_value_label)
        
        layout.addLayout(gain_layout)
        
        # パンノブ
        pan_layout = QVBoxLayout()
        pan_layout.setSpacing(3)
        
        pan_label = QLabel("PAN")
        pan_label.setAlignment(Qt.AlignCenter)
        pan_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        pan_layout.addWidget(pan_label)
        
        self.pan_knob = ProfessionalKnob(size=35)
        self.pan_knob.setValue(self.pan_value)
        self.pan_knob.valueChanged.connect(self.on_pan_changed)
        pan_layout.addWidget(self.pan_knob, alignment=Qt.AlignCenter)
        
        self.pan_value_label = QLabel("C")
        self.pan_value_label.setAlignment(Qt.AlignCenter)
        self.pan_value_label.setStyleSheet("color: #ff8c00; font-size: 7px; font-weight: bold;")
        pan_layout.addWidget(self.pan_value_label)
        
        layout.addLayout(pan_layout)
        
        # レベルメーター
        meter_layout = QVBoxLayout()
        meter_layout.setSpacing(3)
        
        meter_label = QLabel("LEVEL")
        meter_label.setAlignment(Qt.AlignCenter)
        meter_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        meter_layout.addWidget(meter_label)
        
        self.level_meter = ProfessionalLEDMeter(width=45, height=120)
        meter_layout.addWidget(self.level_meter, alignment=Qt.AlignCenter)
        
        layout.addLayout(meter_layout)
        
        # フェーダー
        fader_layout = QVBoxLayout()
        fader_layout.setSpacing(3)
        
        fader_label = QLabel("FADER")
        fader_label.setAlignment(Qt.AlignCenter)
        fader_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        fader_layout.addWidget(fader_label)
        
        self.fader = ProfessionalFader(width=20, height=80)
        self.fader.setValue(self.fader_value)
        self.fader.valueChanged.connect(self.on_fader_changed)
        fader_layout.addWidget(self.fader, alignment=Qt.AlignCenter)
        
        layout.addLayout(fader_layout)
        
        # MUTE/SOLOボタン
        button_layout = QVBoxLayout()
        button_layout.setSpacing(3)
        
        self.mute_button = QPushButton("MUTE")
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedHeight(20)
        self.mute_button.toggled.connect(self.on_mute_toggled)
        button_layout.addWidget(self.mute_button)
        
        self.solo_button = QPushButton("SOLO")
        self.solo_button.setCheckable(True)
        self.solo_button.setFixedHeight(20)
        self.solo_button.toggled.connect(self.on_solo_toggled)
        button_layout.addWidget(self.solo_button)
        
        # 削除ボタン
        self.remove_button = QPushButton("×")
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.clicked.connect(self.removeRequested.emit)
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                font-weight: bold;
                font-size: 10px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)
        button_layout.addWidget(self.remove_button, alignment=Qt.AlignCenter)
        
        layout.addLayout(button_layout)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            ProfessionalChannelStrip {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #3d3d3d,
                                           stop: 0.5 #2b2b2b,
                                           stop: 1 #3d3d3d);
                border-top: 1px solid #4a4a4a;
                border-left: 1px solid #4a4a4a;
                border-right: 1px solid #1a1a1a;
                border-bottom: 1px solid #1a1a1a;
                border-radius: 6px;
                margin: 2px;
            }
            
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a,
                                           stop: 1 #2a2a2a);
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                font-size: 8px;
                padding: 2px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5a5a5a,
                                           stop: 1 #3a3a3a);
            }
            
            QPushButton:checked {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #ff8c00,
                                           stop: 1 #cc6600);
                color: #000000;
            }
        """)
    
    def setup_audio_processor(self):
        """オーディオプロセッサ設定"""
        if not AUDIO_AVAILABLE or not self.device_info:
            return
        
        self.audio_processor = AudioProcessor(
            self.device_info['index'],
            channels=min(self.device_info['max_channels'], 2)
        )
        self.audio_processor.levelUpdated.connect(self.on_audio_level_update)
        self.audio_processor.start_monitoring()
    
    def on_gain_changed(self, value):
        """ゲイン変更"""
        self.gain_value = value
        self.gain_value_label.setText(f"{int(value)}")
        self.gainChanged.emit(value)
    
    def on_pan_changed(self, value):
        """パン変更"""
        self.pan_value = value
        if value < 45:
            self.pan_value_label.setText(f"L{int(50-value)}")
        elif value > 55:
            self.pan_value_label.setText(f"R{int(value-50)}")
        else:
            self.pan_value_label.setText("C")
        self.panChanged.emit(value)
    
    def on_fader_changed(self, value):
        """フェーダー変更"""
        self.fader_value = value
        self.faderChanged.emit(value)
    
    def on_mute_toggled(self, checked):
        """ミュート切り替え"""
        self.is_muted = checked
        self.muteToggled.emit(checked)
    
    def on_solo_toggled(self, checked):
        """ソロ切り替え"""
        self.is_solo = checked
        self.soloToggled.emit(checked)
    
    def on_audio_level_update(self, current_level, peak_level):
        """オーディオレベル更新"""
        # ゲインとミュートを適用
        if self.is_muted:
            current_level = 0.0
            peak_level = 0.0
        else:
            gain_factor = self.gain_value / 50.0  # 50を基準とする
            current_level *= gain_factor
            peak_level *= gain_factor
        
        self.level_meter.setLevel(current_level, peak_level)
    
    def cleanup(self):
        """クリーンアップ"""
        if self.audio_processor:
            self.audio_processor.stop_monitoring()


class DeviceListWidget(QFrame):
    """デバイス一覧ウィジェット"""
    
    deviceSelected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device_manager = DeviceManager()
        self.setup_ui()
        self.refresh_devices()
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(250)
        self.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # タイトル
        title_label = QLabel("AUDIO DEVICES")
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-weight: bold;
                font-size: 12px;
                padding: 4px;
            }
        """)
        layout.addWidget(title_label)
        
        # フィルターボタン
        filter_layout = QHBoxLayout()
        
        self.input_radio = QRadioButton("INPUT")
        self.output_radio = QRadioButton("OUTPUT")
        self.all_radio = QRadioButton("ALL")
        self.all_radio.setChecked(True)
        
        for radio in [self.input_radio, self.output_radio, self.all_radio]:
            radio.setStyleSheet("""
                QRadioButton {
                    color: #ffffff;
                    font-size: 9px;
                    font-weight: bold;
                }
                QRadioButton::indicator {
                    width: 12px;
                    height: 12px;
                }
                QRadioButton::indicator:checked {
                    background-color: #ff8c00;
                    border: 2px solid #ffffff;
                    border-radius: 6px;
                }
                QRadioButton::indicator:unchecked {
                    background-color: #333333;
                    border: 2px solid #666666;
                    border-radius: 6px;
                }
            """)
            radio.toggled.connect(self.filter_devices)
            filter_layout.addWidget(radio)
        
        layout.addLayout(filter_layout)
        
        # デバイスリスト
        self.device_list = QListWidget()
        self.device_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 3px;
                color: #ffffff;
                font-size: 9px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333333;
            }
            QListWidget::item:selected {
                background-color: #ff8c00;
                color: #000000;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
        self.device_list.itemDoubleClicked.connect(self.on_device_selected)
        layout.addWidget(self.device_list)
        
        # 更新ボタン
        refresh_button = QPushButton("REFRESH DEVICES")
        refresh_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a, stop: 1 #2a2a2a);
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                font-size: 9px;
                padding: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5a5a5a, stop: 1 #3a3a3a);
            }
        """)
        refresh_button.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh_button)
    
    def refresh_devices(self):
        """デバイス一覧更新"""
        self.device_manager.refresh_devices()
        self.filter_devices()
    
    def filter_devices(self):
        """デバイスフィルタリング"""
        self.device_list.clear()
        
        devices = []
        if self.input_radio.isChecked():
            devices = self.device_manager.get_input_devices()
        elif self.output_radio.isChecked():
            devices = self.device_manager.get_output_devices()
        else:  # ALL
            devices = self.device_manager.get_input_devices() + self.device_manager.get_output_devices()
        
        for device in devices:
            item_text = f"[{device['type'].upper()}] {device['name']}"
            if len(item_text) > 35:
                item_text = item_text[:32] + "..."
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, device)
            self.device_list.addItem(item)
    
    def on_device_selected(self, item):
        """デバイス選択時"""
        device = item.data(Qt.UserRole)
        if device:
            self.deviceSelected.emit(device)


class MasterSection(QFrame):
    """マスターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_styles()
        
        # マスターレベル
        self.left_level = 0.0
        self.right_level = 0.0
        self.left_peak = 0.0
        self.right_peak = 0.0
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(150)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # マスタータイトル
        title_label = QLabel("MASTER OUTPUT")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-weight: bold;
                font-size: 14px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a, stop: 1 #2a2a2a);
                border: 2px solid #ff8c00;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout.addWidget(title_label)
        
        # ステレオメーター
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(8)
        
        # 左チャンネル
        left_layout = QVBoxLayout()
        left_label = QLabel("L")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        left_layout.addWidget(left_label)
        
        self.left_meter = ProfessionalLEDMeter(width=35, height=200)
        left_layout.addWidget(self.left_meter)
        
        meter_layout.addLayout(left_layout)
        
        # 右チャンネル
        right_layout = QVBoxLayout()
        right_label = QLabel("R")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        right_layout.addWidget(right_label)
        
        self.right_meter = ProfessionalLEDMeter(width=35, height=200)
        right_layout.addWidget(self.right_meter)
        
        meter_layout.addLayout(right_layout)
        
        layout.addLayout(meter_layout)
        
        # マスターフェーダー
        fader_layout = QVBoxLayout()
        fader_label = QLabel("MASTER")
        fader_label.setAlignment(Qt.AlignCenter)
        fader_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        fader_layout.addWidget(fader_label)
        
        self.master_fader = ProfessionalFader(width=30, height=100)
        self.master_fader.setValue(85)
        fader_layout.addWidget(self.master_fader, alignment=Qt.AlignCenter)
        
        layout.addLayout(fader_layout)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            MasterSection {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #3d3d3d,
                                           stop: 0.5 #2b2b2b,
                                           stop: 1 #3d3d3d);
                border: 2px solid #ff8c00;
                border-radius: 8px;
            }
        """)
    
    def update_levels(self, left_level, right_level, left_peak, right_peak):
        """マスターレベル更新"""
        self.left_meter.setLevel(left_level, left_peak)
        self.right_meter.setLevel(right_level, right_peak)


class UADConsoleMainWindow(QMainWindow):
    """UAD Console メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.channel_strips = []
        self.setup_ui()
        self.setup_master_timer()
    
    def setup_ui(self):
        """UI構築"""
        self.setWindowTitle("UAD Console - Professional Audio Manager")
        self.setMinimumSize(1200, 700)
        
        # 背景色設定
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
        """)
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # デバイスリスト
        self.device_list = DeviceListWidget()
        self.device_list.deviceSelected.connect(self.add_channel_strip)
        main_layout.addWidget(self.device_list)
        
        # チャンネルエリア（スクロール可能）
        self.setup_channel_area(main_layout)
        
        # マスターセクション
        self.master_section = MasterSection()
        main_layout.addWidget(self.master_section)
    
    def setup_channel_area(self, main_layout):
        """チャンネルエリア構築"""
        channel_frame = QFrame()
        channel_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 4px;
            }
        """)
        
        channel_layout = QVBoxLayout(channel_frame)
        channel_layout.setContentsMargins(4, 4, 4, 4)
        
        # タイトル
        title_label = QLabel("CHANNEL STRIPS")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-weight: bold;
                font-size: 12px;
                padding: 4px;
            }
        """)
        channel_layout.addWidget(title_label)
        
        # スクロールエリア
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QScrollBar:horizontal {
                border: none;
                background: #3a3a3a;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #ff8c00;
                border-radius: 6px;
                min-width: 20px;
            }
        """)
        
        # チャンネルコンテナ
        self.channel_container = QWidget()
        self.channel_layout = QHBoxLayout(self.channel_container)
        self.channel_layout.setContentsMargins(4, 4, 4, 4)
        self.channel_layout.setSpacing(4)
        self.channel_layout.addStretch()
        
        self.scroll_area.setWidget(self.channel_container)
        channel_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(channel_frame, 1)
    
    def setup_master_timer(self):
        """マスタータイマー設定"""
        self.master_timer = QTimer()
        self.master_timer.timeout.connect(self.update_master_levels)
        self.master_timer.start(33)  # 30fps
    
    def add_channel_strip(self, device_info):
        """チャンネルストリップ追加"""
        channel_name = f"CH {len(self.channel_strips) + 1}"
        
        strip = ProfessionalChannelStrip(channel_name, device_info)
        strip.removeRequested.connect(lambda: self.remove_channel_strip(strip))
        
        # レイアウトに追加（ストレッチの前に挿入）
        self.channel_layout.insertWidget(len(self.channel_strips), strip)
        self.channel_strips.append(strip)
        
        print(f"Added channel: {channel_name} - {device_info['name']}")
    
    def remove_channel_strip(self, strip):
        """チャンネルストリップ削除"""
        if strip in self.channel_strips:
            strip.cleanup()
            self.channel_strips.remove(strip)
            strip.setParent(None)
            strip.deleteLater()
            print(f"Removed channel: {strip.channel_name}")
    
    def update_master_levels(self):
        """マスターレベル更新"""
        if not self.channel_strips:
            self.master_section.update_levels(0.0, 0.0, 0.0, 0.0)
            return
        
        # 全チャンネルのレベルを合成
        total_left = 0.0
        total_right = 0.0
        peak_left = 0.0
        peak_right = 0.0
        active_channels = 0
        
        for strip in self.channel_strips:
            if strip.audio_processor and not strip.is_muted:
                current, peak = strip.audio_processor.get_levels()
                
                # ゲインとフェーダーを適用
                gain_factor = strip.gain_value / 50.0
                fader_factor = strip.fader_value / 100.0
                current *= gain_factor * fader_factor
                peak *= gain_factor * fader_factor
                
                # パン計算
                pan_ratio = strip.pan_value / 100.0  # 0.0(L) - 1.0(R)
                left_gain = 1.0 - pan_ratio
                right_gain = pan_ratio
                
                total_left += current * left_gain
                total_right += current * right_gain
                peak_left = max(peak_left, peak * left_gain)
                peak_right = max(peak_right, peak * right_gain)
                
                active_channels += 1
        
        # 平均化
        if active_channels > 0:
            total_left = min(total_left / active_channels, 1.0)
            total_right = min(total_right / active_channels, 1.0)
        
        # マスターフェーダーを適用
        master_factor = self.master_section.master_fader.getValue() / 100.0
        total_left *= master_factor
        total_right *= master_factor
        peak_left *= master_factor
        peak_right *= master_factor
        
        self.master_section.update_levels(total_left, total_right, peak_left, peak_right)
    
    def closeEvent(self, event):
        """ウィンドウクローズ時"""
        for strip in self.channel_strips:
            strip.cleanup()
        event.accept()


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    
    # アプリケーション設定
    app.setApplicationName("UAD Console Professional")
    app.setApplicationVersion("1.0")
    
    # フォント設定
    font = QFont("Segoe UI", 8)
    app.setFont(font)
    
    # メインウィンドウ作成・表示
    window = UADConsoleMainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())