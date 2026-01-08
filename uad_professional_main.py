"""
UAD Console Professional - メインアプリケーション
実機画像を基にした完全再現版
"""
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
                               QComboBox, QProgressBar, QFileDialog, QMessageBox,
                               QSplitter, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QSize
from PySide6.QtGui import (QFont, QColor, QPalette, QLinearGradient, QBrush, 
                           QPainter, QPen, QIcon, QPixmap)

from uad_professional_data import (SessionManager, SessionData, ChannelSettings, 
                                   SystemSettings, ClockSource, PluginDatabase)


class TopToolbar(QFrame):
    """トップツールバー - セッション管理とシステム情報"""
    
    # シグナル定義
    newSessionRequested = Signal()
    openSessionRequested = Signal()
    saveSessionRequested = Signal()
    settingsRequested = Signal()
    helpRequested = Signal()
    sampleRateChanged = Signal(int)
    clockSourceChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_manager = SessionManager()
        self.setup_ui()
        self.setup_styles()
        
        # システム情報更新タイマー
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_info)
        self.system_timer.start(100)  # 10fps
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedHeight(50)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        
        # 左側: SESSION管理
        session_group = self.create_session_group()
        layout.addWidget(session_group)
        
        # 中央: システム情報
        system_group = self.create_system_group()
        layout.addWidget(system_group, 1)  # 伸縮可能
        
        # 右側: SETTINGS/HELP
        control_group = self.create_control_group()
        layout.addWidget(control_group)
    
    def create_session_group(self) -> QGroupBox:
        """セッション管理グループ"""
        group = QGroupBox("SESSION")
        group.setFixedWidth(200)
        
        layout = QHBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # NEW ボタン
        self.new_button = QPushButton("NEW")
        self.new_button.setFixedSize(50, 30)
        self.new_button.clicked.connect(self.newSessionRequested.emit)
        layout.addWidget(self.new_button)
        
        # OPEN ボタン
        self.open_button = QPushButton("OPEN")
        self.open_button.setFixedSize(50, 30)
        self.open_button.clicked.connect(self.openSessionRequested.emit)
        layout.addWidget(self.open_button)
        
        # SAVE ボタン
        self.save_button = QPushButton("SAVE")
        self.save_button.setFixedSize(50, 30)
        self.save_button.clicked.connect(self.saveSessionRequested.emit)
        layout.addWidget(self.save_button)
        
        return group
    
    def create_system_group(self) -> QGroupBox:
        """システム情報グループ"""
        group = QGroupBox("SYSTEM STATUS")
        
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)
        
        # サンプルレート
        sr_layout = QVBoxLayout()
        sr_label = QLabel("SAMPLE RATE")
        sr_label.setAlignment(Qt.AlignCenter)
        sr_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        sr_layout.addWidget(sr_label)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["44.1kHz", "48kHz", "88.2kHz", "96kHz", "176.4kHz", "192kHz"])
        self.sample_rate_combo.setCurrentText("44.1kHz")
        self.sample_rate_combo.setFixedSize(80, 25)
        self.sample_rate_combo.currentTextChanged.connect(self.on_sample_rate_changed)
        sr_layout.addWidget(self.sample_rate_combo)
        
        layout.addLayout(sr_layout)
        
        # クロックソース
        clock_layout = QVBoxLayout()
        clock_label = QLabel("CLOCK SOURCE")
        clock_label.setAlignment(Qt.AlignCenter)
        clock_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        clock_layout.addWidget(clock_label)
        
        self.clock_combo = QComboBox()
        self.clock_combo.addItems(["INTERNAL", "EXTERNAL", "WORD CLOCK", "S/PDIF", "ADAT"])
        self.clock_combo.setCurrentText("INTERNAL")
        self.clock_combo.setFixedSize(90, 25)
        self.clock_combo.currentTextChanged.connect(self.on_clock_source_changed)
        clock_layout.addWidget(self.clock_combo)
        
        layout.addLayout(clock_layout)
        
        # DSP使用率メーター
        dsp_layout = QVBoxLayout()
        dsp_label = QLabel("DSP USAGE")
        dsp_label.setAlignment(Qt.AlignCenter)
        dsp_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        dsp_layout.addWidget(dsp_label)
        
        self.dsp_meter = QProgressBar()
        self.dsp_meter.setRange(0, 100)
        self.dsp_meter.setValue(0)
        self.dsp_meter.setFixedSize(100, 20)
        self.dsp_meter.setTextVisible(True)
        dsp_layout.addWidget(self.dsp_meter)
        
        layout.addLayout(dsp_layout)
        
        # バッファサイズ
        buffer_layout = QVBoxLayout()
        buffer_label = QLabel("BUFFER SIZE")
        buffer_label.setAlignment(Qt.AlignCenter)
        buffer_label.setStyleSheet("color: #cccccc; font-size: 8px; font-weight: bold;")
        buffer_layout.addWidget(buffer_label)
        
        self.buffer_combo = QComboBox()
        self.buffer_combo.addItems(["64", "128", "256", "512", "1024", "2048"])
        self.buffer_combo.setCurrentText("256")
        self.buffer_combo.setFixedSize(70, 25)
        buffer_layout.addWidget(self.buffer_combo)
        
        layout.addLayout(buffer_layout)
        
        return group
    
    def create_control_group(self) -> QGroupBox:
        """コントロールグループ"""
        group = QGroupBox("CONTROL")
        group.setFixedWidth(150)
        
        layout = QHBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # SETTINGS ボタン
        self.settings_button = QPushButton("SETTINGS")
        self.settings_button.setFixedSize(65, 30)
        self.settings_button.clicked.connect(self.settingsRequested.emit)
        layout.addWidget(self.settings_button)
        
        # HELP ボタン
        self.help_button = QPushButton("HELP")
        self.help_button.setFixedSize(55, 30)
        self.help_button.clicked.connect(self.helpRequested.emit)
        layout.addWidget(self.help_button)
        
        return group
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            TopToolbar {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a,
                                           stop: 0.5 #3a3a3a,
                                           stop: 1 #2a2a2a);
                border-bottom: 2px solid #ff8c00;
            }
            
            QGroupBox {
                color: #ff8c00;
                font-weight: bold;
                font-size: 9px;
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5a5a5a,
                                           stop: 1 #3a3a3a);
                border: 1px solid #666666;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                font-size: 8px;
                padding: 2px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #6a6a6a,
                                           stop: 1 #4a4a4a);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #3a3a3a,
                                           stop: 1 #5a5a5a);
            }
            
            QComboBox {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a,
                                           stop: 1 #2a2a2a);
                border: 1px solid #555555;
                border-radius: 2px;
                color: #ffffff;
                font-weight: bold;
                font-size: 8px;
                padding: 2px 4px;
            }
            
            QComboBox:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #5a5a5a,
                                           stop: 1 #3a3a3a);
            }
            
            QComboBox::drop-down {
                border: none;
                width: 15px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                margin-right: 4px;
            }
            
            QProgressBar {
                background-color: #2a2a2a;
                border: 1px solid #555555;
                border-radius: 2px;
                text-align: center;
                color: #ffffff;
                font-size: 7px;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #00ff00,
                                           stop: 0.7 #ffff00,
                                           stop: 1 #ff0000);
                border-radius: 1px;
            }
        """)
    
    def on_sample_rate_changed(self, text: str):
        """サンプルレート変更"""
        rate_map = {
            "44.1kHz": 44100,
            "48kHz": 48000,
            "88.2kHz": 88200,
            "96kHz": 96000,
            "176.4kHz": 176400,
            "192kHz": 192000
        }
        rate = rate_map.get(text, 44100)
        self.sampleRateChanged.emit(rate)
    
    def on_clock_source_changed(self, text: str):
        """クロックソース変更"""
        self.clockSourceChanged.emit(text)
    
    def update_system_info(self):
        """システム情報更新"""
        # DSP使用率をシミュレート（実際の実装では実際の値を取得）
        import random
        current_value = self.dsp_meter.value()
        # ランダムな変動を加える
        new_value = max(0, min(100, current_value + random.randint(-2, 2)))
        self.dsp_meter.setValue(new_value)
        
        # 使用率に応じて色を変更
        if new_value < 50:
            color = "#00ff00"  # 緑
        elif new_value < 80:
            color = "#ffff00"  # 黄
        else:
            color = "#ff0000"  # 赤
        
        self.dsp_meter.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 1px;
            }}
        """)


class DevicePanel(QFrame):
    """デバイスパネル - 左側のオーディオデバイス一覧"""
    
    deviceSelected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(220)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        
        # タイトル
        title_label = QLabel("AUDIO DEVICES")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-weight: bold;
                font-size: 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a, stop: 1 #2a2a2a);
                border: 1px solid #ff8c00;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(title_label)
        
        # フィルターボタン
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(2)
        
        self.input_button = QPushButton("INPUT")
        self.output_button = QPushButton("OUTPUT")
        self.all_button = QPushButton("ALL")
        
        for btn in [self.input_button, self.output_button, self.all_button]:
            btn.setCheckable(True)
            btn.setFixedHeight(25)
            filter_layout.addWidget(btn)
        
        self.all_button.setChecked(True)
        layout.addLayout(filter_layout)
        
        # デバイスリスト（スクロール可能）
        self.device_scroll = QScrollArea()
        self.device_scroll.setWidgetResizable(True)
        self.device_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.device_widget = QWidget()
        self.device_layout = QVBoxLayout(self.device_widget)
        self.device_layout.setContentsMargins(2, 2, 2, 2)
        self.device_layout.setSpacing(2)
        
        # サンプルデバイス（実際の実装では sounddevice から取得）
        sample_devices = [
            {"name": "Microsoft サウンド マッパー - Output", "type": "output", "index": 0},
            {"name": "スピーカー (Realtek(R) Audio)", "type": "output", "index": 1},
            {"name": "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)", "type": "input", "index": 2},
            {"name": "マイク (Realtek(R) Audio)", "type": "input", "index": 3},
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "type": "output", "index": 4},
        ]
        
        for device in sample_devices:
            device_btn = QPushButton(f"[{device['type'].upper()}] {device['name'][:25]}...")
            device_btn.setFixedHeight(30)
            device_btn.clicked.connect(lambda checked, d=device: self.deviceSelected.emit(d))
            self.device_layout.addWidget(device_btn)
        
        self.device_layout.addStretch()
        self.device_scroll.setWidget(self.device_widget)
        layout.addWidget(self.device_scroll)
        
        # REFRESH ボタン
        refresh_button = QPushButton("REFRESH DEVICES")
        refresh_button.setFixedHeight(30)
        layout.addWidget(refresh_button)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            DevicePanel {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #2a2a2a,
                                           stop: 1 #1a1a1a);
                border-right: 2px solid #ff8c00;
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
                text-align: left;
                padding: 4px;
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
            
            QScrollArea {
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
        """)


class MasterOutputPanel(QFrame):
    """マスターアウトプットパネル - 右側のマスターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """UI構築"""
        self.setFixedWidth(180)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # タイトル
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
                padding: 8px;
            }
        """)
        layout.addWidget(title_label)
        
        # L/R メーター
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(12)
        
        # 左チャンネル
        left_layout = QVBoxLayout()
        left_label = QLabel("L")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
        left_layout.addWidget(left_label)
        
        # 仮のメーター（実際の実装では ProfessionalLEDMeter を使用）
        self.left_meter = QProgressBar()
        self.left_meter.setOrientation(Qt.Orientation.Vertical)
        self.left_meter.setFixedSize(25, 200)
        self.left_meter.setRange(0, 100)
        self.left_meter.setValue(0)
        left_layout.addWidget(self.left_meter)
        
        meter_layout.addLayout(left_layout)
        
        # 右チャンネル
        right_layout = QVBoxLayout()
        right_label = QLabel("R")
        right_label.setAlignment(Qt.AlignCenter)
        right_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12px;")
        right_layout.addWidget(right_label)
        
        self.right_meter = QProgressBar()
        self.right_meter.setOrientation(Qt.Orientation.Vertical)
        self.right_meter.setFixedSize(25, 200)
        self.right_meter.setRange(0, 100)
        self.right_meter.setValue(0)
        right_layout.addWidget(self.right_meter)
        
        meter_layout.addLayout(right_layout)
        
        layout.addLayout(meter_layout)
        
        # マスターフェーダー
        fader_layout = QVBoxLayout()
        fader_label = QLabel("MASTER")
        fader_label.setAlignment(Qt.AlignCenter)
        fader_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 10px;")
        fader_layout.addWidget(fader_label)
        
        # 仮のフェーダー（実際の実装では ProfessionalFader を使用）
        self.master_fader = QProgressBar()
        self.master_fader.setOrientation(Qt.Orientation.Vertical)
        self.master_fader.setFixedSize(30, 100)
        self.master_fader.setRange(0, 100)
        self.master_fader.setValue(85)
        fader_layout.addWidget(self.master_fader, alignment=Qt.AlignCenter)
        
        layout.addLayout(fader_layout)
        
        # Control Room セクション
        control_layout = QVBoxLayout()
        control_label = QLabel("CONTROL ROOM")
        control_label.setAlignment(Qt.AlignCenter)
        control_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 9px;")
        control_layout.addWidget(control_label)
        
        # ALT, DIM, MONO, MUTE, TALKBACK ボタン
        control_buttons = ["ALT", "DIM", "MONO", "MUTE", "TALK"]
        for btn_text in control_buttons:
            btn = QPushButton(btn_text)
            btn.setCheckable(True)
            btn.setFixedHeight(25)
            control_layout.addWidget(btn)
        
        layout.addLayout(control_layout)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            MasterOutputPanel {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #1a1a1a,
                                           stop: 1 #2a2a2a);
                border-left: 2px solid #ff8c00;
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
            
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 2px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0,
                                           stop: 0 #00ff00,
                                           stop: 0.7 #ffff00,
                                           stop: 1 #ff0000);
                border-radius: 1px;
            }
        """)


class ChannelStripsArea(QFrame):
    """チャンネルストリップエリア - 中央のスクロール可能エリア"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channel_strips = []
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """UI構築"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # タイトル
        title_label = QLabel("CHANNEL STRIPS")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: #ff8c00;
                font-weight: bold;
                font-size: 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #4a4a4a, stop: 1 #2a2a2a);
                border: 1px solid #ff8c00;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(title_label)
        
        # スクロールエリア
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # チャンネルコンテナ
        self.channel_container = QWidget()
        self.channel_layout = QHBoxLayout(self.channel_container)
        self.channel_layout.setContentsMargins(4, 4, 4, 4)
        self.channel_layout.setSpacing(2)
        self.channel_layout.addStretch()  # 右端にスペース
        
        self.scroll_area.setWidget(self.channel_container)
        layout.addWidget(self.scroll_area)
    
    def setup_styles(self):
        """スタイル設定"""
        self.setStyleSheet("""
            ChannelStripsArea {
                background-color: #2b2b2b;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            
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
            
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
        """)
    
    def add_channel_strip(self, device_info: Dict[str, Any]):
        """チャンネルストリップ追加"""
        # 仮のチャンネルストリップ（実際の実装では ProfessionalChannelStrip を使用）
        strip = QFrame()
        strip.setFixedSize(85, 500)
        strip.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 #3d3d3d,
                                           stop: 0.5 #2b2b2b,
                                           stop: 1 #3d3d3d);
                border: 1px solid #555555;
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        # チャンネル名ラベル
        strip_layout = QVBoxLayout(strip)
        name_label = QLabel(f"CH {len(self.channel_strips) + 1}")
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 10px;
                background: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 3px;
            }
        """)
        strip_layout.addWidget(name_label)
        strip_layout.addStretch()
        
        # レイアウトに追加（ストレッチの前に挿入）
        self.channel_layout.insertWidget(len(self.channel_strips), strip)
        self.channel_strips.append(strip)
        
        print(f"Added channel strip for device: {device_info['name']}")


class UADProfessionalMainWindow(QMainWindow):
    """UAD Professional メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.session_manager = SessionManager()
        self.setup_ui()
        self.setup_connections()
        
        # 新規セッション作成
        self.session_manager.new_session("Default Session")
    
    def setup_ui(self):
        """UI構築"""
        self.setWindowTitle("UAD Console - Professional Audio Manager")
        self.setMinimumSize(1400, 800)
        
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
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # トップツールバー
        self.toolbar = TopToolbar()
        main_layout.addWidget(self.toolbar)
        
        # メインエリア（3分割）
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側: デバイスパネル
        self.device_panel = DevicePanel()
        main_splitter.addWidget(self.device_panel)
        
        # 中央: チャンネルストリップエリア
        self.channel_area = ChannelStripsArea()
        main_splitter.addWidget(self.channel_area)
        
        # 右側: マスターアウトプットパネル
        self.master_panel = MasterOutputPanel()
        main_splitter.addWidget(self.master_panel)
        
        # スプリッターの初期サイズ設定
        main_splitter.setSizes([220, 800, 180])
        main_splitter.setCollapsible(0, False)  # デバイスパネルは折りたたみ不可
        main_splitter.setCollapsible(2, False)  # マスターパネルは折りたたみ不可
        
        main_layout.addWidget(main_splitter)
    
    def setup_connections(self):
        """シグナル・スロット接続"""
        # ツールバー
        self.toolbar.newSessionRequested.connect(self.new_session)
        self.toolbar.openSessionRequested.connect(self.open_session)
        self.toolbar.saveSessionRequested.connect(self.save_session)
        self.toolbar.settingsRequested.connect(self.show_settings)
        self.toolbar.helpRequested.connect(self.show_help)
        
        # デバイスパネル
        self.device_panel.deviceSelected.connect(self.add_channel)
    
    def new_session(self):
        """新規セッション"""
        session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_manager.new_session(session_name)
        print(f"Created new session: {session_name}")
    
    def open_session(self):
        """セッション読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "UAD Session Files (*.json)"
        )
        if file_path:
            if self.session_manager.load_session(file_path):
                print(f"Loaded session: {file_path}")
                # UI更新処理をここに追加
            else:
                QMessageBox.warning(self, "Error", "Failed to load session file.")
    
    def save_session(self):
        """セッション保存"""
        if not self.session_manager.session_file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Session", "", "UAD Session Files (*.json)"
            )
            if not file_path:
                return
            self.session_manager.session_file_path = file_path
        
        if self.session_manager.save_session():
            print(f"Saved session: {self.session_manager.session_file_path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save session file.")
    
    def show_settings(self):
        """設定画面表示"""
        QMessageBox.information(self, "Settings", "Settings dialog will be implemented here.")
    
    def show_help(self):
        """ヘルプ表示"""
        QMessageBox.information(self, "Help", "UAD Console Professional v1.0\n\nProfessional Audio Interface Manager")
    
    def add_channel(self, device_info: Dict[str, Any]):
        """チャンネル追加"""
        self.channel_area.add_channel_strip(device_info)
        self.session_manager.add_channel(device_info)


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
    window = UADProfessionalMainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())