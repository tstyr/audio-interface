"""
チャンネルストリップウィジェット
UAD Console風の垂直チャンネルコントロール
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QPushButton, QFrame, QDial, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from .level_meter import LevelMeterWidget
from .styles import COLORS


class ChannelStrip(QFrame):
    """チャンネルストリップウィジェット"""
    
    # シグナル定義
    gain_changed = Signal(float)
    pan_changed = Signal(float)
    mute_toggled = Signal(bool)
    solo_toggled = Signal(bool)
    
    def __init__(self, channel_name="Channel", device_info=None, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.device_info = device_info
        self.audio_processor = None
        
        # 状態
        self.is_muted = False
        self.is_solo = False
        self.gain_value = 0.75  # 0.0 - 1.0
        self.pan_value = 0.5    # 0.0(L) - 1.0(R), 0.5=Center
        
        self.setup_ui()
        self.setup_connections()
        
        # レベル更新タイマー
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self.update_level_display)
        self.level_timer.start(33)  # 30fps
    
    def setup_ui(self):
        """UI構築"""
        self.setObjectName("ChannelStrip")
        self.setFixedWidth(80)
        self.setMinimumHeight(400)
        
        # メインレイアウト
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(8)
        
        # チャンネル名ラベル
        self.name_label = QLabel(self.channel_name)
        self.name_label.setObjectName("ChannelLabel")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumHeight(40)
        layout.addWidget(self.name_label)
        
        # ゲインコントロール
        gain_layout = QVBoxLayout()
        gain_layout.setSpacing(4)
        
        gain_label = QLabel("GAIN")
        gain_label.setAlignment(Qt.AlignCenter)
        gain_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px; font-weight: bold;")
        gain_layout.addWidget(gain_label)
        
        self.gain_slider = QSlider(Qt.Vertical)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(75)
        self.gain_slider.setFixedHeight(100)
        gain_layout.addWidget(self.gain_slider)
        
        self.gain_value_label = QLabel("75%")
        self.gain_value_label.setAlignment(Qt.AlignCenter)
        self.gain_value_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 8px; font-weight: bold;")
        gain_layout.addWidget(self.gain_value_label)
        
        layout.addLayout(gain_layout)
        
        # パンコントロール
        pan_layout = QVBoxLayout()
        pan_layout.setSpacing(4)
        
        pan_label = QLabel("PAN")
        pan_label.setAlignment(Qt.AlignCenter)
        pan_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px; font-weight: bold;")
        pan_layout.addWidget(pan_label)
        
        self.pan_dial = QDial()
        self.pan_dial.setRange(-50, 50)
        self.pan_dial.setValue(0)
        self.pan_dial.setFixedSize(40, 40)
        self.pan_dial.setNotchesVisible(True)
        pan_layout.addWidget(self.pan_dial, alignment=Qt.AlignCenter)
        
        self.pan_value_label = QLabel("C")
        self.pan_value_label.setAlignment(Qt.AlignCenter)
        self.pan_value_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 8px; font-weight: bold;")
        pan_layout.addWidget(self.pan_value_label)
        
        layout.addLayout(pan_layout)
        
        # レベルメーター
        self.level_meter = LevelMeterWidget("LEVEL")
        layout.addWidget(self.level_meter)
        
        # スペーサー
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Mute/Soloボタン
        button_layout = QVBoxLayout()
        button_layout.setSpacing(4)
        
        self.mute_button = QPushButton("MUTE")
        self.mute_button.setCheckable(True)
        self.mute_button.setObjectName("mute")
        self.mute_button.setFixedHeight(25)
        button_layout.addWidget(self.mute_button)
        
        self.solo_button = QPushButton("SOLO")
        self.solo_button.setCheckable(True)
        self.solo_button.setObjectName("solo")
        self.solo_button.setFixedHeight(25)
        button_layout.addWidget(self.solo_button)
        
        layout.addLayout(button_layout)
        
        # デバイス情報表示
        if self.device_info:
            info_text = f"Ch: {self.device_info.get('max_channels', 1)}"
            info_label = QLabel(info_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 7px;")
            layout.addWidget(info_label)
    
    def setup_connections(self):
        """シグナル・スロット接続"""
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        self.pan_dial.valueChanged.connect(self.on_pan_changed)
        self.mute_button.toggled.connect(self.on_mute_toggled)
        self.solo_button.toggled.connect(self.on_solo_toggled)
    
    def on_gain_changed(self, value):
        """ゲイン変更時"""
        self.gain_value = value / 100.0
        self.gain_value_label.setText(f"{value}%")
        self.gain_changed.emit(self.gain_value)
    
    def on_pan_changed(self, value):
        """パン変更時"""
        self.pan_value = (value + 50) / 100.0  # -50~50 -> 0.0~1.0
        
        # 表示テキスト
        if value == 0:
            text = "C"
        elif value < 0:
            text = f"L{abs(value)}"
        else:
            text = f"R{value}"
        
        self.pan_value_label.setText(text)
        self.pan_changed.emit(self.pan_value)
    
    def on_mute_toggled(self, checked):
        """ミュート切り替え時"""
        self.is_muted = checked
        if checked:
            self.mute_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #cc0000;
                    color: white;
                    font-weight: bold;
                }}
            """)
        else:
            self.mute_button.setStyleSheet("")
        
        self.mute_toggled.emit(checked)
    
    def on_solo_toggled(self, checked):
        """ソロ切り替え時"""
        self.is_solo = checked
        if checked:
            self.solo_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #ff8800;
                    color: white;
                    font-weight: bold;
                }}
            """)
        else:
            self.solo_button.setStyleSheet("")
        
        self.solo_toggled.emit(checked)
    
    def set_audio_processor(self, processor):
        """オーディオプロセッサを設定"""
        self.audio_processor = processor
        if processor:
            processor.set_level_callback(self.on_audio_level_update)
    
    def on_audio_level_update(self, current_level, peak_level):
        """オーディオレベル更新コールバック"""
        # ゲインとミュートを適用
        if self.is_muted:
            current_level = 0.0
            peak_level = 0.0
        else:
            current_level *= self.gain_value
            peak_level *= self.gain_value
        
        # メーターに反映（UIスレッドで実行される）
        self.level_meter.set_level(current_level, peak_level)
    
    def update_level_display(self):
        """レベル表示更新（タイマー用）"""
        if self.audio_processor:
            current, peak = self.audio_processor.get_levels()
            self.on_audio_level_update(current, peak)
    
    def start_monitoring(self):
        """モニタリング開始"""
        if self.audio_processor:
            return self.audio_processor.start_monitoring()
        return False
    
    def stop_monitoring(self):
        """モニタリング停止"""
        if self.audio_processor:
            self.audio_processor.stop_monitoring()
    
    def get_channel_info(self):
        """チャンネル情報を取得"""
        return {
            'name': self.channel_name,
            'device_info': self.device_info,
            'gain': self.gain_value,
            'pan': self.pan_value,
            'muted': self.is_muted,
            'solo': self.is_solo
        }


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
    from ..ui.styles import COMPLETE_STYLESHEET
    
    app = QApplication(sys.argv)
    app.setStyleSheet(COMPLETE_STYLESHEET)
    
    # テスト用メインウィンドウ
    window = QMainWindow()
    window.setWindowTitle("Channel Strip Test")
    
    central_widget = QWidget()
    layout = QHBoxLayout(central_widget)
    
    # テスト用チャンネルストリップ
    for i in range(3):
        device_info = {
            'name': f'Test Device {i+1}',
            'max_channels': 2,
            'sample_rate': 44100
        }
        
        strip = ChannelStrip(f"CH {i+1}", device_info)
        
        # シグナル接続テスト
        strip.gain_changed.connect(lambda v, ch=i+1: print(f"CH{ch} Gain: {v:.2f}"))
        strip.pan_changed.connect(lambda v, ch=i+1: print(f"CH{ch} Pan: {v:.2f}"))
        strip.mute_toggled.connect(lambda v, ch=i+1: print(f"CH{ch} Mute: {v}"))
        strip.solo_toggled.connect(lambda v, ch=i+1: print(f"CH{ch} Solo: {v}"))
        
        layout.addWidget(strip)
    
    window.setCentralWidget(central_widget)
    window.show()
    
    sys.exit(app.exec())