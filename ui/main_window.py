"""
メインウィンドウ
UAD Console風のオーディオミキサーUI
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                               QScrollArea, QPushButton, QLabel, QComboBox, 
                               QFrame, QMenuBar, QStatusBar, QMessageBox, QDialog,
                               QListWidget, QListWidgetItem, QDialogButtonBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from .channel_strip import ChannelStrip
from .level_meter import LevelMeterWidget
from .styles import COMPLETE_STYLESHEET, COLORS
from ..audio.device_manager import AudioDeviceManager
from ..audio.audio_processor import AudioProcessor


class DeviceSelectionDialog(QDialog):
    """デバイス選択ダイアログ"""
    
    def __init__(self, device_manager, parent=None):
        super().__init__(parent)
        self.device_manager = device_manager
        self.selected_device = None
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("オーディオデバイスを選択")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 説明ラベル
        info_label = QLabel("チャンネルとして追加するオーディオデバイスを選択してください:")
        layout.addWidget(info_label)
        
        # デバイスリスト
        self.device_list = QListWidget()
        
        # 入力デバイスを追加
        input_devices = self.device_manager.get_input_devices()
        for device in input_devices:
            item_text = f"[入力] {device['name']} ({device['max_channels']}ch)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, device)
            self.device_list.addItem(item)
        
        # 出力デバイスを追加
        output_devices = self.device_manager.get_output_devices()
        for device in output_devices:
            item_text = f"[出力] {device['name']} ({device['max_channels']}ch)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, device)
            self.device_list.addItem(item)
        
        layout.addWidget(self.device_list)
        
        # ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # ダブルクリックでOK
        self.device_list.itemDoubleClicked.connect(self.accept)
    
    def accept(self):
        current_item = self.device_list.currentItem()
        if current_item:
            self.selected_device = current_item.data(Qt.UserRole)
        super().accept()


class MasterSection(QFrame):
    """マスターセクション"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        self.setFixedWidth(120)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                           stop: 0 {COLORS['surface']},
                                           stop: 0.5 {COLORS['surface_light']},
                                           stop: 1 {COLORS['surface']});
                border: 2px solid {COLORS['accent']};
                border-radius: 8px;
                margin: 4px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # マスタータイトル
        title_label = QLabel("MASTER")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                font-size: 14px;
                font-weight: bold;
                padding: 4px;
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['accent']};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(title_label)
        
        # マスターレベルメーター（ステレオ）
        meter_layout = QHBoxLayout()
        meter_layout.setSpacing(4)
        
        self.left_meter = LevelMeterWidget("L")
        self.right_meter = LevelMeterWidget("R")
        
        meter_layout.addWidget(self.left_meter)
        meter_layout.addWidget(self.right_meter)
        
        layout.addLayout(meter_layout)
        
        # マスターボリューム
        volume_label = QLabel("MASTER VOL")
        volume_label.setAlignment(Qt.AlignCenter)
        volume_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px; font-weight: bold;")
        layout.addWidget(volume_label)
        
        # マスターコントロールボタン
        control_layout = QVBoxLayout()
        control_layout.setSpacing(4)
        
        self.main_mute_button = QPushButton("MAIN MUTE")
        self.main_mute_button.setCheckable(True)
        self.main_mute_button.setFixedHeight(30)
        control_layout.addWidget(self.main_mute_button)
        
        layout.addLayout(control_layout)
    
    def set_master_levels(self, left_level, right_level, left_peak, right_peak):
        """マスターレベルを設定"""
        self.left_meter.set_level(left_level, left_peak)
        self.right_meter.set_level(right_level, right_peak)


class MainWindow(QMainWindow):
    """メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        self.device_manager = AudioDeviceManager()
        self.channel_strips = []
        self.master_left_level = 0.0
        self.master_right_level = 0.0
        self.master_left_peak = 0.0
        self.master_right_peak = 0.0
        
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        # マスターレベル更新タイマー
        self.master_timer = QTimer()
        self.master_timer.timeout.connect(self.update_master_levels)
        self.master_timer.start(33)  # 30fps
    
    def setup_ui(self):
        """UI構築"""
        self.setWindowTitle("UAD Console風 オーディオミキサー")
        self.setMinimumSize(800, 600)
        
        # スタイルシート適用
        self.setStyleSheet(COMPLETE_STYLESHEET)
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # チャンネルエリア（スクロール可能）
        self.setup_channel_area(main_layout)
        
        # マスターセクション
        self.master_section = MasterSection()
        main_layout.addWidget(self.master_section)
    
    def setup_channel_area(self, main_layout):
        """チャンネルエリア構築"""
        channel_frame = QFrame()
        channel_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['background']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)
        
        channel_layout = QVBoxLayout(channel_frame)
        channel_layout.setContentsMargins(4, 4, 4, 4)
        
        # チャンネル追加ボタン
        add_button = QPushButton("+ チャンネル追加")
        add_button.setFixedHeight(30)
        add_button.clicked.connect(self.add_channel)
        channel_layout.addWidget(add_button)
        
        # スクロールエリア
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlways)
        
        # チャンネルストリップコンテナ
        self.channel_container = QWidget()
        self.channel_layout = QHBoxLayout(self.channel_container)
        self.channel_layout.setContentsMargins(4, 4, 4, 4)
        self.channel_layout.setSpacing(4)
        self.channel_layout.addStretch()  # 右端にスペース
        
        self.scroll_area.setWidget(self.channel_container)
        channel_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(channel_frame, 1)  # 伸縮可能
    
    def setup_menu(self):
        """メニューバー構築"""
        menubar = self.menuBar()
        
        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル(&F)")
        
        refresh_action = QAction("デバイス更新(&R)", self)
        refresh_action.triggered.connect(self.refresh_devices)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("終了(&X)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ(&H)")
        
        about_action = QAction("バージョン情報(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """ステータスバー構築"""
        status_bar = self.statusBar()
        status_bar.showMessage("準備完了")
    
    def add_channel(self):
        """チャンネル追加"""
        dialog = DeviceSelectionDialog(self.device_manager, self)
        
        if dialog.exec() == QDialog.Accepted and dialog.selected_device:
            device = dialog.selected_device
            
            # チャンネル名生成
            channel_name = f"CH {len(self.channel_strips) + 1}"
            
            # チャンネルストリップ作成
            strip = ChannelStrip(channel_name, device)
            
            # オーディオプロセッサ作成（入力デバイスの場合）
            if device['type'] == 'input':
                processor = AudioProcessor(
                    device['index'], 
                    channels=min(device['max_channels'], 2)  # 最大2チャンネル
                )
                strip.set_audio_processor(processor)
            
            # シグナル接続
            strip.gain_changed.connect(lambda v, s=strip: self.on_channel_gain_changed(s, v))
            strip.pan_changed.connect(lambda v, s=strip: self.on_channel_pan_changed(s, v))
            strip.mute_toggled.connect(lambda v, s=strip: self.on_channel_mute_toggled(s, v))
            strip.solo_toggled.connect(lambda v, s=strip: self.on_channel_solo_toggled(s, v))
            
            # レイアウトに追加（ストレッチの前に挿入）
            self.channel_layout.insertWidget(len(self.channel_strips), strip)
            self.channel_strips.append(strip)
            
            # モニタリング開始
            if device['type'] == 'input':
                if strip.start_monitoring():
                    self.statusBar().showMessage(f"チャンネル '{channel_name}' を追加しました")
                else:
                    self.statusBar().showMessage(f"チャンネル '{channel_name}' の追加に失敗しました")
            else:
                self.statusBar().showMessage(f"出力チャンネル '{channel_name}' を追加しました")
    
    def on_channel_gain_changed(self, strip, value):
        """チャンネルゲイン変更時"""
        info = strip.get_channel_info()
        print(f"{info['name']} Gain: {value:.2f}")
    
    def on_channel_pan_changed(self, strip, value):
        """チャンネルパン変更時"""
        info = strip.get_channel_info()
        print(f"{info['name']} Pan: {value:.2f}")
    
    def on_channel_mute_toggled(self, strip, muted):
        """チャンネルミュート切り替え時"""
        info = strip.get_channel_info()
        print(f"{info['name']} Mute: {muted}")
    
    def on_channel_solo_toggled(self, strip, solo):
        """チャンネルソロ切り替え時"""
        info = strip.get_channel_info()
        print(f"{info['name']} Solo: {solo}")
    
    def update_master_levels(self):
        """マスターレベル更新"""
        # 全チャンネルのレベルを合成してマスターレベルを計算
        total_left = 0.0
        total_right = 0.0
        peak_left = 0.0
        peak_right = 0.0
        
        active_channels = 0
        
        for strip in self.channel_strips:
            if strip.audio_processor and not strip.is_muted:
                current, peak = strip.audio_processor.get_levels()
                
                # ゲインとパンを適用
                current *= strip.gain_value
                peak *= strip.gain_value
                
                # パン計算
                left_gain = 1.0 - strip.pan_value  # 0.0(R) - 1.0(L)
                right_gain = strip.pan_value       # 0.0(L) - 1.0(R)
                
                total_left += current * left_gain
                total_right += current * right_gain
                peak_left = max(peak_left, peak * left_gain)
                peak_right = max(peak_right, peak * right_gain)
                
                active_channels += 1
        
        # 平均化（複数チャンネルがある場合）
        if active_channels > 0:
            total_left = min(total_left / active_channels, 1.0)
            total_right = min(total_right / active_channels, 1.0)
        
        # マスターセクションに反映
        self.master_section.set_master_levels(total_left, total_right, peak_left, peak_right)
    
    def refresh_devices(self):
        """デバイス一覧更新"""
        self.device_manager.refresh_devices()
        self.statusBar().showMessage("デバイス一覧を更新しました")
    
    def show_about(self):
        """バージョン情報表示"""
        QMessageBox.about(self, "バージョン情報", 
                         "UAD Console風 オーディオミキサー v1.0\n\n"
                         "PySide6 + sounddevice を使用したプロフェッショナル\n"
                         "オーディオミキサーアプリケーション")
    
    def closeEvent(self, event):
        """ウィンドウクローズ時"""
        # 全チャンネルのモニタリング停止
        for strip in self.channel_strips:
            strip.stop_monitoring()
        
        event.accept()


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())