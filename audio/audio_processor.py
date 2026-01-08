"""
オーディオ処理モジュール
リアルタイム音量レベル計算とオーディオストリーム管理
"""
import numpy as np
import sounddevice as sd
import threading
import time
from typing import Callable, Optional


class AudioProcessor:
    def __init__(self, device_index: int, channels: int = 1, sample_rate: int = 44100):
        self.device_index = device_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_running = False
        self.stream = None
        
        # レベルメーター用
        self.current_level = 0.0
        self.peak_level = 0.0
        self.peak_hold_time = 1.0  # ピークホールド時間（秒）
        self.last_peak_time = 0.0
        
        # コールバック関数
        self.level_callback: Optional[Callable[[float, float], None]] = None
        
        # スレッドセーフティ用
        self.lock = threading.Lock()
    
    def set_level_callback(self, callback: Callable[[float, float], None]) -> None:
        """レベル更新時のコールバック関数を設定"""
        self.level_callback = callback
    
    def audio_callback(self, indata, frames, time, status):
        """オーディオストリームのコールバック"""
        if status:
            print(f"Audio callback status: {status}")
        
        # RMS（Root Mean Square）レベル計算
        if indata.size > 0:
            # ステレオの場合は平均を取る
            if self.channels > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # RMSレベル計算
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # dBに変換（-60dB〜0dBの範囲）
            if rms > 0:
                db_level = 20 * np.log10(rms)
                # -60dB以下は-60dBとする
                db_level = max(db_level, -60.0)
                # 0-1の範囲に正規化
                normalized_level = (db_level + 60.0) / 60.0
            else:
                normalized_level = 0.0
            
            with self.lock:
                self.current_level = normalized_level
                
                # ピークホールド処理
                current_time = time.time()
                if normalized_level > self.peak_level:
                    self.peak_level = normalized_level
                    self.last_peak_time = current_time
                elif current_time - self.last_peak_time > self.peak_hold_time:
                    # ピークホールド時間が経過したら徐々に減衰
                    self.peak_level = max(self.peak_level - 0.01, normalized_level)
                
                # コールバック呼び出し
                if self.level_callback:
                    self.level_callback(self.current_level, self.peak_level)
    
    def start_monitoring(self) -> bool:
        """オーディオモニタリング開始"""
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
            print(f"デバイス {self.device_index} のモニタリング開始")
            return True
            
        except Exception as e:
            print(f"オーディオストリーム開始エラー: {e}")
            return False
    
    def stop_monitoring(self) -> None:
        """オーディオモニタリング停止"""
        if self.stream and self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print(f"デバイス {self.device_index} のモニタリング停止")
    
    def get_levels(self) -> tuple[float, float]:
        """現在のレベルを取得（current_level, peak_level）"""
        with self.lock:
            return self.current_level, self.peak_level
    
    def __del__(self):
        """デストラクタ"""
        self.stop_monitoring()


if __name__ == "__main__":
    # テスト実行
    from device_manager import AudioDeviceManager
    
    manager = AudioDeviceManager()
    input_devices = manager.get_input_devices()
    
    if input_devices:
        # 最初の入力デバイスでテスト
        device = input_devices[0]
        print(f"テスト対象: {device['name']}")
        
        processor = AudioProcessor(device['index'], channels=1)
        
        def level_update(current, peak):
            # 簡易レベル表示
            bar_length = 50
            current_bar = int(current * bar_length)
            peak_pos = int(peak * bar_length)
            
            bar = ['─'] * bar_length
            for i in range(current_bar):
                bar[i] = '█'
            if peak_pos < bar_length:
                bar[peak_pos] = '▲'
            
            print(f"\r{''.join(bar)} {current:.2f} / {peak:.2f}", end='')
        
        processor.set_level_callback(level_update)
        
        if processor.start_monitoring():
            print("音声入力してください... (Ctrl+Cで終了)")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n終了中...")
            finally:
                processor.stop_monitoring()
    else:
        print("入力デバイスが見つかりません")