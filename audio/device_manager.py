"""
オーディオデバイス管理モジュール
システム上のオーディオデバイスを列挙し、入力/出力を分類
"""
import sounddevice as sd
from typing import List, Dict, Any


class AudioDeviceManager:
    def __init__(self):
        self.devices = []
        self.input_devices = []
        self.output_devices = []
        self.refresh_devices()
    
    def refresh_devices(self) -> None:
        """デバイス一覧を更新"""
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
                
                # 入力デバイス（マイクなど）
                if device['max_input_channels'] > 0:
                    device_info['type'] = 'input'
                    device_info['max_channels'] = device['max_input_channels']
                    self.input_devices.append(device_info)
                
                # 出力デバイス（スピーカーなど）
                if device['max_output_channels'] > 0:
                    device_info['type'] = 'output'
                    device_info['max_channels'] = device['max_output_channels']
                    self.output_devices.append(device_info)
                    
        except Exception as e:
            print(f"デバイス取得エラー: {e}")
    
    def get_input_devices(self) -> List[Dict[str, Any]]:
        """入力デバイス一覧を取得"""
        return self.input_devices
    
    def get_output_devices(self) -> List[Dict[str, Any]]:
        """出力デバイス一覧を取得"""
        return self.output_devices
    
    def get_device_by_index(self, index: int) -> Dict[str, Any]:
        """インデックスでデバイス情報を取得"""
        if 0 <= index < len(self.devices):
            return self.devices[index]
        return None
    
    def print_devices(self) -> None:
        """デバイス一覧をコンソールに表示"""
        print("=== オーディオデバイス一覧 ===")
        print("\n【入力デバイス】")
        for device in self.input_devices:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      チャンネル数: {device['max_channels']}")
            print(f"      サンプルレート: {device['sample_rate']} Hz")
            print()
        
        print("【出力デバイス】")
        for device in self.output_devices:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      チャンネル数: {device['max_channels']}")
            print(f"      サンプルレート: {device['sample_rate']} Hz")
            print()


if __name__ == "__main__":
    # テスト実行
    manager = AudioDeviceManager()
    manager.print_devices()