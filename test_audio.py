"""
オーディオデバイステスト用スクリプト
開発・デバッグ用
"""
import sys
import time
from audio.device_manager import AudioDeviceManager
from audio.audio_processor import AudioProcessor


def test_device_enumeration():
    """デバイス一覧テスト"""
    print("=== デバイス一覧テスト ===")
    manager = AudioDeviceManager()
    manager.print_devices()
    return manager


def test_audio_processing(device_manager):
    """オーディオ処理テスト"""
    print("\n=== オーディオ処理テスト ===")
    
    input_devices = device_manager.get_input_devices()
    if not input_devices:
        print("入力デバイスが見つかりません")
        return
    
    # 最初の入力デバイスでテスト
    device = input_devices[0]
    print(f"テスト対象デバイス: {device['name']}")
    
    processor = AudioProcessor(device['index'], channels=1)
    
    # レベル表示用コールバック
    def level_callback(current, peak):
        # プログレスバー風表示
        bar_length = 50
        current_bar = int(current * bar_length)
        peak_pos = int(peak * bar_length)
        
        bar = ['─'] * bar_length
        for i in range(current_bar):
            if i < bar_length * 0.7:
                bar[i] = '█'  # 緑
            elif i < bar_length * 0.9:
                bar[i] = '▓'  # 黄
            else:
                bar[i] = '▒'  # 赤
        
        if peak_pos < bar_length:
            bar[peak_pos] = '▲'
        
        # dB値計算
        if current > 0:
            db_current = 20 * math.log10(current)
            db_current = max(db_current, -60.0)
        else:
            db_current = -60.0
        
        print(f"\r{''.join(bar)} {db_current:5.1f}dB", end='', flush=True)
    
    processor.set_level_callback(level_callback)
    
    if processor.start_monitoring():
        print("音声入力してください... (10秒間測定、Ctrl+Cで中断)")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n測定を中断しました")
        finally:
            processor.stop_monitoring()
            print("\n測定終了")
    else:
        print("オーディオモニタリングの開始に失敗しました")


def test_ui_components():
    """UIコンポーネントテスト"""
    print("\n=== UIコンポーネントテスト ===")
    
    try:
        from PySide6.QtWidgets import QApplication
        from ui.level_meter import LevelMeterWidget
        from ui.channel_strip import ChannelStrip
        
        print("✅ PySide6インポート成功")
        print("✅ UIコンポーネントインポート成功")
        
        # 簡易アプリケーション作成テスト
        app = QApplication([])
        
        # レベルメーターテスト
        meter = LevelMeterWidget("TEST")
        meter.set_level(0.5, 0.7)
        print("✅ レベルメーター作成成功")
        
        # チャンネルストリップテスト
        device_info = {'name': 'Test Device', 'max_channels': 2}
        strip = ChannelStrip("TEST CH", device_info)
        print("✅ チャンネルストリップ作成成功")
        
        app.quit()
        
    except ImportError as e:
        print(f"❌ UIコンポーネントテスト失敗: {e}")
    except Exception as e:
        print(f"❌ UIテストエラー: {e}")


def main():
    """メインテスト"""
    print("UAD Console風オーディオミキサー - テストスクリプト")
    print("=" * 60)
    
    try:
        # 依存関係チェック
        import sounddevice as sd
        import numpy as np
        import math
        print("✅ 必要ライブラリ確認完了")
        
        # デバイステスト
        device_manager = test_device_enumeration()
        
        # UIテスト
        test_ui_components()
        
        # オーディオ処理テスト（オプション）
        print(f"\nオーディオ処理テストを実行しますか? (y/N): ", end='')
        if input().lower().startswith('y'):
            test_audio_processing(device_manager)
        
        print("\n✅ 全テスト完了")
        
    except ImportError as e:
        print(f"❌ 依存関係エラー: {e}")
        print("pip install -r requirements.txt を実行してください")
        return 1
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())