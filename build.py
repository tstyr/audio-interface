"""
PyInstallerを使用した実行ファイル作成スクリプト
"""
import os
import sys
import subprocess
from pathlib import Path


def create_spec_file():
    """PyInstaller用specファイル作成"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sounddevice',
        'numpy',
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AudioMixer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
    
    with open('AudioMixer.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("AudioMixer.spec ファイルを作成しました")


def build_executable():
    """実行ファイルをビルド"""
    print("実行ファイルをビルド中...")
    
    try:
        # PyInstallerでビルド
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            'AudioMixer.spec'
        ], check=True, capture_output=True, text=True)
        
        print("ビルド完了!")
        print(f"実行ファイル: {Path('dist/AudioMixer.exe').absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"ビルドエラー: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    return True


def main():
    """メイン処理"""
    print("UAD Console風オーディオミキサー - ビルドスクリプト")
    print("=" * 50)
    
    # 依存関係チェック
    try:
        import PyInstaller
        print(f"PyInstaller バージョン: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstallerがインストールされていません")
        print("pip install pyinstaller を実行してください")
        return 1
    
    # specファイル作成
    create_spec_file()
    
    # ビルド実行
    if build_executable():
        print("\n✅ ビルド成功!")
        print("dist/AudioMixer.exe を実行してください")
        return 0
    else:
        print("\n❌ ビルド失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())