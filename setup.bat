@echo off
echo UAD Console風オーディオミキサー - セットアップ
echo ================================================

echo.
echo 1. Python環境確認中...
python --version
if %errorlevel% neq 0 (
    echo エラー: Pythonがインストールされていません
    echo Python 3.8以上をインストールしてください
    pause
    exit /b 1
)

echo.
echo 2. 依存関係インストール中...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo エラー: 依存関係のインストールに失敗しました
    pause
    exit /b 1
)

echo.
echo 3. テスト実行中...
python test_audio.py
if %errorlevel% neq 0 (
    echo 警告: テストでエラーが発生しました
)

echo.
echo ✅ セットアップ完了!
echo.
echo 使用方法:
echo   アプリケーション実行: python main.py
echo   実行ファイル作成:     python build.py
echo.
pause