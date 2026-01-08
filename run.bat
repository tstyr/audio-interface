@echo off
echo UAD Console風オーディオミキサー 起動中...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo エラーが発生しました。
    echo 依存関係を確認してください: pip install -r requirements.txt
    pause
)