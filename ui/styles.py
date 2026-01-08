"""
UAD Console風のQSSスタイル定義
ダークテーマ + オレンジ/ゴールドアクセント
"""

# カラーパレット
COLORS = {
    'background': '#2b2b2b',
    'surface': '#3a3a3a',
    'surface_light': '#4a4a4a',
    'accent': '#ff8c00',  # オレンジ
    'accent_hover': '#ffa500',
    'text': '#ffffff',
    'text_secondary': '#cccccc',
    'border': '#555555',
    'meter_bg': '#1a1a1a',
    'meter_green': '#00ff00',
    'meter_yellow': '#ffff00',
    'meter_red': '#ff0000',
}

MAIN_WINDOW_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Segoe UI', Arial, sans-serif;
}}
"""

CHANNEL_STRIP_STYLE = f"""
QFrame.ChannelStrip {{
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                               stop: 0 {COLORS['surface']},
                               stop: 0.5 {COLORS['surface_light']},
                               stop: 1 {COLORS['surface']});
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin: 2px;
    padding: 8px;
}}

QLabel.ChannelLabel {{
    color: {COLORS['text']};
    font-weight: bold;
    font-size: 11px;
    padding: 4px;
    background-color: {COLORS['background']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}
"""

BUTTON_STYLE = f"""
QPushButton {{
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                               stop: 0 {COLORS['surface_light']},
                               stop: 1 {COLORS['surface']});
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    color: {COLORS['text']};
    font-weight: bold;
    padding: 6px 12px;
    min-height: 20px;
}}

QPushButton:hover {{
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                               stop: 0 {COLORS['accent_hover']},
                               stop: 1 {COLORS['accent']});
}}

QPushButton:pressed {{
    background: {COLORS['accent']};
}}

QPushButton.mute {{
    background-color: #ff4444;
}}

QPushButton.solo {{
    background-color: #ffaa00;
}}

QPushButton.mute:checked {{
    background-color: #cc0000;
    color: white;
}}

QPushButton.solo:checked {{
    background-color: #ff8800;
    color: white;
}}
"""

SLIDER_STYLE = f"""
QSlider::groove:vertical {{
    background: {COLORS['meter_bg']};
    width: 20px;
    border-radius: 10px;
    border: 1px solid {COLORS['border']};
}}

QSlider::handle:vertical {{
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                               stop: 0 {COLORS['accent']},
                               stop: 1 #cc6600);
    border: 2px solid {COLORS['border']};
    height: 20px;
    margin: 0 -2px;
    border-radius: 12px;
}}

QSlider::handle:vertical:hover {{
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                               stop: 0 {COLORS['accent_hover']},
                               stop: 1 {COLORS['accent']});
}}

QSlider::sub-page:vertical {{
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                               stop: 0 {COLORS['accent']},
                               stop: 1 #cc6600);
    border-radius: 10px;
}}
"""

LEVEL_METER_STYLE = f"""
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 2px;
    background-color: {COLORS['meter_bg']};
    text-align: center;
}}

QProgressBar.vertical {{
    width: 12px;
}}

QProgressBar::chunk {{
    background-color: {COLORS['meter_green']};
    border-radius: 1px;
}}
"""

SCROLL_AREA_STYLE = f"""
QScrollArea {{
    border: none;
    background-color: {COLORS['background']};
}}

QScrollBar:horizontal {{
    border: none;
    background: {COLORS['surface']};
    height: 15px;
    margin: 0px 20px 0 20px;
    border-radius: 7px;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['accent']};
    min-width: 20px;
    border-radius: 7px;
}}

QScrollBar::add-line:horizontal {{
    border: none;
    background: {COLORS['surface']};
    width: 20px;
    border-radius: 7px;
    subcontrol-position: right;
    subcontrol-origin: margin;
}}

QScrollBar::sub-line:horizontal {{
    border: none;
    background: {COLORS['surface']};
    width: 20px;
    border-radius: 7px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}}
"""

# 全体のスタイルシート
COMPLETE_STYLESHEET = f"""
{MAIN_WINDOW_STYLE}
{CHANNEL_STRIP_STYLE}
{BUTTON_STYLE}
{SLIDER_STYLE}
{LEVEL_METER_STYLE}
{SCROLL_AREA_STYLE}
"""