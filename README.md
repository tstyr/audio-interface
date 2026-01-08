# UAD Pro Console

Professional Audio Interface Manager for Windows

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **5-Band Parametric EQ** - Graphical curve editing with mouse drag
- **Clip Indicators** - Red peak LEDs with hold (click to reset)
- **Large Stereo Meters** - 32-segment L/R meters with dB scale
- **VB-CABLE Routing** - Send processed audio to Discord/OBS
- **Dual Monitor Output** - Headphones + Speakers simultaneously
- **Recording** - Save processed audio to WAV files
- **Session Management** - Save/Load mixer settings (JSON)
- **Noise Gate** - -90dB threshold eliminates silent meter movement

## Installation

```bash
# Clone repository
git clone https://github.com/tstyr/audio-interface.git
cd audio-interface

# Install dependencies
pip install -r requirements.txt

# Run
python uad_console.py
```

## Requirements

- Python 3.9+
- Windows 10/11
- Audio interface or microphone

## Discord Integration (VB-CABLE)

To use processed audio in Discord:

1. Install [VB-CABLE](https://vb-audio.com/Cable/) (free virtual audio cable)
2. In UAD Pro Console, set channel output to **"VB-CABLE Input"**
3. In Discord Settings → Voice & Video → Input Device → **"VB-CABLE Output"**
4. Your EQ'd and processed voice will now be sent to Discord!

## Building Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build single .exe file
pyinstaller --onefile --windowed --name "UAD_Pro_Console" --icon=icon.ico uad_console.py
```

The executable will be in the `dist/` folder.

## Controls

| Control | Function |
|---------|----------|
| 48V | Phantom power indicator |
| PAD | -20dB input attenuation |
| Φ | Phase invert |
| LC | Low cut filter (80Hz HPF) |
| GAIN | Input gain (0-4x) |
| EQ | Open 5-band parametric EQ window |
| FADER | Channel volume |
| PAN | Stereo position |
| S | Solo |
| M | Mute |
| REC | Record to WAV file |

## EQ Window

- **Drag points** to adjust frequency and gain
- **Mouse wheel** on points to adjust Q value
- **Click RESET** to flatten all bands

## License

MIT License - Free for personal and commercial use.

## Author

Created with haka for audio professionals and streamers.
