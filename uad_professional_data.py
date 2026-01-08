"""
UAD Console Professional - データ構造定義
セッション管理、チャンネル設定、ルーティング情報を管理
"""
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class PluginType(Enum):
    """プラグインタイプ"""
    UNISON = "unison"
    INSERT = "insert"
    SEND = "send"


class ClockSource(Enum):
    """クロックソース"""
    INTERNAL = "Internal"
    EXTERNAL = "External"
    WORD_CLOCK = "Word Clock"
    SPDIF = "S/PDIF"
    ADAT = "ADAT"


@dataclass
class PluginSlot:
    """プラグインスロット情報"""
    slot_id: str
    plugin_name: str = ""
    plugin_type: PluginType = PluginType.INSERT
    enabled: bool = False
    bypass: bool = False
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ChannelSettings:
    """チャンネル設定"""
    channel_id: str
    channel_name: str
    device_index: int = -1
    device_name: str = ""
    
    # INPUT セクション
    phantom_48v: bool = False
    pad_enabled: bool = False
    phase_invert: bool = False
    low_cut: bool = False
    auto_gain: bool = False
    
    # UNISON プリアンプ
    unison_plugin: PluginSlot = None
    
    # INSERT エフェクト（5スロット）
    insert_slots: List[PluginSlot] = None
    
    # UAD REC/MON モード
    uad_rec_mode: bool = True  # True=REC, False=MON
    
    # SENDS/AUX (4系統)
    aux_sends: Dict[str, float] = None  # aux1-4: 0.0-1.0
    
    # PAN/FADER
    pan: float = 0.5  # 0.0(L) - 1.0(R)
    fader: float = 0.75  # 0.0 - 1.0 (-inf to +12dB)
    
    # MUTE/SOLO
    muted: bool = False
    solo: bool = False
    
    # CUE BUS 設定（4系統の独立ミックス）
    cue_sends: Dict[str, Dict[str, float]] = None  # cue1-4: {pan, volume}
    
    def __post_init__(self):
        if self.unison_plugin is None:
            self.unison_plugin = PluginSlot(f"{self.channel_id}_unison", plugin_type=PluginType.UNISON)
        
        if self.insert_slots is None:
            self.insert_slots = [
                PluginSlot(f"{self.channel_id}_insert_{i}", plugin_type=PluginType.INSERT)
                for i in range(5)
            ]
        
        if self.aux_sends is None:
            self.aux_sends = {f"aux{i}": 0.0 for i in range(1, 5)}
        
        if self.cue_sends is None:
            self.cue_sends = {
                f"cue{i}": {"pan": 0.5, "volume": 0.0}
                for i in range(1, 5)
            }


@dataclass
class SystemSettings:
    """システム設定"""
    sample_rate: int = 44100
    buffer_size: int = 256
    clock_source: ClockSource = ClockSource.INTERNAL
    dsp_usage: float = 0.0  # 0.0-1.0
    
    # Control Room設定
    alt_monitor_pair: int = 1  # 1-3
    dim_enabled: bool = False
    mono_enabled: bool = False
    control_room_muted: bool = False
    talkback_enabled: bool = False
    
    # Virtual I/O設定
    loopback_enabled: bool = False
    system_audio_channel: str = ""


@dataclass
class SessionData:
    """セッションデータ（保存・復元用）"""
    session_name: str
    created_at: str
    modified_at: str
    
    # システム設定
    system: SystemSettings
    
    # チャンネル設定
    channels: List[ChannelSettings]
    
    # グローバル設定
    master_fader: float = 0.85
    master_muted: bool = False
    
    def to_json(self) -> str:
        """JSON形式で出力"""
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        data = convert_enums(asdict(self))
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionData':
        """JSONから復元"""
        data = json.loads(json_str)
        
        # システム設定を復元
        system_data = data['system']
        system_data['clock_source'] = ClockSource(system_data['clock_source'])
        system = SystemSettings(**system_data)
        
        # チャンネル設定を復元
        channels = []
        for ch_data in data['channels']:
            # UNISON プラグインを復元
            if ch_data['unison_plugin']:
                unison_data = ch_data['unison_plugin']
                unison_data['plugin_type'] = PluginType(unison_data['plugin_type'])
                ch_data['unison_plugin'] = PluginSlot(**unison_data)
            
            # INSERT スロットを復元
            if ch_data['insert_slots']:
                insert_slots = []
                for slot_data in ch_data['insert_slots']:
                    slot_data['plugin_type'] = PluginType(slot_data['plugin_type'])
                    insert_slots.append(PluginSlot(**slot_data))
                ch_data['insert_slots'] = insert_slots
            
            channels.append(ChannelSettings(**ch_data))
        
        return cls(
            session_name=data['session_name'],
            created_at=data['created_at'],
            modified_at=data['modified_at'],
            system=system,
            channels=channels,
            master_fader=data.get('master_fader', 0.85),
            master_muted=data.get('master_muted', False)
        )


class PluginDatabase:
    """プラグインデータベース"""
    
    # UNISON プリアンプ
    UNISON_PREAMPS = [
        "API Vision Channel Strip",
        "Neve 1073 Preamp & EQ",
        "SSL E Channel Strip",
        "Manley VOXBOX Channel Strip",
        "Avalon VT-737sp Channel Strip",
        "UA 610-B Tube Preamp & EQ",
        "Helios Type 69 Preamp & EQ"
    ]
    
    # INSERT エフェクト
    INSERT_EFFECTS = {
        "EQ": [
            "API Vision Channel Strip",
            "Neve 1073 Preamp & EQ", 
            "SSL E Channel Strip",
            "Pultec Pro EQ",
            "Cambridge EQ",
            "Harrison 32C EQ"
        ],
        "Compressor": [
            "1176LN Rev E",
            "LA-2A Classic Leveler",
            "Fairchild 670 Legacy",
            "SSL G Bus Compressor",
            "dbx 160 Compressor/Limiter",
            "Empirical Labs Distressor"
        ],
        "Reverb": [
            "EMT 140 Classic Plate Reverb",
            "Capitol Chambers",
            "Pure Plate Reverb",
            "Galaxy Tape Echo",
            "EP-34 Tape Echo"
        ],
        "Delay": [
            "Galaxy Tape Echo",
            "EP-34 Tape Echo",
            "Brigade Chorus Pedal",
            "KORG SDD-3000 Digital Delay"
        ],
        "Saturation": [
            "Studer A800 Multichannel Tape Recorder",
            "Oxide Tape Recorder",
            "Ampex ATR-102 Mastering Tape Recorder",
            "Precision Channel Strip"
        ]
    }
    
    @classmethod
    def get_all_plugins(cls) -> List[str]:
        """全プラグインリストを取得"""
        all_plugins = cls.UNISON_PREAMPS.copy()
        for category in cls.INSERT_EFFECTS.values():
            all_plugins.extend(category)
        return sorted(list(set(all_plugins)))
    
    @classmethod
    def get_plugins_by_category(cls, category: str) -> List[str]:
        """カテゴリ別プラグインリストを取得"""
        if category == "UNISON":
            return cls.UNISON_PREAMPS
        return cls.INSERT_EFFECTS.get(category, [])


class SessionManager:
    """セッション管理クラス"""
    
    def __init__(self):
        self.current_session: Optional[SessionData] = None
        self.session_file_path: str = ""
        self.auto_save_enabled: bool = True
    
    def new_session(self, session_name: str = "Untitled Session") -> SessionData:
        """新規セッション作成"""
        from datetime import datetime
        
        now = datetime.now().isoformat()
        
        self.current_session = SessionData(
            session_name=session_name,
            created_at=now,
            modified_at=now,
            system=SystemSettings(),
            channels=[],
            master_fader=0.85,
            master_muted=False
        )
        
        return self.current_session
    
    def save_session(self, file_path: str = None) -> bool:
        """セッション保存"""
        if not self.current_session:
            return False
        
        if file_path:
            self.session_file_path = file_path
        elif not self.session_file_path:
            return False
        
        try:
            from datetime import datetime
            self.current_session.modified_at = datetime.now().isoformat()
            
            with open(self.session_file_path, 'w', encoding='utf-8') as f:
                f.write(self.current_session.to_json())
            
            return True
        except Exception as e:
            print(f"Session save error: {e}")
            return False
    
    def load_session(self, file_path: str) -> bool:
        """セッション読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = f.read()
            
            self.current_session = SessionData.from_json(json_data)
            self.session_file_path = file_path
            
            return True
        except Exception as e:
            print(f"Session load error: {e}")
            return False
    
    def add_channel(self, device_info: Dict[str, Any]) -> ChannelSettings:
        """チャンネル追加"""
        if not self.current_session:
            self.new_session()
        
        channel_id = f"ch_{len(self.current_session.channels) + 1}"
        channel_name = f"CH {len(self.current_session.channels) + 1}"
        
        channel = ChannelSettings(
            channel_id=channel_id,
            channel_name=channel_name,
            device_index=device_info.get('index', -1),
            device_name=device_info.get('name', 'Unknown Device')
        )
        
        self.current_session.channels.append(channel)
        
        if self.auto_save_enabled and self.session_file_path:
            self.save_session()
        
        return channel
    
    def remove_channel(self, channel_id: str) -> bool:
        """チャンネル削除"""
        if not self.current_session:
            return False
        
        self.current_session.channels = [
            ch for ch in self.current_session.channels 
            if ch.channel_id != channel_id
        ]
        
        if self.auto_save_enabled and self.session_file_path:
            self.save_session()
        
        return True
    
    def update_channel(self, channel_id: str, **kwargs) -> bool:
        """チャンネル設定更新"""
        if not self.current_session:
            return False
        
        for channel in self.current_session.channels:
            if channel.channel_id == channel_id:
                for key, value in kwargs.items():
                    if hasattr(channel, key):
                        setattr(channel, key, value)
                
                if self.auto_save_enabled and self.session_file_path:
                    self.save_session()
                
                return True
        
        return False


if __name__ == "__main__":
    # テスト実行
    manager = SessionManager()
    
    # 新規セッション作成
    session = manager.new_session("Test Session")
    print("Created new session:", session.session_name)
    
    # チャンネル追加
    device_info = {"index": 0, "name": "Test Microphone", "type": "input"}
    channel = manager.add_channel(device_info)
    print(f"Added channel: {channel.channel_name}")
    
    # プラグイン設定
    channel.unison_plugin.plugin_name = "Neve 1073 Preamp & EQ"
    channel.unison_plugin.enabled = True
    
    channel.insert_slots[0].plugin_name = "1176LN Rev E"
    channel.insert_slots[0].enabled = True
    
    # セッション保存テスト
    manager.session_file_path = "test_session.json"
    if manager.save_session():
        print("Session saved successfully")
        
        # JSON内容を表示
        print("\nSession JSON:")
        print(session.to_json()[:500] + "...")
    
    # プラグインデータベーステスト
    print(f"\nAvailable UNISON preamps: {len(PluginDatabase.UNISON_PREAMPS)}")
    print(f"Available EQ plugins: {len(PluginDatabase.get_plugins_by_category('EQ'))}")
    print(f"Total plugins: {len(PluginDatabase.get_all_plugins())}")