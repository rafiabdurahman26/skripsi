import json

PHOTO_DIR = 'captures/photos'
VIDEO_DIR = 'captures/videos'
TRIM_STEP = 3
TRIM_MAX = 30
DEADZONE = 0.15
SPEED_MODES = (30, 50, 70, 100)
HOLD_DELAY = 0.8
BATTERY_WARN = 20
BATTERY_CRITICAL = 10
CONFIG_FILE = 'config.json'


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            d = json.load(f)
        return d.get('trim_lr', 0), d.get('speed_idx', 3)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, 3


def save_config(trim_lr, speed_idx):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'trim_lr': trim_lr, 'speed_idx': speed_idx}, f)
    except OSError:
        pass
