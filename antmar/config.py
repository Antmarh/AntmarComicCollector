import os, configparser

_CFG_PATH = os.path.join(os.getenv("APPDATA", "."), "AntmarComicCollector", "config.ini")
_cfg = configparser.ConfigParser()
_loaded = False

def _ensure_loaded():
    global _loaded
    if _loaded: return
    os.makedirs(os.path.dirname(_CFG_PATH), exist_ok=True)
    if os.path.exists(_CFG_PATH):
        _cfg.read(_CFG_PATH, encoding="utf-8")
    _loaded = True

def get(key, section="API_KEYS", default=""):
    _ensure_loaded()
    return _cfg.get(section, key, fallback=default)

def set(key, value, section="API_KEYS"):
    _ensure_loaded()
    if section not in _cfg: _cfg[section] = {}
    _cfg[section][key] = value
    with open(_CFG_PATH, "w", encoding="utf-8") as f:
        _cfg.write(f)

def path():
    return _CFG_PATH
