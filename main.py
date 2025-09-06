import os
import sys
import re
import time
import json
import yaml
import signal
import logging
import unicodedata
import datetime
from datetime import timezone, timedelta
from typing import Any, Dict, List, Optional, Set
import html
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import schedule
import cloudscraper
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Index, desc
from sqlalchemy.orm import declarative_base, sessionmaker
from logging.handlers import RotatingFileHandler
import threading
import subprocess
import tempfile
import shutil

DETECTION_WINDOW_SEC = 3600

def load_config(config_path: str = None) -> Dict:
    config_path = config_path or os.environ.get("DEX_BOT_CONFIG", "core/config.yaml")
    defaults = {
        'api_urls': {'tokens': '', 'boosts': '', 'boosts_top': ''},
        'alerts': {
            'cooldown_sec': 900,
            'min_price_rel_change': 0.10,
            'min_vol_rel_change': 0.50,
            'min_ch24_abs_change': 5.0,
            'binning': {'price_pct': 0.02, 'vol_pct': 0.25}
        },
        'coin_blacklist': [],
        'dev_blacklist': [],
        'rugcheck': {'base_url': 'https://api.rugcheck.xyz/v1', 'api_key': '', 'good_status': 'Good'},
        'supply_check': {'bundled_supply_field': 'is_supply_bundled'},
        'fake_volume_detection': {
            'method': 'algorithm',
            'algorithm': {'min_volume_threshold': 10000.0, 'max_volume_change_percentage': 300.0},
            'pocket_universe': {'base_url': '', 'api_key': ''}
        },
        'filters': {
            'min_price_change_percentage_24h': 50.0,
            'max_price_change_percentage_24h': -50.0,
            'monitored_events': ['pumped', 'rugged', 'tier-1', 'listed_on_cex']
        },
        'database': {'url': 'sqlite:///data/coins.db'},
        'runtime': {'json_interval_seconds': 2, 'data_dir': 'data'},
        'telegram': {'bot_token': '', 'chat_id': ''},
        'auto_buy': {'enabled': True, 'allowed_events': ['pumped'], 'cooldown_sec': 120, 'per_token_cooldown_sec': 0, 'validate_mint_rpc': False}
    }
    def deep_merge(a: Dict, b: Dict) -> Dict:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    if not os.path.exists(config_path):
        logging.warning("Config file %s not found. Using defaults.", config_path)
        return defaults
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logging.error("Failed to read %s (%s). Using defaults.", config_path, e)
        raw = {}
    cfg = deep_merge(defaults, raw)
    for key in ['tokens', 'boosts', 'boosts_top']:
        val = (cfg['api_urls'].get(key) or '').strip()
        if not val:
            logging.warning(f"api_urls.{key} está vacío. Algunas funciones pueden no operar.")
        cfg['api_urls'][key] = val
    logging.info("Configuration loaded.")
    return cfg

CONFIG = load_config()

 
try:
    _log_level = str(CONFIG.get('logging', {}).get('level', os.environ.get("DEX_BOT_LOG_LEVEL", 'INFO'))).upper()
    logging.getLogger().setLevel(_log_level)
except Exception:
    pass
 

 

def load_detected_mints() -> Dict[str, str]:
    data = load_from_file(DETECTED_FILE)
    return data if isinstance(data, dict) else {}

def save_detected_mints(m: Dict[str, str]) -> None:
    save_to_file(m, DETECTED_FILE)

def mark_mint_detected(mint: str) -> None:
    if not mint:
        return
    m = load_detected_mints()
    if mint not in m:
        m[mint] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_detected_mints(m)

def is_within_detection_window(mint: str, window_sec: int = DETECTION_WINDOW_SEC) -> bool:
    m = load_detected_mints()
    ts = m.get(mint)
    if not ts:
        return False
    try:
        t0 = datetime.datetime.fromisoformat(ts)
    except Exception:
        return False
    return (datetime.datetime.now(datetime.timezone.utc) - t0) <= datetime.timedelta(seconds=window_sec)

def purge_expired_detected(window_sec: int = DETECTION_WINDOW_SEC) -> int:
    m = load_detected_mints()
    now = datetime.datetime.now(datetime.timezone.utc)
    kept = {}
    removed = 0
    for k, ts in m.items():
        try:
            t0 = datetime.datetime.fromisoformat(ts)
            if (now - t0) <= datetime.timedelta(seconds=window_sec):
                kept[k] = ts
            else:
                removed += 1
        except Exception:
            removed += 1
    save_detected_mints(kept)
    return removed

def list_recent_detected(window_sec: int = DETECTION_WINDOW_SEC) -> List[str]:
    m = load_detected_mints()
    out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for k, ts in m.items():
        try:
            t0 = datetime.datetime.fromisoformat(ts)
            if (now - t0) <= datetime.timedelta(seconds=window_sec):
                out.append(k)
        except Exception:
            pass
    return out

def setup_logging() -> str:
    default_dir = './data'
    log_dir = os.environ.get("DEX_BOT_LOG_DIR", default_dir)
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.environ.get("DEX_BOT_LOG", os.path.join(log_dir, "dexscreener_bot.log"))
    logger = logging.getLogger()
    logger.setLevel(os.environ.get("DEX_BOT_LOG_LEVEL", "INFO").upper())
    logger.handlers.clear()
    fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logging.info("Logging to %s", logfile)
    return logfile

LOG_FILE = setup_logging()

def safe_log_message(message: Any) -> str:
    return unicodedata.normalize("NFKD", str(message)).encode("ascii", "ignore").decode("ascii")

def make_retrying_session(total=3, backoff=0.5, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount('http://', adapter)
    sess.mount('https://', adapter)
    return sess

REQ = make_retrying_session()
SCRAPER = cloudscraper.create_scraper()

def save_to_file(data: Any, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)

def load_from_file(filename: str) -> Any:
    if not os.path.exists(filename):
        logging.warning(f"Archivo no encontrado: {filename}. Iniciando vacío.")
        return []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error al leer {filename}: {e}")
        return []

def _deep_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _pick(d: Dict, *paths):
    for path in paths:
        cur = d
        ok = True
        for p in path.split('.'):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return cur
    return None

def _to_float(x):
    try:
        if x in [None, ""]:
            return 0.0
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0

def _looks_like_url(s: str) -> bool:
    return bool(s and re.match(r"^https?://", s.strip(), flags=re.I))


# --- Globals (define exactly once) ---
Base = declarative_base()
 
DATA_DIR = CONFIG['runtime']['data_dir']
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_FILE = os.path.join(DATA_DIR, "events.json")
DETECTED_FILE = os.path.join(DATA_DIR, "mints.json")  # single source of truth
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "dexscreener.sqlite")
DB_URI_PATH = DB_PATH.replace("\\", "/")
ENGINE = create_engine(
    f"sqlite:///{DB_URI_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True
)
Session = sessionmaker(bind=ENGINE)


class CoinEvent(Base):
    __tablename__ = 'coin_events'
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String, index=True, nullable=False)
    name = Column(String)
    price = Column(Float)
    event_type = Column(String, index=True)
    dev_address = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now(timezone.utc), index=True)
    metadata_json = Column(Text)
    __table_args__ = (
        Index('ix_token_time', 'token_id', 'timestamp'),
        Index('ix_token_event_type_time', 'token_id', 'event_type', 'timestamp'),
    )

Base.metadata.create_all(ENGINE)
logging.info("SQLite: esquemas actualizados.")


DEX_API_URLS = {
    "tokens": CONFIG['api_urls']['tokens'],
    "boosts": CONFIG['api_urls']['boosts'],
    "boosts_top": CONFIG['api_urls']['boosts_top'],
}

TELEGRAM_CFG = CONFIG.get('telegram', {})
ALERTED_FILE = os.path.join(DATA_DIR, "alerted_tokens.json")
ALERT_LOCK = threading.RLock()

_LAST_BUY: Dict[str, float] = {}
AUTO_BUY = CONFIG.get('auto_buy', {})
AUTO_BUY_ENABLED = bool(AUTO_BUY.get('enabled', False))
AUTO_BUY_ALLOWED_EVENTS = set(AUTO_BUY.get('allowed_events', ['pumped']))
AUTO_BUY_COOLDOWN = int(AUTO_BUY.get('cooldown_sec', 120))
AUTO_BUY_PER_TOKEN_COOLDOWN = int(AUTO_BUY.get('per_token_cooldown_sec', 0))
AUTO_BUY_VALIDATE_RPC = bool(AUTO_BUY.get('validate_mint_rpc', True))
_GLOBAL_LAST_BUY_TS = 0.0
_LAST_BUY_BY_TOKEN: Dict[str, float] = {}
_DS_CACHE: Dict[str, Dict[str, Any]] = {}
_DS_TTL = 90
_RUG_CACHE: Dict[str, Dict[str, Any]] = {}
_RUG_TTL = 3600

def send_telegram_message(text: str) -> None:
    token = (TELEGRAM_CFG.get('bot_token') or '').strip()
    chat_id = (TELEGRAM_CFG.get('chat_id') or '').strip()
    if not token or not chat_id:
        return
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML', 'disable_web_page_preview': True}
    try:
        REQ.post(api_url, data=payload, timeout=10).raise_for_status()
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

def send_enriched_telegram_alert(event_type: str, name: str, price: float, ch24: float, vol24: float, contract: str, image_url: str = None, dex_url: str = None, mc_sol: float = None, mc_usd: float = None) -> None:
    token_tg = (TELEGRAM_CFG.get('bot_token') or '').strip()
    chat_id = (TELEGRAM_CFG.get('chat_id') or '').strip()
    if not token_tg or not chat_id:
        return
    if not dex_url:
        dex_url = f"https://dexscreener.com/search?q={contract}"
    title = (event_type or "other").strip().upper()
    if title not in {"PUMPED", "RUGGED", "TIER-1", "LISTED_ON_CEX", "OTHER"}:
        title = "OTHER"
    price_str = f"${price:.6f}"
    ch24_str = f"{ch24:+.2f}%"
    vol24_str = f"${vol24:,.0f}"
    mc_sol_str = f"{mc_sol:,.2f} SOL" if mc_sol is not None else ""
    mc_usd_str = f" (~${mc_usd:,.2f})" if mc_usd is not None else ""
    parts = []
    if image_url and _looks_like_url(image_url):
        parts.append(f'<a href="{image_url}">📊</a>')
    safe_name = html.escape(name or "Unknown")
    name_html = f"<b>{safe_name}</b>" if not _looks_like_url(name) else f'<a href="{name}"><b> </b></a>'
    parts.append(f"{name_html}\n<blockquote><b>🚨 {title}</b>\n\n💰 {price_str}\n🔗 {ch24_str}\n📊 {vol24_str}\n</blockquote>")
    if mc_sol_str:
        parts.append(f"🔖 MC: <b>{mc_sol_str}</b>{mc_usd_str}")
    parts.append(f"<code>{contract}</code>")
    full_text = "\n".join(parts)
    reply_markup = {"inline_keyboard": [[{"text": "🔎 Ver en Dexscreener", "url": dex_url}]]}
    api_url = f"https://api.telegram.org/bot{token_tg}/sendMessage"
    try:
        REQ.post(api_url, data={'chat_id': chat_id, 'text': full_text, 'parse_mode': 'HTML', 'disable_web_page_preview': False, 'reply_markup': json.dumps(reply_markup)}, timeout=10)
        logging.info(f"✅ Alerta enviada: {name} | {title}")
    except Exception as e:
        logging.error(f"Error enviando alerta enriquecida: {e}")

def load_alerted_tokens() -> Dict[str, Dict]:
    with ALERT_LOCK:
        data = load_from_file(ALERTED_FILE)
        return data if isinstance(data, dict) else {}

def save_alerted_token(mint: str, event_type: str, price: float, volume: float, ch24: float):
    with ALERT_LOCK:
        data = load_from_file(ALERTED_FILE)
        if not isinstance(data, dict):
            data = {}
        data[mint] = {'last_event_type': event_type, 'last_price': price, 'last_volume': volume, 'last_ch24': ch24, 'last_alert_time': datetime.datetime.now(timezone.utc).isoformat()}
        save_to_file(data, ALERTED_FILE)

def _rel_change(new: float, old: Optional[float]) -> float:
    if old is None:
        return 1.0
    if abs(old) < 1e-6: 
        return 1.0
    return abs(new - old) / abs(old)


def should_alert_again(contract: str, event_type: str, price: float, vol24: float, ch24: float) -> bool:
    tokens = load_alerted_tokens()
    last_info = tokens.get(contract)
    if not last_info:
        return True

    cooldown = int(CONFIG.get('alerts', {}).get('cooldown_sec', 900))
    last_ts_str = last_info.get('last_alert_time')
    if last_ts_str:
        try:
            last_ts = datetime.datetime.fromisoformat(last_ts_str)
            if datetime.datetime.now(datetime.timezone.utc) - last_ts < datetime.timedelta(seconds=cooldown):
                return False
        except Exception:
            pass

    price_rel_change = _rel_change(price, last_info.get('last_price'))
    vol_rel_change   = _rel_change(vol24, last_info.get('last_volume'))
    ch24_abs_change  = abs(ch24 - (last_info.get('last_ch24') or 0.0))

    min_price_rel = float(CONFIG['alerts'].get('min_price_rel_change', 0.10))
    min_vol_rel   = float(CONFIG['alerts'].get('min_vol_rel_change', 0.50))
    min_ch24_abs  = float(CONFIG['alerts'].get('min_ch24_abs_change', 5.0))

    return not (price_rel_change < min_price_rel and
                vol_rel_change   < min_vol_rel   and
                ch24_abs_change  < min_ch24_abs)


class StrategySuite:
    def __init__(self, config: Dict):
        self.config = config
        self.coin_blacklist: Set[str] = set(str(x).lower() for x in config.get('coin_blacklist', []))
        self.dev_blacklist: Set[str] = set(str(x).lower() for x in config.get('dev_blacklist', []))
        self.monitored_events: Set[str] = set(config['filters'].get('monitored_events', []))

    def is_blacklisted(self, token_id: str, token_name: str, dev_address: str) -> bool:
        return str(token_id).lower() in self.coin_blacklist or str(token_name).lower() in self.coin_blacklist or str(dev_address).lower() in self.dev_blacklist

    def is_token_good_rugcheck(self, token_id: str) -> bool:
        base = (self.config['rugcheck'].get('base_url') or 'https://api.rugcheck.xyz/v1').rstrip('/')
        api_key = (self.config['rugcheck'].get('api_key') or '').strip()
        headers = {'X-API-KEY': api_key} if api_key else {}
        if not token_id:
            return False
        now = time.time()
        c = _RUG_CACHE.get(token_id)
        if c and now - c["ts"] < _RUG_TTL:
            return c["ok"]
        try:
            url = f"{base}/tokens/{token_id}/report/summary"
            r = REQ.get(url, headers=headers, timeout=15)
            if r.status_code == 404:
                _RUG_CACHE[token_id] = {"ts": now, "ok": False}
                return False
            r.raise_for_status()
            j = r.json() or {}
            risk_level = (j.get('risk_level') or j.get('riskLevel') or '').upper()
            status = (j.get('status') or '').upper()
            score = float(j.get('score') or j.get('risk_score') or 0)
            ok = (risk_level in {'LOW', 'GOOD', 'OK'}) or (status in {'GOOD', 'OK'}) or (score >= 70)
            _RUG_CACHE[token_id] = {"ts": now, "ok": ok}
            return ok
        except Exception:
            return False

    def is_supply_bundled(self, token_data: Dict) -> bool:
        field = self.config['supply_check'].get('bundled_supply_field', 'is_supply_bundled')
        return bool(token_data.get(field, False))

    def update_blacklists_if_bundled(self, token_data: Dict) -> None:
        coin_id = token_data.get('id')
        dev_address = token_data.get('developer_address', '')
        coin_name = token_data.get('name', 'Unknown')
        if not coin_id or not dev_address:
            return
        self.config.setdefault('coin_blacklist', [])
        self.config.setdefault('dev_blacklist', [])
        if coin_id not in self.config['coin_blacklist']:
            self.config['coin_blacklist'].append(coin_id)
        if coin_name not in self.config['coin_blacklist']:
            self.config['coin_blacklist'].append(coin_name)
        if dev_address not in self.config['dev_blacklist']:
            self.config['dev_blacklist'].append(dev_address)
        self.coin_blacklist.add(str(coin_id).lower())
        self.coin_blacklist.add(str(coin_name).lower())
        self.dev_blacklist.add(str(dev_address).lower())

    def _is_volume_valid_algorithm(self, coin: Dict) -> bool:
        try:
            vol = float(coin.get('daily_volume', 0) or 0)
            change = float(coin.get('volume_change_percentage_24h', 0) or 0)
            min_vol = float(_deep_get(self.config, ['fake_volume_detection', 'algorithm', 'min_volume_threshold'], 10000.0))
            max_change = float(_deep_get(self.config, ['fake_volume_detection', 'algorithm', 'max_volume_change_percentage'], 300.0))
            if vol < min_vol:
                return False
            if abs(change) > max_change:
                return False
            return True
        except Exception:
            return False

    def _lp_locked_ok(self, item: Dict) -> bool:
        try:
            locker = self.config.get("lp_locker", {})
            if not locker.get("base_url"):
                return True
            return True
        except Exception:
            return False

    def extra_supply_guards(self, item: Dict) -> bool:
        if self.is_supply_bundled(item):
            self.update_blacklists_if_bundled(item)
            return False
        if not self._lp_locked_ok(item):
            return False
        return True

    def is_flow_good_for_scalp(self, enrich: Dict, ticket_usd: float = 300.0) -> bool:
        min_liq = float(self.config.get("scalp", {}).get("min_liquidity_usd", 25000))
        min_tx5m = int(self.config.get("scalp", {}).get("min_tx5m", 30))
        max_fdv = float(self.config.get("scalp", {}).get("max_fdv_usd", 50000000))
        max_m5_spike_pct = float(self.config.get("scalp", {}).get("max_m5_spike_pct", 100.0))
        liq = float(enrich.get("liquidity_usd") or 0)
        tx5m = int(enrich.get("tx5m") or 0)
        fdv = float(enrich.get("fdv") or 0)
        vol5m = float(enrich.get("vol5m") or 0)
        vol1h = float(enrich.get("vol1h") or 0)
        vol24 = float(enrich.get("vol24") or 0)
        if liq < min_liq:
            return False
        if tx5m < min_tx5m:
            return False
        if fdv and fdv > max_fdv:
            return False
        if vol24 > 0 and (vol1h / vol24) < 0.10:
            return False
        if vol1h > 0 and (vol5m / vol1h) > (max_m5_spike_pct / 100.0):
            return False
        slip_est = ticket_usd / max(liq, 1.0)
        max_slip = float(self.config.get("scalp", {}).get("max_slippage_est", 0.015))
        if slip_est > max_slip:
            return False
        return True

    def _is_volume_valid_pocket_universe(self, coin: Dict) -> bool:
        try:
            pu = self.config['fake_volume_detection'].get('pocket_universe', {})
            url = pu.get('base_url', '').strip()
            api_key = pu.get('api_key', '').strip()
            if not url:
                return False
            headers = {'Authorization': f"Bearer {api_key}"} if api_key else {}
            params = {'coin_id': coin.get('id')}
            response = REQ.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            result = response.json() or {}
            return not bool(result.get('is_volume_fake', False))
        except Exception:
            return False

    def is_volume_valid(self, coin: Dict) -> bool:
        method = str(self.config.get('fake_volume_detection', {}).get('method', 'algorithm')).lower()
        if method == 'pocket_universe':
            return self._is_volume_valid_pocket_universe(coin)
        return self._is_volume_valid_algorithm(coin)

    def determine_event_type(self, price_change: float, item: Dict) -> str:
        filters = self.config['filters']
        min_pump = float(filters.get('min_price_change_percentage_24h', 50))
        max_dump = float(filters.get('max_price_change_percentage_24h', -50))
        if price_change >= min_pump:
            return 'pumped'
        if price_change <= max_dump:
            return 'rugged'
        if item.get('is_tier_1', False):
            return 'tier-1'
        if item.get('is_listed_on_cex', False):
            return 'listed_on_cex'
        return 'other'

    def classify_event_simple(self, token: Dict) -> str:
        change = float(token.get("price_change_percentage_24h", 0) or 0)
        thresholds = self.config['filters']
        min_pump = float(thresholds.get('min_price_change_percentage_24h', 50))
        max_dump = float(thresholds.get('max_price_change_percentage_24h', -50))
        if change >= min_pump:
            return 'pumped'
        if change <= max_dump:
            return 'rugged'
        return 'normal'

STRATEGY = StrategySuite(CONFIG)

def fetch_api_data(url: str, use_cloudscraper: bool = True) -> Optional[Dict]:
    if not url:
        return None
    try:
        sess = SCRAPER if use_cloudscraper else REQ
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Error fetching {url}: {safe_log_message(e)}")
        return None

def update_dex_data() -> None:
    for name, url in DEX_API_URLS.items():
        if not url:
            continue
        logging.info(f"📡 Obteniendo datos de {name}")
        data = fetch_api_data(url)
        if data is None:
            continue
        tokens = data.get("tokens", data) if isinstance(data, dict) else data
        filename = os.path.join(DATA_DIR, f"dex_{name}.json")
        save_to_file(tokens, filename)

def fetch_coin_data() -> Optional[Dict]:
    url = CONFIG['api_urls'].get('tokens', '').strip()
    if not url:
        logging.warning("api_urls.tokens vacío. No se puede ejecutar fetch_coin_data().")
        return None
    try:
        response = REQ.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            data = {"tokens": data}
        logging.info("Fetched coin data successfully (SQLAlchemy pipeline).")
        return data
    except requests.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

_DS_LOCK = threading.RLock()
_RUG_LOCK = threading.RLock()

def fetch_ds_metrics_by_address(address: str) -> Optional[Dict]:
    try:
        now = time.time()
        # --- thread-safe cache read ---
        with _DS_LOCK:
            c = _DS_CACHE.get(address)
        if c and now - c["ts"] < _DS_TTL:
            return c["data"]

        r = REQ.get(f"https://api.dexscreener.com/latest/dex/tokens/{address}", timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        pairs = data.get("pairs") or []
        if not pairs:
            return None

        best = max(
            pairs,
            key=lambda p: (
                float((p.get("liquidity") or {}).get("usd") or 0),
                float(((p.get("volume") or {}).get("h24")) or 0)
            )
        )

        price = float(best.get("priceUsd") or 0.0)
        volume = best.get("volume") or {}
        txns = best.get("txns") or {}
        liq_usd = float((best.get("liquidity") or {}).get("usd") or 0.0)
        dex_id = best.get("dexId")
        pair_url = best.get("url")

        result = {
            "price": price,
            "ch24": float((best.get("priceChange") or {}).get("h24") or 0.0),
            "vol24": float(volume.get("h24") or 0.0),
            "vol1h": float(volume.get("h1") or 0.0),
            "vol5m": float(volume.get("m5") or 0.0),
            "tx24": int((txns.get("h24") or {}).get("buys", 0)) + int((txns.get("h24") or {}).get("sells", 0)),
            "tx1h": int((txns.get("h1") or {}).get("buys", 0)) + int((txns.get("h1") or {}).get("sells", 0)),
            "tx5m": int((txns.get("m5") or {}).get("buys", 0)) + int((txns.get("m5") or {}).get("sells", 0)),
            "liquidity_usd": liq_usd,
            "fdv": float(best.get("fdv") or 0.0),
            "chainId": best.get("chainId"),
            "pairAddress": best.get("pairAddress"),
            "dexId": dex_id,
            "url": pair_url,
        }

        # --- thread-safe cache write ---
        with _DS_LOCK:
            _DS_CACHE[address] = {"ts": now, "data": result}

        return result
    except Exception as e:
        logging.debug(f"Dexscreener enrich fail for {address}: {safe_log_message(e)}")
        return None



def build_dex_url(item: Dict, contract: str) -> str:
    chain = _pick(item, "chainId", "chain", "chain.id")
    pair = _pick(item, "pairAddress", "pair.address")
    if chain and pair:
        return f"https://dexscreener.com/{chain}/{pair}"
    try:
        r = REQ.get(f"https://api.dexscreener.com/latest/dex/tokens/{contract}", timeout=15)
        r.raise_for_status()
        pairs = (r.json() or {}).get("pairs") or []
        if pairs:
            best = max(
                pairs,
                key=lambda p: (
                    float(p.get("liquidity", {}).get("usd") or 0),
                    float((p.get("volume", {}) or {}).get("h24") or 0)
                )
            )
            chain = best.get("chainId")
            pair = best.get("pairAddress")
            if chain and pair:
                return f"https://dexscreener.com/{chain}/{pair}"
    except Exception:
        pass
    from urllib.parse import quote
    return f"https://dexscreener.com/search?q={quote(contract)}"

def normalize_dex_for_swap(dex_id: Optional[str], pair_url: Optional[str] = None) -> str:
    if not dex_id:
        if pair_url and "jup.ag" in pair_url.lower():
            return "jupiter"
        return "jupiter"
    d = dex_id.strip().lower()
    mapping = {
        "raydium": "raydium",
        "raydium-clmm": "raydium",
        "orca": "orca",
        "meteora": "meteora",
        "saber": "saber",
        "saros": "saros",
        "crema": "crema",
        "damm": "damm",
        "damm3": "damm",
        "lifinity": "lifinity",
        "jup.ag": "jupiter",
        "jupiter": "jupiter",
        "moonshot": "moonshot",
        "pumpswap": "pump",
        "pump": "pump",
        "heaven": "heaven",
    }
    d = d.replace(" ", "").replace("_", "").replace("-", "")
    if d in mapping:
        return mapping[d]
    if pair_url:
        u = pair_url.lower()
        if "jup.ag" in u or "jupiter" in u:
            return "jupiter"
        if "raydium" in u:
            return "raydium"
        if "orca.so" in u or "/orca/" in u:
            return "orca"
        if "meteora" in u:
            return "meteora"
        if "pump.fun" in u or "pumpswap" in u:
            return "pump"
        if "moonshot" in u:
            return "moonshot"
        if "heaven" in u:
            return "heaven"
    return "jupiter"

def parse_coin_data(raw_data: Dict) -> List[Dict]:
    coins: List[Dict] = []
    tokens_iter = raw_data.get('tokens') or [] if isinstance(raw_data, dict) else []
    for item in tokens_iter:
        try:
            coin_id = item.get('id')
            coin_name = item.get('name', 'Unknown')
            price = float(item.get('price', 0) or 0)
            price_change = float(item.get('price_change_percentage_24h', 0) or 0)
            dev_address = str(item.get('developer_address', '')).lower()
            if not coin_id:
                continue
            if STRATEGY.is_blacklisted(coin_id, coin_name, dev_address):
                continue
            event_type = STRATEGY.determine_event_type(price_change, item)
            if event_type not in STRATEGY.monitored_events:
                continue
            if not STRATEGY.is_token_good_rugcheck(coin_id):
                continue
            if not STRATEGY.extra_supply_guards(item):
                continue
            coin_for_vol = {'id': coin_id, 'daily_volume': item.get('daily_volume'), 'volume_change_percentage_24h': item.get('volume_change_percentage_24h')}
            if not STRATEGY.is_volume_valid(coin_for_vol):
                continue
            coin = {
                'token_id': coin_id,
                'name': coin_name,
                'price': price,
                'event_type': event_type,
                'dev_address': dev_address,
                'timestamp': datetime.datetime.now(timezone.utc),
                'metadata_json': json.dumps(item, default=str),
            }
            mark_mint_detected(coin_id) 
            coins.append(coin)
        except Exception as e:
            logging.error(f"Error parsing coin {item.get('id', 'Unknown')}: {e}")
    logging.info(f"✅ Parsed {len(coins)} valid coins (SQLAlchemy pipeline).")
    return coins

def _event_recent_exists(session, token_id: str, event_type: str, window_sec: int) -> bool:
    cutoff = datetime.datetime.now(timezone.utc) - timedelta(seconds=window_sec)
    exists = (
        session.query(CoinEvent)
        .filter(CoinEvent.token_id == token_id, CoinEvent.event_type == event_type, CoinEvent.timestamp >= cutoff)
        .order_by(desc(CoinEvent.timestamp))
        .first()
    )
    return exists is not None

def save_to_database(coins: List[Dict]) -> None:
    if not coins:
        return
    with Session() as session:
        try:
            cooldown = int(CONFIG.get('alerts', {}).get('cooldown_sec', 900))
            for coin in coins:
                if _event_recent_exists(session, coin['token_id'], coin['event_type'], cooldown):
                    continue
                session.add(CoinEvent(**coin))
            session.commit()
            logging.info("💾 Saved %d events to SQLite database.", len(coins))
        except Exception as e:
            session.rollback()
            logging.error("DB save error: %s", e)


def analyze_data_db() -> None:
    session = Session()
    try:
        data = session.query(CoinEvent).all()
        if not data:
            logging.info("No data to analyze.")
            return
        df = pd.DataFrame([{'event_type': e.event_type, 'price': e.price, 'timestamp': e.timestamp} for e in data])
        counts = df['event_type'].value_counts()
        logging.info(f"📊 Event Counts:\n{counts.to_string()}")
        logging.info(f"📈 Price Stats:\n{df['price'].describe()}")
    except Exception as e:
        logging.error(f"Analysis error: {e}")
    finally:
        session.close()

def job_sqlalchemy() -> None:
    logging.info("🔄 Starting SQLAlchemy pipeline (hourly)…")
    raw_data = fetch_coin_data()
    if raw_data:
        coins = parse_coin_data(raw_data)
        save_to_database(coins)
        analyze_data_db()
    logging.info("✅ SQLAlchemy pipeline completed.")

def append_event_json(event: Dict) -> None:
    events = load_from_file(DATABASE_FILE)
    if not isinstance(events, list):
        events = []
    events.append(event)
    save_to_file(events, DATABASE_FILE)

_BASE58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_RE = re.compile(rf"[{_BASE58_CHARS}]{{32,44}}")
SOL_MINT = "So11111111111111111111111111111111111111112"
STABLE_MINTS = {"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"}
SPL_TOKEN_PROGRAMS = {"TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}

def looks_like_spl_mint(m: str) -> bool:
    if not m:
        return False
    s = m.strip()
    return 32 <= len(s) <= 44 and all(c in _BASE58_CHARS for c in s)

def _best_non_sol_stable(bt: dict, qt: dict) -> str:
    cand = None
    if bt and looks_like_spl_mint(bt.get("address", "")) and bt["address"] not in {SOL_MINT, *STABLE_MINTS}:
        cand = bt["address"]
    if not cand and qt and looks_like_spl_mint(qt.get("address", "")) and qt["address"] not in {SOL_MINT, *STABLE_MINTS}:
        cand = qt["address"]
    return cand or (bt or {}).get("address") or (qt or {}).get("address") or ""

def _from_pump_fun(url: str) -> str:
    m = re.search(r"/coin/([1-9A-HJ-NP-Za-km-z]{32,44})", url)
    return m.group(1) if m else ""

def _from_moonshot(url: str) -> str:
    m = re.search(r"/coin/([1-9A-HJ-NP-Za-km-z]{32,44})", url)
    return m.group(1) if m else ""

def _from_birdeye(url: str) -> str:
    m = re.search(r"/token/([1-9A-HJ-NP-Za-km-z]{32,44})", url)
    return m.group(1) if m else ""

def _from_jup(url: str) -> str:
    m = re.search(r"(?:[?&]outputMint=)([1-9A-HJ-NP-Za-km-z]{32,44})", url)
    if m:
        return m.group(1)
    m2 = re.search(r"/swap/([1-9A-HJ-NP-Za-km-z]{32,44})", url)
    return m2.group(1) if m2 else ""

def _from_dexscreener_pair(url_or_id: str) -> str:
    m = re.search(r"dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})", url_or_id)
    pair = m.group(1) if m else url_or_id
    if not looks_like_spl_mint(pair):
        return ""
    try:
        r = REQ.get(f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair}", timeout=10)
        if r.status_code != 200:
            return ""
        j = r.json() or {}
        p = (j.get("pair") or (j.get("pairs") or [{}]))
        if isinstance(p, list):
            p = p[0] if p else {}
        return _best_non_sol_stable(p.get("baseToken") or {}, p.get("quoteToken") or {})
    except Exception:
        return ""

def resolve_mint_from_any(s: str) -> str:
    if not s:
        return ""
    raw = s.strip()
    if looks_like_spl_mint(raw):
        return raw
    low = raw.lower()
    if low.startswith("http"):
        for extractor in (_from_pump_fun, _from_moonshot, _from_birdeye, _from_jup):
            m = extractor(raw)
            if looks_like_spl_mint(m):
                return m
        if "dexscreener.com/solana/" in low:
            m = _from_dexscreener_pair(raw)
            if looks_like_spl_mint(m):
                return m
        if "raydium.io" in low:
            m = re.search(r"(?:inputCurrency|outputCurrency)=([1-9A-HJ-NP-Za-km-z]{32,44})", raw)
            if m and looks_like_spl_mint(m.group(1)):
                return m.group(1)
    cands = _BASE58_RE.findall(raw)
    if cands:
        if len(cands) == 1:
            maybe_pair = cands[0]
            via_pair = _from_dexscreener_pair(maybe_pair)
            return via_pair or maybe_pair
        for c in cands:
            if c not in {SOL_MINT, *STABLE_MINTS}:
                return c
        return cands[0]
    return ""

def is_sol_or_stable(addr: str) -> bool:
    return addr == SOL_MINT or addr in STABLE_MINTS

def pick_solana_mint(item: dict) -> str:
    chain = (item.get("chainId") or (item.get("chain") or {}).get("id") or item.get("chain") or "")
    if isinstance(chain, str) and chain.lower() not in ("", "solana"):
        return ""
    bt = (item.get("baseToken") or {})
    qt = (item.get("quoteToken") or {})
    cand = None
    if bt.get("address") and not is_sol_or_stable(bt["address"]):
        cand = bt["address"]
    elif qt.get("address") and not is_sol_or_stable(qt["address"]):
        cand = qt["address"]
    if not cand:
        cand = ((item.get("token") or {}).get("address") or (item.get("baseToken") or {}).get("address") or (item.get("quoteToken") or {}).get("address"))
    if not cand:
        raw = item.get("address") or item.get("tokenAddress")
        if raw and raw != item.get("pairAddress"):
            cand = raw
    cand = (cand or "").strip()
    if not looks_like_spl_mint(cand):
        m = re.findall(r"[1-9A-HJ-NP-Za-km-z]{32,44}", cand)
        cand = max(m, key=len) if m else ""
    if cand and cand == (item.get("pairAddress") or ""):
        return ""
    return cand

def _global_cooldown_ok() -> bool:
    global _GLOBAL_LAST_BUY_TS
    now = time.time()
    if now - _GLOBAL_LAST_BUY_TS < AUTO_BUY_COOLDOWN:
        return False
    _GLOBAL_LAST_BUY_TS = now
    return True

def _token_cooldown_ok(mint: str) -> bool:
    now = time.time()
    last = _LAST_BUY_BY_TOKEN.get(mint, 0.0)
    if now - last < AUTO_BUY_PER_TOKEN_COOLDOWN:
        return False
    _LAST_BUY_BY_TOKEN[mint] = now
    return True

def validate_spl_mint_via_rpc(mint: str) -> bool:
    try:
        if not looks_like_spl_mint(mint):
            return False
        rpc = os.getenv("RPC", "") or os.getenv("RPC_ENDPOINT", "")
        if not rpc:
            logging.debug("RPC endpoint not set; skipping on-chain mint validation.")
            return True
        from solders.pubkey import Pubkey
        from solana.rpc.api import Client
        client = Client(rpc)
        info = client.get_account_info(Pubkey.from_string(mint)).value
        if info is None:
            return False
        return str(info.owner) in SPL_TOKEN_PROGRAMS
    except Exception as e:
        logging.debug("RPC validation error: %s", e)
        return False

def _is_valid_spl_mint(mint: str) -> bool:
    if not looks_like_spl_mint(mint):
        return False
    if not AUTO_BUY_VALIDATE_RPC:
        return True
    return validate_spl_mint_via_rpc(mint)

def _reaper(proc: subprocess.Popen, tmpdir: str) -> None:
    try:
        proc.wait()
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            logging.warning("No pude borrar %s: %s", tmpdir, e)

def _launch_buy_subprocess(mint: str, scalp_ok: bool = False, dex_id: Optional[str] = None) -> None:
    try:
        mint_norm = resolve_mint_from_any(mint)
        if not mint_norm:
            logging.warning("Mint vacío/inválido tras normalización: %r", mint); return

        now = time.time()
        if now - _LAST_BUY.get(mint_norm, 0.0) < 10.0:
            logging.info("⏱️ Evito buy duplicado para %s", mint_norm); return
        _LAST_BUY[mint_norm] = now

        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Ejecutar en el repo, sin tmpdir ni copias
        cmd = [sys.executable, os.path.join(base_dir, "buy.py"), "--mint", mint_norm]
        if scalp_ok:
            cmd += ["--sol", "0.006"]

        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")
        # Asegurar rutas/vars críticas
        env.setdefault("STATE_DB", os.path.join(base_dir, "data", "state.db"))

        # Log a archivo para ver errores del hijo
        log_path = os.path.join(base_dir, "data", f"buy-{int(now)}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        lf = open(log_path, "w")

        proc = subprocess.Popen(cmd, cwd=base_dir, env=env,
                                stdout=lf, stderr=lf,
                                close_fds=True, start_new_session=True)
        logging.info("▶️ buy.py lanzado | mint=%s | log=%s | pid=%s", mint_norm, log_path, proc.pid)
        # No borres nada; ya no usamos tmpdir
    except Exception as e:
        logging.error("No se pudo lanzar buy: %s", e)

def maybe_auto_buy(event_type: str, mint: str, name: str, scalp_ok: bool = False, dex_id: str = None):
    if not AUTO_BUY_ENABLED:
        return
    if event_type not in AUTO_BUY_ALLOWED_EVENTS:
        return
    mint = resolve_mint_from_any(mint)
    if not _is_valid_spl_mint(mint):
        logging.info("⛔ Auto-buy abortado: mint inválido %s", mint)
        return
    if not _global_cooldown_ok() or not _token_cooldown_ok(mint):
        logging.info("⏱️ Auto-buy saltado por cooldown (mint=%s)", mint)
        return
    if is_sol_or_stable(mint):
        logging.info("⛔ Auto-buy abortado: es SOL/stable (%s)", mint)
        return
    logging.info("🛒 AUTO-BUY disparado para %s (%s) | DEX: %s | scalp=%s", name, mint, dex_id, scalp_ok)
    _launch_buy_subprocess(mint, scalp_ok=scalp_ok, dex_id=dex_id)

def process_tokens() -> int:
    tokens_file = os.path.join(DATA_DIR, "dex_tokens.json")
    dex_tokens_raw = load_from_file(tokens_file)
    processed = 0
    scalp_cfg = CONFIG.get("scalp", {})
    ticket_usd = float(scalp_cfg.get("ticket_usd", 300.0))
    scalp_tx5m_min = int(scalp_cfg.get("min_tx5m", 30))
    scalp_vol5m_min = float(scalp_cfg.get("min_vol5m_usd", 2000.0))
    scalp_liq_min = float(scalp_cfg.get("min_liquidity_usd", 25000.0))
    for item in dex_tokens_raw:
        if not isinstance(item, dict):
            continue
        raw_contract = pick_solana_mint(item)
        contract = resolve_mint_from_any(raw_contract)
        if not contract:
            continue
        mark_mint_detected(contract)  # <- NUEVO

        name = _pick(item, "name", "header", "baseToken.name", "token.name") or "Unknown"
        price = _to_float(_pick(item, "price", "priceUsd", "priceUSD"))
        ch24 = _to_float(_pick(item, "price_change_percentage_24h", "priceChange24h", "priceChange.h24"))
        vol24 = _to_float(_pick(item, "daily_volume", "volume24h", "volume.h24"))
        vchg24 = _to_float(_pick(item, "volume_change_percentage_24h", "volumeChange24h", "volumeChange.h24"))
        dev = _pick(item, "developer_address") or "N/A"
        if STRATEGY.is_blacklisted(contract, name, dev):
            continue
        if not STRATEGY.is_token_good_rugcheck(contract):
            continue
        if not STRATEGY.extra_supply_guards(item):
            continue
        enrich = None
        if (price == 0.0 and vol24 == 0.0) or ch24 == 0.0:
            enrich = fetch_ds_metrics_by_address(contract)
            if enrich:
                price = enrich.get("price", price)
                ch24 = enrich.get("ch24", ch24)
                vol24 = enrich.get("vol24", vol24)
                item.setdefault("chainId", enrich.get("chainId"))
                item.setdefault("pairAddress", enrich.get("pairAddress"))
        else:
            enrich = fetch_ds_metrics_by_address(contract)
        dex_id = (enrich or {}).get("dexId")
        liq_usd = float((enrich or {}).get("liquidity_usd", 0.0))
        tx5m = int((enrich or {}).get("tx5m", 0))
        vol5m = float((enrich or {}).get("vol5m", 0.0))
        vol1h = float((enrich or {}).get("vol1h", 0.0))
        vol24_e = float((enrich or {}).get("vol24", vol24))
        token = {"id": contract, "name": name, "price": price, "price_change_percentage_24h": ch24, "daily_volume": vol24, "volume_change_percentage_24h": vchg24, "developer_address": dev}
        if not STRATEGY.is_volume_valid(token):
            continue
        scalp_signal = False
        if liq_usd >= scalp_liq_min and tx5m >= scalp_tx5m_min and vol5m >= scalp_vol5m_min:
            if vol24_e > 0 and (vol1h / vol24_e) >= 0.10:
                scalp_signal = True
        event_type = STRATEGY.classify_event_simple(token)
        if scalp_signal:
            event_type = "pumped"
        if event_type == "normal":
            continue
        if event_type not in STRATEGY.monitored_events:
            continue
        scalp_ok = bool(enrich and STRATEGY.is_flow_good_for_scalp(enrich, ticket_usd=ticket_usd))

        event = {"token": token, "event_type": event_type, "timestamp": datetime.datetime.now(timezone.utc).isoformat()}
        append_event_json(event)
        logging.info(f"✅ {event_type.upper()}: {name} (${price:.6f}) | 24h {ch24:+.2f}% | Vol24 ${vol24:,.0f} | liq ${liq_usd:,.0f} | tx5m {tx5m} | vol5m ${vol5m:,.0f} | h1/24 {(vol1h/vol24_e*100 if vol24_e>0 else 0):.1f}% | DEX: {dex_id}")
        try:
            image_url = _pick(item, "image_uri", "image", "imageUrl", "baseToken.logoURI", "token.logoURI", "info.image") or None
            dex_url = build_dex_url(item, contract)
            if is_within_detection_window(contract) and should_alert_again(contract, event_type, price, vol24, ch24):
                send_enriched_telegram_alert(
                    event_type=event_type, name=name, price=price, ch24=ch24, vol24=vol24,
                    contract=contract, image_url=image_url, dex_url=dex_url
                )
                maybe_auto_buy(event_type, contract, name, scalp_ok=scalp_ok, dex_id=dex_id)
                save_alerted_token(contract, event_type, price, vol24, ch24)

        except Exception as e:
            logging.error(f"Error al enviar alerta en process_tokens: {e}")
        processed += 1
    logging.info(f"📊 Procesados {processed} eventos relevantes (JSON pipeline).")
    return processed

def job_json_pipeline() -> None:
    logging.info("🔄 Starting JSON pipelin…")
    try:
      
        update_dex_data()
        process_tokens()
        stats: Dict[str, int] = {"total": 0}
        events = load_from_file(DATABASE_FILE)
        if isinstance(events, list):
            stats["total"] = len(events)
            for e in events:
                etype = (e or {}).get("event_type")
                if etype:
                    stats[etype] = stats.get(etype, 0) + 1
        logging.info(f"📈 Stats: {stats}")
    except Exception as e:
        logging.error(f"Error in JSON pipeline: {e}")

def signal_handler(signum, frame):
    logging.info("🛑 Señal recibida. Cerrando bot de forma segura…")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main() -> None:
    logging.info("🚀 Dexscreener Monitoring Bot Iniciado (Solo Dex, sin Pump)")
    schedule.every(1).hours.do(job_sqlalchemy)
    logging.info("📅 Scheduler: Pipeline SQLAlchemy cada 1 hora.")
    json_interval = int(CONFIG['runtime'].get('json_interval_seconds', 180))
    last_run = 0.0
    job_sqlalchemy()
    job_json_pipeline()
    last_run = time.time() 
    while True:
        try:
            schedule.run_pending()
            now = time.time()
            if now - last_run >= json_interval:
                job_json_pipeline()
                last_run = now
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
