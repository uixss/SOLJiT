#!/usr/bin/env python3
from __future__ import annotations
import argparse
import io
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import requests
from dotenv import load_dotenv
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer

# ==========================
# Tipos
# ==========================
Status = Literal["ok", "pending", "fail"]

class SellResult(TypedDict, total=False):
    ok: bool
    status: Status
    sig: Optional[str]
    bundle_id: Optional[str]
    error: Optional[str]

class BuyResult(TypedDict, total=False):
    ok: bool
    status: Status
    sig: Optional[str]
    bundle_id: Optional[str]
    error: Optional[str]

# ==========================
# Config & Globals
# ==========================
load_dotenv()

BASE58 = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
_SIG_RE = re.compile(r"(?:Firma|Signature|Compra\s+exitosa|Venta\s+exitosa|Tx|Transaction|Sig)\s*[:=]\s*([1-9A-HJ-NP-Za-km-z]{70,100})", re.IGNORECASE)

DEFAULT_RPC = "https://api.mainnet-beta.solana.com"
RPC = os.getenv("RPC") or os.getenv("RPC_HTTPS") or os.getenv("RPC_ENDPOINT") or os.getenv("HELIUS_RPC") or DEFAULT_RPC
JUP_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"

# ---- Estado en archivos (SIN registro/eventos) ----
STATE_DIR = os.getenv("STATE_DIR", "state")
MINTS_PATH = os.path.join(STATE_DIR, "mints.json")

WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "25"))
TRADING_HOURS = os.getenv("TRADING_HOURS", "")
DEFAULT_SOL = float(os.getenv("DEFAULT_SOL", "0.05"))
DEFAULT_SLIPPAGE = int(os.getenv("DEFAULT_SLIPPAGE", "50"))
DEFAULT_MODE = os.getenv("MODE", "speed")
SELL_PLAN: List[Tuple[int, int]] = json.loads(os.getenv("SELL_PLAN", "[[25,15],[45,25],[105,100]]"))

LAMPORTS_PER_SOL = 1_000_000_000
FEE_CUSHION_SOL = float(os.getenv("FEE_CUSHION_SOL", "0"))
MAX_FRACTION_OF_BALANCE = max(0.0, min(1.0, float(os.getenv("MAX_FRACTION_OF_BALANCE", "1.0"))))
MIN_BUY_SOL = float(os.getenv("MIN_BUY_SOL", "0.0"))
ALLOW_DOWNSIZE = os.getenv("ALLOW_DOWNSIZE", "1").lower() in ("1", "true", "t", "yes", "y")
UNIT_PRICE = os.getenv("UNIT_PRICE")
UNIT_BUDGET = os.getenv("UNIT_BUDGET")
USE_JITO = os.getenv("USE_JITO", "0").lower() in ("1", "true", "t", "yes", "y")
TIP_SOL = os.getenv("TIP_SOL")
REPROCESS_ON_ALERT = os.getenv("REPROCESS_ON_ALERT", "0").lower() in ("1", "true", "t", "yes", "y")
TERMINAL_STATES = {"done", "expired", "skipped_out_of_hours", "skipped_insufficient_balance"}
SELL_REQUIRE_FEE_OK = os.getenv("SELL_REQUIRE_FEE_OK", "0").lower() in ("1", "true", "t", "yes", "y")
SELL_MIN_EXTRA_CUSHION_SOL = float(os.getenv("SELL_MIN_EXTRA_CUSHION_SOL", "0.0"))

_LOG_MAP = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING, "warning": logging.WARNING, "error": logging.ERROR}
log = logging.getLogger("runner")

# ==========================
# Logging
# ==========================

def _setup_logging():
    level = _LOG_MAP.get(os.getenv("LOG_LEVEL", "info").lower(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

class EnvError(RuntimeError):
    ...

# ==========================
# Helpers
# ==========================

def _looks_like_sig(s: str) -> bool:
    return isinstance(s, str) and 70 <= len(s) <= 100 and all(c in BASE58 for c in s)

def _normalize_core_result(res) -> dict:
    if isinstance(res, str) and _looks_like_sig(res):
        return {"ok": True, "status": "ok", "sig": res}
    if not isinstance(res, dict):
        return {"ok": False, "status": "fail", "error": "unrecognized_core_response"}
    raw_sig = (res.get("sig") or res.get("signature") or res.get("txid") or res.get("tx") or res.get("hash"))
    raw_bundle = res.get("bundle_id") or res.get("bundleId")
    sig = str(raw_sig) if raw_sig is not None else None
    bundle = str(raw_bundle) if raw_bundle is not None else None
    ok_flag = res.get("ok")
    status = res.get("status")
    pending = res.get("pending")
    if ok_flag is None and status == "ok":
        ok_flag = True
    if not ok_flag and _looks_like_sig(sig or ""):
        ok_flag = True
        status = status or "ok"
    if status is None:
        status = "ok" if ok_flag else ("pending" if pending else "fail")
    out = {"ok": bool(ok_flag), "status": status, "sig": sig, "bundle_id": bundle}
    if res.get("error") is not None:
        out["error"] = str(res["error"])
    return out

# ==========================
# Env sync / wallets
# ==========================

def _sync_env_for_modules() -> None:
    priv = os.getenv("PRIV_KEY") or os.getenv("PRIVATE_KEY") or os.getenv("BASE58_PRIVKEY")
    rpc = os.getenv("RPC") or os.getenv("RPC_HTTPS") or os.getenv("RPC_ENDPOINT") or os.getenv("HELIUS_RPC") or RPC
    if priv:
        for k in ("PRIV_KEY", "PRIVATE_KEY", "BASE58_PRIVKEY"):
            os.environ[k] = priv
    if rpc:
        for k in ("RPC", "RPC_HTTPS", "RPC_ENDPOINT", "HELIUS_RPC"):
            os.environ[k] = rpc
    if not os.getenv("PRIV_KEY"):
        raise EnvError("Falta PRIV_KEY/PRIVATE_KEY/BASE58_PRIVKEY")
    if not os.getenv("RPC"):
        raise EnvError("Falta RPC/RPC_HTTPS/RPC_ENDPOINT/HELIUS_RPC")


def _select_wallet(wallet: Optional[str]) -> None:
    chosen = (wallet or os.getenv("WALLET") or "").strip().lower()
    if not chosen:
        return
    if chosen not in {"fibo", "mc"}:
        raise EnvError(f"Wallet inválida '{wallet}'. Usa 'fibo' o 'mc'.")
    key_by_wallet = {
        "fibo": (os.getenv("PRIV_KEY_FIBO") or os.getenv("PRIVATE_KEY_FIBO") or os.getenv("BASE58_PRIVKEY_FIBO") or "").strip(),
        "mc": (os.getenv("PRIV_KEY_MC") or os.getenv("PRIVATE_KEY_MC") or os.getenv("BASE58_PRIVKEY_MC") or "").strip(),
    }
    sel_key = key_by_wallet[chosen]
    if not sel_key:
        raise EnvError(f"Falta PRIV_KEY_{chosen.upper()} en .env")
    for k in ("PRIV_KEY", "PRIVATE_KEY", "BASE58_PRIVKEY"):
        os.environ[k] = sel_key
    log.info("Wallet activa: %s", chosen)

# ==========================
# Jupiter
# ==========================

def check_jupiter(mint: str) -> bool:
    try:
        params = {"inputMint": "So11111111111111111111111111111111111111112", "outputMint": mint, "amount": 1_000_000, "slippageBps": 50}
        r = requests.get(JUP_QUOTE_URL, params=params, timeout=10)
        if r.status_code != 200:
            return False
        j = r.json()
        return bool(j) and ("outAmount" in j or "routes" in j)
    except Exception as e:
        log.debug("check_jupiter error: %s", e)
        return False

def _exec_sell_jupiter(mint: str, pct: int, slippage_pct: int) -> SellResult:
    from core.jupiter import sell as jup_sell
    log.info("[jupiter] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    try:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            raw = jup_sell(mint, percentage=pct, slippage=slippage_pct)
        printed = f"{buf_out.getvalue()}\n{buf_err.getvalue()}"
        out = _normalize_core_result(raw)
        if not out.get("sig"):
            m = _SIG_RE.search(printed)
            if m:
                out["sig"] = m.group(1)
                out["ok"] = True
                out["status"] = "ok"
        if out.get("ok") and (out.get("sig") or out.get("bundle_id")):
            out.pop("error", None)
            return out
        return {"ok": False, "status": "fail", "error": out.get("error") or "missing_sig_or_bundle"}
    except Exception as e:
        log.exception("[jupiter] Error: %s", e)
        return {"ok": False, "status": "fail", "error": str(e)}

def auto_sell(mint: str, pct: int, slippage_pct: int, _mode: str = "speed") -> SellResult:
    _sync_env_for_modules()
    if not check_jupiter(mint):
        return {"ok": False, "status": "fail", "error": "jupiter_unavailable_or_no_liquidity"}
    res = _exec_sell_jupiter(mint, pct, slippage_pct)
    if res.get("ok") and (res.get("sig") or res.get("bundle_id")):
        res.pop("error", None)
        return res
    return {"ok": False, "status": "fail", "error": res.get("error") or "sell_failed"}

# ==========================
# Compras
# ==========================

def validate_venues(mint: str) -> dict:
    return {"Jupiter": check_jupiter(mint)}

def _quick_order(mint: str, mode: str) -> List[str]:
    return ["jup"]

def _exec_buy_jupiter(mint: str, sol_amount: float, slippage_pct: int) -> BuyResult:
    from core.jupiter import buy as jup_buy
    log.info("[jupiter] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    try:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            raw = jup_buy(mint, sol_in=sol_amount, slippage=slippage_pct)
        printed = f"{buf_out.getvalue()}\n{buf_err.getvalue()}"
        out = _normalize_core_result(raw)
        if not out.get("sig"):
            m = _SIG_RE.search(printed)
            if m:
                out["sig"] = m.group(1)
                out["ok"] = True
                out["status"] = "ok"
        if out.get("ok") and (out.get("sig") or out.get("bundle_id")):
            out.pop("error", None)
            return out
        return {"ok": False, "status": "fail", "error": out.get("error") or "missing_sig_or_bundle"}
    except Exception as e:
        log.exception("[jupiter] Error: %s", e)
        return {"ok": False, "status": "fail", "error": str(e)}

def auto_buy(mint: str, sol_amount: float, slippage_pct: int, mode: str = "speed") -> BuyResult:
    _sync_env_for_modules()
    order = _quick_order(mint, mode)
    log.info("orden_inicial=%s", order)
    last_error = None
    for name in order:
        log.info("Intentando compra con %s...", name)
        try:
            if name == "jup":
                res = _exec_buy_jupiter(mint, sol_amount, slippage_pct)
            else:
                res = {"ok": False, "status": "fail", "error": "route_not_supported"}
            if res.get("ok") and (res.get("sig") or res.get("bundle_id")):
                log.info("Compra exitosa con %s: %s", name, res.get("sig", res.get("bundle_id")))
                res.pop("error", None)
                return res
            log.warning("%s no devolvió transacción válida: %s", name, res.get("error"))
            last_error = res.get("error")
        except Exception as e:
            log.exception("%s falló: %s", name, e)
            last_error = str(e)
            continue
    return {"ok": False, "status": "fail", "error": last_error or "all_routes_failed"}

# ==========================
# Almacenamiento simple (sin eventos)
# ==========================

_state_lock = threading.RLock()
_mints: Dict[str, Dict[str, Any]] = {}

def _ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)


def _atomic_write(path: str, data: str):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def _load_mints_from_disk():
    _ensure_state_dir()
    if not os.path.exists(MINTS_PATH):
        return
    try:
        with open(MINTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _mints.update(data)
    except Exception as e:
        log.error("No se pudo leer %s: %s", MINTS_PATH, e)


def _flush_mints_to_disk():
    try:
        _atomic_write(MINTS_PATH, json.dumps(_mints, ensure_ascii=False, sort_keys=True))
    except Exception as e:
        log.error("No se pudo escribir %s: %s", MINTS_PATH, e)


def _init_state():
    with _state_lock:
        _load_mints_from_disk()


def _get_mint(mint: str) -> Optional[Dict[str, Any]]:
    with _state_lock:
        rec = _mints.get(mint)
        if rec:
            return json.loads(json.dumps(rec))
        return None


def _put_mint(mint: str, rec: Dict[str, Any]):
    with _state_lock:
        _mints[mint] = rec
        _flush_mints_to_disk()


def _update_mint(mint: str, **fields):
    if not fields:
        return
    with _state_lock:
        base = _mints.get(mint, {})
        base.update(fields)
        _mints[mint] = base
        _flush_mints_to_disk()


def _append_trade(mint: str, tr: Dict[str, Any]):
    if "sig" in tr and tr["sig"] is not None:
        tr["sig"] = str(tr["sig"])
    with _state_lock:
        rec = _mints.get(mint)
        if not rec:
            return
        trades = rec.get("trades_done", [])
        trades.append(tr)
        rec["trades_done"] = trades
        rec["last_update"] = _now_s()
        _mints[mint] = rec
        _flush_mints_to_disk()

# ==========================
# Utilidades negocio
# ==========================

def _now_s() -> int:
    return int(time.time())


def _reset_for_new_round(mint: str) -> Dict[str, Any]:
    now = _now_s()
    deadline = now + WINDOW_MINUTES * 60
    row = _get_mint(mint)
    if not row:
        return _upsert_mint_on_alert(mint)
    new_round = int(row.get("round", 1)) + 1
    new_rec = {
        "mint": mint,
        "first_seen": now,
        "status": "new",
        "last_update": now,
        "buy_sig": None,
        "window_deadline": deadline,
        "trades_done": [],
        "notes": None,
        "round": new_round,
        # persistimos último saldo/fees conocidos
        "last_cap_info": row.get("last_cap_info"),
    }
    _put_mint(mint, new_rec)
    return new_rec


def _has_pending_sells(row: Dict[str, Any]) -> bool:
    if not row:
        return False
    done_pcts = {int(t.get("pct", -1)) for t in row.get("trades_done", []) if t.get("type") == "sell" and t.get("ok")}
    plan_pcts = {int(pct) for _, pct in SELL_PLAN}
    return (row.get("status") == "selling") or (not plan_pcts.issubset(done_pcts))


def _within_window(deadline: int) -> bool:
    return _now_s() <= deadline


def _upsert_mint_on_alert(mint: str) -> Dict[str, Any]:
    now = _now_s()
    row = _get_mint(mint)
    if row:
        # SI SE REPITE: ejecutar igual ⇒ reseteamos round siempre
        return _reset_for_new_round(mint)

    deadline = now + WINDOW_MINUTES * 60
    new_rec = {
        "mint": mint,
        "first_seen": now,
        "status": "new",
        "last_update": now,
        "buy_sig": None,
        "window_deadline": deadline,
        "trades_done": [],
        "notes": None,
        "round": 1,
        "last_cap_info": None,
    }
    _put_mint(mint, new_rec)
    return new_rec

# ==========================
# Horarios, fees y límites (guardando saldo)
# ==========================

def _within_trading_hours(ts: Optional[int] = None) -> bool:
    if not TRADING_HOURS:
        return True
    ts = ts or _now_s()
    local = time.localtime(ts)
    start_s, end_s = TRADING_HOURS.split("-")
    def _hm(s):
        h, m = s.split(":")
        return int(h), int(m)
    sh, sm = _hm(start_s)
    eh, em = _hm(end_s)
    t_now = local.tm_hour * 60 + local.tm_min
    t_start = sh * 60 + sm
    t_end = eh * 60 + em
    return t_start <= t_now <= t_end


def lamports(x_sol: float) -> int:
    return int(round(max(0.0, float(x_sol)) * LAMPORTS_PER_SOL))


def to_sol(x_lamports: int) -> float:
    return x_lamports / LAMPORTS_PER_SOL


def _get_rpc_url() -> str:
    for k in ("RPC_HTTPS", "RPC", "RPC_ENDPOINT", "HELIUS_RPC", "SOL_RPC_URL"):
        v = os.getenv(k)
        if v:
            return v
    return DEFAULT_RPC


def _load_keypair_from_env() -> Keypair:
    val = os.getenv("PRIVATE_KEY") or os.getenv("PRIV_KEY") or os.getenv("BASE58_PRIVKEY")
    if not val:
        sys.exit("❌ Falta PRIVATE_KEY/PRIV_KEY/BASE58_PRIVKEY.")
    try:
        return Keypair.from_base58_string(val)
    except Exception:
        pass
    try:
        arr = json.loads(val)
        if isinstance(arr, list) and len(arr) in (64, 32):
            b = bytes(arr)
            if len(b) == 64:
                return Keypair.from_bytes(b)
    except Exception:
        pass
    sys.exit("❌ PRIVATE_KEY inválida. Base58 o JSON [64 bytes].")


def _estimate_base_fee(client: Client, payer: Pubkey, to_pub: Pubkey) -> int:
    try:
        bh = client.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(payer, [transfer(TransferParams(from_pubkey=payer, to_pubkey=to_pub, lamports=1))], [], bh)
        fee = client.get_fee_for_message(msg).value
        return int(fee) if fee is not None else 5000
    except Exception:
        return 5000


def _estimate_priority_fee_lamports() -> int:
    if not UNIT_PRICE or not UNIT_BUDGET:
        return 0
    try:
        micro = float(UNIT_PRICE)
        budget = float(UNIT_BUDGET)
        micro_total = micro * budget
        return int(round(micro_total / 1_000_000.0))
    except Exception:
        return 0


def _estimate_jito_tip_lamports() -> int:
    if not USE_JITO:
        return 0
    if not TIP_SOL:
        return 0
    try:
        return lamports(float(TIP_SOL))
    except Exception:
        return 0


def _compute_buy_cap_sol() -> Tuple[float, Dict[str, Any]]:
    rpc = _get_rpc_url()
    kp = _load_keypair_from_env()
    payer = kp.pubkey()
    client = Client(rpc)
    to_env = os.getenv("TO_PUBKEY")
    try:
        to_pub = Pubkey.from_string(to_env) if to_env else payer
    except Exception:
        to_pub = payer
    balance_lamports = int(client.get_balance(payer).value)
    base_fee = _estimate_base_fee(client, payer, to_pub)
    prio_fee = _estimate_priority_fee_lamports()
    jito_tip = _estimate_jito_tip_lamports()
    total_fee = base_fee + prio_fee + jito_tip
    cushion_lamports = lamports(FEE_CUSHION_SOL)
    spendable_after_fee = max(0, balance_lamports - total_fee)
    saldo_real_movible = max(0, spendable_after_fee - cushion_lamports)
    cap_por_porcentaje = int(balance_lamports * MAX_FRACTION_OF_BALANCE)
    cap_para_swap = min(saldo_real_movible, cap_por_porcentaje)
    cap_sol = to_sol(cap_para_swap)
    out = {
        "owner_pubkey": str(payer),
        "rpc": rpc,
        "balance_lamports": balance_lamports,
        "balance_sol": round(to_sol(balance_lamports), 9),
        "fees": {
            "base_fee_lamports": base_fee,
            "priority_fee_lamports": prio_fee,
            "jito_tip_lamports": jito_tip,
            "total_fee_lamports": total_fee,
        },
        "cushion_sol": FEE_CUSHION_SOL,
        "saldo_real_movible_sol": round(to_sol(saldo_real_movible), 9),
        "max_fraction_of_balance": MAX_FRACTION_OF_BALANCE,
        "cap_para_swap_sol": round(cap_sol, 9),
    }
    return cap_sol, out


def _can_afford_tx_fees(min_extra_cushion_sol: float = 0.0) -> Tuple[bool, Dict[str, Any]]:
    rpc = _get_rpc_url()
    kp = _load_keypair_from_env()
    payer = kp.pubkey()
    client = Client(rpc)
    balance_lamports = int(client.get_balance(payer).value)
    base_fee = _estimate_base_fee(client, payer, payer)
    prio_fee = _estimate_priority_fee_lamports()
    jito_tip = _estimate_jito_tip_lamports()
    total_fee = base_fee + prio_fee + jito_tip
    needed = total_fee + lamports(FEE_CUSHION_SOL + min_extra_cushion_sol)
    ok = balance_lamports >= needed
    return ok, {"balance_lamports": balance_lamports, "total_fee_lamports": total_fee, "needed_lamports": needed, "ok": ok}

# ==========================
# Worker de compra/venta
# ==========================

_sell_locks: Dict[str, threading.Lock] = {}

def _get_sell_lock(mint: str) -> threading.Lock:
    if mint not in _sell_locks:
        _sell_locks[mint] = threading.Lock()
    return _sell_locks[mint]


def _do_buy_and_schedule_sells(mint: str, sol_amount: float, slippage: int, mode: str):
    row = _get_mint(mint)
    if not row:
        log.warning("[%s] no existe en estado al iniciar worker", mint)
        return
    if not _within_window(row["window_deadline"]):
        _update_mint(mint, status="expired")
        log.info("[%s] ventana expirada, no se compra", mint)
        return
    if not _within_trading_hours():
        _update_mint(mint, status="skipped_out_of_hours")
        log.info("[%s] fuera de horario %s, no se compra", mint, TRADING_HOURS)
        return

    status = row["status"]
    if status in ("new", "failed"):
        cap_sol, cap_raw = _compute_buy_cap_sol()
        # Guardar SIEMPRE la info de saldo/fees más reciente
        _update_mint(mint, last_cap_info=cap_raw)
        requested_sol = float(sol_amount)
        if cap_sol <= 0.0:
            _update_mint(mint, status="skipped_insufficient_balance", notes="cap<=0 por fees/reserva")
            log.info("[%s] skip compra: cap<=0 (fees/reserva).", mint)
            return
        if requested_sol > cap_sol:
            if ALLOW_DOWNSIZE and cap_sol >= MIN_BUY_SOL:
                log.info("[%s] downsizing compra: solicitado=%.6f SOL, cap=%.6f SOL", mint, requested_sol, cap_sol)
                sol_to_use = cap_sol
            else:
                _update_mint(mint, status="skipped_insufficient_balance", notes=f"requested={requested_sol} > cap={cap_sol} (min_buy={MIN_BUY_SOL}, allow_downsize={ALLOW_DOWNSIZE})")
                log.info("[%s] skip compra: requested(%.6f) > cap(%.6f).", mint, requested_sol, cap_sol)
                return
        else:
            sol_to_use = requested_sol
        if sol_to_use < MIN_BUY_SOL:
            _update_mint(mint, status="skipped_insufficient_balance", notes=f"sol_to_use={sol_to_use} < min_buy_sol={MIN_BUY_SOL}")
            log.info("[%s] skip compra: sol_to_use(%.6f) < min_buy_sol(%.6f).", mint, sol_to_use, MIN_BUY_SOL)
            return
        _update_mint(mint, status="buying")
        log.info("[%s] BUY start (sol=%.6f, slip=%d%%)", mint, sol_to_use, slippage)
        buy_res = auto_buy(mint, sol_to_use, slippage, mode)
        ok = bool(buy_res.get("ok")) and (buy_res.get("sig") or buy_res.get("bundle_id"))
        if ok:
            sig = buy_res.get("sig") or buy_res.get("bundle_id")
            _update_mint(mint, status="bought", buy_sig=sig, balance_sol_at_buy=(cap_raw.get("balance_sol") if isinstance(cap_raw, dict) else None))
            _append_trade(mint, {"t": _now_s(), "type": "buy", "ok": True, "sig": sig, "sol": sol_to_use})
            log.info("[%s] BUY ok: %s", mint, sig)
        else:
            _update_mint(mint, status="failed", notes=str(buy_res.get("error")))
            _append_trade(mint, {"t": _now_s(), "type": "buy", "ok": False, "err": buy_res.get("error"), "sol": sol_to_use})
            log.warning("[%s] BUY fail: %s", mint, buy_res.get("error"))
            return
    elif status in ("bought", "selling"):
        log.info("[%s] ya comprado previamente; continúo con ventas si aplica", mint)
    elif status in ("done", "expired", "skipped_out_of_hours", "skipped_insufficient_balance"):
        log.info("[%s] estado=%s; no hago nada", mint, status)
        return

    _update_mint(mint, status="selling")
    sell_lock = _get_sell_lock(mint)
    with sell_lock:
        start_ts = _now_s()
        for delay_sec, pct in SELL_PLAN:
            left = (_get_mint(mint) or {}).get("window_deadline", _now_s()) - _now_s()
            log.info("[%s] Venta %d%% en %ds (resto ventana=%ds)", mint, pct, delay_sec, max(0, left))
            sleep_for = max(0, delay_sec - (_now_s() - start_ts))
            if sleep_for:
                time.sleep(sleep_for)
            if SELL_REQUIRE_FEE_OK:
                ok_fee, fee_info = _can_afford_tx_fees(SELL_MIN_EXTRA_CUSHION_SOL)
                if not ok_fee:
                    log.warning("[%s] SELL %d%% omitida por fee guard: %s", mint, pct, fee_info)
                    continue
            sell_res = auto_sell(mint, pct, slippage, mode)
            ok = bool(sell_res.get("ok")) and (sell_res.get("sig") or sell_res.get("bundle_id"))
            if ok:
                sig = sell_res.get("sig") or sell_res.get("bundle_id")
                _append_trade(mint, {"t": _now_s(), "type": "sell", "pct": pct, "ok": True, "sig": sig})
                log.info("[%s] SELL %d%% ok: %s", mint, pct, sig)
            else:
                _append_trade(mint, {"t": _now_s(), "type": "sell", "pct": pct, "ok": False, "err": sell_res.get("error")})
                log.warning("[%s] SELL %d%% fail: %s", mint, pct, sell_res.get("error"))
    _update_mint(mint, status="done")
    log.info("[%s] Flujo completo finalizado", mint)

# ==========================
# Alertas y ejecución (SIN de-dupe: si se repite, ejecuta igual)
# ==========================

_pending_workers: Dict[str, threading.Thread] = {}

def on_alert(mint: str, sol_amount: Optional[float] = None, slippage: Optional[int] = None, mode: Optional[str] = None):
    sol_amount = sol_amount if sol_amount is not None else DEFAULT_SOL
    slippage = slippage if slippage is not None else DEFAULT_SLIPPAGE
    mode = mode or DEFAULT_MODE
    _upsert_mint_on_alert(mint)  # siempre resetea round al alertar

    # SIN de-dupe: siempre lanzamos un worker nuevo
    th = threading.Thread(target=_do_buy_and_schedule_sells, args=(mint, sol_amount, slippage, mode), daemon=True)
    _pending_workers[f"{mint}:{int(time.time())}"] = th
    th.start()

# ==========================
# Main
# ==========================

def main():
    _setup_logging()
    _init_state()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--mint")
    ap.add_argument("--sol", type=float, default=DEFAULT_SOL)
    ap.add_argument("--slippage", type=int, default=DEFAULT_SLIPPAGE)
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["price", "speed"])
    args = ap.parse_args()

    if args.mint:
        on_alert(args.mint, args.sol, args.slippage, args.mode)
        # join solo a los threads creados en este main
        for key, t in list(_pending_workers.items()):
            t.join()
        return

    if args.stdin:
        log.info("Leyendo mints por STDIN. CTRL+C para salir.")
        try:
            for line in sys.stdin:
                m = line.strip()
                if not m:
                    continue
                on_alert(m, args.sol, args.slippage, args.mode)
        except KeyboardInterrupt:
            pass
        for key, t in list(_pending_workers.items()):
            t.join()
    else:
        log.warning("No se especificó --mint ni --stdin; nada que hacer.")

if __name__ == "__main__":
    main()
