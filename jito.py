#!/usr/bin/env python3
# main_jito.py — sender modular para Jito (legacy + v0)

import os, time, base64, random, logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter, Retry

from base58 import b58decode

# ---------- solana-py (legacy tx) ----------
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.transaction import Transaction, Transaction as LegacyTx
from solana.publickey import PublicKey, PublicKey as SolPub
from solana.keypair import Keypair, Keypair as SolKeypair
from solana.system_program import TransferParams, transfer, transfer as sys_transfer
from solana.rpc.commitment import Processed

# ---------- solders (versioned tx v0) ----------
from solders.transaction import VersionedTransaction  # para v0
# (MessageV0 lo creás en tus módulos Raydium/Pump.fun y lo pasás a este archivo)

# =========================
# Config por entorno
# =========================
BLOCK_ENGINE_URL = os.getenv("JITO_BLOCK_ENGINE_URL", "https://ny.mainnet.block-engine.jito.wtf").rstrip("/")
RPC_HTTPS        = os.getenv("RPC_HTTPS")
PRIVATE_KEY_B58  = os.getenv("PRIVATE_KEY")  # base58 (32B seed o 64B secret)
TIP_ACCOUNT_STR  = os.getenv("TIP_ACCOUNT", "JitoTip1111111111111111111111111111111111")
TIP_SOL          = min(max(float(os.getenv("JITO_TIP_VALUE", "0.004")), 0.0), 0.1)  # 0..0.1
USE_JITO         = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
SKIP_PREFLIGHT   = os.getenv("SKIP_PREFLIGHT", "true").lower() in ("1", "true", "yes")
JITO_API_KEY     = os.getenv("JITO_API_KEY")  # opcional

assert RPC_HTTPS and PRIVATE_KEY_B58, "Faltan RPC_HTTPS o PRIVATE_KEY"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LAMPORTS_PER_SOL = 1_000_000_000

# =========================
# Utils básicos
# =========================
def now() -> str:
    return f"{datetime.utcnow().isoformat()}Z"

def load_wallet() -> Keypair:
    raw = b58decode(PRIVATE_KEY_B58)
    if len(raw) == 64:
        return Keypair.from_secret_key(raw)
    if len(raw) == 32:
        return Keypair.from_seed(raw)
    raise ValueError("PRIVATE_KEY debe ser base58 de 32 o 64 bytes")

def get_latest_blockhash_and_height(rpc: Client) -> Tuple[str, Optional[int]]:
    resp = rpc.get_latest_blockhash()
    # soporta dict (algunos providers) y objetos de solana-py
    if isinstance(resp, dict):
        v = resp.get("result", {}).get("value", {}) or resp.get("value", {})
        bh = v.get("blockhash"); h = v.get("lastValidBlockHeight")
        if bh:
            return str(bh), (int(h) if h is not None else None)
    return str(resp.value.blockhash), int(resp.value.last_valid_block_height)  # type: ignore

# =========================
# Cliente Jito JSON-RPC
# =========================
class JitoClient:
    def __init__(self, block_engine_url: str, api_key: Optional[str] = None):
        self.url = block_engine_url.rstrip("/")
        self.s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.4, status_forcelist=[500,502,503,504], allowed_methods=["POST"])
        self.s.mount("https://", HTTPAdapter(max_retries=retries))
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-KEY"] = api_key  # algunos BE usan este header

    def send_bundle(self, b64_txs: List[str]) -> Optional[str]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "sendBundle", "params": [b64_txs]}
        r = self.s.post(f"{self.url}/api/v1/bundles", json=payload, timeout=(2, 8), headers=self.headers)
        if r.ok:
            return r.json().get("result")
        logging.warning("[JITO] sendBundle %s: %s", r.status_code, r.text)
        return None

    def get_bundle_status(self, bundle_id: str) -> Optional[Dict]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBundleStatuses", "params": [[bundle_id]]}
        r = self.s.post(f"{self.url}/api/v1/bundles", json=payload, timeout=(2, 8), headers=self.headers)
        if not r.ok:
            logging.warning("[JITO] getBundleStatuses %s: %s", r.status_code, r.text)
            return None
        j = r.json()
        res = j.get("result", {})
        arr = res.get("value", []) if isinstance(res, dict) else (res or [])
        return arr[0] if arr else None

def parse_jito_state(status: Dict) -> Tuple[Optional[str], Optional[str]]:
    conf = status.get("confirmation_status") or status.get("state")
    err  = status.get("bundle_error") or status.get("error")
    if not err:
        meta = status.get("status") or {}
        if isinstance(meta, dict):
            err = meta.get("err") or meta.get("error")
    return (str(conf) if conf else None, str(err) if err else None)

# =========================
# Legacy: construir y enviar
# =========================
def build_main_tx(instructions: List, wallet: Keypair, blockhash: str) -> Transaction:
    tx = Transaction(recent_blockhash=blockhash, fee_payer=wallet.public_key)
    # (opcional) compute budget primero:
    # tx.add(set_compute_unit_limit(100_000))
    # tx.add(set_compute_unit_price(1_000_000))
    for ix in instructions:
        tx.add(ix)
    tx.sign(wallet)
    return tx

def build_tip_tx(sender: Keypair, tip_account: PublicKey, lamports: int, blockhash: str) -> Transaction:
    tx = Transaction(recent_blockhash=blockhash, fee_payer=sender.public_key)
    tx.add(transfer(TransferParams(
        from_pubkey=sender.public_key,
        to_pubkey=tip_account,
        lamports=lamports,
    )))
    tx.sign(sender)
    return tx

def send_via_rpc(rpc: Client, tx: Transaction) -> str:
    resp = rpc.send_transaction(tx, opts=TxOpts(skip_preflight=SKIP_PREFLIGHT, preflight_commitment=Processed))
    return str(resp.get("result") or resp.get("value") or resp)

def send_with_jito_bundle(instructions: List, wait_timeout_s: float = 12.0, retries: int = 2) -> Dict:
    """
    Envia una TX legacy como bundle [main_tx_legacy, tip_tx_legacy].
    Devuelve {"bundle_id": str|None, "confirmed": bool, "error": str|None}
    """
    wallet = load_wallet()
    try:
        tip_pubkey = PublicKey(TIP_ACCOUNT_STR)
    except Exception as e:
        return {"bundle_id": None, "confirmed": False, "error": f"TIP_ACCOUNT inválido: {e}"}

    rpc = Client(RPC_HTTPS)
    jito = JitoClient(BLOCK_ENGINE_URL, JITO_API_KEY)
    tip_lamports = int(TIP_SOL * LAMPORTS_PER_SOL)

    attempt = 0
    base_delay = 0.4

    while attempt <= retries:
        attempt += 1
        blockhash, last_valid_height = get_latest_blockhash_and_height(rpc)
        logging.info("Using blockhash=%s lastValidHeight=%s (attempt %d/%d)", blockhash, last_valid_height, attempt, retries+1)

        main_tx = build_main_tx(instructions, wallet, blockhash)
        tip_tx  = build_tip_tx(wallet, tip_pubkey, tip_lamports, blockhash)

        bundle = [
            base64.b64encode(main_tx.serialize()).decode(),
            base64.b64encode(tip_tx.serialize()).decode()
        ]
        logging.info("[JITO] tip=%.6f SOL → %s | bundle_len=%d", TIP_SOL, str(tip_pubkey), len(bundle))

        bundle_id = jito.send_bundle(bundle)
        if not bundle_id:
            delay = base_delay * (1.7 ** (attempt - 1)) + random.uniform(0, 0.15)
            logging.warning("[JITO] no bundle_id; retrying in %.2fs", delay)
            time.sleep(delay); continue

        started = time.time()
        while time.time() - started < wait_timeout_s:
            st = jito.get_bundle_status(bundle_id)
            if st:
                conf, err = parse_jito_state(st)
                if conf in ("confirmed", "finalized"):
                    logging.info("[JITO] ✅ Bundle confirmed")
                    return {"bundle_id": bundle_id, "confirmed": True, "error": None}
                if err:
                    logging.error("[JITO] ❌ %s", err)
                    if "blockhash" in err.lower():
                        break  # refrescar y reintentar
                    return {"bundle_id": bundle_id, "confirmed": False, "error": err}
            time.sleep(0.4)

        logging.warning("[JITO] ⏳ Timeout esperando confirmación; nuevo intento...")

    logging.error("[JITO] ❌ Agotados reintentos")
    return {"bundle_id": None, "confirmed": False, "error": "retries_exhausted"}

# =========================
# v0: bundle [v0 + tip legacy]
# =========================
def bundle_v0_with_tip(
    signed_vtx: VersionedTransaction,
    payer_solders_kp,               # solders.Keypair
    tip_account: str,
    tip_sol_lamports: int,
) -> List[str]:
    """Devuelve [main_v0_b64, tip_legacy_b64] compartiendo el mismo blockhash."""
    import base64 as _b64
    recent_bh = str(signed_vtx.message.recent_blockhash)
    payer_solana = SolKeypair.from_secret_key(payer_solders_kp.to_bytes())
    tip_pubkey   = SolPub(tip_account)

    tip_tx = LegacyTx(recent_blockhash=recent_bh, fee_payer=payer_solana.public_key)
    tip_tx.add(sys_transfer(TransferParams(
        from_pubkey=payer_solana.public_key,
        to_pubkey=tip_pubkey,
        lamports=tip_sol_lamports,
    )))
    tip_tx.sign(payer_solana)

    main_b64 = _b64.b64encode(bytes(signed_vtx)).decode()
    tip_b64  = _b64.b64encode(tip_tx.serialize()).decode()
    return [main_b64, tip_b64]

def send_v0_with_jito_bundle(
    signed_vtx: VersionedTransaction,
    payer_solders_kp,                 # solders.Keypair
    wait_timeout_s: float = 12.0,
    retries: int = 2,
) -> Dict:
    """
    Envia una v0 (ya firmada) por bundle con tip legacy.
    Si error='blockhash_expired' → el caller debe recompilar MessageV0 y reintentar.
    """
    jito = JitoClient(BLOCK_ENGINE_URL, JITO_API_KEY)
    tip_lamports = int(TIP_SOL * LAMPORTS_PER_SOL)

    attempt = 0
    base_delay = 0.4

    while attempt <= retries:
        attempt += 1
        b64_txs = bundle_v0_with_tip(
            signed_vtx=signed_vtx,
            payer_solders_kp=payer_solders_kp,
            tip_account=TIP_ACCOUNT_STR,
            tip_sol_lamports=tip_lamports,
        )
        logging.info("[JITO v0] tip=%.6f SOL → %s | bundle_len=%d", TIP_SOL, TIP_ACCOUNT_STR, len(b64_txs))

        bundle_id = jito.send_bundle(b64_txs)
        if not bundle_id:
            delay = base_delay * (1.7 ** (attempt - 1)) + random.uniform(0, 0.15)
            logging.warning("[JITO v0] no bundle_id; retrying in %.2fs", delay)
            time.sleep(delay); continue

        started = time.time()
        while time.time() - started < wait_timeout_s:
            st = jito.get_bundle_status(bundle_id)
            if st:
                conf, err = parse_jito_state(st)
                if conf in ("confirmed", "finalized"):
                    logging.info("[JITO v0] ✅ Bundle confirmed")
                    return {"bundle_id": bundle_id, "confirmed": True, "error": None}
                if err:
                    logging.error("[JITO v0] ❌ %s", err)
                    if "blockhash" in err.lower() or "expired" in err.lower():
                        return {"bundle_id": bundle_id, "confirmed": False, "error": "blockhash_expired"}
                    return {"bundle_id": bundle_id, "confirmed": False, "error": err}
            time.sleep(0.4)

        logging.warning("[JITO v0] ⏳ Timeout esperando confirmación; reintento...")

    logging.error("[JITO v0] ❌ Agotados reintentos")
    return {"bundle_id": None, "confirmed": False, "error": "retries_exhausted"}

def maybe_send_v0_with_jito_or_rpc(
    message_v0,                 # solders.MessageV0
    payer_solders_kp,           # solders.Keypair
) -> Tuple[Optional[str], Optional[str]]:
    """
    Firma y envía:
      - Si USE_JITO => Jito (devuelve (signature=None, bundle_id=str))
      - Si no => RPC normal con VersionedTransaction (devuelve (signature=str, bundle_id=None))
    """
    vtx = VersionedTransaction(message_v0, [payer_solders_kp])

    if USE_JITO:
        res = send_v0_with_jito_bundle(vtx, payer_solders_kp)
        if res.get("confirmed"):
            return None, res.get("bundle_id")
        if res.get("error") == "blockhash_expired":
            logging.warning("[maybe_send] blockhash_expired → recompilar MessageV0 con blockhash fresco y reintentar en el caller")
        return None, res.get("bundle_id")
    else:
        rpc = Client(RPC_HTTPS)
        try:
            sig = rpc.send_transaction(vtx, opts=TxOpts(skip_preflight=SKIP_PREFLIGHT)).value
            logging.info("📤 RPC v0 Tx: %s", str(sig))
            return str(sig), None
        except Exception as e:
            logging.error("[maybe_send] RPC error: %s", e)
            return None, None

# =========================
# Ejemplo mínimo (legacy)
# =========================
if __name__ == "__main__":
    kp = load_wallet()
    dummy_ix = transfer(TransferParams(
        from_pubkey=kp.public_key,
        to_pubkey=kp.public_key,
        lamports=1_000_000,  # 0.001 SOL
    ))

    if USE_JITO:
        res = send_with_jito_bundle([dummy_ix])
        print("Result:", res)
    else:
        rpc = Client(RPC_HTTPS)
        blockhash, _ = get_latest_blockhash_and_height(rpc)
        tx = build_main_tx([dummy_ix], kp, blockhash)
        sig = send_via_rpc(rpc, tx)
        print("RPC signature:", sig)
