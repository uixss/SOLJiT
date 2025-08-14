#!/usr/bin/env python3
 
from __future__ import annotations
import os
import time
import base64
import logging
from typing import Optional, Tuple, Dict

import requests
from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.instruction import Instruction
from solders.system_program import transfer, TransferParams
from solders.signature import Signature

# ---------------- ENV & defaults ----------------
RPC                = os.getenv("RPC", "") or os.getenv("RPC_ENDPOINT", "")
BLOCK_ENGINE_URL   = os.getenv("BLOCK_ENGINE_URL", "https://mainnet.block-engine.jito.wtf")
JITO_API_KEY       = os.getenv("JITO_API_KEY", "")  # opcional
USE_JITO           = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
SKIP_PREFLIGHT     = os.getenv("SKIP_PREFLIGHT", "false").lower() in ("1", "true", "yes")

TIP_SOL_DEFAULT    = 0.004
try:
    TIP_SOL = float(os.getenv("TIP_SOL", str(TIP_SOL_DEFAULT)))
    TIP_SOL = max(0.0, min(0.1, TIP_SOL))  # clamp [0, 0.1]
except Exception:
    TIP_SOL = TIP_SOL_DEFAULT

TIP_ACCOUNT_STR    = os.getenv("TIP_ACCOUNT", "DMBm1aAq5H3eR9j6nqMmf1tKj9xwQ2rceYbrY9jito11")  # ejemplo
LAMPORTS_PER_SOL   = 10**9

HTTP_CONNECT_TO    = float(os.getenv("JITO_CONNECT_TIMEOUT", "2.0"))
HTTP_READ_TO       = float(os.getenv("JITO_READ_TIMEOUT", "8.0"))
CONFIRM_RETRIES    = int(os.getenv("JITO_CONFIRM_RETRIES", "40"))
CONFIRM_SLEEP_SECS = float(os.getenv("JITO_CONFIRM_SLEEP", "1.5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
logging.basicConfig(
    level={"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}.get(LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("jito")

if not RPC:
    raise RuntimeError("RPC no configurado (RPC / RPC_ENDPOINT)")

# -------------- HTTP helper ---------------
def _post_json(url: str, json_body: dict) -> requests.Response:
    headers = {"content-type": "application/json", "accept": "application/json"}
    if JITO_API_KEY:
        headers["X-API-KEY"] = JITO_API_KEY
    r = requests.post(url, json=json_body, headers=headers, timeout=(HTTP_CONNECT_TO, HTTP_READ_TO))
    r.raise_for_status()
    return r

# -------------- RPC helpers ---------------
def _get_client() -> Client:
    return Client(RPC)

def _latest_blockhash_str(client: Client) -> str:
    # Devuelve el blockhash como str (para v0)
    bh = client.get_latest_blockhash().value.blockhash
    return str(bh)

def _confirm_sig(sig_str: str) -> bool:
    client = _get_client()
    tries = 0
    sig = Signature.from_string(sig_str)
    while tries < CONFIRM_RETRIES:
        try:
            st = client.get_signature_statuses([sig]).value
            if st and st[0] is not None:
                if st[0].err is None:
                    log.info("✔ Confirmado: %s", sig_str)
                    return True
                else:
                    log.error("✗ Error en tx: %s", st[0].err)
                    return False
        except Exception as e:
            log.debug("status err: %s", e)
        tries += 1
        time.sleep(CONFIRM_SLEEP_SECS)
    log.warning("Tiempo de confirmación agotado para %s", sig_str)
    return False

# -------------- Tip transactions ----------
def _build_tip_v0_tx(payer: Keypair, tip_account: Pubkey, lamports: int, recent_blockhash: str) -> VersionedTransaction:
    ix: Instruction = transfer(TransferParams(from_pubkey=payer.pubkey(), to_pubkey=tip_account, lamports=lamports))
    msg = MessageV0.try_compile(payer.pubkey(), [ix], [], recent_blockhash)
    return VersionedTransaction(msg, [payer])

# -------------- Bundles (REST minimal) ----
def _send_bundle_base64(b64_txs: list[str]) -> Dict:
    """
    Envía la lista de transacciones (base64) al Block Engine.
    Usamos la REST API /api/v1/bundles (soportada por la mayoría de BE gateways).
    """
    url = f"{BLOCK_ENGINE_URL.rstrip('/')}/api/v1/bundles"
    body = {"transactions": b64_txs}
    try:
        r = _post_json(url, body)
        j = r.json()
        # Muchos BE devuelven {"bundleId": "..."}; si no, devolvemos el JSON crudo
        bundle_id = j.get("bundleId") or j.get("bundle_id") or None
        return {"ok": True, "bundle_id": bundle_id, "raw": j}
    except Exception as e:
        log.error("send bundle error: %s", e)
        return {"ok": False, "bundle_id": None, "error": str(e)}

# -------------- API pública ----------------
def send_v0_with_jito_bundle(
    signed_vtx: VersionedTransaction,
    payer_solders_kp: Keypair,
    *,
    tip_account_override: Optional[str] = None,
    tip_lamports_override: Optional[int] = None,
    confirm: bool = True
) -> Dict:
    """
    Envía un bundle con:
      [ signed_vtx (tu swap/acción principal), tip_tx (v0, system transfer) ]
    Retorna: {"bundle_id": str|None, "confirmed": bool, "signature": str, "error": str|None}
    """
    # 1) validar TIP_ACCOUNT
    tip_acc_str = (tip_account_override or TIP_ACCOUNT_STR or "").strip()
    try:
        tip_acc = Pubkey.from_string(tip_acc_str)
    except Exception as e:
        return {"bundle_id": None, "confirmed": False, "signature": None, "error": f"invalid_tip_account: {e}"}

    # 2) calcular tip
    tip_lamports = tip_lamports_override if tip_lamports_override is not None else int(TIP_SOL * LAMPORTS_PER_SOL)
    if tip_lamports < 0:
        tip_lamports = 0

    # 3) armar tip tx v0 con blockhash reciente
    client = _get_client()
    recent_bh = _latest_blockhash_str(client)
    tip_tx = _build_tip_v0_tx(payer_solders_kp, tip_acc, tip_lamports, recent_bh)

    # 4) serializar ambas
    main_b64 = base64.b64encode(bytes(signed_vtx)).decode("utf-8")
    tip_b64  = base64.b64encode(bytes(tip_tx)).decode("utf-8")
    b64_list = [main_b64, tip_b64]

    # La firma “principal” (para confirmar por RPC)
    try:
        main_sig = str(signed_vtx.signatures[0])
    except Exception:
        # último recurso: None
        main_sig = None

    # 5) Si USE_JITO = false, fallback RPC directo
    if not USE_JITO:
        try:
            sig = Client(RPC).send_raw_transaction(bytes(signed_vtx), opts=TxOpts(skip_preflight=SKIP_PREFLIGHT, preflight_commitment=Processed)).value
            sig_str = str(sig)
            ok = _confirm_sig(sig_str) if confirm else True
            return {"bundle_id": None, "confirmed": ok, "signature": sig_str, "error": None}
        except Exception as e:
            return {"bundle_id": None, "confirmed": False, "signature": None, "error": f"rpc_fallback_error: {e}"}

    # 6) Enviar bundle a Jito
    resp = _send_bundle_base64(b64_list)
    if not resp.get("ok"):
        return {"bundle_id": None, "confirmed": False, "signature": main_sig, "error": resp.get("error") or "bundle_send_failed"}

    bundle_id = resp.get("bundle_id")
    log.info("Bundle enviado (id=%s) tip=%.6f SOL → %s", bundle_id, tip_lamports / LAMPORTS_PER_SOL, tip_acc_str)

    # 7) Confirmación por RPC usando la firma principal
    if confirm and main_sig:
        ok = _confirm_sig(main_sig)
        return {"bundle_id": bundle_id, "confirmed": ok, "signature": main_sig, "error": None if ok else "tx_not_confirmed"}
    else:
        # si no tenemos firma principal, no podemos confirmar por RPC
        return {"bundle_id": bundle_id, "confirmed": False if confirm else True, "signature": main_sig, "error": None if not confirm else "no_main_signature"}

def maybe_send_v0_with_jito_or_rpc(message_v0: MessageV0, payer_solders_kp: Keypair) -> Tuple[Optional[str], Optional[str]]:
    """
    Conveniencia:
      - Si USE_JITO => bundle v0 + tip (devuelve (None, bundle_id))
      - Si NO => RPC raw (devuelve (sig, None))
    """
    vtx = VersionedTransaction(message_v0, [payer_solders_kp])

    if USE_JITO:
        res = send_v0_with_jito_bundle(vtx, payer_solders_kp)
        return (None, res.get("bundle_id"))
    else:
        client = _get_client()
        try:
            sig = client.send_raw_transaction(bytes(vtx), opts=TxOpts(skip_preflight=SKIP_PREFLIGHT, preflight_commitment=Processed)).value
            sig_str = str(sig)
            log.info("RPC v0 Tx: %s", sig_str)
            return (sig_str, None)
        except Exception as e:
            log.error("[maybe_send] RPC error: %s", e)
            return (None, None)
 
def send_with_jito_bundle(*args, **kwargs) -> Dict:
 
    return {"bundle_id": None, "confirmed": False, "signature": None, "error": "legacy_not_implemented"}
