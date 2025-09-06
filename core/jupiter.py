#!/usr/bin/env python3
# Jupiter v6 — buy/sell con Versioned TX; envío por Jito opcional vía core/jito_sender

import os
import sys
import json
import base64
import time
import random
from typing import Any, Dict, Optional, Union, Tuple

import requests
from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import TxOpts

from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction
from solders.signature import Signature

# -------- Config ----------
PRIV_KEY = os.getenv("PRIV_KEY", "")
RPC      = os.getenv("RPC", "")
USE_JITO = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
SKIP_PREFLIGHT = os.getenv("SKIP_PREFLIGHT", "false").lower() in ("1", "true", "yes")
ALLOW_MULTI_HOPS = os.getenv("JUP_ALLOW_MULTI_HOPS", "false").lower() in ("1", "true", "yes")

if not PRIV_KEY or not RPC:
    raise RuntimeError("PRIV_KEY / RPC no configurados")

SOL_MINT = "So11111111111111111111111111111111111111112"
HTTP_TIMEOUT = 25
CONFIRM_RETRIES = 32
CONFIRM_SLEEP_SECS = 1.2

JUP_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUP_SWAP_URL  = "https://quote-api.jup.ag/v6/swap"

 
 
_session = requests.Session()

def _http_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    """Pequeño helper con backoff exponencial + jitter para 429/5xx."""
    max_tries = 4
    for i in range(max_tries):
        try:
            resp = _session.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if i == max_tries - 1:
                raise
            time.sleep(0.4 * (2 ** i) + random.random() * 0.25)

# -------- Init ----------
def _require_config():
    if not PRIV_KEY:
        raise RuntimeError("PRIV_KEY vacío")
    if not RPC:
        raise RuntimeError("RPC vacío")

def get_client_and_payer() -> Tuple[Client, Keypair]:
    _require_config()
    return Client(RPC), Keypair.from_base58_string(PRIV_KEY)

# -------- Helpers ----------
def get_token_balance_lamports(mint_str: str, owner_pubkey: str) -> int:
    """Suma TODAS las ATAs del mint para ese owner (en lamports/unidades base del token)."""
    payload = {
        "id": 1, "jsonrpc": "2.0", "method": "getTokenAccountsByOwner",
        "params": [owner_pubkey, {"mint": mint_str}, {"encoding": "jsonParsed"}],
    }
    headers = {"accept": "application/json", "content-type": "application/json"}
    try:
        r = _http_with_retries("POST", RPC, json=payload, headers=headers)
        j = r.json()
        total = 0
        for v in (j.get("result", {}).get("value", []) or []):
            amt = v["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"]
            total += int(amt or 0)
        return total
    except Exception:
        return 0

# jupiter.py

def confirm_txn(client: Client, sig_str: str) -> bool:
    tries = 0
    while tries < CONFIRM_RETRIES:
        try:
            # ✅ clave: incluir historial
            res = client.get_signature_statuses(
                [Signature.from_string(sig_str)],
                search_transaction_history=True
            )
            st = res.value[0] if res.value else None
            if st is not None:
                if st.err is None:
                    print(f"[✓] Confirmado: {sig_str}")
                    return True
                else:
                    print(f"[x] Error en transacción: {st.err}")
                    return False
            else:
                # 🔁 Fallback: get_transaction
                try:
                    tx = client.get_transaction(
                        Signature.from_string(sig_str),
                        encoding="json",
                        commitment="confirmed",
                        max_supported_transaction_version=0
                    )
                    if tx.value is not None:
                        import json as _json
                        err = _json.loads(tx.value.transaction.meta.to_json()).get("err")
                        if err is None:
                            print(f"[✓] Confirmado (fallback): {sig_str}")
                            return True
                        else:
                            print(f"[x] Error en transacción (fallback): {err}")
                            return False
                except Exception:
                    pass
        except Exception:
            pass
        tries += 1
        print(f"[.] Esperando confirmación... ({tries}/{CONFIRM_RETRIES})")
        time.sleep(CONFIRM_SLEEP_SECS)
    print("[x] Tiempo de confirmación agotado.")
    return False

# -------- Jupiter v6 ----------
def get_quote(input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict[str, Any]]:
    try:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippage_bps,
            # Direct routes = True por defecto (más rápidas); permitir multi-hop por env.
            "onlyDirectRoutes": "true" if ALLOW_MULTI_HOPS else "true",
        }
        r = _http_with_retries("GET", JUP_QUOTE_URL, params=params, headers={"Accept": "application/json"})
        return r.json()
    except requests.RequestException as e:
        print(f"[!] Error en get_quote: {e}")
        return None

def get_swap(user_public_key: str, quote_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        payload = {
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "useSharedAccounts": True,
            "quoteResponse": quote_response,
        }
        r = _http_with_retries(
            "POST", JUP_SWAP_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            data=json.dumps(payload),
        )
        resp = r.json()
        if "swapTransaction" not in resp:
            print(f"[!] Respuesta inesperada de /swap: {resp}")
            return None
        return resp
    except requests.RequestException as e:
        print(f"[!] Error en get_swap: {e}")
        return None

def _sign_swap_tx(payer: Keypair, swap_tx_b64: str) -> VersionedTransaction:
    raw_tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_b64))
    msg_bytes = to_bytes_versioned(raw_tx.message)
    signature = payer.sign_message(msg_bytes)
    return VersionedTransaction.populate(raw_tx.message, [signature])

def swap(input_mint: str, output_mint: str, amount_lamports: int, slippage_bps: int):
    if amount_lamports <= 0:
        print("[x] Monto inválido.")
        return {"ok": False, "status": "fail"}

    client, payer = get_client_and_payer()
    user_pk = str(payer.pubkey())

    quote = get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
    if not quote:
        print("[x] No se obtuvo quote.")
        return {"ok": False, "status": "fail"}

    swap_tx = get_swap(user_pk, quote)
    if not swap_tx:
        print("[x] No se obtuvo swapTransaction.")
        return {"ok": False, "status": "fail"}

    signed_tx = _sign_swap_tx(payer, swap_tx["swapTransaction"])

    opts = TxOpts(skip_preflight=SKIP_PREFLIGHT, preflight_commitment=Processed)
    sig_str = client.send_raw_transaction(bytes(signed_tx), opts=opts).value
    print(f"[→] Firma: {sig_str}")

    return {"ok": True, "status": "ok", "sig": sig_str}
    # <-- devolvemos el sig!
def buy(token_address: str, sol_in: Union[int, float], slippage: int = 5):
    if sol_in <= 0:
        print("[x] SOL inválido.")
        return {"ok": False, "status": "fail"}
    if slippage < 0:
        slippage = 0
    amount_lamports = int(sol_in * 1e9)
    slippage_bps = int(slippage) * 100
    return swap(SOL_MINT, token_address, amount_lamports, slippage_bps)

def sell(token_address: str, percentage: int = 100, slippage: int = 5):
    if not (1 <= int(percentage) <= 100):
        print("[x] percentage debe estar entre 1 y 100.")
        return {"ok": False, "status": "fail"}
    if slippage < 0:
        slippage = 0

    _, payer = get_client_and_payer()
    owner = str(payer.pubkey())
    bal = get_token_balance_lamports(token_address, owner)
    if bal <= 0:
        print("[x] Sin balance para vender.")
        return {"ok": False, "status": "fail"}

    sell_amount = int(bal * (int(percentage) / 100.0))
    if sell_amount <= 0:
        print("[x] Monto a vender no positivo.")
        return {"ok": False, "status": "fail"}

    slippage_bps = int(slippage) * 100
    return swap(token_address, SOL_MINT, sell_amount, slippage_bps)
if __name__ == "__main__":
    # CLI opcional (compat)
    def usage():
        print("python jupiter.py buy <MINT> <SOL> [SLIPPAGE%]\npython jupiter.py sell <MINT> <PCT> [SLIPPAGE%]")
    if len(sys.argv) < 2 or sys.argv[1] not in {"buy", "sell"}:
        usage(); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "buy":
        mint = sys.argv[2]; sol = float(sys.argv[3]); sl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        sys.exit(0 if buy(mint, sol, sl) else 2)
    else:
        mint = sys.argv[2]; pct = int(sys.argv[3]); sl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        sys.exit(0 if sell(mint, pct, sl) else 2)
