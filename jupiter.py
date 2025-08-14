#!/usr/bin/env python3
# Jupiter Swap (v6) – CLI: buy/sell (Versioned TX + ALTs via /swap) + Jito bundle opcional

import os
import sys
import time
import json
import base64
from typing import Any, Dict, Optional, Union

import requests
from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import TxOpts

from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction
from solders.signature import Signature

# ===== JITO CONFIG / HELPER =====

USE_JITO        = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
JITO_BE_URL     = os.getenv("JITO_BLOCK_ENGINE_URL", "https://ny.mainnet.block-engine.jito.wtf").rstrip("/")
JITO_TIP_VALUE  = float(os.getenv("JITO_TIP_VALUE", "0.004"))
JITO_TIP_VALUE  = max(0.0, min(JITO_TIP_VALUE, 0.1))
JITO_TIP_ACC    = os.getenv("TIP_ACCOUNT", "JitoTip1111111111111111111111111111111111")
JITO_API_KEY    = os.getenv("JITO_API_KEY")

import logging, time as _time
from typing import Optional as _Optional
import requests as _requests
from solana.transaction import Transaction as _LegacyTx
from solana.system_program import transfer as _transfer, TransferParams as _TP
from solana.keypair import Keypair as _SolKeypair
from solders.pubkey import Pubkey as _Pubkey

def _jito_send_versioned_with_tip(
    signed_vtx: VersionedTransaction,
    payer_base58_privkey: str,
    rpc_url: str,
    tip_sol: float = JITO_TIP_VALUE,
    tip_account: str = JITO_TIP_ACC,
    wait_timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Empaqueta [signed_vtx, tip_tx] y envía a Jito. Devuelve {"bundle_id", "confirmed", "error"}.
    """
    try:
        # 1) blockhash de la TX versionada (tiene que ser el MISMO para la tip)
        recent_bh = str(signed_vtx.message.recent_blockhash)

        # 2) armar tip TX legacy con el mismo blockhash
        rpc = Client(rpc_url)
        payer_solders = Keypair.from_base58_string(payer_base58_privkey)
        payer_solana  = _SolKeypair.from_secret_key(payer_solders.to_bytes())
        tip_lamports  = int(tip_sol * 1_000_000_000)
        tip_pubkey    = _Pubkey.from_string(tip_account)

        tip_tx = _LegacyTx(recent_blockhash=recent_bh, fee_payer=payer_solana.public_key)
        tip_tx.add(_transfer(_TP(
            from_pubkey=payer_solana.public_key,
            to_pubkey=tip_pubkey,
            lamports=tip_lamports
        )))
        tip_tx.sign(payer_solana)

        main_b64 = base64.b64encode(bytes(signed_vtx)).decode()
        tip_b64  = base64.b64encode(tip_tx.serialize()).decode()

        # 3) cliente Jito ultra simple
        headers = {"Accept":"application/json","Content-Type":"application/json"}
        if JITO_API_KEY:
            headers["X-API-KEY"] = JITO_API_KEY

        def _post(method: str, params):
            payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
            return _requests.post(f"{JITO_BE_URL}/api/v1/bundles", json=payload, headers=headers, timeout=10)

        r = _post("sendBundle", [[main_b64, tip_b64]])
        if not r.ok:
            return {"bundle_id": None, "confirmed": False, "error": f"sendBundle {r.status_code}: {r.text}"}
        bundle_id = r.json().get("result")
        if not bundle_id:
            return {"bundle_id": None, "confirmed": False, "error": "sendBundle result vacío"}

        started = _time.time()
        while _time.time() - started < wait_timeout_s:
            rr = _post("getBundleStatuses", [[bundle_id]])
            if rr.ok:
                j = rr.json().get("result", {})
                arr = j.get("value", []) if isinstance(j, dict) else (j or [])
                st = arr[0] if arr else None
                if st:
                    conf = st.get("confirmation_status") or st.get("state")
                    err  = st.get("bundle_error") or st.get("error") or (st.get("status", {}) or {}).get("err")
                    if conf in ("confirmed", "finalized"):
                        return {"bundle_id": bundle_id, "confirmed": True, "error": None}
                    if err:
                        return {"bundle_id": bundle_id, "confirmed": False, "error": str(err)}
            _time.sleep(0.4)

        return {"bundle_id": bundle_id, "confirmed": False, "error": "timeout"}

    except Exception as e:
        return {"bundle_id": None, "confirmed": False, "error": f"exception: {e}"}

# ============================
# CONFIG
# ============================

PRIV_KEY = os.getenv("PRIV_KEY", "base_58_priv_key_str")
RPC = os.getenv("RPC", "rpc_url_here")

SOL_MINT = "So11111111111111111111111111111111111111112"
HTTP_TIMEOUT = 25
CONFIRM_RETRIES = 40
CONFIRM_SLEEP_SECS = 1.5

JUP_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUP_SWAP_URL  = "https://quote-api.jup.ag/v6/swap"

# ============================
# INIT
# ============================

def _require_config():
    if not PRIV_KEY or PRIV_KEY == "base_58_priv_key_str":
        raise RuntimeError("PRIV_KEY vacío. Seteá PRIV_KEY (Base58) o editá el archivo.")
    if not RPC or RPC == "rpc_url_here":
        raise RuntimeError("RPC vacío. Seteá RPC (URL) o editá el archivo.")

def get_client_and_payer() -> tuple[Client, Keypair]:
    _require_config()
    return Client(RPC), Keypair.from_base58_string(PRIV_KEY)

# ============================
# HELPERS
# ============================

def find_data(data, field: str):
    if isinstance(data, dict):
        if field in data:
            return data[field]
        for v in data.values():
            r = find_data(v, field)
            if r is not None:
                return r
    elif isinstance(data, list):
        for it in data:
            r = find_data(it, field)
            if r is not None:
                return r
    return None

def get_token_balance_lamports(mint_str: str, owner_pubkey: str) -> int:
    payload = {
        "id": 1, "jsonrpc": "2.0", "method": "getTokenAccountsByOwner",
        "params": [owner_pubkey, {"mint": mint_str}, {"encoding": "jsonParsed"}],
    }
    headers = {"accept": "application/json", "content-type": "application/json"}
    try:
        r = requests.post(RPC, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        amount = find_data(r.json(), "amount")
        return int(amount) if amount is not None else 0
    except Exception as e:
        print(f"[!] Error al obtener balance token: {e}")
        return 0

def confirm_txn(client: Client, sig_str: str) -> bool:
    tries = 0
    while tries < CONFIRM_RETRIES:
        try:
            res = client.get_signature_statuses([Signature.from_string(sig_str)])
            st = res.value[0] if res.value else None
            if st is not None:
                if st.err is None:
                    print(f"[✓] Confirmado: {sig_str}")
                    return True
                else:
                    print(f"[x] Error en transacción: {st.err}")
                    return False
        except Exception:
            pass
        tries += 1
        print(f"[.] Esperando confirmación... ({tries}/{CONFIRM_RETRIES})")
        time.sleep(CONFIRM_SLEEP_SECS)
    print("[x] Tiempo de confirmación agotado.")
    return False

# ============================
# JUPITER v6
# ============================

def get_quote(input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict[str, Any]]:
    try:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": "true",
        }
        r = requests.get(JUP_QUOTE_URL, params=params, headers={"Accept": "application/json"}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
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
            # "prioritizationFeeLamports": 10000,  # opcional
        }
        r = requests.post(JUP_SWAP_URL, headers={"Content-Type": "application/json", "Accept": "application/json"},
                          data=json.dumps(payload), timeout=HTTP_TIMEOUT)
        r.raise_for_status()
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

def swap(input_mint: str, output_mint: str, amount_lamports: int, slippage_bps: int) -> bool:
    client, payer = get_client_and_payer()
    user_pk = str(payer.pubkey())

    quote = get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
    if not quote:
        print("[x] No se obtuvo quote.")
        return False
    print("[i] Quote OK.")

    swap_tx = get_swap(user_pk, quote)
    if not swap_tx:
        print("[x] No se obtuvo swapTransaction.")
        return False

    try:
        # 1) Firmar TX versionada de Jupiter
        signed_tx = _sign_swap_tx(payer, swap_tx["swapTransaction"])

        if USE_JITO:
            # 2) Enviar por Jito bundle
            res = _jito_send_versioned_with_tip(
                signed_vtx=signed_tx,
                payer_base58_privkey=PRIV_KEY,
                rpc_url=RPC,
                tip_sol=JITO_TIP_VALUE,
                tip_account=JITO_TIP_ACC,
            )
            print("Jito result:", res)

            # 2a) Si expiró el blockhash en el BE, reintenta UNA vez con un /swap nuevo
            err = (res.get("error") or "").lower()
            if (not res.get("confirmed")) and ("blockhash" in err or "expired" in err):
                print("[i] Reintentando con blockhash fresco desde /swap ...")
                # pedir nueva swapTransaction y re-firmar
                swap_tx2 = get_swap(user_pk, quote)  # podrías volver a pedir quote, pero suele valer el mismo
                if not swap_tx2:
                    print("[x] Segundo intento: no se obtuvo swapTransaction.")
                    return False
                signed_tx2 = _sign_swap_tx(payer, swap_tx2["swapTransaction"])
                res2 = _jito_send_versioned_with_tip(
                    signed_vtx=signed_tx2,
                    payer_base58_privkey=PRIV_KEY,
                    rpc_url=RPC,
                    tip_sol=JITO_TIP_VALUE,
                    tip_account=JITO_TIP_ACC,
                )
                print("Jito result (retry):", res2)
                return bool(res2.get("confirmed"))

            return bool(res.get("confirmed"))

        else:
            # 3) Fallback: envío por RPC normal
            opts = TxOpts(skip_preflight=False, preflight_commitment=Processed)
            sig_str = client.send_raw_transaction(bytes(signed_tx), opts=opts).value
            print(f"[→] Firma: {sig_str}")
            ok = confirm_txn(client, sig_str)
            print(f"[✓] Resultado: {ok}")
            return ok

    except Exception as e:
        print(f"[x] Fallo al firmar/enviar: {e}")
        return False

# ============================
# WRAPPERS
# ============================

def buy(token_address: str, sol_in: Union[int, float], slippage: int = 5) -> bool:
    amount_lamports = int(sol_in * 1e9)
    slippage_bps = slippage * 100
    return swap(SOL_MINT, token_address, amount_lamports, slippage_bps)

def sell(token_address: str, percentage: int = 100, slippage: int = 5) -> bool:
    if not (1 <= percentage <= 100):
        print("[x] percentage debe estar entre 1 y 100.")
        return False
    _, payer = get_client_and_payer()
    owner = str(payer.pubkey())
    bal = get_token_balance_lamports(token_address, owner)
    print(f"[i] Balance del token: {bal}")
    if bal <= 0:
        print("[x] Sin balance para vender.")
        return False
    sell_amount = int(bal * (percentage / 100.0))
    slippage_bps = slippage * 100
    return swap(token_address, SOL_MINT, sell_amount, slippage_bps)

# ============================
# CLI
# ============================

def print_usage():
    print("""
Uso:
  python main.py buy <TOKEN_MINT> <SOL_AMOUNT> [SLIPPAGE_%]
  python main.py sell <TOKEN_MINT> <PERCENTAGE> [SLIPPAGE_%]

Ejemplos:
  python main.py buy 7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr 0.1 10
  python main.py sell 7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr 100 10

Config:
  export PRIV_KEY="tu_clave_privada_base58"
  export RPC="https://tu-rpc.solana.example"
  # Opcional Jito:
  export USE_JITO=true
  export JITO_TIP_VALUE=0.004
  export TIP_ACCOUNT="JitoTip1111111111111111111111111111111111"
  export JITO_BLOCK_ENGINE_URL="https://ny.mainnet.block-engine.jito.wtf"
  # export JITO_API_KEY="si tu BE lo requiere"
""")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"buy", "sell"}:
        print_usage(); sys.exit(1)

    try:
        _require_config()
    except Exception as e:
        print(f"[x] Config error: {e}"); sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "buy":
        if len(sys.argv) < 4:
            print_usage(); sys.exit(1)
        token = sys.argv[2]
        sol_amount = float(sys.argv[3])
        slippage = int(sys.argv[4]) if len(sys.argv) >= 5 else 5
        ok = buy(token, sol_amount, slippage)
        sys.exit(0 if ok else 2)

    if cmd == "sell":
        if len(sys.argv) < 4:
            print_usage(); sys.exit(1)
        token = sys.argv[2]
        percentage = int(sys.argv[3])
        slippage = int(sys.argv[4]) if len(sys.argv) >= 5 else 5
        ok = sell(token, percentage, slippage)
        sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
