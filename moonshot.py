# main.py
# Swap on Moonswap with Python — "moonshot_py" + Jito bundle opcional
# Actualizado para enviar Versioned TX + tip como bundle a Jito

from __future__ import annotations
import os
import time
import json
import struct
import requests

# =========================
# Configuración básica
# =========================
from solana.rpc.api import Client
from solders.keypair import Keypair  # type: ignore

# Lee de entorno si están definidos; si no, usa los literales (por compat)
PRIV_KEY = os.getenv("PRIV_KEY", "") or ""   # Base58 string del private key
RPC      = os.getenv("RPC", "") or ""        # URL de tu RPC (Helius o QuickNode recomendado)

if not PRIV_KEY or not RPC:
    raise RuntimeError("Config incompleta: setea PRIV_KEY y RPC (env o literal).")

client = Client(RPC)
payer_keypair = Keypair.from_base58_string(PRIV_KEY)

# =========================
# Flags/Config Jito (env)
# =========================
USE_JITO        = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
JITO_BE_URL     = os.getenv("JITO_BLOCK_ENGINE_URL", "https://ny.mainnet.block-engine.jito.wtf").rstrip("/")
JITO_TIP_SOL    = max(0.0, min(float(os.getenv("JITO_TIP_VALUE", "0.004")), 0.1))
JITO_TIP_ACC    = os.getenv("TIP_ACCOUNT", "JitoTip1111111111111111111111111111111111")
JITO_API_KEY    = os.getenv("JITO_API_KEY")

# =========================
# Constantes y direcciones
# =========================
from solders.pubkey import Pubkey  # type: ignore

DEX_FEE = Pubkey.from_string("3udvfL24waJcLhskRAsStNMoNUvtyXdxrWQz4hgi953N")
HELIO_FEE = Pubkey.from_string("5K5RtTWzzLp4P8Npi84ocf7F1vBsAu29N1irG4iiUnzt")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_ACC_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
CONFIG_ACCOUNT = Pubkey.from_string("36Eru7v11oU5Pfrojyn5oY3nETA1a1iqsw2WUu6afkM9")
MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")

# Compute Budget (puedes ajustar)
UNIT_PRICE = 1_000_000
UNIT_BUDGET = 100_000

# =========================
# Imports de Solana/SPL
# =========================
from solana.rpc.types import TokenAccountOpts, TxOpts
from solana.transaction import TransactionInstruction  # compat (no se usa directamente)
from solders.compute_budget import (  # type: ignore
    set_compute_unit_limit,
    set_compute_unit_price,
)
from solders.instruction import Instruction  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from spl.token.instructions import (
    create_associated_token_account,
    get_associated_token_address,
)
from solders.signature import Signature  # type: ignore

# =========================
# Helper Jito: bundle [versioned_tx, tip_tx]
# =========================
import base64 as _b64
import requests as _rq
from typing import Any, Dict as _Dict
from solana.transaction import Transaction as _LegacyTx
from solana.system_program import transfer as _transfer, TransferParams as _TP
from solana.keypair import Keypair as _SolKeypair
from solders.pubkey import Pubkey as _SPub

def _jito_send_versioned_with_tip(
    signed_vtx: VersionedTransaction,
    payer_base58_privkey: str,
    rpc_url: str,
    tip_sol: float = JITO_TIP_SOL,
    tip_account: str = JITO_TIP_ACC,
    wait_timeout_s: float = 12.0,
) -> _Dict[str, Any]:
    """
    Empaqueta [signed_vtx, tip_tx] y envía a Jito.
    Devuelve {"bundle_id": str|None, "confirmed": bool, "error": str|None}
    """
    try:
        recent_bh = str(signed_vtx.message.recent_blockhash)

        # Tip TX legacy con MISMO blockhash
        rpc = Client(rpc_url)
        payer_solders = Keypair.from_base58_string(payer_base58_privkey)
        payer_solana  = _SolKeypair.from_secret_key(payer_solders.to_bytes())
        tip_lamports  = int(max(0.0, min(tip_sol, 0.1)) * 1_000_000_000)
        tip_pubkey    = _SPub.from_string(tip_account)

        tip_tx = _LegacyTx(recent_blockhash=recent_bh, fee_payer=payer_solana.public_key)
        tip_tx.add(_transfer(_TP(
            from_pubkey=payer_solana.public_key,
            to_pubkey=tip_pubkey,
            lamports=tip_lamports
        )))
        tip_tx.sign(payer_solana)

        main_b64 = _b64.b64encode(bytes(signed_vtx)).decode()
        tip_b64  = _b64.b64encode(tip_tx.serialize()).decode()

        headers = {"Accept":"application/json","Content-Type":"application/json"}
        if JITO_API_KEY:
            headers["X-API-KEY"] = JITO_API_KEY

        def _post(method: str, params):
            payload = {"jsonrpc":"2.0","id":1,"method":method,"params":params}
            return _rq.post(f"{JITO_BE_URL}/api/v1/bundles", json=payload, headers=headers, timeout=10)

        r = _post("sendBundle", [[main_b64, tip_b64]])
        if not r.ok:
            return {"bundle_id": None, "confirmed": False, "error": f"sendBundle {r.status_code}: {r.text}"}
        bundle_id = r.json().get("result")
        if not bundle_id:
            return {"bundle_id": None, "confirmed": False, "error": "sendBundle sin result"}

        start = time.time()
        while time.time() - start < wait_timeout_s:
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
            time.sleep(0.4)

        return {"bundle_id": bundle_id, "confirmed": False, "error": "timeout"}

    except Exception as e:
        return {"bundle_id": None, "confirmed": False, "error": f"exception: {e}"}

# =========================
# Utils
# =========================
def derive_curve_accounts(mint: Pubkey):
    try:
        SEED = "token".encode()
        curve_account, _ = Pubkey.find_program_address(
            [SEED, bytes(mint)],
            MOONSHOT_PROGRAM,
        )
        curve_token_account = get_associated_token_address(curve_account, mint)
        return curve_account, curve_token_account
    except Exception:
        return None, None

def find_data(data, field):
    if isinstance(data, dict):
        if field in data:
            return data[field]
        for value in data.values():
            result = find_data(value, field)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_data(item, field)
            if result is not None:
                return result
    return None

def get_token_balance(mint_str: str):
    try:
        pubkey_str = str(payer_keypair.pubkey())
        headers = {"accept": "application/json", "content-type": "application/json"}
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "getTokenAccountsByOwner",
            "params": [
                pubkey_str,
                {"mint": mint_str},
                {"encoding": "jsonParsed"},
            ],
        }
        response = requests.post(RPC, json=payload, headers=headers, timeout=15)
        ui_amount = find_data(response.json(), "uiAmount")
        return float(ui_amount) if ui_amount is not None else None
    except Exception:
        return None

def confirm_txn(txn_sig, max_retries=20, retry_interval=3):
    retries = 0
    if isinstance(txn_sig, str):
        txn_sig = Signature.from_string(txn_sig)
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(
                txn_sig,
                encoding="json",
                commitment="confirmed",
                max_supported_transaction_version=0,
            )
            if txn_res.value is None:
                print("Awaiting confirmation... try count:", retries + 1)
                retries += 1
                time.sleep(retry_interval)
                continue
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            if txn_json.get("err") is None:
                print("Transaction confirmed... try count:", retries + 1)
                return True
            print("Error: Transaction not confirmed.")
            return False
        except Exception:
            print("Awaiting confirmation... try count:", retries + 1)
        retries += 1
        time.sleep(retry_interval)
    print("Max retries reached. Transaction confirmation failed.")
    return None

# =========================
# API Moonshot: datos de token
# =========================
def get_token_data(token_address: str):
    url = f"https://api.moonshot.cc/token/v1/solana/{token_address}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

# =========================
# BUY
# =========================
from solana.transaction import Transaction  # compat
from solana.transaction import AccountMeta  # compat
from solders.instruction import AccountMeta as SoldersAccountMeta  # type: ignore

def buy(mint_str: str, sol_in: float = 0.01, slippage_bps: int = 500):
    try:
        token_data = get_token_data(mint_str)
        if token_data is None:
            print("No se pudo obtener data del token.")
            return

        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data["priceNative"]
        tokens_out = float(sol_in) / float(token_price)
        token_amount = int(tokens_out * token_decimal)
        collateral_amount = int(sol_in * sol_decimal)
        print(f"Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")

        SENDER = payer_keypair.pubkey()
        MINT = Pubkey.from_string(mint_str)

        token_account = None
        token_account_instruction = None
        try:
            account_data = client.get_token_accounts_by_owner(SENDER, TokenAccountOpts(MINT))
            token_account = account_data.value[0].pubkey
        except Exception:
            token_account = get_associated_token_address(SENDER, MINT)
            token_account_instruction = create_associated_token_account(SENDER, SENDER, MINT)

        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        if CURVE_ACCOUNT is None or CURVE_TOKEN_ACCOUNT is None:
            print("No se pudo derivar curve accounts.")
            return

        SENDER_TOKEN_ACCOUNT = token_account

        keys = [
            SoldersAccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            SoldersAccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=MINT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack("<Q", token_amount))
        data.extend(struct.pack("<Q", collateral_amount))
        data.extend(struct.pack("<B", 0))
        data.extend(struct.pack("<Q", slippage_bps))
        swap_instruction = Instruction(MOONSHOT_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_price(UNIT_PRICE),
            set_compute_unit_limit(UNIT_BUDGET),
        ]
        if token_account_instruction:
            instructions.append(token_account_instruction)
        instructions.append(swap_instruction)

        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            client.get_latest_blockhash().value.blockhash,
        )
        transaction = VersionedTransaction(compiled_message, [payer_keypair])

        if USE_JITO:
            res = _jito_send_versioned_with_tip(
                signed_vtx=transaction,
                payer_base58_privkey=PRIV_KEY,
                rpc_url=RPC,
                tip_sol=JITO_TIP_SOL,
                tip_account=JITO_TIP_ACC,
            )
            print("Jito result:", res)
            if not res.get("confirmed"):
                err = (res.get("error") or "").lower()
                if "blockhash" in err or "expired" in err:
                    print("[i] Reintentando con blockhash fresco...")
                    compiled_message = MessageV0.try_compile(
                        payer_keypair.pubkey(), instructions, [], client.get_latest_blockhash().value.blockhash
                    )
                    transaction2 = VersionedTransaction(compiled_message, [payer_keypair])
                    res2 = _jito_send_versioned_with_tip(
                        signed_vtx=transaction2,
                        payer_base58_privkey=PRIV_KEY,
                        rpc_url=RPC,
                        tip_sol=JITO_TIP_SOL,
                        tip_account=JITO_TIP_ACC,
                    )
                    print("Jito result (retry):", res2)
                    return
            return
        else:
            txn_sig = client.send_transaction(
                transaction,
                opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"),
            ).value
            print(f"Transaction Signature: {txn_sig}")
            confirm = confirm_txn(txn_sig)
            print(f"Transaction Confirmation: {confirm}")

    except Exception as e:
        print(e)

# =========================
# SELL
# =========================
def sell(mint_str: str, token_balance=None, slippage_bps: int = 500):
    try:
        if token_balance is None:
            token_balance = get_token_balance(mint_str)
        print(f"Token Balance: {token_balance}")

        if not token_balance or float(token_balance) == 0.0:
            print("Balance 0; no hay nada que vender.")
            return

        token_data = get_token_data(mint_str)
        if token_data is None:
            print("No se pudo obtener data del token.")
            return

        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data["priceNative"]
        token_value = float(token_balance) * float(token_price)
        collateral_amount = int(token_value * sol_decimal)
        token_amount = int(float(token_balance) * token_decimal)
        print(f"Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")

        MINT = Pubkey.from_string(mint_str)
        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        if CURVE_ACCOUNT is None or CURVE_TOKEN_ACCOUNT is None:
            print("No se pudo derivar curve accounts.")
            return

        SENDER = payer_keypair.pubkey()
        SENDER_TOKEN_ACCOUNT = get_associated_token_address(SENDER, MINT)

        keys = [
            SoldersAccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            SoldersAccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=MINT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack("<Q", token_amount))
        data.extend(struct.pack("<Q", collateral_amount))
        data.extend(struct.pack("<B", 0))
        data.extend(struct.pack("<Q", slippage_bps))
        swap_instruction = Instruction(MOONSHOT_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_price(UNIT_PRICE),
            set_compute_unit_limit(UNIT_BUDGET),
            swap_instruction,
        ]

        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            client.get_latest_blockhash().value.blockhash,
        )
        transaction = VersionedTransaction(compiled_message, [payer_keypair])

        if USE_JITO:
            res = _jito_send_versioned_with_tip(
                signed_vtx=transaction,
                payer_base58_privkey=PRIV_KEY,
                rpc_url=RPC,
                tip_sol=JITO_TIP_SOL,
                tip_account=JITO_TIP_ACC,
            )
            print("Jito result:", res)
            if not res.get("confirmed"):
                err = (res.get("error") or "").lower()
                if "blockhash" in err or "expired" in err:
                    print("[i] Reintentando con blockhash fresco...")
                    compiled_message = MessageV0.try_compile(
                        payer_keypair.pubkey(), instructions, [], client.get_latest_blockhash().value.blockhash
                    )
                    transaction2 = VersionedTransaction(compiled_message, [payer_keypair])
                    res2 = _jito_send_versioned_with_tip(
                        signed_vtx=transaction2,
                        payer_base58_privkey=PRIV_KEY,
                        rpc_url=RPC,
                        tip_sol=JITO_TIP_SOL,
                        tip_account=JITO_TIP_ACC,
                    )
                    print("Jito result (retry):", res2)
                    return
            return
        else:
            txn_sig = client.send_transaction(
                transaction,
                opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"),
            ).value
            print(f"Transaction Signature: {txn_sig}")
            confirm = confirm_txn(txn_sig)
            print(f"Transaction Confirmation: {confirm}")

    except Exception as e:
        print(e)

# =========================
# CLI simple opcional
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Moonshot buy/sell via Python")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_buy = sub.add_parser("buy", help="Comprar token")
    p_buy.add_argument("--mint", required=True, help="Mint del token")
    p_buy.add_argument("--sol", type=float, default=0.01, help="Cantidad de SOL a usar")
    p_buy.add_argument("--slippage_bps", type=int, default=500, help="Slippage en bps")

    p_sell = sub.add_parser("sell", help="Vender token")
    p_sell.add_argument("--mint", required=True, help="Mint del token")
    p_sell.add_argument("--balance", type=float, default=None, help="Balance del token (UI)")
    p_sell.add_argument("--slippage_bps", type=int, default=500, help="Slippage en bps")

    args = parser.parse_args()

    if args.cmd == "buy":
        buy(mint_str=args.mint, sol_in=args.sol, slippage_bps=args.slippage_bps)
    elif args.cmd == "sell":
        sell(mint_str=args.mint, token_balance=args.balance, slippage_bps=args.slippage_bps)
    else:
        parser.print_help()
