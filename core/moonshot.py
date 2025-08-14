#!/usr/bin/env python3
# Moonshot — buy/sell con Versioned TX; envío por Jito opcional vía core/jito_sender

from __future__ import annotations
import os
import time
import json
import struct
from typing import Any, Dict, Optional, Tuple

import requests
from solana.rpc.api import Client
from solana.rpc.types import TokenAccountOpts, TxOpts
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import Instruction, AccountMeta as SoldersAccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from spl.token.instructions import create_associated_token_account, get_associated_token_address

# -------- Config ----------
PRIV_KEY = os.getenv("PRIV_KEY", "")
RPC      = os.getenv("RPC", "")
USE_JITO = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")

if not PRIV_KEY or not RPC:
    raise RuntimeError("PRIV_KEY / RPC no configurados")

client = Client(RPC)
payer_keypair = Keypair.from_base58_string(PRIV_KEY)

# -------- Jito sender (modular) ----------
try:
    from core.jito_sender import send_v0_with_jito_bundle
except Exception:
    send_v0_with_jito_bundle = None

# -------- Constantes/direcciones ----------
DEX_FEE = Pubkey.from_string("3udvfL24waJcLhskRAsStNMoNUvtyXdxrWQz4hgi953N")
HELIO_FEE = Pubkey.from_string("5K5RtTWzzLp4P8Npi84ocf7F1vBsAu29N1irG4iiUnzt")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_ACC_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
CONFIG_ACCOUNT = Pubkey.from_string("36Eru7v11oU5Pfrojyn5oY3nETA1a1iqsw2WUu6afkM9")
MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")

UNIT_PRICE  = int(os.getenv("UNIT_PRICE",  "1000000"))
UNIT_BUDGET = int(os.getenv("UNIT_BUDGET", "100000"))

# -------- Utils ----------
def derive_curve_accounts(mint: Pubkey):
    try:
        curve_account, _ = Pubkey.find_program_address([b"token", bytes(mint)], MOONSHOT_PROGRAM)
        curve_token_account = get_associated_token_address(curve_account, mint)
        return curve_account, curve_token_account
    except Exception:
        return None, None

def get_token_balance(mint_str: str) -> float:
    try:
        owner = str(payer_keypair.pubkey())
        payload = {
            "id": 1, "jsonrpc": "2.0", "method": "getTokenAccountsByOwner",
            "params": [owner, {"mint": mint_str}, {"encoding": "jsonParsed"}],
        }
        r = requests.post(RPC, json=payload, headers={"accept":"application/json","content-type":"application/json"}, timeout=15)
        r.raise_for_status()
        j = r.json()
        total = 0.0
        for v in (j.get("result", {}).get("value", []) or []):
            ui = v["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"] or 0
            total += float(ui)
        return total
    except Exception:
        return 0.0

def confirm_txn(txn_sig, max_retries=20, retry_interval=3):
    retries = 0
    if isinstance(txn_sig, str):
        txn_sig = Signature.from_string(txn_sig)
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(
                txn_sig, encoding="json", commitment="confirmed", max_supported_transaction_version=0
            )
            if txn_res.value is None:
                retries += 1; time.sleep(retry_interval); continue
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            return txn_json.get("err") is None
        except Exception:
            pass
        retries += 1
        time.sleep(retry_interval)
    return False

def get_token_data(token_address: str):
    url = f"https://api.moonshot.cc/token/v1/solana/{token_address}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[moonshot api] {e}")
        return None

# -------- Core (buy/sell) ----------
def _compile_and_send(instructions) -> bool:
    compiled_message = MessageV0.try_compile(
        payer_keypair.pubkey(), instructions, [], client.get_latest_blockhash().value.blockhash
    )
    vtx = VersionedTransaction(compiled_message, [payer_keypair])

    if USE_JITO and send_v0_with_jito_bundle is not None:
        res = send_v0_with_jito_bundle(vtx, payer_keypair)
        print("Jito result:", res)
        return bool(res.get("confirmed"))

    sig = client.send_transaction(vtx, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")).value
    print("Signature:", sig)
    return confirm_txn(sig)

def buy(mint_str: str, sol_in: float = 0.01, slippage_bps: int = 500):
    try:
        slippage_bps = max(0, int(slippage_bps))

        token_data = get_token_data(mint_str)
        if not token_data:
            print("No se pudo obtener data del token."); return False

        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = float(token_data["priceNative"])
        tokens_out = float(sol_in) / token_price
        token_amount = int(tokens_out * token_decimal)
        collateral_amount = int(sol_in * sol_decimal)
        print(f"Collateral: {collateral_amount} | Tokens: {token_amount} | Slippage(bps): {slippage_bps}")

        SENDER = payer_keypair.pubkey()
        MINT = Pubkey.from_string(mint_str)

        token_account = None
        token_account_ix = None
        try:
            account_data = client.get_token_accounts_by_owner(
                SENDER, TokenAccountOpts(mint=MINT)
            )
            token_account = account_data.value[0].pubkey
        except Exception:
            token_account = get_associated_token_address(SENDER, MINT)
            token_account_ix = create_associated_token_account(SENDER, SENDER, MINT)

        curve_acc, curve_token_acc = derive_curve_accounts(MINT)
        if not curve_acc or not curve_token_acc:
            print("No se pudo derivar curve accounts."); return False

        keys = [
            SoldersAccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            SoldersAccountMeta(pubkey=token_account, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=curve_acc, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=curve_token_acc, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=MINT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),     # <- no writable
            SoldersAccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack("<Q", token_amount))
        data.extend(struct.pack("<Q", collateral_amount))
        data.extend(struct.pack("<B", 0))
        data.extend(struct.pack("<Q", slippage_bps))
        swap_ix = Instruction(MOONSHOT_PROGRAM, bytes(data), keys)

        instructions = [set_compute_unit_price(UNIT_PRICE), set_compute_unit_limit(UNIT_BUDGET)]
        if token_account_ix:
            instructions.append(token_account_ix)
        instructions.append(swap_ix)

        return _compile_and_send(instructions)

    except Exception as e:
        print("[buy] error:", e)
        return False

def sell(mint_str: str, percentage: int = 100, slippage_bps: int = 500):
    try:
        slippage_bps = max(0, int(slippage_bps))

        current_bal = get_token_balance(mint_str)
        if current_bal <= 0:
            print("Balance 0; no hay nada que vender."); return False

        if not (1 <= int(percentage) <= 100):
            print("percentage inválido"); return False

        token_data = get_token_data(mint_str)
        if not token_data:
            print("No se pudo obtener data del token."); return False

        token_balance = current_bal * (float(percentage) / 100.0)

        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = float(token_data["priceNative"])
        token_value = float(token_balance) * token_price
        collateral_amount = int(token_value * sol_decimal)
        token_amount = int(float(token_balance) * token_decimal)
        print(f"Collateral: {collateral_amount} | Tokens: {token_amount} | Slippage(bps): {slippage_bps}")

        MINT = Pubkey.from_string(mint_str)
        curve_acc, curve_token_acc = derive_curve_accounts(MINT)
        if not curve_acc or not curve_token_acc:
            print("No se pudo derivar curve accounts."); return False

        SENDER = payer_keypair.pubkey()
        sender_token_acc = get_associated_token_address(SENDER, MINT)

        keys = [
            SoldersAccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            SoldersAccountMeta(pubkey=sender_token_acc, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=curve_acc, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=curve_token_acc, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=MINT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),     # <- no writable
            SoldersAccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack("<Q", token_amount))
        data.extend(struct.pack("<Q", collateral_amount))
        data.extend(struct.pack("<B", 0))
        data.extend(struct.pack("<Q", slippage_bps))
        swap_ix = Instruction(MOONSHOT_PROGRAM, bytes(data), keys)

        instructions = [set_compute_unit_price(UNIT_PRICE), set_compute_unit_limit(UNIT_BUDGET), swap_ix]
        return _compile_and_send(instructions)

    except Exception as e:
        print("[sell] error:", e)
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Moonshot buy/sell via Python")
    sub = parser.add_subparsers(dest="cmd", required=False)
    p_buy = sub.add_parser("buy");  p_buy.add_argument("--mint", required=True); p_buy.add_argument("--sol", type=float, default=0.01); p_buy.add_argument("--slippage_bps", type=int, default=500)
    p_sell = sub.add_parser("sell"); p_sell.add_argument("--mint", required=True); p_sell.add_argument("--pct", type=int, default=100); p_sell.add_argument("--slippage_bps", type=int, default=500)
    args = parser.parse_args()
    if args.cmd == "buy":
        buy(mint_str=args.mint, sol_in=args.sol, slippage_bps=args.slippage_bps)
    elif args.cmd == "sell":
        sell(mint_str=args.mint, percentage=args.pct, slippage_bps=args.slippage_bps)
    else:
        parser.print_help()
