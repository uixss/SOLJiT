#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pump.fun Buy/Sell Bot (Unified) + Jito bundle opcional

Comandos CLI:
  python main.py price   --mint <MINT>
  python main.py balance --mint <MINT>
  python main.py buy     --mint <MINT> --sol 0.1 --slippage 5
  python main.py sell    --mint <MINT> --pct 50  --slippage 5

Requisitos (ejemplo):
  pip install "solana>=0.30" solders construct "spl-token-instructions>=0.3"
"""

import os
import sys
import base64
import json
import time
import struct
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any, Dict

from construct import Flag, Int64ul, Padding, Struct
from solders.pubkey import Pubkey  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.instruction import Instruction  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore

from solana.rpc.api import Client
from solana.rpc.commitment import Processed, Confirmed
from solana.rpc.types import TokenAccountOpts, TxOpts
from solana.transaction import AccountMeta, Transaction as LegacyTx
from solana.system_program import transfer as sys_transfer, TransferParams as SysTP
from solana.publickey import PublicKey as SolPub
from solana.keypair import Keypair as SolKeypair

from spl.token.instructions import (
    CloseAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
)
from spl.token.client import Token

# ====================
# Configuración
# ====================

RPC_ENDPOINT     = os.getenv("RPC_ENDPOINT", "https://rpc_url_here")     # <-- Cambiar
BASE58_PRIVKEY   = os.getenv("BASE58_PRIVKEY", "base58_priv_str_here")   # <-- Cambiar

# Presupuestos de compute
UNIT_BUDGET = int(os.getenv("UNIT_BUDGET", "100000"))   # 100_000
UNIT_PRICE  = int(os.getenv("UNIT_PRICE",  "1000000"))  # 1_000_000 micro-lamports/CU

# Jito (opcional)
USE_JITO          = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
JITO_BE_URL       = os.getenv("JITO_BLOCK_ENGINE_URL", "https://ny.mainnet.block-engine.jito.wtf").rstrip("/")
JITO_TIP_VALUE    = max(0.0, min(float(os.getenv("JITO_TIP_VALUE", "0.004")), 0.1))  # 0..0.1 SOL
JITO_TIP_ACCOUNT  = os.getenv("TIP_ACCOUNT", "JitoTip1111111111111111111111111111111111")
JITO_API_KEY      = os.getenv("JITO_API_KEY")  # opcional

client = Client(RPC_ENDPOINT)
payer_keypair = Keypair.from_base58_string(BASE58_PRIVKEY)

LAMPORTS_PER_SOL = 10**9
TOKEN_DECIMALS   = 10**6  # Pump.fun suele usar 6 decimales

# ====================
# Constantes (Pump.fun)
# ====================

GLOBAL            = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
FEE_RECIPIENT     = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
SYSTEM_PROGRAM    = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM     = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_PROG  = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
RENT              = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
EVENT_AUTHORITY   = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
PUMP_FUN_PROGRAM  = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
WSOL              = Pubkey.from_string("So11111111111111111111111111111111111111112")

# ====================
# Dataclass
# ====================

@dataclass
class CoinData:
    mint: Pubkey
    bonding_curve: Pubkey
    associated_bonding_curve: Pubkey
    virtual_token_reserves: int
    virtual_sol_reserves: int
    token_total_supply: int
    complete: bool

# ====================
# Helper Jito: bundle [versioned_tx, tip_tx]
# ====================

import requests as _rq
import base64 as _b64

def _jito_send_versioned_with_tip(
    signed_vtx: VersionedTransaction,
    payer_solders_kp: Keypair,
    rpc_url: str,
    tip_sol: float = JITO_TIP_VALUE,
    tip_account: str = JITO_TIP_ACCOUNT,
    wait_timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Empaqueta [signed_vtx(versioned), tip_tx(legacy)] con mismo blockhash y lo envía a Jito.
    Devuelve {"bundle_id": str|None, "confirmed": bool, "error": str|None}
    """
    try:
        # Blockhash reciente de la TX versionada
        recent_bh = str(signed_vtx.message.recent_blockhash)

        # Convertimos a solana-py Keypair para firmar la legacy tip TX
        payer_solana = SolKeypair.from_secret_key(payer_solders_kp.to_bytes())
        tip_lamports = int(max(0.0, min(tip_sol, 0.1)) * LAMPORTS_PER_SOL)
        tip_pubkey   = SolPub(tip_account)

        # Tip TX legacy con el MISMO blockhash
        tip_tx = LegacyTx(recent_blockhash=recent_bh, fee_payer=payer_solana.public_key)
        tip_tx.add(sys_transfer(SysTP(
            from_pubkey=payer_solana.public_key,
            to_pubkey=tip_pubkey,
            lamports=tip_lamports,
        )))
        tip_tx.sign(payer_solana)

        main_b64 = _b64.b64encode(bytes(signed_vtx)).decode()
        tip_b64  = _b64.b64encode(tip_tx.serialize()).decode()

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if JITO_API_KEY:
            headers["X-API-KEY"] = JITO_API_KEY

        def _post(method: str, params):
            payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
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

# ====================
# Helpers on-chain
# ====================

def _decode_account_data(value_data: Union[bytes, bytearray, list, tuple]) -> bytes:
    if isinstance(value_data, (bytes, bytearray)):
        return bytes(value_data)
    if isinstance(value_data, (list, tuple)) and len(value_data) >= 1:
        return base64.b64decode(value_data[0])
    raise ValueError("Formato de data de cuenta no reconocido")

def get_virtual_reserves(bonding_curve: Pubkey):
    bonding_curve_struct = Struct(
        Padding(8),
        "virtualTokenReserves" / Int64ul,
        "virtualSolReserves" / Int64ul,
        "realTokenReserves" / Int64ul,
        "realSolReserves" / Int64ul,
        "tokenTotalSupply" / Int64ul,
        "complete" / Flag,
    )
    try:
        account_info = client.get_account_info(bonding_curve)
        if account_info.value is None:
            return None
        raw = _decode_account_data(account_info.value.data)
        return bonding_curve_struct.parse(raw)
    except Exception as e:
        print(f"[get_virtual_reserves] Error: {e}")
        return None

def derive_bonding_curve_accounts(mint_str: str) -> Tuple[Optional[Pubkey], Optional[Pubkey]]:
    try:
        mint = Pubkey.from_string(mint_str)
        bonding_curve, _ = Pubkey.find_program_address(
            [b"bonding-curve", bytes(mint)],
            PUMP_FUN_PROGRAM,
        )
        associated_bonding_curve = get_associated_token_address(bonding_curve, mint)
        return bonding_curve, associated_bonding_curve
    except Exception as e:
        print(f"[derive_bonding_curve_accounts] Error: {e}")
        return None, None

def get_coin_data(mint_str: str) -> Optional[CoinData]:
    bonding_curve, associated_bonding_curve = derive_bonding_curve_accounts(mint_str)
    if not bonding_curve or not associated_bonding_curve:
        print("Failed to derive bonding curve accounts.")
        return None

    virtual = get_virtual_reserves(bonding_curve)
    if not virtual:
        print("Failed to fetch virtual reserves.")
        return None

    try:
        return CoinData(
            mint=Pubkey.from_string(mint_str),
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            virtual_token_reserves=int(virtual.virtualTokenReserves),
            virtual_sol_reserves=int(virtual.virtualSolReserves),
            token_total_supply=int(virtual.tokenTotalSupply),
            complete=bool(virtual.complete),
        )
    except Exception as e:
        print(f"[get_coin_data] Error: {e}")
        return None

# ====================
# AMM (x*y = k)
# ====================

def sol_for_tokens(sol_spent: float, sol_reserves: float, token_reserves: float) -> float:
    new_sol_reserves = sol_reserves + sol_spent
    new_token_reserves = (sol_reserves * token_reserves) / new_sol_reserves
    token_received = token_reserves - new_token_reserves
    return token_received

def tokens_for_sol(tokens_to_sell: float, sol_reserves: float, token_reserves: float) -> float:
    new_token_reserves = token_reserves + tokens_to_sell
    new_sol_reserves = (sol_reserves * token_reserves) / new_token_reserves
    sol_received = sol_reserves - new_sol_reserves
    return sol_received

# ====================
# Utils
# ====================

def get_token_balance(mint_str: str) -> Optional[float]:
    try:
        mint = Pubkey.from_string(mint_str)
        owner = payer_keypair.pubkey()
        resp = client.get_token_accounts_by_owner_json_parsed(
            owner, TokenAccountOpts(mint=mint), commitment=Processed
        )
        accounts = resp.value
        if accounts:
            ui_amount = accounts[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
            return float(ui_amount)
        return 0.0
    except Exception as e:
        print(f"[get_token_balance] Error: {e}")
        return None

def confirm_txn(txn_sig, max_retries: int = 20, retry_interval: int = 3) -> bool:
    tries = 1
    while tries <= max_retries:
        try:
            res = client.get_transaction(
                txn_sig,
                encoding="json",
                commitment=Confirmed,
                max_supported_transaction_version=0,
            )
            if res.value is None:
                raise Exception("No transaction yet")
            meta_json = json.loads(res.value.transaction.meta.to_json())
            if meta_json.get("err") is None:
                print(f"✅ Confirmed ({tries}): https://solscan.io/tx/{txn_sig}")
                return True
            else:
                print(f"❌ Transaction failed: {meta_json.get('err')}")
                return False
        except Exception:
            print(f"⏳ Awaiting confirmation... try {tries}/{max_retries}")
        tries += 1
        time.sleep(retry_interval)
    print("❌ Max retries reached. Not confirmed.")
    return False

def get_token_price(mint_str: str) -> Optional[float]:
    try:
        cd = get_coin_data(mint_str)
        if not cd:
            print("Failed to retrieve coin data.")
            return None
        sol_res  = cd.virtual_sol_reserves / LAMPORTS_PER_SOL
        tok_res  = cd.virtual_token_reserves / TOKEN_DECIMALS
        price = (sol_res / tok_res) if tok_res > 0 else 0.0
        print(f"📊 Token price: {price:.12f} SOL")
        return price
    except Exception as e:
        print(f"[get_token_price] Error: {e}")
        return None

# ====================
# Instrucciones de swap
# ====================

def _send_vtx_with_optional_jito(message: MessageV0) -> Optional[str]:
    """Firma el message y envía: si USE_JITO => bundle a Jito; else send_transaction normal. Devuelve signature o None."""
    vtx = VersionedTransaction(message, [payer_keypair])

    if USE_JITO:
        res = _jito_send_versioned_with_tip(
            signed_vtx=vtx,
            payer_solders_kp=payer_keypair,
            rpc_url=RPC_ENDPOINT,
            tip_sol=JITO_TIP_VALUE,
            tip_account=JITO_TIP_ACCOUNT,
        )
        print("Jito result:", res)
        if res.get("confirmed"):
            # Jito no devuelve firma de la TX principal; solo bundle_id.
            # Confirmado por BE ⇒ listo.
            return None
        # Reintento rápido si el error sugiere blockhash expirado
        err = (res.get("error") or "").lower()
        if "blockhash" in err or "expired" in err:
            print("[i] Reintentando con blockhash fresco...")
            bh = client.get_latest_blockhash().value.blockhash
            msg2 = MessageV0.try_compile(payer_keypair.pubkey(), message.instructions, [], bh)
            vtx2 = VersionedTransaction(msg2, [payer_keypair])
            res2 = _jito_send_versioned_with_tip(
                signed_vtx=vtx2,
                payer_solders_kp=payer_keypair,
                rpc_url=RPC_ENDPOINT,
                tip_sol=JITO_TIP_VALUE,
                tip_account=JITO_TIP_ACCOUNT,
            )
            print("Jito result (retry):", res2)
        return None
    else:
        sig = client.send_transaction(
            txn=vtx,
            opts=TxOpts(skip_preflight=True),
        ).value
        print(f"📤 Tx: https://solscan.io/tx/{sig}")
        return str(sig)

def buy(mint_str: str, sol_in: float = 0.01, slippage: int = 5) -> bool:
    try:
        print(f"🚀 Buy start – mint: {mint_str}")
        cd = get_coin_data(mint_str)
        if not cd:
            print("❌ Failed to get coin data.")
            return False
        if cd.complete:
            print("⚠️ Token bonded. Trade on Raydium.")
            return False

        MINT = cd.mint
        BONDING_CURVE = cd.bonding_curve
        ASSOCIATED_BONDING_CURVE = cd.associated_bonding_curve
        USER = payer_keypair.pubkey()

        token_account_instruction = None
        try:
            accs = client.get_token_accounts_by_owner(USER, TokenAccountOpts(mint=MINT)).value
            if accs:
                ASSOCIATED_USER = accs[0].pubkey
                print(f"✅ ATA found: {ASSOCIATED_USER}")
            else:
                raise Exception("No ATA")
        except Exception:
            ASSOCIATED_USER = get_associated_token_address(USER, MINT)
            token_account_instruction = create_associated_token_account(USER, USER, MINT)
            print(f"🆕 Creating ATA: {ASSOCIATED_USER}")

        sol_res = cd.virtual_sol_reserves   / LAMPORTS_PER_SOL
        tok_res = cd.virtual_token_reserves / TOKEN_DECIMALS

        token_out_ui = sol_for_tokens(sol_in, sol_res, tok_res)
        amount_tokens = int(round(token_out_ui * TOKEN_DECIMALS))
        max_sol_cost  = int(round(sol_in * (1 + slippage / 100) * LAMPORTS_PER_SOL))

        print(f"💱 Receive ~{token_out_ui:.6f} tokens; max spend {sol_in*(1+slippage/100):.6f} SOL")

        keys = [
            AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_USER, is_signer=False, is_writable=True),
            AccountMeta(pubkey=USER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=RENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))      # discriminator buy
        data.extend(struct.pack("<Q", amount_tokens))       # amount (tokens)
        data.extend(struct.pack("<Q", max_sol_cost))        # max SOL (lamports)
        swap_ix = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
        ]
        if token_account_instruction:
            instructions.append(token_account_instruction)
        instructions.append(swap_ix)

        blockhash = client.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(USER, instructions, [], blockhash)

        sig = _send_vtx_with_optional_jito(msg)
        return True if (sig is None or confirm_txn(sig)) else False

    except Exception as e:
        print(f"[buy] Error: {e}")
        return False

def sell(mint_str: str, percentage: int = 100, slippage: int = 5) -> bool:
    if not (1 <= percentage <= 100):
        print("❌ Percentage must be between 1 and 100.")
        return False
    try:
        print(f"🔥 Sell start – mint: {mint_str}")
        cd = get_coin_data(mint_str)
        if not cd:
            print("❌ Failed to get coin data.")
            return False
        if cd.complete:
            print("⚠️ Token bonded. Trade on Raydium.")
            return False

        MINT = cd.mint
        BONDING_CURVE = cd.bonding_curve
        ASSOCIATED_BONDING_CURVE = cd.associated_bonding_curve
        USER = payer_keypair.pubkey()
        ASSOCIATED_USER = get_associated_token_address(USER, MINT)

        bal = get_token_balance(mint_str)
        if not bal or bal == 0:
            print("❌ No token balance to sell.")
            return False

        amount_to_sell_ui = bal * (percentage / 100.0)
        amount_tokens = int(round(amount_to_sell_ui * TOKEN_DECIMALS))

        sol_res = cd.virtual_sol_reserves   / LAMPORTS_PER_SOL
        tok_res = cd.virtual_token_reserves / TOKEN_DECIMALS
        sol_out_ui = tokens_for_sol(amount_to_sell_ui, sol_res, tok_res)
        min_sol_out = int(round(sol_out_ui * (1 - slippage / 100) * LAMPORTS_PER_SOL))

        print(f"💸 Will receive ~{sol_out_ui:.6f} SOL; min {sol_out_ui*(1-slippage/100):.6f} SOL")

        keys = [
            AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_USER, is_signer=False, is_writable=True),
            AccountMeta(pubkey=USER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=ASSOC_TOKEN_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))      # discriminator sell
        data.extend(struct.pack("<Q", min(amount_tokens, (1<<64)-1)))
        data.extend(struct.pack("<Q", min_sol_out))
        swap_ix = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
            swap_ix,
        ]

        if int(percentage) == 100:
            print("🧹 Closing ATA after full sell...")
            close_ix = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM,
                    account=ASSOCIATED_USER,
                    destination=USER,
                    owner=USER,
                )
            )
            instructions.append(close_ix)

        blockhash = client.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(USER, instructions, [], blockhash)

        sig = _send_vtx_with_optional_jito(msg)
        return True if (sig is None or confirm_txn(sig)) else False

    except Exception as e:
        print(f"[sell] Error: {e}")
        return False

# ====================
# CLI
# ====================

def main():
    parser = argparse.ArgumentParser(description="Pump.fun unified bot (+ Jito bundle opcional)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_price = sub.add_parser("price", help="Mostrar precio en SOL")
    p_price.add_argument("--mint", required=True)

    p_bal = sub.add_parser("balance", help="Mostrar balance del token")
    p_bal.add_argument("--mint", required=True)

    p_buy = sub.add_parser("buy", help="Comprar token con SOL")
    p_buy.add_argument("--mint", required=True)
    p_buy.add_argument("--sol", type=float, default=0.01)
    p_buy.add_argument("--slippage", type=int, default=5)

    p_sell = sub.add_parser("sell", help="Vender token")
    p_sell.add_argument("--mint", required=True)
    p_sell.add_argument("--pct", type=int, default=100)
    p_sell.add_argument("--slippage", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "price":
        px = get_token_price(args.mint)
        sys.exit(0 if px is not None else 1)

    if args.cmd == "balance":
        bal = get_token_balance(args.mint)
        if bal is None:
            sys.exit(1)
        print(f"Balance: {bal}")
        sys.exit(0)

    if args.cmd == "buy":
        ok = buy(args.mint, args.sol, args.slippage)
        sys.exit(0 if ok else 1)

    if args.cmd == "sell":
        ok = sell(args.mint, args.pct, args.slippage)
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
