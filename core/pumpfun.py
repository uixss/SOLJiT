#!/usr/bin/env python3
# Pump.fun — buy/sell con Versioned TX; envío por Jito opcional vía core/jito_sender

import os, sys, time, json, base64, struct, argparse
from dataclasses import dataclass
from typing import Optional, Tuple

from construct import Flag, Int64ul, Padding, Struct
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import Instruction, AccountMeta as SoldersAccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction

from solana.rpc.api import Client
from solana.rpc.commitment import Processed, Confirmed
from solana.rpc.types import TokenAccountOpts, TxOpts

from spl.token.instructions import (
    CloseAccountParams, close_account, create_associated_token_account, get_associated_token_address,
)
from spl.token.client import Token  # (no usado directamente pero útil si amplías)

# ---- Jito sender (modular)
try:
    from core.jito_sender import send_v0_with_jito_bundle
except Exception:
    send_v0_with_jito_bundle = None

# ---- Config
RPC_ENDPOINT   = os.getenv("RPC_ENDPOINT") or os.getenv("RPC", "")
BASE58_PRIVKEY = os.getenv("BASE58_PRIVKEY") or os.getenv("PRIV_KEY", "")
if not RPC_ENDPOINT or not BASE58_PRIVKEY:
    raise RuntimeError("Faltan RPC_ENDPOINT/BASE58_PRIVKEY (o RPC/PRIV_KEY).")

client        = Client(RPC_ENDPOINT)
payer_keypair = Keypair.from_base58_string(BASE58_PRIVKEY)

UNIT_BUDGET = int(os.getenv("UNIT_BUDGET", "100000"))
UNIT_PRICE  = int(os.getenv("UNIT_PRICE",  "1000000"))
USE_JITO    = os.getenv("USE_JITO", "true").lower() in ("1","true","yes")

LAMPORTS_PER_SOL = 10**9
TOKEN_DECIMALS   = 10**6  # Pump.fun tokens suelen tener 6 decimales

# ---- Constantes Pump.fun
GLOBAL            = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
FEE_RECIPIENT     = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
SYSTEM_PROGRAM    = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM     = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_PROG  = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
RENT              = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
EVENT_AUTHORITY   = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
PUMP_FUN_PROGRAM  = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
WSOL              = Pubkey.from_string("So11111111111111111111111111111111111111112")

@dataclass
class CoinData:
    mint: Pubkey
    bonding_curve: Pubkey
    associated_bonding_curve: Pubkey
    virtual_token_reserves: int
    virtual_sol_reserves: int
    token_total_supply: int
    complete: bool

def _decode_account_data(value_data) -> bytes:
    if isinstance(value_data, (bytes, bytearray)): return bytes(value_data)
    if isinstance(value_data, (list, tuple)) and len(value_data) >= 1: return base64.b64decode(value_data[0])
    raise ValueError("Formato de data no reconocido")

def get_virtual_reserves(bonding_curve: Pubkey):
    layout = Struct(
        Padding(8),
        "virtualTokenReserves" / Int64ul,
        "virtualSolReserves" / Int64ul,
        "realTokenReserves" / Int64ul,
        "realSolReserves" / Int64ul,
        "tokenTotalSupply" / Int64ul,
        "complete" / Flag,
    )
    info = client.get_account_info(bonding_curve)
    if info.value is None: return None
    return layout.parse(_decode_account_data(info.value.data))

def derive_bonding_curve_accounts(mint_str: str) -> Tuple[Optional[Pubkey], Optional[Pubkey]]:
    try:
        mint = Pubkey.from_string(mint_str)
        bonding_curve, _ = Pubkey.find_program_address([b"bonding-curve", bytes(mint)], PUMP_FUN_PROGRAM)
        associated_bonding_curve = get_associated_token_address(bonding_curve, mint)
        return bonding_curve, associated_bonding_curve
    except Exception:
        return None, None

def get_coin_data(mint_str: str) -> Optional[CoinData]:
    bonding_curve, associated_bonding_curve = derive_bonding_curve_accounts(mint_str)
    if not bonding_curve or not associated_bonding_curve: return None
    v = get_virtual_reserves(bonding_curve)
    if not v: return None
    return CoinData(
        mint=Pubkey.from_string(mint_str),
        bonding_curve=bonding_curve,
        associated_bonding_curve=associated_bonding_curve,
        virtual_token_reserves=int(v.virtualTokenReserves),
        virtual_sol_reserves=int(v.virtualSolReserves),
        token_total_supply=int(v.tokenTotalSupply),
        complete=bool(v.complete),
    )

# x*y=k aproximado (virtual reserves)
def sol_for_tokens(sol_spent: float, sol_res: float, tok_res: float) -> float:
    new_sol = sol_res + sol_spent
    new_tok = (sol_res * tok_res) / new_sol
    return max(0.0, tok_res - new_tok)

def tokens_for_sol(tok_sell: float, sol_res: float, tok_res: float) -> float:
    new_tok = tok_res + tok_sell
    new_sol = (sol_res * tok_res) / new_tok
    return max(0.0, sol_res - new_sol)

def get_token_balance(mint_str: str) -> float:
    try:
        mint = Pubkey.from_string(mint_str)
        owner = payer_keypair.pubkey()
        resp = client.get_token_accounts_by_owner_json_parsed(owner, TokenAccountOpts(mint=mint), commitment=Processed)
        if resp.value:
            ui = resp.value[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
            return float(ui or 0.0)
        return 0.0
    except Exception:
        return 0.0

def confirm_txn(sig: str, max_retries: int = 20, retry_interval: int = 3) -> bool:
    tries = 1
    while tries <= max_retries:
        try:
            r = client.get_transaction(sig, encoding="json", commitment=Confirmed, max_supported_transaction_version=0)
            if r.value is not None:
                err = json.loads(r.value.transaction.meta.to_json()).get("err")
                return err is None
        except Exception:
            pass
        time.sleep(retry_interval); tries += 1
    return False

def _send_vtx(message: MessageV0) -> Optional[str]:
    vtx = VersionedTransaction(message, [payer_keypair])
    if USE_JITO and send_v0_with_jito_bundle is not None:
        res = send_v0_with_jito_bundle(vtx, payer_keypair)
        print("Jito result:", res)
        # Asumimos bundle OK => no esperamos confirmación RPC
        return None
    sig = client.send_transaction(vtx, opts=TxOpts(skip_preflight=True)).value
    print("Tx:", sig)
    return str(sig)

def buy(mint_str: str, sol_in: float = 0.01, slippage: int = 5) -> bool:
    slippage = max(0, int(slippage))
    cd = get_coin_data(mint_str)
    if not cd or cd.complete: return False

    USER = payer_keypair.pubkey()
    MINT = cd.mint

    try:
        ata_resp = client.get_token_accounts_by_owner(USER, TokenAccountOpts(mint=MINT)).value
        ASSOCIATED_USER = ata_resp[0].pubkey if ata_resp else get_associated_token_address(USER, MINT)
        ata_ix = None if ata_resp else create_associated_token_account(USER, USER, MINT)
    except Exception:
        ASSOCIATED_USER = get_associated_token_address(USER, MINT)
        ata_ix = create_associated_token_account(USER, USER, MINT)

    sol_res = cd.virtual_sol_reserves / LAMPORTS_PER_SOL
    tok_res = cd.virtual_token_reserves / TOKEN_DECIMALS
    token_out_ui = sol_for_tokens(sol_in, sol_res, tok_res)
    amount_tokens = int(round(token_out_ui * TOKEN_DECIMALS))
    max_sol_cost  = int(round(sol_in * (1 + slippage/100) * LAMPORTS_PER_SOL))

    keys = [
        SoldersAccountMeta(pubkey=GLOBAL,            is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=FEE_RECIPIENT,     is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=MINT,              is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=cd.bonding_curve,  is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=cd.associated_bonding_curve, is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=ASSOCIATED_USER,   is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=USER,              is_signer=True,  is_writable=True),
        SoldersAccountMeta(pubkey=SYSTEM_PROGRAM,    is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=TOKEN_PROGRAM,     is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=RENT,              is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=EVENT_AUTHORITY,   is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=PUMP_FUN_PROGRAM,  is_signer=False, is_writable=False),
    ]
    data = bytearray()
    data.extend(bytes.fromhex("66063d1201daebea"))
    data.extend(struct.pack("<Q", amount_tokens))
    data.extend(struct.pack("<Q", max_sol_cost))
    ix = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

    bh = client.get_latest_blockhash().value.blockhash
    instructions = [set_compute_unit_limit(UNIT_BUDGET), set_compute_unit_price(UNIT_PRICE)]
    if ata_ix: instructions.append(ata_ix)
    instructions.append(ix)
    msg = MessageV0.try_compile(USER, instructions, [], bh)

    sig = _send_vtx(msg)
    return True if (sig is None or confirm_txn(sig)) else False

def sell(mint_str: str, percentage: int = 100, slippage: int = 5) -> bool:
    if not (1 <= int(percentage) <= 100): return False
    slippage = max(0, int(slippage))

    cd = get_coin_data(mint_str)
    if not cd or cd.complete: return False

    USER = payer_keypair.pubkey()
    MINT = cd.mint
    ASSOCIATED_USER = get_associated_token_address(USER, MINT)

    bal = get_token_balance(mint_str)
    if bal <= 0: return False

    amount_ui = bal * (int(percentage)/100.0)
    amount_tokens = int(round(amount_ui * TOKEN_DECIMALS))

    sol_res = cd.virtual_sol_reserves / LAMPORTS_PER_SOL
    tok_res = cd.virtual_token_reserves / TOKEN_DECIMALS
    sol_out_ui = tokens_for_sol(amount_ui, sol_res, tok_res)
    min_sol_out = int(round(sol_out_ui * (1 - slippage/100) * LAMPORTS_PER_SOL))

    keys = [
        SoldersAccountMeta(pubkey=GLOBAL,            is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=FEE_RECIPIENT,     is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=MINT,              is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=cd.bonding_curve,  is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=cd.associated_bonding_curve, is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=ASSOCIATED_USER,   is_signer=False, is_writable=True),
        SoldersAccountMeta(pubkey=USER,              is_signer=True,  is_writable=True),
        SoldersAccountMeta(pubkey=SYSTEM_PROGRAM,    is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=ASSOC_TOKEN_PROG,  is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=TOKEN_PROGRAM,     is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=EVENT_AUTHORITY,   is_signer=False, is_writable=False),
        SoldersAccountMeta(pubkey=PUMP_FUN_PROGRAM,  is_signer=False, is_writable=False),
    ]
    data = bytearray()
    data.extend(bytes.fromhex("33e685a4017f83ad"))
    data.extend(struct.pack("<Q", min(amount_tokens, (1<<64)-1)))
    data.extend(struct.pack("<Q", min_sol_out))
    ix = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

    instructions = [set_compute_unit_limit(UNIT_BUDGET), set_compute_unit_price(UNIT_PRICE), ix]
    if int(percentage) == 100:
        instructions.append(
            close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM,
                    account=ASSOCIATED_USER,
                    destination=USER,
                    owner=USER
                )
            )
        )

    bh = client.get_latest_blockhash().value.blockhash
    msg = MessageV0.try_compile(USER, instructions, [], bh)

    sig = _send_vtx(msg)
    return True if (sig is None or confirm_txn(sig)) else False

# CLI opcional de prueba local
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b1 = sub.add_parser("buy");  b1.add_argument("--mint", required=True); b1.add_argument("--sol", type=float, default=0.01); b1.add_argument("--slippage", type=int, default=5)
    s1 = sub.add_parser("sell"); s1.add_argument("--mint", required=True); s1.add_argument("--pct", type=int, default=100); s1.add_argument("--slippage", type=int, default=5)
    a = p.parse_args()
    if a.cmd == "buy":
        sys.exit(0 if buy(a.mint, a.sol, a.slippage) else 2)
    else:
        sys.exit(0 if sell(a.mint, a.pct, a.slippage) else 2)
