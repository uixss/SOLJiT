#!/usr/bin/env python3
"""
main.py — Raydium AMM v4 (pares SOL) + envío opcional por Jito (bundle con tip).
- Un solo archivo: sin dependencias locales.
- CLI buy/sell, resolución de par por API o RPC.
- Versioned TX + tip TX legacy para bundles Jito (mismo blockhash).

Requisitos (pip):
    solana
    solders
    base58
    requests
    construct
    spl-token
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import requests
from construct import (
    Bytes, Int32ul, Int8ul, Int64ul, Padding, BitsInteger, BitsSwapped, BitStruct,
    Const, Flag, BytesInteger, Struct as cStruct
)

# ---------- Solana / Solders ----------
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts, TxOpts, MemcmpOpts
from solana.transaction import AccountMeta, Signature as LegacySig, Transaction as LegacyTx
from solana.system_program import transfer as sys_transfer, TransferParams as SysTP
from solana.publickey import PublicKey as SolPub
from solana.keypair import Keypair as SolKeypair

from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed
from solders.transaction import VersionedTransaction

# ---------- SPL Token ----------
from spl.token.client import Token
from spl.token.instructions import (
    CloseAccountParams,
    InitializeAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
    initialize_account,
)

# =====================
# Constantes públicas
# =====================
WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
RAY_V4 = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
RAY_AUTHORITY_V4 = Pubkey.from_string("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1")
OPEN_BOOK_PROGRAM = Pubkey.from_string("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
SOL_DECIMAL = 1e9

# =====================
# Layouts (construct)
# =====================
LIQUIDITY_STATE_LAYOUT_V4 = cStruct(
    "status" / Int64ul,
    "nonce" / Int64ul,
    "orderNum" / Int64ul,
    "depth" / Int64ul,
    "coinDecimals" / Int64ul,
    "pcDecimals" / Int64ul,
    "state" / Int64ul,
    "resetFlag" / Int64ul,
    "minSize" / Int64ul,
    "volMaxCutRatio" / Int64ul,
    "amountWaveRatio" / Int64ul,
    "coinLotSize" / Int64ul,
    "pcLotSize" / Int64ul,
    "minPriceMultiplier" / Int64ul,
    "maxPriceMultiplier" / Int64ul,
    "systemDecimalsValue" / Int64ul,
    "minSeparateNumerator" / Int64ul,
    "minSeparateDenominator" / Int64ul,
    "tradeFeeNumerator" / Int64ul,
    "tradeFeeDenominator" / Int64ul,
    "pnlNumerator" / Int64ul,
    "pnlDenominator" / Int64ul,
    "swapFeeNumerator" / Int64ul,
    "swapFeeDenominator" / Int64ul,
    "needTakePnlCoin" / Int64ul,
    "needTakePnlPc" / Int64ul,
    "totalPnlPc" / Int64ul,
    "totalPnlCoin" / Int64ul,
    "poolOpenTime" / Int64ul,
    "punishPcAmount" / Int64ul,
    "punishCoinAmount" / Int64ul,
    "orderbookToInitTime" / Int64ul,
    "swapCoinInAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapPcOutAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapCoin2PcFee" / Int64ul,
    "swapPcInAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapCoinOutAmount" / BytesInteger(16, signed=False, swapped=True),
    "swapPc2CoinFee" / Int64ul,
    "poolCoinTokenAccount" / Bytes(32),
    "poolPcTokenAccount" / Bytes(32),
    "coinMintAddress" / Bytes(32),
    "pcMintAddress" / Bytes(32),
    "lpMintAddress" / Bytes(32),
    "ammOpenOrders" / Bytes(32),
    "serumMarket" / Bytes(32),
    "serumProgramId" / Bytes(32),
    "ammTargetOrders" / Bytes(32),
    "poolWithdrawQueue" / Bytes(32),
    "poolTempLpTokenAccount" / Bytes(32),
    "ammOwner" / Bytes(32),
    "pnlOwner" / Bytes(32),
)

ACCOUNT_FLAGS_LAYOUT = BitsSwapped(
    BitStruct(
        "initialized" / Flag,
        "market" / Flag,
        "open_orders" / Flag,
        "request_queue" / Flag,
        "event_queue" / Flag,
        "bids" / Flag,
        "asks" / Flag,
        Const(0, BitsInteger(57)),
    )
)

MARKET_STATE_LAYOUT_V3 = cStruct(
    Padding(5),
    "account_flags" / ACCOUNT_FLAGS_LAYOUT,
    "own_address" / Bytes(32),
    "vault_signer_nonce" / Int64ul,
    "base_mint" / Bytes(32),
    "quote_mint" / Bytes(32),
    "base_vault" / Bytes(32),
    "base_deposits_total" / Int64ul,
    "base_fees_accrued" / Int64ul,
    "quote_vault" / Bytes(32),
    "quote_deposits_total" / Int64ul,
    "quote_fees_accrued" / Int64ul,
    "quote_dust_threshold" / Int64ul,
    "request_queue" / Bytes(32),
    "event_queue" / Bytes(32),
    "bids" / Bytes(32),
    "asks" / Bytes(32),
    "base_lot_size" / Int64ul,
    "quote_lot_size" / Int64ul,
    "fee_rate_bps" / Int64ul,
    "referrer_rebate_accrued" / Int64ul,
    Padding(7),
)

PUBLIC_KEY_LAYOUT = Bytes(32)
ACCOUNT_LAYOUT = cStruct(
    "mint" / PUBLIC_KEY_LAYOUT,
    "owner" / PUBLIC_KEY_LAYOUT,
    "amount" / Int64ul,
    "delegate_option" / Int32ul,
    "delegate" / PUBLIC_KEY_LAYOUT,
    "state" / Int8ul,
    "is_native_option" / Int32ul,
    "is_native" / Int64ul,
    "delegated_amount" / Int64ul,
    "close_authority_option" / Int32ul,
    "close_authority" / PUBLIC_KEY_LAYOUT,
)

SWAP_LAYOUT = cStruct(
    "instruction" / Int8ul,
    "amount_in" / Int64ul,
    "min_amount_out" / Int64ul,
)

# =====================
# Tipos y helpers
# =====================
@dataclass
class PoolKeys:
    amm_id: Pubkey
    base_mint: Pubkey
    quote_mint: Pubkey
    base_decimals: int
    quote_decimals: int
    open_orders: Pubkey
    target_orders: Pubkey
    base_vault: Pubkey
    quote_vault: Pubkey
    market_id: Pubkey
    market_authority: Pubkey
    market_base_vault: Pubkey
    market_quote_vault: Pubkey
    bids: Pubkey
    asks: Pubkey
    event_queue: Pubkey


def bytes_of(value: int) -> bytes:
    if not (0 <= value < 2**64):
        raise ValueError("Value must be u64 (0..2^64-1)")
    return struct.pack('<Q', value)

# =====================
# Config CLI/ENV
# =====================

# Priority fees por defecto (puedes ajustar vía CLI)
DEFAULT_UNIT_BUDGET = int(os.getenv("UNIT_BUDGET", "100000"))
DEFAULT_UNIT_PRICE  = int(os.getenv("UNIT_PRICE",  "1000000"))

# Jito (opcional)
USE_JITO         = os.getenv("USE_JITO", "true").lower() in ("1", "true", "yes")
JITO_BE_URL      = os.getenv("JITO_BLOCK_ENGINE_URL", "https://ny.mainnet.block-engine.jito.wtf").rstrip("/")
JITO_TIP_VALUE   = max(0.0, min(float(os.getenv("JITO_TIP_VALUE", "0.004")), 0.1))  # 0..0.1 SOL
JITO_TIP_ACCOUNT = os.getenv("TIP_ACCOUNT", "JitoTip1111111111111111111111111111111111")
JITO_API_KEY     = os.getenv("JITO_API_KEY")  # opcional

# =====================
# RPC/REST utils
# =====================

def get_pair_address_from_api(client: Client, mint: str) -> Optional[str]:
    url = f"https://api-v3.raydium.io/pools/info/mint?mint1={mint}&poolType=all&poolSortField=default&sortType=desc&pageSize=1&page=1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        pools = data.get('data', {}).get('data', [])
        if not pools:
            return None
        pool = pools[0]
        if pool.get('programId') == str(RAY_V4):
            return pool.get('id')
        return None
    except Exception as e:
        print(f"API request failed: {e}")
        return None

def get_pair_address_from_rpc(client: Client, token_address: str) -> Optional[str]:
    print("Getting pair address from RPC...")
    BASE_OFFSET = 400
    QUOTE_OFFSET = 432
    DATA_LENGTH_FILTER = 752
    QUOTE_MINT = str(WSOL)
    RAYDIUM_PROGRAM_ID = RAY_V4

    def fetch_amm_id(base_mint: str, quote_mint: str) -> Optional[str]:
        memcmp_base = MemcmpOpts(offset=BASE_OFFSET, bytes=base_mint)
        memcmp_quote = MemcmpOpts(offset=QUOTE_OFFSET, bytes=quote_mint)
        try:
            resp = client.get_program_accounts(
                RAYDIUM_PROGRAM_ID,
                commitment=Processed,
                filters=[DATA_LENGTH_FILTER, memcmp_base, memcmp_quote],
            )
            accounts = resp.value
            if accounts:
                return str(accounts[0].pubkey)
        except Exception as e:
            print(f"Error fetching AMM ID: {e}")
        return None

    pair = fetch_amm_id(token_address, QUOTE_MINT)
    if not pair:
        pair = fetch_amm_id(QUOTE_MINT, token_address)
    return pair

def _decode_account_data(value_data):
    # value_data: client.get_account_info(...).value.data
    import base64
    if isinstance(value_data, (bytes, bytearray)):
        return bytes(value_data)
    if isinstance(value_data, (list, tuple)) and len(value_data) >= 1:
        return base64.b64decode(value_data[0])
    raise ValueError("Formato de data no reconocido")

def fetch_pool_keys(client: Client, pair_address: str) -> Optional[PoolKeys]:
    try:
        amm_id = Pubkey.from_string(pair_address)
        amm_info = client.get_account_info(amm_id, commitment=Processed).value
        if amm_info is None:
            return None
        amm_bytes = _decode_account_data(amm_info.data)
        amm_dec = LIQUIDITY_STATE_LAYOUT_V4.parse(amm_bytes)
        market_id = Pubkey.from_bytes(amm_dec.serumMarket)

        market_info = client.get_account_info(market_id, commitment=Processed).value
        if market_info is None:
            return None
        market_bytes = _decode_account_data(market_info.data)
        market_dec = MARKET_STATE_LAYOUT_V3.parse(market_bytes)
        nonce = market_dec.vault_signer_nonce

        market_auth = Pubkey.create_program_address(
            [bytes(market_id), nonce.to_bytes(8, "little")], OPEN_BOOK_PROGRAM
        )

        return PoolKeys(
            amm_id=amm_id,
            base_mint=Pubkey.from_bytes(market_dec.base_mint),
            quote_mint=Pubkey.from_bytes(market_dec.quote_mint),
            base_decimals=int(amm_dec.coinDecimals),
            quote_decimals=int(amm_dec.pcDecimals),
            open_orders=Pubkey.from_bytes(amm_dec.ammOpenOrders),
            target_orders=Pubkey.from_bytes(amm_dec.ammTargetOrders),
            base_vault=Pubkey.from_bytes(amm_dec.poolCoinTokenAccount),
            quote_vault=Pubkey.from_bytes(amm_dec.poolPcTokenAccount),
            market_id=market_id,
            market_authority=market_auth,
            market_base_vault=Pubkey.from_bytes(market_dec.base_vault),
            market_quote_vault=Pubkey.from_bytes(market_dec.quote_vault),
            bids=Pubkey.from_bytes(market_dec.bids),
            asks=Pubkey.from_bytes(market_dec.asks),
            event_queue=Pubkey.from_bytes(market_dec.event_queue),
        )
    except Exception as e:
        print(f"Error fetching pool keys: {e}")
        return None


# =====================
# AMM (x*y=k) helpers
# =====================

def sol_for_tokens(spend_sol_amount: float, base_vault_balance: float, quote_vault_balance: float, swap_fee: float = 0.25) -> float:
    effective_sol = spend_sol_amount * (1 - swap_fee / 100)
    k = base_vault_balance * quote_vault_balance
    new_base = k / (quote_vault_balance + effective_sol)
    tokens = base_vault_balance - new_base
    return round(tokens, 9)

def tokens_for_sol(sell_token_amount: float, base_vault_balance: float, quote_vault_balance: float, swap_fee: float = 0.25) -> float:
    effective_tokens = sell_token_amount * (1 - swap_fee / 100)
    k = base_vault_balance * quote_vault_balance
    new_quote = k / (base_vault_balance + effective_tokens)
    sol = quote_vault_balance - new_quote
    return round(sol, 9)

def get_token_reserves(client: Client, pool_keys: PoolKeys) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    try:
        balances_resp = client.get_multiple_accounts_json_parsed(
            [pool_keys.base_vault, pool_keys.quote_vault], Processed
        )
        balances = balances_resp.value
        token_acc = balances[0]
        sol_acc = balances[1]
        token_ui = token_acc.data.parsed['info']['tokenAmount']['uiAmount']
        sol_ui = sol_acc.data.parsed['info']['tokenAmount']['uiAmount']
        if token_ui is None or sol_ui is None:
            return None, None, None

        if pool_keys.base_mint == WSOL:
            base_reserve = sol_ui
            quote_reserve = token_ui
            token_decimal = pool_keys.quote_decimals
        else:
            base_reserve = token_ui
            quote_reserve = sol_ui
            token_decimal = pool_keys.base_decimals

        print(f"Base Mint: {pool_keys.base_mint} | Quote Mint: {pool_keys.quote_mint}")
        print(f"Base Reserve: {base_reserve} | Quote Reserve: {quote_reserve} | Token Decimal: {token_decimal}")
        return float(base_reserve), float(quote_reserve), int(token_decimal)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None

def get_token_balance(client: Client, owner: Pubkey, mint_str: str) -> Optional[float]:
    try:
        mint = Pubkey.from_string(mint_str)
        resp = client.get_token_accounts_by_owner_json_parsed(
            owner, TokenAccountOpts(mint=mint), commitment=Processed
        )
        accounts = resp.value
        if accounts:
            ui = accounts[0].account.data.parsed['info']['tokenAmount']['uiAmount']
            if ui is not None:
                return float(ui)
        return None
    except Exception as e:
        print(f"Error fetching token balance: {e}")
        return None

def confirm_txn(client: Client, txn_sig: LegacySig, max_retries: int = 20, retry_interval: int = 3) -> bool:
    retries = 1
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(
                txn_sig, encoding="json", commitment=Confirmed, max_supported_transaction_version=0
            )
            if txn_res.value is None:
                raise Exception("Transaction not found in ledger")
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            if txn_json['err'] is None:
                print(f"Transaction confirmed after {retries} attempt(s).")
                return True
            print("Transaction failed:", txn_json['err'])
            return False
        except Exception as e:
            print(f"Awaiting confirmation... ({retries}/{max_retries}) | {e}")
            retries += 1
            time.sleep(retry_interval)
    print("Max retries reached. Transaction confirmation failed.")
    return False

def make_swap_instruction(
    amount_in: int,
    minimum_amount_out: int,
    token_account_in: Pubkey,
    token_account_out: Pubkey,
    accounts: PoolKeys,
    owner: Keypair,
) -> Optional[Instruction]:
    try:
        keys = [
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.amm_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=RAY_AUTHORITY_V4, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.open_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.target_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=OPEN_BOOK_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.market_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.bids, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.asks, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.event_queue, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_authority, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True),
            AccountMeta(pubkey=owner.pubkey(), is_signer=True, is_writable=False),
        ]
        data = SWAP_LAYOUT.build(dict(instruction=9, amount_in=amount_in, min_amount_out=minimum_amount_out))
        return Instruction(RAY_V4, data, keys)
    except Exception as e:
        print(f"Error building swap instruction: {e}")
        return None

# =====================
# Jito helpers (bundle con tip)
# =====================
import requests as _rq
import base64 as _b64

def _jito_send_versioned_with_tip(
    signed_vtx: VersionedTransaction,
    payer_solders_kp: Keypair,
    tip_sol: float = JITO_TIP_VALUE,
    tip_account: str = JITO_TIP_ACCOUNT,
    wait_timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Empaqueta [signed_vtx(versioned), tip_tx(legacy)] con el mismo blockhash y lo envía a Jito.
    Devuelve {"bundle_id": str|None, "confirmed": bool, "error": str|None}
    """
    try:
        recent_bh = str(signed_vtx.message.recent_blockhash)

        # Firmante legacy para tip
        payer_solana = SolKeypair.from_secret_key(payer_solders_kp.to_bytes())
        tip_lamports = int(max(0.0, min(tip_sol, 0.1)) * SOL_DECIMAL)
        tip_pubkey   = SolPub(tip_account)

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

def _send_vtx_with_optional_jito(client: Client, message: MessageV0, payer_kp: Keypair) -> Optional[str]:
    """
    Firma y envía:
      - Si USE_JITO => bundle a Jito (no devuelve signature de la TX principal).
      - Si no, send_transaction normal y devuelve signature.
    """
    vtx = VersionedTransaction(message, [payer_kp])

    if USE_JITO:
        res = _jito_send_versioned_with_tip(
            signed_vtx=vtx,
            payer_solders_kp=payer_kp,
            tip_sol=JITO_TIP_VALUE,
            tip_account=JITO_TIP_ACCOUNT,
        )
        print("Jito result:", res)
        if res.get("confirmed"):
            return None
        # Reintento rápido si parece blockhash expirado
        err = (res.get("error") or "").lower()
        if "blockhash" in err or "expired" in err:
            print("[i] Reintentando con blockhash fresco...")
            new_bh = client.get_latest_blockhash().value.blockhash
            msg2 = MessageV0.try_compile(payer_kp.pubkey(), message.instructions, [], new_bh)
            vtx2 = VersionedTransaction(msg2, [payer_kp])
            res2 = _jito_send_versioned_with_tip(vtx2, payer_kp, JITO_TIP_VALUE, JITO_TIP_ACCOUNT)
            print("Jito result (retry):", res2)
            if res2.get("confirmed"):
                return None
        return None
    else:
        sig = client.send_transaction(vtx, opts=TxOpts(skip_preflight=True)).value
        print(f"📤 Tx: https://solscan.io/tx/{sig}")
        return str(sig)

# =====================
# Operaciones buy/sell
# =====================

def buy(
    client: Client,
    payer_keypair: Keypair,
    pair_address: str,
    sol_in: float = 0.01,
    slippage: float = 5.0,
    unit_budget: int = DEFAULT_UNIT_BUDGET,
    unit_price: int = DEFAULT_UNIT_PRICE,
) -> bool:
    try:
        print(f"🚀 Starting buy for pair: {pair_address}")
        pool_keys = fetch_pool_keys(client, pair_address)
        if not pool_keys:
            print("❌ No pool keys found.")
            return False

        mint = pool_keys.base_mint if pool_keys.base_mint != WSOL else pool_keys.quote_mint
        amount_in = int(sol_in * SOL_DECIMAL)

        print("📊 Fetching reserves...")
        base_reserve, quote_reserve, token_decimal = get_token_reserves(client, pool_keys)
        if not base_reserve or not quote_reserve:
            print("❌ Failed to fetch reserves.")
            return False

        estimated_out = sol_for_tokens(sol_in, base_reserve, quote_reserve)
        min_out = int(estimated_out * (1 - slippage / 100) * (10 ** token_decimal))
        print(f"💸 Buy: {sol_in:.4f} SOL → min {min_out / 10**token_decimal:.9f} tokens")

        # Token account
        tok_acc_check = client.get_token_accounts_by_owner(
            payer_keypair.pubkey(), TokenAccountOpts(mint=mint), Processed
        )
        if tok_acc_check.value:
            token_account = tok_acc_check.value[0].pubkey
            create_token_account_instr = None
        else:
            token_account = get_associated_token_address(payer_keypair.pubkey(), mint)
            create_token_account_instr = create_associated_token_account(
                payer_keypair.pubkey(), payer_keypair.pubkey(), mint
            )

        # WSOL temp
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode('utf-8')
        wsol_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
        rent = Token.get_min_balance_rent_for_exempt_for_account(client)

        create_wsol = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(rent + amount_in),
                space=ACCOUNT_LAYOUT.sizeof(),
                owner=TOKEN_PROGRAM_ID,
            )
        )
        init_wsol = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        swap_ix = make_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=min_out,
            token_account_in=wsol_account,
            token_account_out=token_account,
            accounts=pool_keys,
            owner=payer_keypair,
        )
        if not swap_ix:
            return False

        close_wsol = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_wsol,
            init_wsol,
        ]
        if create_token_account_instr:
            instructions.append(create_token_account_instr)
        instructions += [swap_ix, close_wsol]

        blockhash = client.get_latest_blockhash().value.blockhash
        message = MessageV0.try_compile(
            payer_keypair.pubkey(), instructions, [], blockhash
        )

        sig = _send_vtx_with_optional_jito(client, message, payer_keypair)
        return True if (sig is None or confirm_txn(client, LegacySig.from_string(sig))) else False

    except Exception as e:
        print(f"❌ Buy error: {e}")
        return False

def sell(
    client: Client,
    payer_keypair: Keypair,
    pair_address: str,
    percentage: float = 100.0,
    slippage: float = 5.0,
    unit_budget: int = DEFAULT_UNIT_BUDGET,
    unit_price: int = DEFAULT_UNIT_PRICE,
) -> bool:
    try:
        print(f"🚀 Starting sell for pair: {pair_address}")
        if not (1 <= percentage <= 100):
            print("❌ Percentage must be between 1 and 100.")
            return False

        pool_keys = fetch_pool_keys(client, pair_address)
        if not pool_keys:
            print("❌ No pool keys found.")
            return False

        mint = pool_keys.base_mint if pool_keys.base_mint != WSOL else pool_keys.quote_mint
        balance = get_token_balance(client, payer_keypair.pubkey(), str(mint))
        print(f"💰 Token balance: {balance}")
        if not balance or balance == 0:
            print("❌ No tokens to sell.")
            return False

        amount_to_sell = balance * (percentage / 100)
        base_reserve, quote_reserve, token_decimal = get_token_reserves(client, pool_keys)
        if not base_reserve or not quote_reserve:
            print("❌ Failed to fetch reserves.")
            return False

        sol_out = tokens_for_sol(amount_to_sell, base_reserve, quote_reserve)
        min_sol_out = int(sol_out * (1 - slippage / 100) * SOL_DECIMAL)
        amount_in = int(amount_to_sell * 10 ** token_decimal)
        print(f"💸 Sell: {amount_to_sell:.9f} tokens → min {min_sol_out / SOL_DECIMAL:.9f} SOL")

        token_account = get_associated_token_address(payer_keypair.pubkey(), mint)

        # WSOL temp
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode('utf-8')
        wsol_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
        rent = Token.get_min_balance_rent_for_exempt_for_account(client)

        create_wsol = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(rent),
                space=ACCOUNT_LAYOUT.sizeof(),
                owner=TOKEN_PROGRAM_ID,
            )
        )
        init_wsol = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        swap_ix = make_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=min_sol_out,
            token_account_in=token_account,
            token_account_out=wsol_account,
            accounts=pool_keys,
            owner=payer_keypair,
        )
        if not swap_ix:
            return False

        close_wsol = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_wsol,
            init_wsol,
            swap_ix,
            close_wsol,
        ]

        if int(percentage) == 100:
            close_token = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM_ID,
                    account=token_account,
                    dest=payer_keypair.pubkey(),
                    owner=payer_keypair.pubkey(),
                )
            )
            instructions.append(close_token)

        blockhash = client.get_latest_blockhash().value.blockhash
        message = MessageV0.try_compile(
            payer_keypair.pubkey(), instructions, [], blockhash
        )

        sig = _send_vtx_with_optional_jito(client, message, payer_keypair)
        return True if (sig is None or confirm_txn(client, LegacySig.from_string(sig))) else False

    except Exception as e:
        print(f"❌ Sell error: {e}")
        return False

# =====================
# CLI
# =====================

def resolve_pair(client: Client, mint: Optional[str], pair: Optional[str], source: str) -> Optional[str]:
    if pair:
        return pair
    if not mint:
        print("❌ You must pass --mint if --pair is not provided.")
        return None
    if source == "api":
        return get_pair_address_from_api(client, mint)
    elif source == "rpc":
        return get_pair_address_from_rpc(client, mint)
    else:  # auto
        pair = get_pair_address_from_api(client, mint)
        return pair or get_pair_address_from_rpc(client, mint)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trade on Raydium AMM v4 (SOL pairs only) + Jito bundles opcionales.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--private-key", required=True, help="Clave privada base58")
    common.add_argument("--rpc", required=True, help="RPC URL (Helius/QuickNode recomendado)")
    common.add_argument("--unit-budget", type=int, default=DEFAULT_UNIT_BUDGET, help="Compute Unit limit")
    common.add_argument("--unit-price", type=int, default=DEFAULT_UNIT_PRICE, help="Micro-lamports por CU (priority fee)")
    common.add_argument("--pair", help="AMM ID de Raydium v4 si ya lo tienes")
    common.add_argument("--mint", help="Mint del token (si no pasas --pair)")
    common.add_argument("--pair-source", choices=["auto", "api", "rpc"], default="auto", help="Cómo resolver el par desde un mint")
    common.add_argument("--slippage", type=float, default=5.0, help="Slippage en %")

    pb = sub.add_parser("buy", parents=[common], help="Comprar token con SOL")
    pb.add_argument("--sol", type=float, required=True, help="Cantidad de SOL a gastar")

    ps = sub.add_parser("sell", parents=[common], help="Vender token a SOL")
    ps.add_argument("--pct", type=float, required=True, help="Porcentaje del balance a vender (1-100)")

    return p.parse_args()

def main():
    args = parse_args()

    client = Client(args.rpc)
    try:
        payer_keypair = Keypair.from_base58_string(args.private_key)
    except Exception:
        print("❌ Invalid private key.")
        raise SystemExit(1)

    pair_address = resolve_pair(client, args.mint, args.pair, args.pair_source)
    if not pair_address:
        print("❌ Error: Pair address not found.")
        raise SystemExit(1)

    print(f"🔗 Pair Address: {pair_address}")

    if args.cmd == "buy":
        success = buy(client, payer_keypair, pair_address, args.sol, args.slippage, args.unit_budget, args.unit_price)
    else:
        success = sell(client, payer_keypair, pair_address, args.pct, args.slippage, args.unit_budget, args.unit_price)

    raise SystemExit(0 if success else 2)

if __name__ == "__main__":
    main()
