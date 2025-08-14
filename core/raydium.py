#!/usr/bin/env python3
# Raydium AMM v4 (pares SOL) — buy/sell modulares; Jito vía core/jito_sender

import os, time, json, base64, struct
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import requests
from construct import Bytes, Int32ul, Int8ul, Int64ul, Padding, BitsInteger, BitsSwapped, BitStruct, Const, Flag, BytesInteger, Struct as cStruct

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts, TxOpts, MemcmpOpts
from solders.instruction import Instruction, AccountMeta as SoldersAccountMeta
 
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
 
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed
from solders.transaction import VersionedTransaction
from spl.token.client import Token
from spl.token.instructions import CloseAccountParams, InitializeAccountParams, close_account, create_associated_token_account, get_associated_token_address, initialize_account

# ---- Jito sender (modular)
try:
    from core.jito_sender import send_v0_with_jito_bundle
except Exception:
    send_v0_with_jito_bundle = None

# ---- Config
PRIV_KEY    = os.getenv("PRIV_KEY", "")
RPC         = os.getenv("RPC", "")
if not PRIV_KEY or not RPC:
    raise RuntimeError("Faltan PRIV_KEY/RPC")
UNIT_BUDGET = int(os.getenv("UNIT_BUDGET", "100000"))
UNIT_PRICE  = int(os.getenv("UNIT_PRICE",  "1000000"))
USE_JITO    = os.getenv("USE_JITO", "true").lower() in ("1","true","yes")

client        = Client(RPC)
payer_keypair = Keypair.from_base58_string(PRIV_KEY)
SOL_DECIMAL   = 1e9

WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
RAY_V4 = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
RAY_AUTHORITY_V4 = Pubkey.from_string("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1")
OPEN_BOOK_PROGRAM = Pubkey.from_string("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX")
TOKEN_PROGRAM_ID  = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

LIQUIDITY_STATE_LAYOUT_V4 = cStruct(
    "status" / Int64ul, "nonce" / Int64ul, "orderNum" / Int64ul, "depth" / Int64ul,
    "coinDecimals" / Int64ul, "pcDecimals" / Int64ul, "state" / Int64ul, "resetFlag" / Int64ul,
    "minSize" / Int64ul, "volMaxCutRatio" / Int64ul, "amountWaveRatio" / Int64ul,
    "coinLotSize" / Int64ul, "pcLotSize" / Int64ul, "minPriceMultiplier" / Int64ul, "maxPriceMultiplier" / Int64ul,
    "systemDecimalsValue" / Int64ul, "minSeparateNumerator" / Int64ul, "minSeparateDenominator" / Int64ul,
    "tradeFeeNumerator" / Int64ul, "tradeFeeDenominator" / Int64ul, "pnlNumerator" / Int64ul, "pnlDenominator" / Int64ul,
    "swapFeeNumerator" / Int64ul, "swapFeeDenominator" / Int64ul, "needTakePnlCoin" / Int64ul, "needTakePnlPc" / Int64ul,
    "totalPnlPc" / Int64ul, "totalPnlCoin" / Int64ul, "poolOpenTime" / Int64ul, "punishPcAmount" / Int64ul,
    "punishCoinAmount" / Int64ul, "orderbookToInitTime" / Int64ul,
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

ACCOUNT_FLAGS_LAYOUT = BitsSwapped(BitStruct(
    "initialized" / Flag, "market" / Flag, "open_orders" / Flag, "request_queue" / Flag, "event_queue" / Flag, "bids" / Flag, "asks" / Flag, Const(0, BitsInteger(57))
))
MARKET_STATE_LAYOUT_V3 = cStruct(
    Padding(5), "account_flags" / ACCOUNT_FLAGS_LAYOUT, "own_address" / Bytes(32), "vault_signer_nonce" / Int64ul,
    "base_mint" / Bytes(32), "quote_mint" / Bytes(32), "base_vault" / Bytes(32),
    "base_deposits_total" / Int64ul, "base_fees_accrued" / Int64ul, "quote_vault" / Bytes(32),
    "quote_deposits_total" / Int64ul, "quote_fees_accrued" / Int64ul, "quote_dust_threshold" / Int64ul,
    "request_queue" / Bytes(32), "event_queue" / Bytes(32), "bids" / Bytes(32), "asks" / Bytes(32),
    "base_lot_size" / Int64ul, "quote_lot_size" / Int64ul, "fee_rate_bps" / Int64ul, "referrer_rebate_accrued" / Int64ul, Padding(7),
)

def _decode_account_data(value_data):
    if isinstance(value_data, (bytes, bytearray)): return bytes(value_data)
    if isinstance(value_data, (list, tuple)) and len(value_data) >= 1: return base64.b64decode(value_data[0])
    raise ValueError("Formato de data no reconocido")

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

def get_pair_address_from_api(mint: str) -> Optional[str]:
    url = f"https://api-v3.raydium.io/pools/info/mint?mint1={mint}&poolType=all&poolSortField=default&sortType=desc&pageSize=1&page=1"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        pools = r.json().get("data", {}).get("data", []) or []
        if not pools: return None
        pool = pools[0]
        return pool.get("id") if pool.get("programId") == str(RAY_V4) else None
    except Exception:
        return None

def get_pair_address_from_rpc(token_address: str) -> Optional[str]:
    BASE_OFFSET = 400; QUOTE_OFFSET = 432; DATA_LENGTH_FILTER = 752
    QUOTE_MINT = str(WSOL)

    def fetch_amm_id(base_mint: str, quote_mint: str) -> Optional[str]:
        try:
            filters = [
                {"dataSize": DATA_LENGTH_FILTER},
                MemcmpOpts(offset=BASE_OFFSET, bytes=base_mint),
                MemcmpOpts(offset=QUOTE_OFFSET, bytes=quote_mint),
            ]
            resp = client.get_program_accounts(RAY_V4, commitment=Processed, filters=filters)
            if resp.value:
                return str(resp.value[0].pubkey)
        except Exception:
            pass
        return None
    return fetch_amm_id(token_address, QUOTE_MINT) or fetch_amm_id(QUOTE_MINT, token_address)

def fetch_pool_keys(pair_address: str) -> Optional[PoolKeys]:
    try:
        amm_id = Pubkey.from_string(pair_address)
        amm_info = client.get_account_info(amm_id, commitment=Processed).value
        if amm_info is None: return None
        amm_dec = LIQUIDITY_STATE_LAYOUT_V4.parse(_decode_account_data(amm_info.data))
        market_id = Pubkey.from_bytes(amm_dec.serumMarket)
        market_info = client.get_account_info(market_id, commitment=Processed).value
        if market_info is None: return None
        market_dec = MARKET_STATE_LAYOUT_V3.parse(_decode_account_data(market_info.data))
        nonce = market_dec.vault_signer_nonce
        market_auth = Pubkey.create_program_address([bytes(market_id), int(nonce).to_bytes(8, "little")], OPEN_BOOK_PROGRAM)
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
    except Exception:
        return None

def sol_for_tokens(sol_in: float, base_res: float, quote_res: float, fee_pct: float = 0.25) -> float:
    eff = sol_in * (1 - fee_pct/100); k = base_res * quote_res
    new_base = k / (quote_res + eff); return max(0.0, base_res - new_base)

def tokens_for_sol(tok_in: float, base_res: float, quote_res: float, fee_pct: float = 0.25) -> float:
    eff = tok_in * (1 - fee_pct/100); k = base_res * quote_res
    new_quote = k / (base_res + eff); return max(0.0, quote_res - new_quote)

def get_token_reserves(pool: PoolKeys) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    try:
        balances = client.get_multiple_accounts_json_parsed([pool.base_vault, pool.quote_vault], Processed).value
        token_ui = balances[0].data.parsed["info"]["tokenAmount"]["uiAmount"]
        sol_ui   = balances[1].data.parsed["info"]["tokenAmount"]["uiAmount"]
        if token_ui is None or sol_ui is None: return None, None, None
        if pool.base_mint == WSOL:
            base_res, quote_res, token_dec = sol_ui, token_ui, pool.quote_decimals
        else:
            base_res, quote_res, token_dec = token_ui, sol_ui, pool.base_decimals
        return float(base_res), float(quote_res), int(token_dec)
    except Exception:
        return None, None, None

def get_token_balance(owner: Pubkey, mint_str: str) -> float:
    try:
        mint = Pubkey.from_string(mint_str)
        resp = client.get_token_accounts_by_owner_json_parsed(owner, TokenAccountOpts(mint=mint), commitment=Processed)
        if resp.value:
            ui = resp.value[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
            return float(ui or 0.0)
        return 0.0
    except Exception:
        return 0.0

def make_swap_instruction(amount_in: int, min_out: int, token_in: Pubkey, token_out: Pubkey, acc: PoolKeys, owner: Keypair) -> Optional[Instruction]:
    try:
        keys = [
            SoldersAccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=acc.amm_id, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=RAY_AUTHORITY_V4, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=acc.open_orders, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.target_orders, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.base_vault, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.quote_vault, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=OPEN_BOOK_PROGRAM, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=acc.market_id, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.bids, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.asks, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.event_queue, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.market_base_vault, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.market_quote_vault, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=acc.market_authority, is_signer=False, is_writable=False),
            SoldersAccountMeta(pubkey=token_in, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=token_out, is_signer=False, is_writable=True),
            SoldersAccountMeta(pubkey=owner.pubkey(), is_signer=True, is_writable=False),
        ]
        data = cStruct("instruction"/Int8ul, "amount_in"/Int64ul, "min_amount_out"/Int64ul).build(dict(instruction=9, amount_in=amount_in, min_amount_out=min_out))
        return Instruction(RAY_V4, data, keys)
    except Exception:
        return None

def _send_vtx(message: MessageV0) -> Optional[str]:
    vtx = VersionedTransaction(message, [payer_keypair])
    if USE_JITO and send_v0_with_jito_bundle is not None:
        res = send_v0_with_jito_bundle(vtx, payer_keypair)
        print("Jito result:", res)
        return "JITO_CONFIRMED" if res and res.get("confirmed") else None
    sig = client.send_transaction(vtx, opts=TxOpts(skip_preflight=True)).value
    print("Tx:", sig)
    return str(sig)


def _resolve_pair(mint: str) -> Optional[str]:
    return get_pair_address_from_api(mint) or get_pair_address_from_rpc(mint)

# --------- API pública (uniforme con el router) ---------

def buy(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    pair = _resolve_pair(mint)
    if not pair: return False
    pool = fetch_pool_keys(pair); 
    if not pool: return False

    real_token_mint = pool.base_mint if pool.base_mint != WSOL else pool.quote_mint
    amount_in = int(sol_amount * SOL_DECIMAL)

    base_res, quote_res, token_dec = get_token_reserves(pool)
    if base_res is None or quote_res is None or token_dec is None: return False
    estimated_out = sol_for_tokens(sol_amount, base_res, quote_res)
    min_out = int(estimated_out * (1 - max(0, slippage_pct)/100) * (10 ** token_dec))

    # token account dest
    tok_accs = client.get_token_accounts_by_owner(payer_keypair.pubkey(), TokenAccountOpts(mint=real_token_mint), Processed).value
    if tok_accs:
        token_account = tok_accs[0].pubkey; create_tok_ix = None
    else:
        token_account = get_associated_token_address(payer_keypair.pubkey(), real_token_mint)
        create_tok_ix = create_associated_token_account(payer_keypair.pubkey(), payer_keypair.pubkey(), real_token_mint)

    # WSOL temp
    seed = os.urandom(16).hex()  # 32 chars exactos, siempre válido

    wsol_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
    rent = Token.get_min_balance_rent_for_exempt_for_account(client)
    create_wsol = create_account_with_seed(CreateAccountWithSeedParams(from_pubkey=payer_keypair.pubkey(), to_pubkey=wsol_account, base=payer_keypair.pubkey(), seed=seed, lamports=int(rent + amount_in), space=165, owner=TOKEN_PROGRAM_ID))
    init_wsol   = initialize_account(InitializeAccountParams(program_id=TOKEN_PROGRAM_ID, account=wsol_account, mint=WSOL, owner=payer_keypair.pubkey()))

    swap_ix = make_swap_instruction(amount_in, min_out, wsol_account, token_account, pool, payer_keypair)
    if not swap_ix: return False
    close_wsol = close_account(CloseAccountParams(program_id=TOKEN_PROGRAM_ID, account=wsol_account, dest=payer_keypair.pubkey(), owner=payer_keypair.pubkey()))

    ixs = [set_compute_unit_limit(UNIT_BUDGET), set_compute_unit_price(UNIT_PRICE), create_wsol, init_wsol]
    if create_tok_ix: ixs.append(create_tok_ix)
    ixs += [swap_ix, close_wsol]
    bh = client.get_latest_blockhash().value.blockhash
    msg = MessageV0.try_compile(payer_keypair.pubkey(), ixs, [], bh)
    sig = _send_vtx(msg)
    return True if (sig == "JITO_CONFIRMED" or (sig and _confirm(sig))) else False

def sell(mint: str, percentage: int, slippage_pct: int) -> bool:
    if not (1 <= int(percentage) <= 100): return False
    pair = _resolve_pair(mint)
    if not pair: return False
    pool = fetch_pool_keys(pair)
    if not pool: return False

    real_token_mint = pool.base_mint if pool.base_mint != WSOL else pool.quote_mint
    bal = get_token_balance(payer_keypair.pubkey(), str(real_token_mint))
    if bal <= 0: return False

    amount_to_sell = bal * (int(percentage)/100.0)
    base_res, quote_res, token_dec = get_token_reserves(pool)
    if base_res is None or quote_res is None or token_dec is None: return False
    sol_out = tokens_for_sol(amount_to_sell, base_res, quote_res)
    min_sol_out = int(sol_out * (1 - max(0, slippage_pct)/100) * SOL_DECIMAL)
    amount_in = int(amount_to_sell * (10 ** token_dec))

    token_account = get_associated_token_address(payer_keypair.pubkey(), real_token_mint)

    seed = os.urandom(16).hex()  # 32 chars exactos, siempre válido

    wsol_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
    rent = Token.get_min_balance_rent_for_exempt_for_account(client)
    create_wsol = create_account_with_seed(CreateAccountWithSeedParams(from_pubkey=payer_keypair.pubkey(), to_pubkey=wsol_account, base=payer_keypair.pubkey(), seed=seed, lamports=int(rent), space=165, owner=TOKEN_PROGRAM_ID))
    init_wsol   = initialize_account(InitializeAccountParams(program_id=TOKEN_PROGRAM_ID, account=wsol_account, mint=WSOL, owner=payer_keypair.pubkey()))

    swap_ix = make_swap_instruction(amount_in, min_sol_out, token_account, wsol_account, pool, payer_keypair)
    if not swap_ix: return False
    close_wsol = close_account(CloseAccountParams(program_id=TOKEN_PROGRAM_ID, account=wsol_account, dest=payer_keypair.pubkey(), owner=payer_keypair.pubkey()))

    ixs = [set_compute_unit_limit(UNIT_BUDGET), set_compute_unit_price(UNIT_PRICE), create_wsol, init_wsol, swap_ix, close_wsol]
    if int(percentage) == 100:
        close_token = close_account(CloseAccountParams(program_id=TOKEN_PROGRAM_ID, account=token_account, dest=payer_keypair.pubkey(), owner=payer_keypair.pubkey()))
        ixs.append(close_token)

    bh = client.get_latest_blockhash().value.blockhash
    msg = MessageV0.try_compile(payer_keypair.pubkey(), ixs, [], bh)
    sig = _send_vtx(msg)
    return True if (sig == "JITO_CONFIRMED" or (sig and _confirm(sig))) else False

def _confirm(sig_str: str) -> bool:
    tries = 1
    while tries < 20:
        try:
            res = client.get_transaction(sig_str, encoding="json", commitment=Confirmed, max_supported_transaction_version=0)
            if res.value is not None:
                err = json.loads(res.value.transaction.meta.to_json()).get("err")
                return err is None
        except Exception:
            pass
        time.sleep(1.5); tries += 1
    return False
