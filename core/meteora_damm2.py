import base64
import os
import struct

from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import TokenAccountOpts, TxOpts

from spl.token.client import Token
from spl.token.instructions import (
    CloseAccountParams,
    InitializeAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
    initialize_account,
)

from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.instruction import AccountMeta, Instruction  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from enum import Enum
from decimal import Decimal, getcontext
from typing import NamedTuple, Optional
from typing import Optional
from solana.rpc.api import Client
from solders.pubkey import Pubkey  # type: ignore

from solana.rpc.commitment import Processed
from solana.rpc.types import MemcmpOpts
from dataclasses import dataclass
from typing import List
from construct import Container, Struct, Int8ul, Int16ul, Int32ul, Int64ul, Array, Bytes, Padding
from construct.core import Construct
from solders.pubkey import Pubkey  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
import json
import time

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts

from solders.pubkey import Pubkey  # type: ignore
from solders.signature import Signature  # type: ignore

def get_token_balance(client: Client, pub_key: Pubkey, mint: Pubkey) -> float | None:
    response = client.get_token_accounts_by_owner_json_parsed(
        pub_key,
        TokenAccountOpts(mint=mint),
        commitment=Processed
    )

    if response.value:
        accounts = response.value
        if accounts:
            token_amount = accounts[0].account.data.parsed['info']['tokenAmount']['amount']
            if token_amount:
                return int(token_amount)
    return None

def confirm_txn(client: Client, txn_sig: Signature, max_retries: int = 20, retry_interval: int = 3) -> bool:
    retries = 1
    
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(
                txn_sig, 
                encoding="json", 
                commitment=Confirmed, 
                max_supported_transaction_version=0)
            
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            
            if txn_json['err'] is None:
                print("Transaction confirmed... try count:", retries)
                return True
            
            print("Error: Transaction not confirmed. Retrying...")
            if txn_json['err']:
                print("Transaction failed.")
                return False
        except Exception as e:
            print("Awaiting confirmation... try count:", retries)
            retries += 1
            time.sleep(retry_interval)
    
    print("Max retries reached. Transaction confirmation failed.")
    return None

METEORA_DAMM2_PROGRAM = Pubkey.from_string("cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
POOL_AUTHORITY = Pubkey.from_string("HLnpSz9h2S4hiLQ43rnSD9XkcUThA7B8hQMKmDaiTLcC")
REFERRAL_TOKEN_ACC = Pubkey.from_string("cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG")
EVENT_AUTH = Pubkey.from_string("3rmHSu74h1ZcmAisVcWerTCiRDQbUrBKmcwptYGjHfet")
ACCOUNT_SPACE = 165

class Int128ul(Construct):
    def _parse(self, stream, context, path):
        data = stream.read(16)
        return int.from_bytes(data, byteorder="little")

    def _build(self, obj, stream, context, path):
        stream.write(obj.to_bytes(16, byteorder="little"))
        return obj

    def _sizeof(self, context, path):
        return 16

BASE_FEE_STRUCT_LAYOUT = Struct(
    "cliff_fee_numerator"   / Int64ul,
    "fee_scheduler_mode"    / Int8ul,
    "padding_0"             / Array(5, Int8ul),
    "number_of_period"      / Int16ul,
    "period_frequency"      / Int64ul,
    "reduction_factor"      / Int64ul,
    "padding_1"             / Int64ul,
)

DYNAMIC_FEE_STRUCT_LAYOUT = Struct(
    "initialized"                 / Int8ul,
    "padding"                     / Array(7, Int8ul),
    "max_volatility_accumulator"  / Int32ul,
    "variable_fee_control"        / Int32ul,
    "bin_step"                    / Int16ul,
    "filter_period"               / Int16ul,
    "decay_period"                / Int16ul,
    "reduction_factor"            / Int16ul,
    "last_update_timestamp"       / Int64ul,
    "bin_step_u128"               / Int128ul(),
    "sqrt_price_reference"        / Int128ul(),
    "volatility_accumulator"      / Int128ul(),
    "volatility_reference"        / Int128ul(),
)

POOL_FEES_STRUCT_LAYOUT = Struct(
    "base_fee"               / BASE_FEE_STRUCT_LAYOUT,
    "protocol_fee_percent"   / Int8ul,
    "partner_fee_percent"    / Int8ul,
    "referral_fee_percent"   / Int8ul,
    "padding_0"              / Array(5, Int8ul),
    "dynamic_fee"            / DYNAMIC_FEE_STRUCT_LAYOUT,
    "padding_1"              / Array(2, Int64ul),
)

POOL_METRICS_LAYOUT = Struct(
    "total_lp_a_fee"        / Int128ul(),
    "total_lp_b_fee"        / Int128ul(),
    "total_protocol_a_fee"  / Int64ul,
    "total_protocol_b_fee"  / Int64ul,
    "total_partner_a_fee"   / Int64ul,
    "total_partner_b_fee"   / Int64ul,
    "total_position"        / Int64ul,
    "padding"               / Int64ul,
)

REWARD_INFO_LAYOUT = Struct(
    "initialized"                                  / Int8ul,
    "reward_token_flag"                            / Int8ul,
    "_padding_0"                                   / Array(6, Int8ul),
    "_padding_1"                                   / Array(8, Int8ul),
    "mint"                                         / Bytes(32),
    "vault"                                        / Bytes(32),
    "funder"                                       / Bytes(32),
    "reward_duration"                              / Int64ul,
    "reward_duration_end"                          / Int64ul,
    "reward_rate"                                  / Int128ul(),
    "reward_per_token_stored"                      / Bytes(32),
    "last_update_time"                             / Int64ul,
    "cumulative_seconds_with_empty_liquidity_reward" / Int64ul,
)

POOL_LAYOUT = Struct(
    Padding(8),
    "pool_fees"                   / POOL_FEES_STRUCT_LAYOUT,
    "token_a_mint"                / Bytes(32),
    "token_b_mint"                / Bytes(32),
    "token_a_vault"               / Bytes(32),
    "token_b_vault"               / Bytes(32),
    "whitelisted_vault"           / Bytes(32),
    "partner"                     / Bytes(32),
    "liquidity"                   / Int128ul(),
    "_padding"                    / Int128ul(),
    "protocol_a_fee"              / Int64ul,
    "protocol_b_fee"              / Int64ul,
    "partner_a_fee"               / Int64ul,
    "partner_b_fee"               / Int64ul,
    "sqrt_min_price"              / Int128ul(),
    "sqrt_max_price"              / Int128ul(),
    "sqrt_price"                  / Int128ul(),
    "activation_point"            / Int64ul,
    "activation_type"             / Int8ul,
    "pool_status"                 / Int8ul,
    "token_a_flag"                / Int8ul,
    "token_b_flag"                / Int8ul,
    "collect_fee_mode"            / Int8ul,
    "pool_type"                   / Int8ul,
    "_padding_0"                  / Array(2, Int8ul),
    "fee_a_per_liquidity"         / Bytes(32),
    "fee_b_per_liquidity"         / Bytes(32),
    "permanent_lock_liquidity"    / Int128ul(),
    "metrics"                     / POOL_METRICS_LAYOUT,
    "creator"                     / Bytes(32),
    "_padding_1"                  / Array(6, Int64ul),
    "reward_infos"                / Array(2, REWARD_INFO_LAYOUT),
)

@dataclass
class BaseFeeStruct:
    cliff_fee_numerator: int
    fee_scheduler_mode: int
    padding_0: List[int]
    number_of_period: int
    period_frequency: int
    reduction_factor: int
    padding_1: int

@dataclass
class DynamicFeeStruct:
    initialized: int
    padding: List[int]
    max_volatility_accumulator: int
    variable_fee_control: int
    bin_step: int
    filter_period: int
    decay_period: int
    reduction_factor: int
    last_update_timestamp: int
    bin_step_u128: int
    sqrt_price_reference: int
    volatility_accumulator: int
    volatility_reference: int

@dataclass
class PoolFeesStruct:
    base_fee: BaseFeeStruct
    protocol_fee_percent: int
    partner_fee_percent: int
    referral_fee_percent: int
    padding_0: List[int]
    dynamic_fee: DynamicFeeStruct
    padding_1: List[int]

@dataclass
class PoolMetrics:
    total_lp_a_fee: int
    total_lp_b_fee: int
    total_protocol_a_fee: int
    total_protocol_b_fee: int
    total_partner_a_fee: int
    total_partner_b_fee: int
    total_position: int
    padding: int

@dataclass
class RewardInfo:
    initialized: int
    reward_token_flag: int
    _padding_0: List[int]
    _padding_1: List[int]
    mint: Pubkey
    vault: Pubkey
    funder: Pubkey
    reward_duration: int
    reward_duration_end: int
    reward_rate: int
    reward_per_token_stored: bytes
    last_update_time: int
    cumulative_seconds_with_empty_liquidity_reward: int

@dataclass
class Pool:
    pool: Pubkey
    pool_fees: PoolFeesStruct
    token_a_mint: Pubkey
    token_b_mint: Pubkey
    token_a_vault: Pubkey
    token_b_vault: Pubkey
    whitelisted_vault: Pubkey
    partner: Pubkey
    liquidity: int
    _padding: int
    protocol_a_fee: int
    protocol_b_fee: int
    partner_a_fee: int
    partner_b_fee: int
    sqrt_min_price: int
    sqrt_max_price: int
    sqrt_price: int
    activation_point: int
    activation_type: int
    pool_status: int
    token_a_flag: int
    token_b_flag: int
    collect_fee_mode: int
    pool_type: int
    _padding_0: List[int]
    fee_a_per_liquidity: bytes
    fee_b_per_liquidity: bytes
    permanent_lock_liquidity: int
    metrics: PoolMetrics
    creator: Pubkey
    _padding_1: List[int]
    reward_infos: List[RewardInfo]

def parse_pool(pool_pubkey: Pubkey, c: Container) -> Pool:
    return Pool(
        pool=pool_pubkey,
        pool_fees=PoolFeesStruct(
            base_fee=BaseFeeStruct(
                cliff_fee_numerator=c.pool_fees.base_fee.cliff_fee_numerator,
                fee_scheduler_mode=c.pool_fees.base_fee.fee_scheduler_mode,
                padding_0=list(c.pool_fees.base_fee.padding_0),
                number_of_period=c.pool_fees.base_fee.number_of_period,
                period_frequency=c.pool_fees.base_fee.period_frequency,
                reduction_factor=c.pool_fees.base_fee.reduction_factor,
                padding_1=c.pool_fees.base_fee.padding_1,
            ),
            protocol_fee_percent=c.pool_fees.protocol_fee_percent,
            partner_fee_percent=c.pool_fees.partner_fee_percent,
            referral_fee_percent=c.pool_fees.referral_fee_percent,
            padding_0=list(c.pool_fees.padding_0),
            dynamic_fee=DynamicFeeStruct(
                initialized=c.pool_fees.dynamic_fee.initialized,
                padding=list(c.pool_fees.dynamic_fee.padding),
                max_volatility_accumulator=c.pool_fees.dynamic_fee.max_volatility_accumulator,
                variable_fee_control=c.pool_fees.dynamic_fee.variable_fee_control,
                bin_step=c.pool_fees.dynamic_fee.bin_step,
                filter_period=c.pool_fees.dynamic_fee.filter_period,
                decay_period=c.pool_fees.dynamic_fee.decay_period,
                reduction_factor=c.pool_fees.dynamic_fee.reduction_factor,
                last_update_timestamp=c.pool_fees.dynamic_fee.last_update_timestamp,
                bin_step_u128=c.pool_fees.dynamic_fee.bin_step_u128,
                sqrt_price_reference=c.pool_fees.dynamic_fee.sqrt_price_reference,
                volatility_accumulator=c.pool_fees.dynamic_fee.volatility_accumulator,
                volatility_reference=c.pool_fees.dynamic_fee.volatility_reference,
            ),
            padding_1=list(c.pool_fees.padding_1),
        ),
        token_a_mint=Pubkey.from_bytes(c.token_a_mint),
        token_b_mint=Pubkey.from_bytes(c.token_b_mint),
        token_a_vault=Pubkey.from_bytes(c.token_a_vault),
        token_b_vault=Pubkey.from_bytes(c.token_b_vault),
        whitelisted_vault=Pubkey.from_bytes(c.whitelisted_vault),
        partner=Pubkey.from_bytes(c.partner),
        liquidity=c.liquidity,
        _padding=c._padding,
        protocol_a_fee=c.protocol_a_fee,
        protocol_b_fee=c.protocol_b_fee,
        partner_a_fee=c.partner_a_fee,
        partner_b_fee=c.partner_b_fee,
        sqrt_min_price=c.sqrt_min_price,
        sqrt_max_price=c.sqrt_max_price,
        sqrt_price=c.sqrt_price,
        activation_point=c.activation_point,
        activation_type=c.activation_type,
        pool_status=c.pool_status,
        token_a_flag=c.token_a_flag,
        token_b_flag=c.token_b_flag,
        collect_fee_mode=c.collect_fee_mode,
        pool_type=c.pool_type,
        _padding_0=list(c._padding_0),
        fee_a_per_liquidity=c.fee_a_per_liquidity,
        fee_b_per_liquidity=c.fee_b_per_liquidity,
        permanent_lock_liquidity=c.permanent_lock_liquidity,
        metrics=PoolMetrics(
            total_lp_a_fee=c.metrics.total_lp_a_fee,
            total_lp_b_fee=c.metrics.total_lp_b_fee,
            total_protocol_a_fee=c.metrics.total_protocol_a_fee,
            total_protocol_b_fee=c.metrics.total_protocol_b_fee,
            total_partner_a_fee=c.metrics.total_partner_a_fee,
            total_partner_b_fee=c.metrics.total_partner_b_fee,
            total_position=c.metrics.total_position,
            padding=c.metrics.padding,
        ),
        creator=Pubkey.from_bytes(c.creator),
        _padding_1=list(c._padding_1),
        reward_infos=[
            RewardInfo(
                initialized=ri.initialized,
                reward_token_flag=ri.reward_token_flag,
                _padding_0=list(ri._padding_0),
                _padding_1=list(ri._padding_1),
                mint=Pubkey.from_bytes(ri.mint),
                vault=Pubkey.from_bytes(ri.vault),
                funder=Pubkey.from_bytes(ri.funder),
                reward_duration=ri.reward_duration,
                reward_duration_end=ri.reward_duration_end,
                reward_rate=ri.reward_rate,
                reward_per_token_stored=ri.reward_per_token_stored,
                last_update_time=ri.last_update_time,
                cumulative_seconds_with_empty_liquidity_reward=ri.cumulative_seconds_with_empty_liquidity_reward,
            )
            for ri in c.reward_infos
        ],
    )

 

def fetch_pool_state(client: Client, pool_str: str):
    pool_pubkey = Pubkey.from_string(pool_str)
    info = client.get_account_info_json_parsed(pool_pubkey)
    raw_data = info.value.data
    decoded = POOL_LAYOUT.parse(raw_data)
    return parse_pool(pool_pubkey, decoded)


def fetch_pool_from_rpc(
    client: Client,
    base_mint: str,
    quote_mint: str = "So11111111111111111111111111111111111111112",
) -> Optional[str]:
    try:
        f_base = MemcmpOpts(offset=168, bytes=base_mint)
        f_quote = MemcmpOpts(offset=200, bytes=quote_mint)

        resp = client.get_program_accounts(
            METEORA_DAMM2_PROGRAM,
            commitment=Processed,
            filters=[f_base, f_quote],
        )

        best: Optional[str] = None
        max_liq = -1
        for acct in resp.value:
            pk = acct.pubkey
            state = fetch_pool_state(client, str(pk))
            if state.liquidity > max_liq:
                max_liq = state.liquidity
                best = str(pk)
        return best
    except:
        return None

getcontext().prec = 50

BASIS_POINT_MAX                    = 10_000      # e.g. 10000 bps = 100%
FEE_DENOMINATOR                    = 1_000_000_000  # if your fees are out of 1e9
MAX_FEE_NUMERATOR                  = 500_000_000  # or whatever your protocol max is
SCALE_OFFSET                       = 64          # Q64.64 fixed-point

class FeeSchedulerMode(Enum):
    Constant    = 0
    Linear      = 1
    Exponential = 2

class Rounding(Enum):
    Down = 0
    Up   = 1

class FeeMode(NamedTuple):
    fee_on_input: bool
    fees_on_token_a: bool

class SwapResult(NamedTuple):
    amount_out: int
    total_fee: int
    next_sqrt_price: int

def mul_div(numer: int, mul: int, denom: int, rounding: Rounding) -> int:
    prod = numer * mul
    if rounding == Rounding.Up:
        return (prod + denom - 1) // denom
    else:
        return prod // denom

def get_next_sqrt_price(amount: int, sqrt_price: int, liquidity: int, a_to_b: bool) -> int:
    if a_to_b:
        product     = amount * sqrt_price
        denominator = liquidity + product
        numerator   = liquidity * sqrt_price
        return (numerator + (denominator - 1)) // denominator
    else:
        quotient = (amount << (SCALE_OFFSET * 2)) // liquidity
        return sqrt_price + quotient

def get_amount_a_from_liquidity_delta(
    liquidity: int, cur_sp: int, max_sp: int, rounding: Rounding
) -> int:
    product     = liquidity * (max_sp - cur_sp)
    denominator = cur_sp * max_sp
    if rounding == Rounding.Up:
        return (product + (denominator - 1)) // denominator
    return product // denominator

def get_amount_b_from_liquidity_delta(
    liquidity: int, cur_sp: int, min_sp: int, rounding: Rounding
) -> int:
    one         = 1 << (SCALE_OFFSET * 2)
    delta_price = cur_sp - min_sp
    result      = liquidity * delta_price
    if rounding == Rounding.Up:
        return (result + (one - 1)) // one
    return result >> (SCALE_OFFSET * 2)

def get_next_sqrt_price_from_output(
    sqrt_price: int, liquidity: int, out_amount: int, is_b: bool
) -> int:
    if sqrt_price == 0:
        raise ValueError("sqrt price must be > 0")
    if is_b:
        # √P' = √P - Δy / L  (rounding up)
        quotient = ((out_amount << (SCALE_OFFSET * 2)) + liquidity - 1) // liquidity
        res = sqrt_price - quotient
        if res < 0:
            raise ValueError("sqrt price negative")
        return res
    else:
        # √P' = (L * √P) / (L - Δx * √P)  (rounding down)
        if out_amount == 0:
            return sqrt_price
        prod       = out_amount * sqrt_price
        denom      = liquidity - prod
        if denom <= 0:
            raise ValueError("invalid denom in √P calc")
        num        = liquidity * sqrt_price
        return num // denom

def get_base_fee_numerator(
    mode: FeeSchedulerMode,
    cliff: int,
    period: int,
    reduction: int
) -> int:
    if mode == FeeSchedulerMode.Linear:
        return max(0, cliff - period * reduction)
    else:
        # exponential: cliff * (1 - reduction/BASIS_POINT_MAX)^period
        bps = Decimal(1) - Decimal(reduction) / BASIS_POINT_MAX
        factor = bps ** period
        return int((Decimal(cliff) * factor).to_integral_value(rounding="ROUND_FLOOR"))

def get_dynamic_fee_numerator(
    volatility_acc: int,
    bin_step: int,
    variable_fee_ctrl: int
) -> int:
    if variable_fee_ctrl == 0:
        return 0
    square = volatility_acc * bin_step
    square = square * square
    vfee   = variable_fee_ctrl * square
    # match: (vfee + 1e11 - 1) / 1e11
    return (vfee + 100_000_000_000 - 1) // 100_000_000_000

def get_fee_numerator(
    current_point: int,
    activation_point: int,
    number_of_period: int,
    period_freq: int,
    mode: FeeSchedulerMode,
    cliff_fee: int,
    reduction: int,
    dynamic_params: Optional[dict] = None
) -> int:
    if period_freq == 0 or current_point < activation_point:
        return cliff_fee
    period = min(
        number_of_period,
        (current_point - activation_point) // period_freq
    )
    fee_num = get_base_fee_numerator(mode, cliff_fee, period, reduction)
    if dynamic_params:
        df = get_dynamic_fee_numerator(
            dynamic_params["volatility_accumulator"],
            dynamic_params["bin_step"],
            dynamic_params["variable_fee_control"],
        )
        fee_num += df
    return min(fee_num, MAX_FEE_NUMERATOR)

def get_fee_mode(collect_fee_mode: int, b_to_a: bool) -> FeeMode:
    fee_on_input   = b_to_a and collect_fee_mode == 1  # e.g. OnlyB==1
    fees_on_token_a = b_to_a and collect_fee_mode == 0 # e.g. BothToken==0
    return FeeMode(fee_on_input, fees_on_token_a)

def get_total_fee_on_amount(amount: int, fee_num: int) -> int:
    return mul_div(amount, fee_num, FEE_DENOMINATOR, Rounding.Up)

def get_swap_amount(
    in_amount: int,
    sqrt_price: int,
    liquidity: int,
    trade_fee_numerator: int,
    a_to_b: bool,
    collect_fee_mode: int
) -> SwapResult:
    fee_mode    = get_fee_mode(collect_fee_mode, not a_to_b)
    actual_in   = in_amount
    total_fee   = 0

    # fee on input?
    if fee_mode.fee_on_input:
        total_fee   = get_total_fee_on_amount(in_amount, trade_fee_numerator)
        actual_in   = in_amount - total_fee

    next_sp     = get_next_sqrt_price(actual_in, sqrt_price, liquidity, a_to_b)

    # compute raw out before fees
    if a_to_b:
        out_amount = get_amount_b_from_liquidity_delta(liquidity, sqrt_price, next_sp, Rounding.Down)
    else:
        out_amount = get_amount_a_from_liquidity_delta(liquidity, sqrt_price, next_sp, Rounding.Down)

    # fee on output?
    if not fee_mode.fee_on_input:
        total_fee  = get_total_fee_on_amount(out_amount, trade_fee_numerator)
        out_amount = out_amount - total_fee

    return SwapResult(
        amount_out      = out_amount,
        total_fee       = total_fee,
        next_sqrt_price = next_sp,
    )


 

def buy(
    client: Client,
    payer_keypair: Keypair,
    pool_str: str,
    quote_in: float = 0.1,
    unit_budget: int = 100_000,
    unit_price: int = 1_000_000,
) -> bool:
    try:
        print(f"Starting buy transaction for pool: {pool_str}")
        quote_amount_in = int(quote_in * 10**9)
        min_base_amount_out = 0
        
        print("Fetching pool state...")
        pool_state: Pool = fetch_pool_state(client, pool_str)

        print("Checking for existing base token account...")
        base_account_check = client.get_token_accounts_by_owner(
            payer_keypair.pubkey(),
            TokenAccountOpts(pool_state.token_a_mint),
            Processed,
        )
        if base_account_check.value:
            base_token_account = base_account_check.value[0].pubkey
            base_account_ix = None
            print("Existing base token account found:", base_token_account)
        else:
            base_token_account = get_associated_token_address(
                payer_keypair.pubkey(),
                pool_state.token_a_mint,
            )
            base_account_ix = create_associated_token_account(
                payer_keypair.pubkey(),
                payer_keypair.pubkey(),
                pool_state.token_a_mint,
            )
            print("Will create base token ATA:", base_token_account)

        print("Generating seed for quote token account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        quote_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(),
            seed,
            TOKEN_PROGRAM_ID,
        )
        quote_rent = Token.get_min_balance_rent_for_exempt_for_account(client)

        print("Creating and initializing quote token account...")
        create_quote_token_account_ix = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=quote_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(quote_rent + quote_amount_in),
                space=ACCOUNT_SPACE,
                owner=TOKEN_PROGRAM_ID,
            )
        )
        
        init_quote_token_account_ix = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=quote_token_account,
                mint=pool_state.token_b_mint,
                owner=payer_keypair.pubkey(),
            )
        )

        print("Creating swap instruction...")
        accounts = [
            AccountMeta(POOL_AUTHORITY, False, False),
            AccountMeta(pool_state.pool, False, True),
            AccountMeta(quote_token_account, False, True),
            AccountMeta(base_token_account, False, True),
            AccountMeta(pool_state.token_a_vault, False, True),
            AccountMeta(pool_state.token_b_vault, False, True),
            AccountMeta(pool_state.token_a_mint, False, False),
            AccountMeta(pool_state.token_b_mint, False, False),
            AccountMeta(payer_keypair.pubkey(), True, True),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(REFERRAL_TOKEN_ACC, False, False),
            AccountMeta(EVENT_AUTH, False, False),
            AccountMeta(METEORA_DAMM2_PROGRAM, False, False),
        ]
        data = bytearray.fromhex("f8c69e91e17587c8")
        data.extend(struct.pack("<Q", quote_amount_in))
        data.extend(struct.pack("<Q", min_base_amount_out))
        swap_instr = Instruction(METEORA_DAMM2_PROGRAM, bytes(data), accounts)

        print("Preparing to close quote token account after swap...")
        close_quote_token_account_ix = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=quote_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_quote_token_account_ix,
            init_quote_token_account_ix,
        ]
        if base_account_ix:
            instructions.append(base_account_ix)
        instructions.extend([swap_instr, close_quote_token_account_ix])

        print("Compiling transaction message...")
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            client.get_latest_blockhash().value.blockhash,
        )
        print("Sending transaction...")
        txn_sig = client.send_transaction(
            txn=VersionedTransaction(compiled_message, [payer_keypair]),
            opts=TxOpts(skip_preflight=False),
        ).value
        print("Transaction Signature:", txn_sig)

        print("Confirming transaction...")
        confirmed = confirm_txn(client, txn_sig)
        print("Transaction confirmed:", confirmed)
        return confirmed

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False


def sell(
    client: Client,
    payer_keypair: Keypair,
    pool_str: str,
    percentage: int = 100,
    unit_budget: int = 100_000,
    unit_price: int = 1_000_000,
) -> bool:
    try:
        print(f"Starting sell transaction for pool: {pool_str}")

        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        print("Fetching pool state...")
        pool_state = fetch_pool_state(client, pool_str)

        print("Retrieving base token balance...")
        base_balance = get_token_balance(
            client, payer_keypair.pubkey(), pool_state.token_a_mint
        )
        if not base_balance:
            print("Base token balance is zero. Nothing to sell.")
            return False

        base_amount_in = int(base_balance * (percentage / 100))
        min_quote_amount_out = 0

        print("Getting associated base token account address...")
        base_token_account = get_associated_token_address(
            payer_keypair.pubkey(), pool_state.token_a_mint
        )

        print("Generating seed for quote token account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        quote_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )
        quote_rent = Token.get_min_balance_rent_for_exempt_for_account(client)

        print("Creating and initializing quote token account...")
        create_quote_token_account_ix = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=quote_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(quote_rent),
                space=ACCOUNT_SPACE,
                owner=TOKEN_PROGRAM_ID,
            )
        )
        
        init_quote_token_account_ix = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=quote_token_account,
                mint=pool_state.token_b_mint,
                owner=payer_keypair.pubkey(),
            )
        )

        print("Creating swap instruction...")
        accounts = [
            AccountMeta(POOL_AUTHORITY, False, False),
            AccountMeta(pool_state.pool, False, True),
            AccountMeta(base_token_account, False, True),
            AccountMeta(quote_token_account, False, True),
            AccountMeta(pool_state.token_a_vault, False, True),
            AccountMeta(pool_state.token_b_vault, False, True),
            AccountMeta(pool_state.token_a_mint, False, False),
            AccountMeta(pool_state.token_b_mint, False, False),
            AccountMeta(payer_keypair.pubkey(), True, True),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(REFERRAL_TOKEN_ACC, False, False),
            AccountMeta(EVENT_AUTH, False, False),
            AccountMeta(METEORA_DAMM2_PROGRAM, False, False),
        ]
        data = bytearray.fromhex("f8c69e91e17587c8")
        data.extend(struct.pack("<Q", base_amount_in))
        data.extend(struct.pack("<Q", min_quote_amount_out))
        swap_ix = Instruction(METEORA_DAMM2_PROGRAM, bytes(data), accounts)

        print("Preparing to close quote token account after swap...")
        close_quote_token_account_ix = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=quote_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_quote_token_account_ix,
            init_quote_token_account_ix,
            swap_ix,
            close_quote_token_account_ix,
        ]

        if percentage == 100:
            print("Preparing to close base token account (100% sell)...")
            close_base_token_account_ix = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM_ID,
                    account=base_token_account,
                    dest=payer_keypair.pubkey(),
                    owner=payer_keypair.pubkey(),
                )
            )
            instructions.append(close_base_token_account_ix)

        print("Compiling transaction message...")
        blockhash = client.get_latest_blockhash().value.blockhash
        compiled_msg = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            blockhash,
        )
        print("Sending transaction...")
        sig = client.send_transaction(
            txn=VersionedTransaction(compiled_msg, [payer_keypair]),
            opts=TxOpts(skip_preflight=False),
        ).value
        print("Transaction Signature:", sig)

        print("Confirming transaction...")
        confirmed = confirm_txn(client, sig)
        print("Transaction confirmed:", confirmed)
        return confirmed

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False
