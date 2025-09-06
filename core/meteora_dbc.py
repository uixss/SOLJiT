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
 
from solana.rpc.api import Client
from solders.pubkey import Pubkey  # type: ignore

from solana.rpc.commitment import Processed
from solana.rpc.types import MemcmpOpts
from dataclasses import dataclass
from typing import List
from construct import Container, Struct, Int64ul, Int8ul, Array, Bytes, Padding
from construct.core import Construct
from solders.pubkey import Pubkey # type: ignore
from dataclasses import dataclass
from typing import List
from construct import Container, Struct, Int8ul, Int16ul, Int32ul, Int64ul, Array, Bytes, Padding
from construct.core import Construct
from solders.pubkey import Pubkey #type: ignore
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

METEORA_DBC_PROGRAM = Pubkey.from_string("dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
POOL_AUTHORITY = Pubkey.from_string("FhVo3mqL8PW5pH5U2CN4XE33DokiyZnUwuGpH2hmHLuM")
REFERRAL_TOKEN_ACC = Pubkey.from_string("dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN")
EVENT_AUTH = Pubkey.from_string("8Ks12pbrD6PXxfty1hVQiE9sc289zgU1zHkvXhrSdriF")
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

BASE_FEE_CONFIG_LAYOUT = Struct(
    "cliff_fee_numerator" / Int64ul,
    "second_factor"       / Int64ul,
    "third_factor"        / Int64ul,
    "first_factor"        / Int16ul,
    "base_fee_mode"       / Int8ul,
    "padding_0"           / Array(5, Int8ul),
)

DYNAMIC_FEE_CONFIG_LAYOUT = Struct(
    "initialized"              / Int8ul,
    "padding"                  / Array(7, Int8ul),
    "max_volatility_accumulator"/ Int32ul,
    "variable_fee_control"     / Int32ul,
    "bin_step"                 / Int16ul,
    "filter_period"            / Int16ul,
    "decay_period"             / Int16ul,
    "reduction_factor"         / Int16ul,
    "padding2"                 / Array(8, Int8ul),
    "bin_step_u128"            / Int128ul(),
)

POOL_FEES_CONFIG_LAYOUT = Struct(
    "base_fee"             / BASE_FEE_CONFIG_LAYOUT,
    "dynamic_fee"          / DYNAMIC_FEE_CONFIG_LAYOUT,
    "padding_0"            / Array(5, Int64ul),
    "padding_1"            / Array(6, Int8ul),
    "protocol_fee_percent" / Int8ul,
    "referral_fee_percent" / Int8ul,
)

LOCKED_VESTING_CONFIG_LAYOUT = Struct(
    "amount_per_period"                  / Int64ul,
    "cliff_duration_from_migration_time" / Int64ul,
    "frequency"                         / Int64ul,
    "number_of_period"                  / Int64ul,
    "cliff_unlock_amount"               / Int64ul,
    "_padding"                          / Int64ul,
)

LIQUIDITY_DISTRIBUTION_CONFIG_LAYOUT = Struct(
    "sqrt_price" / Int128ul(),
    "liquidity"  / Int128ul(),
)

POOL_CONFIG_LAYOUT = Struct(
    Padding(8),
    "quote_mint"                    / Bytes(32),
    "fee_claimer"                   / Bytes(32),
    "leftover_receiver"             / Bytes(32),
    "pool_fees"                     / POOL_FEES_CONFIG_LAYOUT,
    "collect_fee_mode"              / Int8ul,
    "migration_option"              / Int8ul,
    "activation_type"               / Int8ul,
    "token_decimal"                 / Int8ul,
    "version"                       / Int8ul,
    "token_type"                    / Int8ul,
    "quote_token_flag"              / Int8ul,
    "partner_locked_lp_percentage"  / Int8ul,
    "partner_lp_percentage"         / Int8ul,
    "creator_locked_lp_percentage"  / Int8ul,
    "creator_lp_percentage"         / Int8ul,
    "migration_fee_option"          / Int8ul,
    "fixed_token_supply_flag"       / Int8ul,
    "creator_trading_fee_percentage"/ Int8ul,
    "token_update_authority"        / Int8ul,
    "migration_fee_percentage"      / Int8ul,
    "creator_migration_fee_percentage"/ Int8ul,
    "_padding_1"                    / Array(7, Int8ul),
    "swap_base_amount"              / Int64ul,
    "migration_quote_threshold"     / Int64ul,
    "migration_base_threshold"      / Int64ul,
    "migration_sqrt_price"          / Int128ul(),
    "locked_vesting_config"         / LOCKED_VESTING_CONFIG_LAYOUT,
    "pre_migration_token_supply"    / Int64ul,
    "post_migration_token_supply"   / Int64ul,
    "_padding_2"                    / Array(2, Int128ul()),
    "sqrt_start_price"              / Int128ul(),
    "curve"                         / Array(20, LIQUIDITY_DISTRIBUTION_CONFIG_LAYOUT),
)

@dataclass
class BaseFeeConfig:
    cliff_fee_numerator: int
    second_factor: int
    third_factor: int
    first_factor: int
    base_fee_mode: int

@dataclass
class DynamicFeeConfig:
    initialized: int
    max_volatility_accumulator: int
    variable_fee_control: int
    bin_step: int
    filter_period: int
    decay_period: int
    reduction_factor: int
    bin_step_u128: int

@dataclass
class PoolFeesConfig:
    base_fee: BaseFeeConfig
    dynamic_fee: DynamicFeeConfig
    protocol_fee_percent: int
    referral_fee_percent: int

@dataclass
class LockedVestingConfig:
    amount_per_period: int
    cliff_duration_from_migration_time: int
    frequency: int
    number_of_period: int
    cliff_unlock_amount: int

@dataclass
class LiquidityDistributionConfig:
    sqrt_price: int
    liquidity: int

@dataclass
class PoolConfig:
    quote_mint: Pubkey
    fee_claimer: Pubkey
    leftover_receiver: Pubkey
    pool_fees: PoolFeesConfig
    collect_fee_mode: int
    migration_option: int
    activation_type: int
    token_decimal: int
    version: int
    token_type: int
    quote_token_flag: int
    partner_locked_lp_percentage: int
    partner_lp_percentage: int
    creator_locked_lp_percentage: int
    creator_lp_percentage: int
    migration_fee_option: int
    fixed_token_supply_flag: int
    creator_trading_fee_percentage: int
    token_update_authority: int
    migration_fee_percentage: int
    creator_migration_fee_percentage: int
    swap_base_amount: int
    migration_quote_threshold: int
    migration_base_threshold: int
    migration_sqrt_price: int
    locked_vesting_config: LockedVestingConfig
    pre_migration_token_supply: int
    post_migration_token_supply: int
    sqrt_start_price: int
    curve: List[LiquidityDistributionConfig]

def parse_pool_config(c: Container) -> PoolConfig:
    return PoolConfig(
        quote_mint=Pubkey.from_bytes(c.quote_mint),
        fee_claimer=Pubkey.from_bytes(c.fee_claimer),
        leftover_receiver=Pubkey.from_bytes(c.leftover_receiver),
        pool_fees=PoolFeesConfig(
            base_fee=BaseFeeConfig(
                cliff_fee_numerator=c.pool_fees.base_fee.cliff_fee_numerator,
                second_factor=c.pool_fees.base_fee.second_factor,
                third_factor=c.pool_fees.base_fee.third_factor,
                first_factor=c.pool_fees.base_fee.first_factor,
                base_fee_mode=c.pool_fees.base_fee.base_fee_mode,
            ),
            dynamic_fee=DynamicFeeConfig(
                initialized=c.pool_fees.dynamic_fee.initialized,
                max_volatility_accumulator=c.pool_fees.dynamic_fee.max_volatility_accumulator,
                variable_fee_control=c.pool_fees.dynamic_fee.variable_fee_control,
                bin_step=c.pool_fees.dynamic_fee.bin_step,
                filter_period=c.pool_fees.dynamic_fee.filter_period,
                decay_period=c.pool_fees.dynamic_fee.decay_period,
                reduction_factor=c.pool_fees.dynamic_fee.reduction_factor,
                bin_step_u128=c.pool_fees.dynamic_fee.bin_step_u128,
            ),
            protocol_fee_percent=c.pool_fees.protocol_fee_percent,
            referral_fee_percent=c.pool_fees.referral_fee_percent,
        ),
        collect_fee_mode=c.collect_fee_mode,
        migration_option=c.migration_option,
        activation_type=c.activation_type,
        token_decimal=c.token_decimal,
        version=c.version,
        token_type=c.token_type,
        quote_token_flag=c.quote_token_flag,
        partner_locked_lp_percentage=c.partner_locked_lp_percentage,
        partner_lp_percentage=c.partner_lp_percentage,
        creator_locked_lp_percentage=c.creator_locked_lp_percentage,
        creator_lp_percentage=c.creator_lp_percentage,
        migration_fee_option=c.migration_fee_option,
        fixed_token_supply_flag=c.fixed_token_supply_flag,
        creator_trading_fee_percentage=c.creator_trading_fee_percentage,
        token_update_authority=c.token_update_authority,
        migration_fee_percentage=c.migration_fee_percentage,
        creator_migration_fee_percentage=c.creator_migration_fee_percentage,
        swap_base_amount=c.swap_base_amount,
        migration_quote_threshold=c.migration_quote_threshold,
        migration_base_threshold=c.migration_base_threshold,
        migration_sqrt_price=c.migration_sqrt_price,
        locked_vesting_config=LockedVestingConfig(
            amount_per_period=c.locked_vesting_config.amount_per_period,
            cliff_duration_from_migration_time=c.locked_vesting_config.cliff_duration_from_migration_time,
            frequency=c.locked_vesting_config.frequency,
            number_of_period=c.locked_vesting_config.number_of_period,
            cliff_unlock_amount=c.locked_vesting_config.cliff_unlock_amount,
        ),
        pre_migration_token_supply=c.pre_migration_token_supply,
        post_migration_token_supply=c.post_migration_token_supply,
        sqrt_start_price=c.sqrt_start_price,
        curve=[
            LiquidityDistributionConfig(pt.sqrt_price, pt.liquidity)
            for pt in c.curve
            if pt.sqrt_price != 0
        ],
    )

class Int128ul(Construct):
    def _parse(self, stream, context, path):
        data = stream.read(16)
        return int.from_bytes(data, byteorder="little")
    def _build(self, obj, stream, context, path):
        stream.write(obj.to_bytes(16, byteorder="little"))
        return obj
    def _sizeof(self, context, path):
        return 16

POOL_STATE_LAYOUT = Struct(
    Padding(8),
    "volatility_tracker" / Struct(
        "last_update_timestamp" / Int64ul,
        "padding" / Array(8, Int8ul),
        "sqrt_price_reference" / Int128ul(),
        "volatility_accumulator" / Int128ul(),
        "volatility_reference" / Int128ul(),
    ),
    "config" / Bytes(32),
    "creator" / Bytes(32),
    "base_mint" / Bytes(32),
    "base_vault" / Bytes(32),
    "quote_vault" / Bytes(32),
    "base_reserve" / Int64ul,
    "quote_reserve" / Int64ul,
    "protocol_base_fee" / Int64ul,
    "protocol_quote_fee" / Int64ul,
    "partner_base_fee" / Int64ul,
    "partner_quote_fee" / Int64ul,
    "sqrt_price" / Int128ul(),
    "activation_point" / Int64ul,
    "pool_type" / Int8ul,
    "is_migrated" / Int8ul,
    "is_partner_withdraw_surplus" / Int8ul,
    "is_protocol_withdraw_surplus" / Int8ul,
    "migration_progress" / Int8ul,
    "is_withdraw_leftover" / Int8ul,
    "is_creator_withdraw_surplus" / Int8ul,
    "migration_fee_withdraw_status" / Int8ul,
    "metrics" / Struct(
        "total_protocol_base_fee" / Int64ul,
        "total_protocol_quote_fee" / Int64ul,
        "total_trading_base_fee" / Int64ul,
        "total_trading_quote_fee" / Int64ul,
    ),
    "finish_curve_timestamp" / Int64ul,
    "creator_base_fee" / Int64ul,
    "creator_quote_fee" / Int64ul,
    "_padding_1" / Array(7, Int64ul),
)


@dataclass
class VolatilityTracker:
    last_update_timestamp: int
    sqrt_price_reference: int
    volatility_accumulator: int
    volatility_reference: int


@dataclass
class PoolMetrics:
    total_protocol_base_fee: int
    total_protocol_quote_fee: int
    total_trading_base_fee: int
    total_trading_quote_fee: int


@dataclass
class PoolState:
    pool: Pubkey
    volatility_tracker: VolatilityTracker
    config: Pubkey
    creator: Pubkey
    base_mint: Pubkey
    base_vault: Pubkey
    quote_vault: Pubkey
    base_reserve: int
    quote_reserve: int
    protocol_base_fee: int
    protocol_quote_fee: int
    partner_base_fee: int
    partner_quote_fee: int
    sqrt_price: int
    activation_point: int
    pool_type: int
    is_migrated: int
    is_partner_withdraw_surplus: int
    is_protocol_withdraw_surplus: int
    migration_progress: int
    is_withdraw_leftover: int
    is_creator_withdraw_surplus: int
    migration_fee_withdraw_status: int
    metrics: PoolMetrics
    finish_curve_timestamp: int
    creator_base_fee: int
    creator_quote_fee: int
    _padding_1: List[int]

def parse_pool_state(pool_pubkey: Pubkey, c: Container) -> PoolState:
    return PoolState(
        pool=pool_pubkey,
        volatility_tracker=VolatilityTracker(
            last_update_timestamp=c.volatility_tracker.last_update_timestamp,
            sqrt_price_reference=c.volatility_tracker.sqrt_price_reference,
            volatility_accumulator= c.volatility_tracker.volatility_accumulator,
            volatility_reference=c.volatility_tracker.volatility_reference
        ),
        config=Pubkey.from_bytes(c.config),
        creator=Pubkey.from_bytes(c.creator),
        base_mint=Pubkey.from_bytes(c.base_mint),
        base_vault=Pubkey.from_bytes(c.base_vault),
        quote_vault=Pubkey.from_bytes(c.quote_vault),
        base_reserve=c.base_reserve,
        quote_reserve=c.quote_reserve,
        protocol_base_fee=c.protocol_base_fee,
        protocol_quote_fee=c.protocol_quote_fee,
        partner_base_fee=c.partner_base_fee,
        partner_quote_fee=c.partner_quote_fee,
        sqrt_price=c.sqrt_price,
        activation_point=c.activation_point,
        pool_type=c.pool_type,
        is_migrated=c.is_migrated,
        is_partner_withdraw_surplus=c.is_partner_withdraw_surplus,
        is_protocol_withdraw_surplus=c.is_protocol_withdraw_surplus,
        migration_progress=c.migration_progress,
        is_withdraw_leftover=c.is_withdraw_leftover,
        is_creator_withdraw_surplus=c.is_creator_withdraw_surplus,
        migration_fee_withdraw_status=c.migration_fee_withdraw_status,
        metrics=PoolMetrics(
            total_protocol_base_fee=c.metrics.total_protocol_base_fee,
            total_protocol_quote_fee=c.metrics.total_protocol_quote_fee,
            total_trading_base_fee=c.metrics.total_trading_base_fee,
            total_trading_quote_fee=c.metrics.total_trading_quote_fee,
        ),
        finish_curve_timestamp=c.finish_curve_timestamp,
        creator_base_fee=c.creator_base_fee,
        creator_quote_fee=c.creator_quote_fee,
        _padding_1=list(c._padding_1),
    )
 
def fetch_pool_state(client: Client, pool_str: str):
    pool_pubkey = Pubkey.from_string(pool_str)
    account_info = client.get_account_info_json_parsed(pool_pubkey)
    account_data = account_info.value.data
    decoded_data = POOL_STATE_LAYOUT.parse(account_data)
    pool_state = parse_pool_state(pool_pubkey, decoded_data)
    return pool_state

def fetch_pool_config(client: Client, pool_config: Pubkey):
    account_info = client.get_account_info_json_parsed(pool_config)
    account_data = account_info.value.data
    decoded_data = POOL_CONFIG_LAYOUT.parse(account_data)
    pool_config = parse_pool_config(decoded_data)
    return pool_config

def fetch_pool_from_rpc(client: Client, base_mint: str) -> str | None:
    memcmp_filter_base = MemcmpOpts(offset=136, bytes=base_mint)

    try:
        response = client.get_program_accounts(
            METEORA_DBC_PROGRAM,
            commitment=Processed,
            filters=[memcmp_filter_base],
        )
        accounts = response.value
        if accounts:
            return str(accounts[0].pubkey)
    except:
        return None
    
    return None


RES      = 64        
FEE_DEN  = 10**9     # fee numerator denominator

def ceildiv(a: int, b: int) -> int:
    return -(-a // b)

def _walk_curve(
    amt_in:         int,
    cur_sqrt:       int,
    curve:          list[tuple[int,int]],
    base_for_quote: bool
) -> tuple[int,int]:

    left      = amt_in
    total_out = 0
    sqrt      = cur_sqrt
    shift     = RES * 2

    if base_for_quote:
        # SELL base → quote: walk bins descending
        for lower, liq in reversed(curve):
            if liq == 0 or lower >= sqrt:
                continue
            # max quote in this bin:
            max_q = (liq * (sqrt - lower)) >> shift
            max_b = (liq * (sqrt - lower)) // (sqrt * lower)
            if left < max_b:
                nxt = (liq * sqrt) // (liq + left * sqrt)
                total_out += (liq * (sqrt - nxt)) >> shift
                sqrt       = nxt
                left       = 0
                break
            total_out += max_q
            left       -= max_b
            sqrt        = lower

        # if still left, drain into lowest bin
        if left:
            lower0, liq0 = curve[0]
            nxt = (liq0 * sqrt) // (liq0 + left * sqrt)
            total_out += (liq0 * (sqrt - nxt)) >> shift
            sqrt = nxt

    else:
        # BUY base with quote: walk bins ascending
        for upper, liq in curve:
            if liq == 0 or upper <= sqrt:
                continue
            max_q = (liq * (upper - sqrt)) >> shift
            max_b = (liq * (upper - sqrt)) // (upper * sqrt)
            if left < max_q:
                nxt = sqrt + (left << shift) // liq
                total_out += (liq * (nxt - sqrt)) // (sqrt * nxt)
                sqrt       = nxt
                left       = 0
                break
            total_out += max_b
            left       -= max_q
            sqrt        = upper

        if left:
            upper0, liq0 = curve[-1]
            nxt = sqrt + (left << shift) // liq0
            total_out += (liq0 * (nxt - sqrt)) // (sqrt * nxt)
            sqrt = nxt

    return total_out, sqrt

def swap_base_to_quote(
    amount_in:         int,
    cliff_fee_num:     int,
    protocol_fee_pct:  int,
    referral_fee_pct:  int,
    cur_sqrt:          int,
    curve:             list[tuple[int,int]]
) -> dict:
    raw_q, nxt = _walk_curve(amount_in, cur_sqrt, curve, True)
    gross_fee    = ceildiv(raw_q * cliff_fee_num, FEE_DEN)
    proto_gross  = gross_fee * protocol_fee_pct // 100
    referral     = proto_gross * referral_fee_pct // 100
    proto_net    = proto_gross - referral
    trading_net  = gross_fee - proto_gross

    return {
        "actualInputAmount": str(amount_in),
        "outputAmount":      str(raw_q - gross_fee),
        "nextSqrtPrice":     str(nxt),
        "tradingFee":        str(trading_net),
        "protocolFee":       str(proto_net),
        "referralFee":       str(referral),
    }

def swap_quote_to_base(
    amount_in:         int,
    cliff_fee_num:     int,
    protocol_fee_pct:  int,
    referral_fee_pct:  int,
    cur_sqrt:          int,
    curve:             list[tuple[int,int]]
) -> dict:
    gross_fee   = ceildiv(amount_in * cliff_fee_num, FEE_DEN)
    net_q       = amount_in - gross_fee
    proto_gross = gross_fee * protocol_fee_pct // 100
    referral    = proto_gross * referral_fee_pct // 100
    proto_net   = proto_gross - referral
    trading_net = gross_fee - proto_gross

    raw_b, nxt = _walk_curve(net_q, cur_sqrt, curve, False)
    return {
        "actualInputAmount": str(net_q),
        "outputAmount":      str(raw_b),
        "nextSqrtPrice":     str(nxt),
        "tradingFee":        str(trading_net),
        "protocolFee":       str(proto_net),
        "referralFee":       str(referral),
    }


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

        print("Fetching pool state...")
        pool_state: PoolState = fetch_pool_state(client, pool_str)
        print("Fetching pool config...")
        pool_config: PoolConfig = fetch_pool_config(client, pool_state.config)

        quote_token_decimals = pool_config.token_decimal
        quote_amount_in = int(quote_in * 10 ** quote_token_decimals)
        min_base_amount_out = 0

        curve: list[tuple[int, int]] = [
            (pt.sqrt_price, pt.liquidity)
            for pt in pool_config.curve
            if pt.sqrt_price != 0
        ]
        cliff_fee_num    = pool_config.pool_fees.base_fee.cliff_fee_numerator
        protocol_fee_pct = pool_config.pool_fees.protocol_fee_percent
        referral_fee_pct = pool_config.pool_fees.referral_fee_percent

        estimate = swap_quote_to_base(
            amount_in=quote_amount_in,
            cliff_fee_num=cliff_fee_num,
            protocol_fee_pct=protocol_fee_pct,
            referral_fee_pct=referral_fee_pct,
            cur_sqrt=pool_state.sqrt_price,
            curve=curve
        )

        print(f"Quote→Base estimate: {estimate}")

        print("Checking for existing base token account...")
        base_account_check = client.get_token_accounts_by_owner(
            payer_keypair.pubkey(),
            TokenAccountOpts(pool_state.base_mint),
            Processed,
        )
        if base_account_check.value:
            base_token_account = base_account_check.value[0].pubkey
            base_account_ix = None
            print("Existing base token account found:", base_token_account)
        else:
            base_token_account = get_associated_token_address(
                payer_keypair.pubkey(),
                pool_state.base_mint,
            )
            base_account_ix = create_associated_token_account(
                payer_keypair.pubkey(),
                payer_keypair.pubkey(),
                pool_state.base_mint,
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
                mint=pool_config.quote_mint,
                owner=payer_keypair.pubkey(),
            )
        )

        print("Creating swap instruction...")
        accounts = [
            AccountMeta(POOL_AUTHORITY, False, False),
            AccountMeta(pool_state.config, False, False),
            AccountMeta(pool_state.pool, False, True),
            AccountMeta(quote_token_account, False, True),
            AccountMeta(base_token_account, False, True),
            AccountMeta(pool_state.base_vault, False, True),
            AccountMeta(pool_state.quote_vault, False, True),
            AccountMeta(pool_state.base_mint, False, False),
            AccountMeta(pool_config.quote_mint, False, False),
            AccountMeta(payer_keypair.pubkey(), True, True),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(REFERRAL_TOKEN_ACC, False, False),
            AccountMeta(EVENT_AUTH, False, False),
            AccountMeta(METEORA_DBC_PROGRAM, False, False),
        ]
        data = bytearray.fromhex("f8c69e91e17587c8")
        data.extend(struct.pack("<Q", quote_amount_in))
        data.extend(struct.pack("<Q", min_base_amount_out))
        swap_instr = Instruction(METEORA_DBC_PROGRAM, bytes(data), accounts)

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
        pool_state: PoolState = fetch_pool_state(client, pool_str)
        print("Fetching pool config...")
        pool_config: PoolConfig = fetch_pool_config(client, pool_state.config)

        curve: list[tuple[int, int]] = [
            (pt.sqrt_price, pt.liquidity)
            for pt in pool_config.curve
            if pt.sqrt_price != 0
        ]
        cliff_fee_num    = pool_config.pool_fees.base_fee.cliff_fee_numerator
        protocol_fee_pct = pool_config.pool_fees.protocol_fee_percent
        referral_fee_pct = pool_config.pool_fees.referral_fee_percent

        print("Retrieving base token balance...")
        base_balance = get_token_balance(
            client, payer_keypair.pubkey(), pool_state.base_mint
        )
        if not base_balance:
            print("Base token balance is zero. Nothing to sell.")
            return False

        base_amount_in = int(base_balance * (percentage / 100))
        min_quote_amount_out = 0

        estimate = swap_base_to_quote(
            amount_in=base_amount_in,
            cliff_fee_num=cliff_fee_num,
            protocol_fee_pct=protocol_fee_pct,
            referral_fee_pct=referral_fee_pct,
            cur_sqrt=pool_state.sqrt_price,
            curve=curve,
        )
        
        print(f"Base→Quote estimate: {estimate}")

        print("Getting associated base token account address...")
        base_token_account = get_associated_token_address(
            payer_keypair.pubkey(), pool_state.base_mint
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
                mint=pool_config.quote_mint,
                owner=payer_keypair.pubkey(),
            )
        )

        print("Creating swap instruction...")
        accounts = [
            AccountMeta(POOL_AUTHORITY, False, False),
            AccountMeta(pool_state.config, False, False),
            AccountMeta(pool_state.pool, False, True),
            AccountMeta(base_token_account, False, True),
            AccountMeta(quote_token_account, False, True),
            AccountMeta(pool_state.base_vault, False, True),
            AccountMeta(pool_state.quote_vault, False, True),
            AccountMeta(pool_state.base_mint, False, False),
            AccountMeta(pool_config.quote_mint, False, False),
            AccountMeta(payer_keypair.pubkey(), True, True),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(TOKEN_PROGRAM_ID, False, False),
            AccountMeta(REFERRAL_TOKEN_ACC, False, False),
            AccountMeta(EVENT_AUTH, False, False),
            AccountMeta(METEORA_DBC_PROGRAM, False, False),
        ]
        data = bytearray.fromhex("f8c69e91e17587c8")
        data.extend(struct.pack("<Q", base_amount_in))
        data.extend(struct.pack("<Q", min_quote_amount_out))
        swap_ix = Instruction(METEORA_DBC_PROGRAM, bytes(data), accounts)

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

