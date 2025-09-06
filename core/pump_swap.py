import json
import time

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts
import base64
import os
import struct
from typing import Optional

from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import TokenAccountOpts, TxOpts

from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.instruction import AccountMeta, Instruction  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
from solders.system_program import (
    CreateAccountWithSeedParams,
    create_account_with_seed,
)
from solders.transaction import VersionedTransaction  # type: ignore

from spl.token.client import Token
from spl.token.instructions import (
    CloseAccountParams,
    InitializeAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
    initialize_account,
)

from solders.pubkey import Pubkey  # type: ignore
from solders.signature import Signature  # type: ignore
from solders.pubkey import Pubkey #type: ignore

GLOBAL_CONFIG = Pubkey.from_string("ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
ASSOCIATED_TOKEN_PROGRAM = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
PROTOCOL_FEE_RECIPIENT = Pubkey.from_string("62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV")
PROTOCOL_FEE_RECIPIENT_TOKEN_ACCOUNT = Pubkey.from_string("94qWNrtmfn42h3ZjUZwWvK1MEo9uVmmrBPd2hpNjYDjb")
EVENT_AUTH = Pubkey.from_string("GS4CU59F31iL7aR2Q8zVS8DRrcRnXX1yjQ66TqNVQnaR")
PF_AMM = Pubkey.from_string("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA")
GLOBAL_VOL_ACC = Pubkey.from_string("C2aFPdENg4A2HQsmrd5rTw5TaYBX5Ku887cWjbFKtZpw")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
ACCOUNT_SPACE = 165
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


 

def buy(client: Client, payer_keypair: Keypair, pair_address: str, sol_in: float = 0.1, slippage: int = 5, unit_budget: int = 150_000, unit_price: int = 1_000_000) -> bool:
    try:
        print(f"Starting buy transaction for pair address: {pair_address}")

        print("Fetching pool keys...")
        pool_keys: Optional[PoolKeys] = fetch_pool_keys(client, pair_address)
        
        if pool_keys is None:
            print("No pool keys found, aborting transaction.")
            return False
        print("Pool keys fetched successfully.")

        print("Fetching creator vault info...")
        creator_vault_authority, creator_vault_ata = get_creator_vault_info(client, pool_keys.creator)
        if creator_vault_authority is None or creator_vault_ata is None:
            print("No creator vault info found, aborting transaction.")
            return False
        print("Creator vault info fetched successfully.")

        mint = pool_keys.base_mint
        token_info = client.get_account_info_json_parsed(mint).value
        base_token_program = token_info.owner
        decimal = token_info.data.parsed['info']['decimals']

        print("Calculating transaction amounts...")
        sol_decimal = 1e9
        token_decimal = 10**decimal
        slippage_adjustment = 1 + (slippage / 100)
        max_quote_amount_in = int((sol_in * slippage_adjustment) * sol_decimal)

        base_reserve, quote_reserve = get_pool_reserves(client, pool_keys)
        raw_sol_in = int(sol_in * sol_decimal)
        base_amount_out = sol_for_tokens(raw_sol_in, base_reserve, quote_reserve)
        print(f"Max Quote Amount In: {max_quote_amount_in / sol_decimal} | Base Amount Out: {base_amount_out / token_decimal}")

        print("Checking for existing token account...")
        token_account_check = client.get_token_accounts_by_owner(payer_keypair.pubkey(), TokenAccountOpts(mint), Processed)
        
        if token_account_check.value:
            token_account = token_account_check.value[0].pubkey
            token_account_instruction = None
            print("Existing token account found.")
        else:
            token_account = get_associated_token_address(payer_keypair.pubkey(), mint, base_token_program)
            token_account_instruction = create_associated_token_account(payer_keypair.pubkey(), payer_keypair.pubkey(), mint, base_token_program)
            print("No existing token account found; creating associated token account.")

        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
        balance_needed = Token.get_min_balance_rent_for_exempt_for_account(client)

        print("Creating and initializing WSOL account...")
        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(balance_needed + max_quote_amount_in),
                space=ACCOUNT_SPACE,
                owner=TOKEN_PROGRAM_ID,
            )
        )

        init_wsol_account_instruction = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        user_volume_accumulator = Pubkey.find_program_address([b"user_volume_accumulator", bytes(payer_keypair.pubkey())], PF_AMM)[0]

        print("Creating swap instructions...")
        keys = [
            AccountMeta(pubkey=pool_keys.amm, is_signer=False, is_writable=True),
            AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=GLOBAL_CONFIG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=pool_keys.base_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=pool_keys.quote_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=wsol_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=pool_keys.pool_base_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=pool_keys.pool_quote_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=PROTOCOL_FEE_RECIPIENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PROTOCOL_FEE_RECIPIENT_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=base_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=ASSOCIATED_TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTH, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PF_AMM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=creator_vault_ata, is_signer=False, is_writable=True),
            AccountMeta(pubkey=creator_vault_authority, is_signer=False, is_writable=False),
            AccountMeta(pubkey=GLOBAL_VOL_ACC, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user_volume_accumulator, is_signer=False, is_writable=True),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack('<Q', base_amount_out))
        data.extend(struct.pack('<Q', max_quote_amount_in))
        swap_instruction = Instruction(PF_AMM, bytes(data), keys)

        print("Preparing to close WSOL account after swap...")
        close_wsol_account_instruction = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_wsol_account_instruction,
            init_wsol_account_instruction,
        ]

        if token_account_instruction:
            instructions.append(token_account_instruction)

        instructions.append(swap_instruction)
        instructions.append(close_wsol_account_instruction)
        
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
            opts=TxOpts(skip_preflight=False)
        ).value
        print(f"Transaction Signature: {txn_sig}")
        
        print("Confirming transaction...")
        confirmed = confirm_txn(client, txn_sig)
        
        print(f"Transaction confirmed: {confirmed}")
        return confirmed
    except Exception as e:
        print("Error occurred during transaction:", e)
        return False

def sell(client: Client, payer_keypair: Keypair, pair_address: str, percentage: int = 100, slippage: int = 5, unit_budget: int = 150_000, unit_price: int = 1_000_000) -> bool:
    try:
        print(f"Starting sell transaction for pair address: {pair_address} with percentage: {percentage}%")
        
        print("Fetching pool keys...")
        pool_keys: Optional[PoolKeys] = fetch_pool_keys(client, pair_address)
        if pool_keys is None:
            print("No pool keys found, aborting transaction.")
            return False
        print("Pool keys fetched successfully.")

        print("Fetching creator vault info...")
        creator_vault_authority, creator_vault_ata = get_creator_vault_info(client, pool_keys.creator)
        if creator_vault_authority is None or creator_vault_ata is None:
            print("No creator vault info found, aborting transaction.")
            return False
        print("Creator vault info fetched successfully.")

        mint = pool_keys.base_mint
        token_info = client.get_account_info_json_parsed(mint).value
        base_token_program = token_info.owner
        decimal = token_info.data.parsed['info']['decimals']

        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        token_account = get_associated_token_address(payer_keypair.pubkey(), mint, base_token_program)

        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID)
        balance_needed = Token.get_min_balance_rent_for_exempt_for_account(client)

        print("Creating and initializing WSOL account...")
        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(balance_needed),
                space=ACCOUNT_SPACE,
                owner=TOKEN_PROGRAM_ID,
            )
        )

        init_wsol_account_instruction = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        print("Retrieving token balance...")
        token_balance = get_token_balance(client, payer_keypair.pubkey(), mint)
        if token_balance == 0 or token_balance is None:
            print("Token balance is zero. Nothing to sell.")
            return False

        print("Calculating transaction amounts...")
        sol_decimal = 1e9
        token_decimal = 10**decimal
        base_amount_in = int(token_balance * (percentage / 100))
        base_reserve, quote_reserve = get_pool_reserves(client, pool_keys)
        sol_out = tokens_for_sol(base_amount_in, base_reserve, quote_reserve)
        slippage_adjustment = 1 - (slippage / 100)
        min_quote_amount_out = int((sol_out * slippage_adjustment))
        print(f"Base Amount In: {base_amount_in / token_decimal}, Minimum Quote Amount Out: {min_quote_amount_out / sol_decimal}")

        print("Creating swap instructions...")    
        keys = [
            AccountMeta(pubkey=pool_keys.amm, is_signer=False, is_writable=True),
            AccountMeta(pubkey=payer_keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=GLOBAL_CONFIG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=pool_keys.base_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=pool_keys.quote_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=wsol_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=pool_keys.pool_base_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=pool_keys.pool_quote_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=PROTOCOL_FEE_RECIPIENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PROTOCOL_FEE_RECIPIENT_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=base_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=ASSOCIATED_TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTH, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PF_AMM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=creator_vault_ata, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=creator_vault_authority, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack('<Q', base_amount_in))
        data.extend(struct.pack('<Q', min_quote_amount_out))
        
        swap_instruction = Instruction(PF_AMM, bytes(data), keys)

        print("Preparing to close WSOL account after swap...")
        close_wsol_account_instruction = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(unit_budget),
            set_compute_unit_price(unit_price),
            create_wsol_account_instruction,
            init_wsol_account_instruction,
            swap_instruction,
            close_wsol_account_instruction
        ]

        if percentage == 100:
            print("Preparing to close token account after swap (selling 100%).")
            close_account_instruction = close_account(
                CloseAccountParams(
                    base_token_program, token_account, payer_keypair.pubkey(), payer_keypair.pubkey()
                )
            )
            instructions.append(close_account_instruction)

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
            opts=TxOpts(skip_preflight=False)
        ).value
        print(f"Transaction Signature: {txn_sig}")
        
        print("Confirming transaction...")
        confirmed = confirm_txn(client, txn_sig)
        
        print(f"Transaction confirmed: {confirmed}")
        return confirmed
    except Exception as e:
        print("Error occurred during transaction:", e)
        return False
    
from dataclasses import dataclass
from typing import List, Optional

from construct import Bytes, Int16ul, Int64ul, Int8ul, Padding, Struct

from solana.rpc.api import Client
from solana.rpc.commitment import Processed
from solana.rpc.types import MemcmpOpts, TokenAccountOpts

from solders.pubkey import Pubkey  # type: ignore
from solders.rpc.responses import RpcKeyedAccount  # type: ignore
 
POOL_LAYOUT = Struct(
    Padding(8),
    "pool_bump" / Int8ul,  # u8
    "index" / Int16ul,  # u16
    "creator" / Bytes(32),  # pubkey (32 bytes)
    "base_mint" / Bytes(32),  # pubkey (32 bytes)
    "quote_mint" / Bytes(32),  # pubkey (32 bytes)
    "lp_mint" / Bytes(32),  # pubkey (32 bytes)
    "pool_base_token_account" / Bytes(32),  # pubkey (32 bytes)
    "pool_quote_token_account" / Bytes(32),  # pubkey (32 bytes)
    "lp_supply" / Int64ul,  # u64
    "coin_creator" / Bytes(32),  # pubkey (32 bytes)
)

@dataclass
class PoolKeys:
    amm: Pubkey
    base_mint: Pubkey
    quote_mint: Pubkey
    pool_base_token_account: Pubkey
    pool_quote_token_account: Pubkey
    creator: Pubkey

def fetch_pool_keys(client: Client, pair_address: str):
    try:
        amm = Pubkey.from_string(pair_address)
        account_info = client.get_account_info_json_parsed(amm, commitment=Processed)
        amm_data = account_info.value.data
        decoded_data = POOL_LAYOUT.parse(amm_data)

        return PoolKeys(
            amm=amm,
            base_mint=Pubkey.from_bytes(decoded_data.base_mint),
            quote_mint=Pubkey.from_bytes(decoded_data.quote_mint),
            pool_base_token_account=Pubkey.from_bytes(decoded_data.pool_base_token_account),
            pool_quote_token_account=Pubkey.from_bytes(decoded_data.pool_quote_token_account),
            creator=Pubkey.from_bytes(decoded_data.coin_creator),
        )
    except:
        return None

def get_pool_reserves(client: Client, pool_keys: PoolKeys):
    try:
        
        base_vault = pool_keys.pool_base_token_account
        quote_vault = pool_keys.pool_quote_token_account # SOL
        
        balances_response = client.get_multiple_accounts_json_parsed(
            [base_vault, quote_vault], 
            Processed
        )
        
        balances = balances_response.value

        base_account = balances[0]
        quote_account = balances[1]
        
        base_account_balance = int(base_account.data.parsed['info']['tokenAmount']['amount'])
        quote_account_balance = int(quote_account.data.parsed['info']['tokenAmount']['amount'])
        
        if base_account_balance is None or quote_account_balance is None:
            return None, None
        
        return base_account_balance, quote_account_balance

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None

def fetch_pair_from_rpc(client: Client, base_str: str) -> Optional[str]:
    quote_str: str = "So11111111111111111111111111111111111111112"
    filters: List[List[MemcmpOpts]] = [
        [MemcmpOpts(offset=43, bytes=base_str), MemcmpOpts(offset=75, bytes=quote_str)],
        [MemcmpOpts(offset=43, bytes=quote_str), MemcmpOpts(offset=75, bytes=base_str)]
    ]
    pools: List[RpcKeyedAccount] = []
    for f in filters:
        try:
            resp = client.get_program_accounts(PF_AMM, filters=f)
            pools.extend(resp.value)
        except Exception as e:
            print(f"Error fetching program accounts with filters {f}: {e}")
            continue

    if not pools:
        return None

    best_pool_addr: Optional[str] = None
    max_liquidity: int = 0

    for pool in pools:
        try:
            pool_data: bytes = pool.account.data
            base_token_account: Pubkey = Pubkey.from_bytes(pool_data[139:171])
            quote_token_account: Pubkey = Pubkey.from_bytes(pool_data[171:203])
        except Exception as e:
            print(f"Error processing pool {pool.pubkey}: {e}")
            continue

        try:
            base_resp = client.get_token_account_balance(base_token_account)
            quote_resp = client.get_token_account_balance(quote_token_account)
        except Exception as e:
            print(f"Error fetching token account balance: {e}")
            continue

        if base_resp.value is None or quote_resp.value is None:
            continue

        try:
            base_balance: int = int(base_resp.value.amount)
            quote_balance: int = int(quote_resp.value.amount)
        except Exception as e:
            print(f"Error converting token balances to int: {e}")
            continue

        liquidity: int = base_balance * quote_balance
        if liquidity > max_liquidity:
            max_liquidity = liquidity
            best_pool_addr = str(pool.pubkey)
    return best_pool_addr

def sol_for_tokens(quote_amount_in, pool_base_token_reserves, pool_quote_token_reserves):
    base_amount_out = pool_base_token_reserves - (pool_base_token_reserves * pool_quote_token_reserves) // (pool_quote_token_reserves + quote_amount_in)
    return int(base_amount_out)

def tokens_for_sol(base_amount_in, pool_base_token_reserves, pool_quote_token_reserves):
    quote_amount_out = pool_quote_token_reserves - (pool_base_token_reserves * pool_quote_token_reserves) // (pool_base_token_reserves + base_amount_in)
    lp_fee = int(quote_amount_out * .002)
    protocol_fee = int(quote_amount_out * .0005)
    fees = lp_fee + protocol_fee
    return int(quote_amount_out - fees)

def get_creator_vault_info(client: Client, creator: Pubkey) -> tuple[Pubkey|None, Pubkey|None]:
    try:
        creator_vault_authority = Pubkey.find_program_address([b"creator_vault", bytes(creator)], PF_AMM)[0]
        creator_vault_ata = client.get_token_accounts_by_owner_json_parsed(
            creator_vault_authority,
            TokenAccountOpts(
                mint=WSOL,
                program_id=TOKEN_PROGRAM_ID
            )
        ).value[0].pubkey
        return creator_vault_authority, creator_vault_ata
    except:
        return None, None