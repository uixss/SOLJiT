import struct
import requests
import json
from solana.rpc.types import TokenAccountOpts, TxOpts
from solana.transaction import AccountMeta
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.instruction import Instruction  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from spl.token.instructions import create_associated_token_account, get_associated_token_address
from solana.rpc.api import Client
from solders.keypair import Keypair #type: ignore
from solders.pubkey import Pubkey #type: ignore
import requests
import time
from solana.transaction import Signature
from solders.pubkey import Pubkey # type: ignore
from spl.token.instructions import get_associated_token_address
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import os
def derive_curve_accounts(mint: Pubkey):
    try:
        MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")
        SEED = "token".encode()

        curve_account, _ = Pubkey.find_program_address(
            [SEED, bytes(mint)],
            MOONSHOT_PROGRAM
        )

        curve_token_account = get_associated_token_address(curve_account, mint)
        return curve_account, curve_token_account
    except Exception:
        return None, None

def find_data(data, field):
    if isinstance(data, dict):
        if field in data:
            return data[field]
        else:
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
        
        response = requests.post(RPC, json=payload, headers=headers)
        ui_amount = find_data(response.json(), "uiAmount")
        return float(ui_amount)
    except Exception as e:
        return None

def confirm_txn(txn_sig, max_retries=20, retry_interval=3):
    retries = 0
    if isinstance(txn_sig, str):
        txn_sig = Signature.from_string(txn_sig)
    while retries < max_retries:
        try:
            txn_res = client.get_transaction(txn_sig, encoding="json", commitment="confirmed", max_supported_transaction_version=0)
            txn_json = json.loads(txn_res.value.transaction.meta.to_json())
            if txn_json['err'] is None:
                print("Transaction confirmed... try count:", retries+1)
                return True
            print("Error: Transaction not confirmed. Retrying...")
            if txn_json['err']:
                print("Transaction failed.")
                return False
        except Exception as e:
            print("Awaiting confirmation... try count:", retries+1)
            retries += 1
            time.sleep(retry_interval)
    print("Max retries reached. Transaction confirmation failed.")
    return None

DEX_FEE = Pubkey.from_string("3udvfL24waJcLhskRAsStNMoNUvtyXdxrWQz4hgi953N")
HELIO_FEE = Pubkey.from_string("5K5RtTWzzLp4P8Npi84ocf7F1vBsAu29N1irG4iiUnzt")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_ACC_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
CONFIG_ACCOUNT = Pubkey.from_string("36Eru7v11oU5Pfrojyn5oY3nETA1a1iqsw2WUu6afkM9")
MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")

UNIT_PRICE =  1_000_000
UNIT_BUDGET =  100_000

PRIV_KEY = os.getenv("PRIV_KEY", "")
RPC      = os.getenv("RPC", "")
client = Client(RPC)
payer_keypair = Keypair.from_base58_string(PRIV_KEY)
def get_token_data(token_address: str):
    url = f"https://api.moonshot.cc/token/v1/solana/{token_address}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        token_data = response.json() 
        
        return token_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None

def buy(mint_str: str, sol_in: float = 0.01, slippage_bps: int = 500):
    try:
        # Get token data
        token_data = get_token_data(mint_str)
        if token_data == None:
            return

        # Calculate token amount and collateral amount
        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data['priceNative']
        tokens_out = float(sol_in) / float(token_price)
        token_amount = int(tokens_out * token_decimal)
        collateral_amount = int(sol_in * sol_decimal)
        print(f"Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")

        # Define sender and mint
        SENDER = payer_keypair.pubkey()
        MINT = Pubkey.from_string(mint_str)
        
        # Attempt to retrieve token account, otherwise create associated token account
        token_account, token_account_instructions = None, None
        try:
            account_data = client.get_token_accounts_by_owner(SENDER, TokenAccountOpts(MINT))
            token_account = account_data.value[0].pubkey
            token_account_instructions = None
        except:
            token_account = get_associated_token_address(SENDER, MINT)
            token_account_instructions = create_associated_token_account(SENDER, SENDER, MINT)

        # Define account keys required for the swap
        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        SENDER_TOKEN_ACCOUNT = token_account

        # Build account key list 
        keys = [
            AccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False)
        ]

        # Construct the swap instruction
        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack('<Q', token_amount))
        data.extend(struct.pack('<Q', collateral_amount))
        data.extend(struct.pack('<B', 0))
        data.extend(struct.pack('<Q', slippage_bps))
        data = bytes(data)
        swap_instruction = Instruction(MOONSHOT_PROGRAM, data, keys)

        # Create transaction instructions
        instructions = []
        instructions.append(set_compute_unit_price(UNIT_PRICE))
        instructions.append(set_compute_unit_limit(UNIT_BUDGET))
        
        # Add token account creation instruction if needed
        if token_account_instructions:
            instructions.append(token_account_instructions)
        
        # Add the swap instruction
        instructions.append(swap_instruction)

        # Compile message
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],  
            client.get_latest_blockhash().value.blockhash,
        )

        # Create transaction
        transaction = VersionedTransaction(compiled_message, [payer_keypair])
        
        # Send the transaction and print the transaction signature
        txn_sig = client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")).value
        print(f"Transaction Signature: {txn_sig}")
        
        # Confirm transaction and print the confirmation result
        confirm = confirm_txn(txn_sig)
        print(f"Transaction Confirmation: {confirm}")
    except Exception as e:
        print(e)
        
def sell(mint_str: str, token_balance = None, slippage_bps: int = 500):
    try:
        # Retrieve token balance if not provided
        if token_balance is None:
            token_balance = get_token_balance(mint_str)
        print(f"Token Balance: {token_balance}")
        
        # Check if the token balance is zero
        if token_balance == 0:
            return

        # Get token data
        token_data = get_token_data(mint_str)
        if token_data == None:
            return

        # Calculate token amount and collateral amount
        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data['priceNative']
        token_value = float(token_balance) * float(token_price)
        collateral_amount = int(token_value * sol_decimal)
        token_amount = int(token_balance * token_decimal)
        print(f"Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")
        
        # Define account keys required for the swap
        MINT = Pubkey.from_string(mint_str)
        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        SENDER = payer_keypair.pubkey()
        SENDER_TOKEN_ACCOUNT = get_associated_token_address(SENDER, MINT)

        # Build account key list 
        keys = [
            AccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False)
        ]

        # Construct the swap instruction
        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack('<Q', token_amount))
        data.extend(struct.pack('<Q', collateral_amount))
        data.extend(struct.pack('<B', 0))
        data.extend(struct.pack('<Q', slippage_bps))
        data = bytes(data)
        swap_instruction = Instruction(MOONSHOT_PROGRAM, data, keys)
        
        # Create transaction instructions
        instructions = []
        instructions.append(set_compute_unit_price(UNIT_PRICE))
        instructions.append(set_compute_unit_limit(UNIT_BUDGET))
        instructions.append(swap_instruction)

        # Compile message
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],  
            client.get_latest_blockhash().value.blockhash,
        )

        # Create transaction
        transaction = VersionedTransaction(compiled_message, [payer_keypair])
        
        # Send the transaction and print the transaction signature
        txn_sig = client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed")).value
        print(f"Transaction Signature: {txn_sig}")

        # Confirm transaction and print the confirmation result
        confirm = confirm_txn(txn_sig)
        print(f"Transaction Confirmation: {confirm}")
    except Exception as e:
        print(e)


