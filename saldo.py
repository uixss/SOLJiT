# saldo_real_env_plus.py
import os, sys, json
from typing import Optional
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.message import MessageV0
from dotenv import load_dotenv
load_dotenv()
LAMPORTS_PER_SOL = 1_000_000_000

def env_bool(v: Optional[str]) -> bool:
    if v is None: return False
    return str(v).strip().lower() in {"1","true","t","yes","y"}

def lamports(x_sol: float) -> int:
    return int(round(max(0.0, float(x_sol)) * LAMPORTS_PER_SOL))

def to_sol(x_lamports: int) -> float:
    return x_lamports / LAMPORTS_PER_SOL

def load_keypair_from_env() -> Keypair:
    candidates = [os.getenv("PRIVATE_KEY") ]
    val = next((c for c in candidates if c), None)
    if not val:
        sys.exit("❌ Falta PRIVATE_KEY (o PRIV_KEY/BASE58_PRIVKEY).")

    # 1) base58 (Phantom/Backpack)
    try:
        return Keypair.from_base58_string(val)
    except Exception:
        pass

    # 2) JSON [64 bytes] (solana-keygen)
    try:
        arr = json.loads(val)
        if isinstance(arr, list) and len(arr) in (64, 32):
            b = bytes(arr)
            # solana-keygen suele ser 64 bytes (priv+pub); algunos formatos exportan 32
            if len(b) == 64:
                return Keypair.from_bytes(b)
    except Exception:
        pass

    sys.exit("❌ PRIVATE_KEY inválida. Debe ser base58 o JSON [64 bytes].")

def get_rpc_url() -> str:
    for k in ("RPC_HTTPS","RPC","RPC_ENDPOINT","HELIUS_RPC","SOL_RPC_URL"):
        v = os.getenv(k)
        if v: return v
    return "https://api.mainnet-beta.solana.com"

def estimate_base_fee(client: Client, payer: Pubkey, to_pub: Pubkey) -> int:
    # Fee base según tamaño del mensaje (no incluye priority fee ni tips externos)
    try:
        bh = client.get_latest_blockhash().value.blockhash
        msg = MessageV0.try_compile(
            payer,
            [transfer(TransferParams(from_pubkey=payer, to_pubkey=to_pub, lamports=1))],
            [],
            bh
        )
        fee = client.get_fee_for_message(msg).value
        return int(fee) if fee is not None else 5000
    except Exception:
        return 5000

def estimate_priority_fee_lamports() -> int:
    # Priority fee = UNIT_PRICE (µLamports/CU) * UNIT_BUDGET (CU)
    unit_price = os.getenv("UNIT_PRICE")  # en µLamports
    unit_budget = os.getenv("UNIT_BUDGET")  # en CU
    if not unit_price or not unit_budget:
        return 0
    try:
        micro = float(unit_price)          # µLamports/CU
        budget = float(unit_budget)        # CU
        micro_total = micro * budget       # µLamports
        return int(round(micro_total / 1_000_000.0))  # a Lamports
    except Exception:
        return 0

def estimate_jito_tip_lamports() -> int:
    if not env_bool(os.getenv("USE_JITO")):
        return 0
    tip_sol = os.getenv("TIP_SOL")
    if not tip_sol:
        return 0
    try:
        return lamports(float(tip_sol))
    except Exception:
        return 0

def main():
    # ---- Entradas desde env
    rpc = get_rpc_url()
    kp = load_keypair_from_env()
    payer = kp.pubkey()

    # Destino para estimación de fee (si no hay, usamos self-transfer)
    to_env = os.getenv("TO_PUBKEY")
    try:
        to_pub = Pubkey.from_string(to_env) if to_env else payer
    except Exception:
        sys.exit("❌ TO_PUBKEY inválido.")

    # Parámetros de control
    cushion_sol = float(os.getenv("FEE_CUSHION_SOL", "0"))
    max_fraction = float(os.getenv("MAX_FRACTION_OF_BALANCE", "1.0"))
    if max_fraction < 0: max_fraction = 0.0
    if max_fraction > 1: max_fraction = 1.0

    client = Client(rpc)

    # ---- Balance
    try:
        balance_lamports = int(client.get_balance(payer).value)
    except Exception as e:
        sys.exit(f"❌ Error consultando balance: {e}")

    # ---- Fees estimados
    base_fee = estimate_base_fee(client, payer, to_pub)
    prio_fee = estimate_priority_fee_lamports()
    jito_tip = estimate_jito_tip_lamports()
    total_fee = base_fee + prio_fee + jito_tip

    # ---- Colchón y límites
    cushion_lamports = lamports(cushion_sol)
    spendable_after_fee = max(0, balance_lamports - total_fee)
    saldo_real_movible = max(0, spendable_after_fee - cushion_lamports)
    cap_por_porcentaje = int(balance_lamports * max_fraction)
    cap_para_swap = min(saldo_real_movible, cap_por_porcentaje)

    out = {
        "owner_pubkey": str(payer),
        "rpc": rpc,
        "balance_lamports": balance_lamports,
        "balance_sol": round(to_sol(balance_lamports), 9),

        "fees": {
            "base_fee_lamports": base_fee,
            "base_fee_sol": round(to_sol(base_fee), 9),
            "priority_fee_lamports": prio_fee,
            "priority_fee_sol": round(to_sol(prio_fee), 9),
            "jito_tip_lamports": jito_tip,
            "jito_tip_sol": round(to_sol(jito_tip), 9),
            "total_fee_lamports": total_fee,
            "total_fee_sol": round(to_sol(total_fee), 9),
        },

        "cushion_sol": cushion_sol,
        "cushion_lamports": cushion_lamports,

        "saldo_real_movible_lamports": saldo_real_movible,
        "saldo_real_movible_sol": round(to_sol(saldo_real_movible), 9),

        "max_fraction_of_balance": max_fraction,
        "cap_para_swap_lamports": cap_para_swap,
        "cap_para_swap_sol": round(to_sol(cap_para_swap), 9),
    }

    # JSON si seteás JSON_OUTPUT=1
    if os.getenv("JSON_OUTPUT") == "1":
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"Cuenta: {out['owner_pubkey']}")
        print(f"RPC: {out['rpc']}")
        print(f"Balance: {out['balance_lamports']} lamports ({out['balance_sol']:.9f} SOL)")
        print("---- Fees estimados ----")
        print(f"  Base fee: {base_fee} lamports ({to_sol(base_fee):.9f} SOL)")
        print(f"  Priority fee: {prio_fee} lamports ({to_sol(prio_fee):.9f} SOL)")
        print(f"  Jito tip: {jito_tip} lamports ({to_sol(jito_tip):.9f} SOL)")
        print(f"  TOTAL fees: {total_fee} lamports ({to_sol(total_fee):.9f} SOL)")
        print("------------------------")
        print(f"Colchón: {cushion_lamports} lamports ({to_sol(cushion_lamports):.9f} SOL)")
        print(f"Saldo REAL movible ahora: {saldo_real_movible} lamports ({to_sol(saldo_real_movible):.9f} SOL)")
        print(f"Tope por porcentaje ({max_fraction*100:.1f}%): {cap_por_porcentaje} lamports ({to_sol(cap_por_porcentaje):.9f} SOL)")
        print(f"CAP PARA SWAP: {cap_para_swap} lamports ({to_sol(cap_para_swap):.9f} SOL)")

if __name__ == "__main__":
    main()
