#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
import traceback
import signal
from dataclasses import dataclass, asdict
from typing import Optional, Literal

# Opcional: cargar .env automáticamente
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Solana SDK
from solana.rpc.api import Client
from solders.keypair import Keypair  # type: ignore
from solders.pubkey import Pubkey    # type: ignore
from core.meteora_dbc import fetch_pool_from_rpc as meteora_dbc_fetch_pool
from core.meteora_dbc import buy as meteora_dbc_buy, sell as meteora_dbc_sell
from core.meteora_damm2 import fetch_pool_from_rpc as meteora_mm2_fetch_pool 
from core.meteora_damm2 import buy as meteora_mm2_buy, sell as meteora_mm2_sell
from core.pump_swap import fetch_pair_from_rpc as pump_fetch_pair
from core.pump_swap import buy as pump_buy, sell as pump_sell
from core.raydium import buy as ray_ammv4_buy, sell as ray_ammv4_sell
from core.raydium import buy as ray_cpmm_buy, sell as ray_cpmm_sell

import core.moonshot as moon_mod  

# =========================
# Tipos
# =========================
DexName = Literal["meteoradbc", "meteoradmm2", "pump", "raydium_ammv4", "raydium_cpmm", "moon"]
Side = Literal["buy", "sell"]

@dataclass
class UniversalConfig:
    # Requeridos
    private_key: str
    rpc_url: str

    # Modo de operación
    side: Side
    dex: DexName
    mint_or_pair: str
    amount_sol: Optional[float] = None
    sell_percentage: Optional[int] = None

    # Slippage %
    slippage_pct: Optional[float] = None

    # Compute Units
    unit_budget: Optional[int] = None
    unit_price_micro_lamports: Optional[int] = None

    # Límites de saldo
    max_fraction_of_balance: float = 0.4
    fee_cushion_sol: float = 0.01

    # Extras
    require_proof_for_postsells: bool = True
    mode: Literal["speed", "price"] = "speed"


# =========================
# Utils / Helpers
# =========================

def read_env_defaults() -> dict:
    return {
        "PRIVATE_KEY": os.getenv("PRIVATE_KEY", "").strip(),
        "RPC": os.getenv("RPC", os.getenv("RPC_HTTPS", os.getenv("RPC_ENDPOINT", ""))).strip(),
        "SLIPPAGE": float(os.getenv("SLIPPAGE", "30")),
        "UNIT_BUDGET": int(os.getenv("UNIT_BUDGET", "300000")),
        "UNIT_PRICE": int(os.getenv("UNIT_PRICE", "1000")),
        "MAX_FRACTION_OF_BALANCE": float(os.getenv("MAX_FRACTION_OF_BALANCE", "0.4")),
        "FEE_CUSHION_SOL": float(os.getenv("FEE_CUSHION_SOL", "0.01")),
        "MODE": os.getenv("MODE", "speed"),
        "REQUIRE_PROOF_FOR_POSTSELLS": os.getenv("REQUIRE_PROOF_FOR_POSTSELLS", "true").lower() == "true",
    }


def is_base58(s: str) -> bool:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return len(s) >= 32 and all(c in alphabet for c in s)


def setup_signal_handlers():
    def handler(sig, frame):
        print("Interrumpido por usuario (SIGINT).", file=sys.stderr)
        sys.exit(130)
    signal.signal(signal.SIGINT, handler)


def log_json(obj, level="info"):
    try:
        payload = obj if isinstance(obj, dict) else {"msg": str(obj)}
        print(json.dumps({"level": level, **payload}, ensure_ascii=False))
    except Exception:
        print(str(obj))


def lamports(sol: float) -> int:
    return int(round(sol * 1_000_000_000))


def get_balance_sol(client: Client, owner: Pubkey) -> float:
    try:
        res = client.get_balance(owner)
        if hasattr(res, "value"):
            lam = res.value
        elif isinstance(res, dict):
            lam = res.get("result", {}).get("value", 0)
        else:
            lam = getattr(res, "result", {}).get("value", 0)
        return float(lam) / 1_000_000_000
    except Exception as e:
        raise RuntimeError(f"No pude leer balance RPC: {e}")


def enforce_balance_limits(client: Client, owner: Pubkey, want_sol: float, max_fraction: float, cushion_sol: float) -> float:
    bal = get_balance_sol(client, owner)
    max_usable = max(bal - cushion_sol, 0.0)
    hard_cap = bal * max_fraction
    allowed = min(max_usable, hard_cap)
    if allowed <= 0:
        raise RuntimeError(f"Saldo insuficiente: balance={bal:.6f} SOL, cushion={cushion_sol} SOL")
    return round(min(want_sol, allowed), 9)


def _ensure(val, name: str):
    if val is None:
        raise ValueError(f"Parámetro requerido faltante: {name}")
    return val


# =========================
# Adaptador universal → DEX
# =========================

def route_order(cfg: UniversalConfig):
    client = Client(cfg.rpc_url)
    payer = Keypair.from_base58_string(cfg.private_key)
    owner = payer.pubkey()
    envd = read_env_defaults()

    # Aplicar valores por defecto
    unit_budget = cfg.unit_budget if cfg.unit_budget is not None else envd["UNIT_BUDGET"]
    unit_price = cfg.unit_price_micro_lamports if cfg.unit_price_micro_lamports is not None else envd["UNIT_PRICE"]
    slippage_pct = cfg.slippage_pct if cfg.slippage_pct is not None else envd["SLIPPAGE"]

    # Balance checks solo en buy
    if cfg.side == "buy":
        sol_in = _ensure(cfg.amount_sol, "amount_sol")
        sol_in = enforce_balance_limits(
            client=client,
            owner=owner,
            want_sol=sol_in,
            max_fraction=cfg.max_fraction_of_balance if cfg.max_fraction_of_balance is not None else envd["MAX_FRACTION_OF_BALANCE"],
            cushion_sol=cfg.fee_cushion_sol if cfg.fee_cushion_sol is not None else envd["FEE_CUSHION_SOL"],
        )
    else:
        percentage = _ensure(cfg.sell_percentage, "sell_percentage")
        if not (1 <= percentage <= 100):
            raise ValueError("sell_percentage debe estar entre 1 y 100")
        sol_in = None

    # Despacho por DEX
    if cfg.dex == "meteoradbc":
        pool_str = meteora_dbc_fetch_pool(client, cfg.mint_or_pair)
        if not pool_str:
            raise RuntimeError("No se encontró pool para ese mint en Meteora dBC.")
        return meteora_dbc_buy(client, payer, pool_str, sol_in, unit_budget, unit_price) if cfg.side == "buy" \
            else meteora_dbc_sell(client, payer, pool_str, int(cfg.sell_percentage), unit_budget, unit_price)

    elif cfg.dex == "meteoradmm2":
        pool_str = meteora_mm2_fetch_pool(client, cfg.mint_or_pair)
        if not pool_str:
            raise RuntimeError("No se encontró pair/pool para ese mint en Meteora DAMM2.")
        return meteora_mm2_buy(client, payer, pool_str, sol_in, unit_budget, unit_price) if cfg.side == "buy" \
            else meteora_mm2_sell(client, payer, pool_str, int(cfg.sell_percentage), unit_budget, unit_price)

    elif cfg.dex == "pump":
        pair_addr = pump_fetch_pair(client, cfg.mint_or_pair)
        if not pair_addr:
            raise RuntimeError("No se encontró pair para ese mint en Pump.fun.")
        return pump_buy(client, payer, pair_addr, sol_in, float(slippage_pct), unit_budget, unit_price) if cfg.side == "buy" \
            else pump_sell(client, payer, pair_addr, int(cfg.sell_percentage), float(slippage_pct), unit_budget, unit_price)

    elif cfg.dex == "moon":
        moon_mod.PRIV_KEY = cfg.private_key
        moon_mod.RPC = cfg.rpc_url
        moon_mod.UNIT_BUDGET = unit_budget
        moon_mod.UNIT_PRICE = unit_price
        moon_mod.client = Client(moon_mod.RPC)
        moon_mod.payer_keypair = Keypair.from_base58_string(moon_mod.PRIV_KEY)
        slippage_bps = int(round(float(slippage_pct) * 100))

        if cfg.side == "buy":
            return moon_mod.buy(mint_str=cfg.mint_or_pair, sol_in=float(sol_in), slippage_bps=slippage_bps)
        else:
            bal_tokens = moon_mod.get_token_balance(cfg.mint_or_pair)
            if bal_tokens is None:
                raise RuntimeError("Moon: no pude obtener tu balance del token.")
            sell_tokens = float(bal_tokens) * (int(cfg.sell_percentage) / 100.0)
            if sell_tokens <= 0:
                raise RuntimeError("Moon: el balance a vender es 0.")
            return moon_mod.sell(mint_str=cfg.mint_or_pair, token_balance=sell_tokens, slippage_bps=slippage_bps)

    elif cfg.dex == "raydium_ammv4":
        pair_addr = cfg.mint_or_pair
        return ray_ammv4_buy(pair_addr, float(sol_in), float(slippage_pct)) if cfg.side == "buy" \
            else ray_ammv4_sell(pair_addr, int(cfg.sell_percentage), float(slippage_pct))

    elif cfg.dex == "raydium_cpmm":
        pair_addr = cfg.mint_or_pair
        return ray_cpmm_buy(pair_addr, float(sol_in), float(slippage_pct)) if cfg.side == "buy" \
            else ray_cpmm_sell(pair_addr, int(cfg.sell_percentage), float(slippage_pct))

    else:
        raise ValueError(f"DEX no soportado: {cfg.dex}")


# =========================
# Build config desde ENV
# =========================

def build_config_from_env(
    side: Side,
    dex: DexName,
    mint_or_pair: str,
    amount_sol: Optional[float] = None,
    sell_percentage: Optional[int] = None,
    slippage_pct: Optional[float] = None,
    unit_budget: Optional[int] = None,
    unit_price_micro_lamports: Optional[int] = None,
) -> UniversalConfig:
    envd = read_env_defaults()
    if not envd["PRIVATE_KEY"] or not envd["RPC"]:
        raise ValueError("Faltan PRIVATE_KEY o RPC en el entorno.")

    return UniversalConfig(
        private_key=envd["PRIVATE_KEY"],
        rpc_url=envd["RPC"],
        side=side,
        dex=dex,
        mint_or_pair=mint_or_pair,
        amount_sol=amount_sol,
        sell_percentage=sell_percentage,
        slippage_pct=slippage_pct if slippage_pct is not None else envd["SLIPPAGE"],
        unit_budget=unit_budget if unit_budget is not None else envd["UNIT_BUDGET"],
        unit_price_micro_lamports=unit_price_micro_lamports if unit_price_micro_lamports is not None else envd["UNIT_PRICE"],
        max_fraction_of_balance=envd["MAX_FRACTION_OF_BALANCE"],
        fee_cushion_sol=envd["FEE_CUSHION_SOL"],
        require_proof_for_postsells=envd["REQUIRE_PROOF_FOR_POSTSELLS"],
        mode=envd["MODE"],
    )


# =========================
# PROBE (resolver pool/pair)
# =========================

def resolve_probe(dex: str, addr: str, rpc: str):
    client = Client(rpc)

    if dex == "meteoradbc":
        pool = meteora_dbc_fetch_pool(client, addr)
        return {"dex": dex, "input": addr, "pool_or_pair": pool}
    elif dex == "meteoradmm2":
        pool = meteora_mm2_fetch_pool(client, addr)
        return {"dex": dex, "input": addr, "pool_or_pair": pool}
    elif dex == "pump":
        pair = pump_fetch_pair(client, addr)
        return {"dex": dex, "input": addr, "pool_or_pair": pair}
    elif dex in ("raydium_ammv4", "raydium_cpmm", "moon"):
        return {"dex": dex, "input": addr, "pool_or_pair": addr}
    else:
        raise ValueError(f"DEX no soportado: {dex}")


# =========================
# CLI
# =========================

def main():
    setup_signal_handlers()

    parser = argparse.ArgumentParser(
        prog="universal-swap",
        description="CLI unificado para comprar/vender en varios DEX de Solana"
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"),
                        choices=["debug", "info", "warn", "error"], help="Nivel de logs")
    parser.add_argument("--json-logs", action="store_true", help="Emite logs en JSON")
    parser.add_argument("--simulate", action="store_true", help="No envía tx; corta antes del submit (si aplica)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--dex", required=True,
                      choices=["meteoradbc", "meteoradmm2", "pump", "raydium_ammv4", "raydium_cpmm", "moon"],
                      help="DEX objetivo")
        p.add_argument("--addr", required=True,
                      help="Mint del token (meteora/pump/moon) o pair_address (raydium)")
        p.add_argument("--slippage", type=float, default=None, help="Slippage en % (override de env)")
        p.add_argument("--unit-budget", type=int, default=None, help="Compute Unit budget (override)")
        p.add_argument("--unit-price", type=int, default=None, help="µLamports por CU (override)")
        p.add_argument("--dry-run", action="store_true", help="No manda tx, solo muestra lo que haría")
        p.add_argument("--probe", action="store_true", help="Solo resuelve pool/pair y muestra")
        p.add_argument("--print-config", action="store_true", help="Imprime la config final efectiva")

    buy = sub.add_parser("buy", help="Comprar con SOL")
    add_common(buy)
    buy.add_argument("--sol", type=float, required=True, help="Cantidad de SOL a gastar")

    sell = sub.add_parser("sell", help="Vender porcentaje del token")
    add_common(sell)
    sell.add_argument("--pct", type=int, required=True, help="Porcentaje a vender (1-100)")

    args = parser.parse_args()

    # Configuración de logging
    lvl = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARN, "error": logging.ERROR}[args.log_level]
    logging.basicConfig(level=lvl, format="%(levelname)s | %(message)s")

    # Normalización
    args.dex = args.dex.strip().lower().replace("-", "_")
    if not is_base58(args.addr):
        logging.warning("El --addr no parece Base58 válido. Asegúrate de pasar mint/pair correcto.")

    # Leer entorno una vez
    envd = read_env_defaults()
    if not envd["PRIVATE_KEY"] or not envd["RPC"]:
        msg = "Faltan PRIVATE_KEY o RPC (via .env o variables de entorno)."
        log_json({"error": msg}, level="error") if args.json_logs else logging.error(msg)
        sys.exit(1)

    # Modo PROBE
    if args.probe:
        try:
            info = resolve_probe(args.dex, args.addr, envd["RPC"])
            output = json.dumps(info, indent=2) if not args.json_logs else {"probe": info}
            log_json(output) if args.json_logs else print(output)
        except Exception as e:
            log_json({"error": str(e)}, level="error") if args.json_logs else logging.error(f"Probe falló: {e}")
            sys.exit(2)

        if args.dry_run or args.simulate:
            return

    # Construcción de configuración
    try:
        cfg = build_config_from_env(
            side=args.cmd,
            dex=args.dex,
            mint_or_pair=args.addr,
            amount_sol=args.sol if args.cmd == "buy" else None,
            sell_percentage=args.pct if args.cmd == "sell" else None,
            slippage_pct=args.slippage,
            unit_budget=args.unit_budget,
            unit_price_micro_lamports=args.unit_price,
        )
    except Exception as e:
        log_json({"error": str(e)}, level="error") if args.json_logs else logging.error(e)
        sys.exit(1)

    # Impresión de configuración
    if args.print_config or args.log_level == "debug":
        output = json.dumps(asdict(cfg), indent=2) if not args.json_logs else {"effective_config": asdict(cfg)}
        log_json(output) if args.json_logs else print(output)

    # Modo simulación o dry-run
    if args.dry_run or args.simulate:
        tag = "DRY RUN" if args.dry_run else "SIMULATE"
        msg = f"{tag}: no se enviará ninguna transacción."
        log_json({"msg": msg}) if args.json_logs else logging.info(msg)
        return

    # Ejecución
    try:
        res = route_order(cfg)
        output = res if isinstance(res, str) else json.dumps(res, indent=2, default=str)
        log_json({"result": output}) if args.json_logs else print(output)
        logging.info("Hecho.")
    except Exception as e:
        error_payload = {"error": str(e)}
        if args.log_level == "debug":
            error_payload["trace"] = traceback.format_exc()
        log_json(error_payload, level="error") if args.json_logs else logging.error(f"Fallo: {e}")
        if args.log_level == "debug" and not args.json_logs:
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()