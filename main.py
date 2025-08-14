#!/usr/bin/env python3
"""
main_auto.py — Trade runner que compra o vende con SOLO pasar el mint

- Usa tus módulos locales: jito.py, jupiter.py, pumpfun.py, moonshot.py, raydium.py
- Rotea automáticamente según el tipo de token/dex disponible.
- Respeta las variables de entorno que compartiste en "Core" y "Trading params".

Uso básico:
  python main_auto.py --mint <MINT>            # compra automática (por defecto)
  python main_auto.py --mint <MINT> --sell     # venta automática (100% por defecto)

Variables de entorno relevantes (ejemplos):
  # ===== Core =====
  RPC_HTTPS=https://mainnet.helius-rpc.com/?api-key=...
  PRIVATE_KEY=BASE58_PRIV
  LOG_LEVEL=info
  SLIPPAGE=30                     # en PORCENTAJE
  UNIT_PRICE=1000                 # micro-lamports/CU
  UNIT_LIMIT=300000
  USE_JITO=true
  JITO_BLOCK_ENGINE_URL=https://ny.mainnet.block-engine.jito.wtf
  JITO_TIP_VALUE=0.004
  TIP_ACCOUNT=JitoTip1111111111111111111111111111111111

  # Montos por defecto
  SOL_AMOUNT=0.01                 # compra: SOL a usar si no se pasa por CLI
  SELL_PERCENT=100                # venta: % del balance a vender si --sell

Notas:
- Intenta en este orden:
  1) Pump.fun (si el bonding curve NO está completo)
  2) Moonshot (si el token existe en su API)
  3) Jupiter (fallback universal)
- Prioridad de fees / Jito se propaga vía variables de entorno hacia los módulos.
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import argparse
from typing import Optional

# ==============================
# Logging
# ==============================
_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
_LOG_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARN,
    "error": logging.ERROR,
}
logging.basicConfig(
    level=_LOG_MAP.get(_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("main_auto")

# ==============================
# Sync de variables para módulos
# ==============================

def _sync_env_for_modules():
    """Mapea tus variables "Core" a las esperadas por cada módulo."""
    rpc = os.getenv("RPC_HTTPS")
    priv = os.getenv("PRIVATE_KEY")
    if not rpc or not priv:
        raise SystemExit("Faltan RPC_HTTPS o PRIVATE_KEY en el entorno.")

    # Jupiter espera PRIV_KEY / RPC
    os.environ.setdefault("PRIV_KEY", priv)
    os.environ.setdefault("RPC", rpc)

    # Pump.fun espera BASE58_PRIVKEY / RPC_ENDPOINT
    os.environ.setdefault("BASE58_PRIVKEY", priv)
    os.environ.setdefault("RPC_ENDPOINT", rpc)

    # Moonshot usa PRIV_KEY / RPC (ya cubierto arriba)

    # Raydium consume UNIT_BUDGET/UNIT_PRICE por defecto; ya vienen de tu entorno
    if os.getenv("UNIT_LIMIT") and not os.getenv("UNIT_BUDGET"):
        os.environ["UNIT_BUDGET"] = os.environ["UNIT_LIMIT"]

    # Slippage: % → módulos
    if os.getenv("SLIPPAGE") and not os.getenv("SLIPPAGE_PERCENT"):
        os.environ["SLIPPAGE_PERCENT"] = os.environ["SLIPPAGE"]  # para referencia

    # Jito: los módulos ya leen USE_JITO, JITO_* y TIP_ACCOUNT directamente del entorno

# ==============================
# Detecciones rápidas por módulo
# ==============================

def _is_pumpfun_buyable(mint: str) -> Optional[bool]:
    try:
        from pumpfun import get_coin_data
        cd = get_coin_data(mint)
        # Si no hay datos, devolvemos None para que otro router intente.
        if not cd:
            return None
        # Si complete == False => aún en bonding curve => usar pump.fun
        return (not bool(cd.complete))
    except Exception as e:
        log.debug("pumpfun detect error: %s", e)
        return None


def _is_moonshot_supported(mint: str) -> Optional[bool]:
    try:
        # Import local del módulo; este mismo hace la query a la API
        from moonshot import get_token_data
        data = get_token_data(mint)
        return bool(data and ("priceNative" in data))
    except Exception as e:
        log.debug("moonshot detect error: %s", e)
        return None

# ==============================
# Ejecutores
# ==============================

def _exec_buy_pumpfun(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from pumpfun import buy as pf_buy
    log.info("[pump.fun] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    return bool(pf_buy(mint, sol_amount, slippage_pct))


def _exec_sell_pumpfun(mint: str, pct: int, slippage_pct: int) -> bool:
    from pumpfun import sell as pf_sell
    log.info("[pump.fun] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    return bool(pf_sell(mint, pct, slippage_pct))


def _exec_buy_moonshot(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from moonshot import buy as ms_buy
    # moonshot usa bps
    slippage_bps = int(max(0, slippage_pct) * 100)
    log.info("[moonshot] BUY mint=%s sol=%.6f slippage_bps=%d", mint, sol_amount, slippage_bps)
    ms_buy(mint, sol_in=sol_amount, slippage_bps=slippage_bps)
    return True


def _exec_sell_moonshot(mint: str, pct: int, slippage_pct: int) -> bool:
    from moonshot import sell as ms_sell
    slippage_bps = int(max(0, slippage_pct) * 100)
    log.info("[moonshot] SELL mint=%s pct=%d slippage_bps=%d", mint, pct, slippage_bps)
    ms_sell(mint, token_balance=None, slippage_bps=slippage_bps)
    return True


def _exec_buy_jupiter(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from jupiter import buy as jup_buy
    log.info("[jupiter] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    return bool(jup_buy(mint, sol_amount, slippage_pct))


def _exec_sell_jupiter(mint: str, pct: int, slippage_pct: int) -> bool:
    from jupiter import sell as jup_sell
    log.info("[jupiter] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    return bool(jup_sell(mint, pct, slippage_pct))

# ==============================
# Router principal
# ==============================

def auto_buy(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    _sync_env_for_modules()

    # 1) Pump.fun si aún en curva
    pf = _is_pumpfun_buyable(mint)
    if pf is True:
        return _exec_buy_pumpfun(mint, sol_amount, slippage_pct)

    # 2) Moonshot si soportado
    ms = _is_moonshot_supported(mint)
    if ms is True:
        return _exec_buy_moonshot(mint, sol_amount, slippage_pct)

    # 3) Fallback Jupiter (universal)
    return _exec_buy_jupiter(mint, sol_amount, slippage_pct)


def auto_sell(mint: str, pct: int, slippage_pct: int) -> bool:
    _sync_env_for_modules()

    # Intento 1: Pump.fun si aún en curva
    pf = _is_pumpfun_buyable(mint)
    if pf is True:
        return _exec_sell_pumpfun(mint, pct, slippage_pct)

    # Intento 2: Moonshot si soportado
    ms = _is_moonshot_supported(mint)
    if ms is True:
        return _exec_sell_moonshot(mint, pct, slippage_pct)

    # Intento 3: Jupiter fallback
    return _exec_sell_jupiter(mint, pct, slippage_pct)

# ==============================
# CLI
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto trade by mint (buy/sell)")
    p.add_argument("--mint", required=True, help="Mint del token")
    p.add_argument("--sell", action="store_true", help="Vender en vez de comprar")
    p.add_argument("--sol", type=float, default=float(os.getenv("SOL_AMOUNT", "0.01")), help="SOL a usar en compra")
    p.add_argument("--pct", type=int, default=int(os.getenv("SELL_PERCENT", "100")), help="% a vender (1..100)")
    p.add_argument("--slippage", type=int, default=int(os.getenv("SLIPPAGE", "30")), help="Slippage en %")
    return p.parse_args()


def main():
    args = parse_args()
    mint = args.mint
    slippage = max(0, int(args.slippage))

    try:
        if args.sell:
            pct = min(100, max(1, int(args.pct)))
            ok = auto_sell(mint, pct, slippage)
            log.info("SELL result: %s", ok)
            sys.exit(0 if ok else 2)
        else:
            sol_amount = float(args.sol)
            ok = auto_buy(mint, sol_amount, slippage)
            log.info("BUY result: %s", ok)
            sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        log.warning("Interrumpido por el usuario")
        sys.exit(130)
    except SystemExit as e:
        raise e
    except Exception as e:
        log.exception("Fallo inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
