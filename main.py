#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import logging
import argparse
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
_LOG_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
logging.basicConfig(level=_LOG_MAP.get(_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("main_router")


def _sync_env_for_modules() -> None:

    priv = (
        os.getenv("PRIV_KEY")
        or os.getenv("PRIVATE_KEY")
        or os.getenv("BASE58_PRIVKEY")
    )
    rpc = (
        os.getenv("RPC")
        or os.getenv("RPC_HTTPS")
        or os.getenv("RPC_ENDPOINT")
        or os.getenv("HELIUS_RPC")
    )

    if priv:
        os.environ["PRIV_KEY"] = priv
        os.environ["PRIVATE_KEY"] = priv
        os.environ["BASE58_PRIVKEY"] = priv
    if rpc:
        os.environ["RPC"] = rpc
        os.environ["RPC_HTTPS"] = rpc
        os.environ["RPC_ENDPOINT"] = rpc
        os.environ["HELIUS_RPC"] = rpc

    if not os.getenv("PRIV_KEY"):
        log.error("Falta PRIV_KEY/PRIVATE_KEY/BASE58_PRIVKEY")
    if not os.getenv("RPC"):
        log.error("Falta RPC/RPC_HTTPS/RPC_ENDPOINT/HELIUS_RPC")

# ---------------------- Detección de rutas ----------------------

def _pumpfun_active(mint: str) -> Optional[bool]:
    """True si bonding curve activa en Pump.fun; False si completa; None si desconocido."""
    try:
        from core.pumpfun import get_coin_data  # type: ignore
        cd = get_coin_data(mint)
        if not cd:
            return None
        return not bool(getattr(cd, "complete", True))
    except Exception as e:
        log.debug("pumpfun detect falló: %s", e)
        return None


def _moonshot_supported(mint: str) -> Optional[bool]:
    """True si la API de Moonshot devuelve priceNative; None si falla."""
    try:
        from core.moonshot import get_token_data  # type: ignore
        td = get_token_data(mint)
        return bool(td and "priceNative" in td)
    except Exception as e:
        log.debug("moonshot detect falló: %s", e)
        return None


def _raydium_has_pool(mint: str) -> Optional[bool]:
    try:
        from core.raydium import _resolve_pair  # type: ignore
        pair = _resolve_pair(mint)
        return bool(pair)
    except Exception as e:
        log.debug("raydium detect falló: %s", e)
        return None

# ---------------------- Ejecutores por ruta ----------------------

def _exec_buy_pumpfun(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from core.pumpfun import buy as pf_buy  # type: ignore
    log.info("[pumpfun] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    return bool(pf_buy(mint, sol_in=sol_amount, slippage=int(slippage_pct)))


def _exec_sell_pumpfun(mint: str, pct: int, slippage_pct: int) -> bool:
    from core.pumpfun import sell as pf_sell  # type: ignore
    log.info("[pumpfun] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    return bool(pf_sell(mint, percentage=int(pct), slippage=int(slippage_pct)))


def _exec_buy_moonshot(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from core.moonshot import buy as ms_buy  # type: ignore
    slippage_bps = int(max(0, slippage_pct) * 100)
    log.info("[moonshot] BUY mint=%s sol=%.6f slippage_bps=%d", mint, sol_amount, slippage_bps)
    return bool(ms_buy(mint, sol_in=sol_amount, slippage_bps=slippage_bps))


def _exec_sell_moonshot(mint: str, pct: int, slippage_pct: int) -> bool:
    from core.moonshot import sell as ms_sell  # type: ignore
    slippage_bps = int(max(0, slippage_pct) * 100)
    log.info("[moonshot] SELL mint=%s pct=%d slippage_bps=%d", mint, pct, slippage_bps)
    return bool(ms_sell(mint, percentage=int(pct), slippage_bps=slippage_bps))


def _exec_buy_jupiter(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from core.jupiter import buy as jup_buy  # type: ignore
    log.info("[jupiter] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    return bool(jup_buy(mint, sol_in=sol_amount, slippage=int(slippage_pct)))


def _exec_sell_jupiter(mint: str, pct: int, slippage_pct: int) -> bool:
    from core.jupiter import sell as jup_sell  # type: ignore
    log.info("[jupiter] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    return bool(jup_sell(mint, percentage=int(pct), slippage=int(slippage_pct)))


def _exec_buy_raydium(mint: str, sol_amount: float, slippage_pct: int) -> bool:
    from core.raydium import buy as ray_buy  # type: ignore
    log.info("[raydium] BUY mint=%s sol=%.6f slippage=%d%%", mint, sol_amount, slippage_pct)
    return bool(ray_buy(mint, sol_amount=sol_amount, slippage_pct=int(slippage_pct)))


def _exec_sell_raydium(mint: str, pct: int, slippage_pct: int) -> bool:
    from core.raydium import sell as ray_sell  # type: ignore
    log.info("[raydium] SELL mint=%s pct=%d slippage=%d%%", mint, pct, slippage_pct)
    return bool(ray_sell(mint, percentage=int(pct), slippage_pct=int(slippage_pct)))

# ---------------------- Router ----------------------

def auto_buy(mint: str, sol_amount: float, slippage_pct: int, mode: str = "price") -> bool:
    """Estrategia: Pump/Moonshot si bonding → luego Jupiter (precio) → Raydium (fallback).
    Modo speed: Pump/Moonshot → Raydium → Jupiter."""
    _sync_env_for_modules()

    pf = _pumpfun_active(mint)
    ms = _moonshot_supported(mint)
    rd = _raydium_has_pool(mint)

    routes = []
    if mode == "speed":
        routes = [
            ("pump", bool(pf)),
            ("moon", (pf is False and bool(ms))),
            ("ray", bool(rd)),
            ("jup", True),
        ]
    else:  # price (default)
        routes = [
            ("pump", bool(pf)),
            ("moon", (pf is False and bool(ms))),
            ("jup", True),
            ("ray", bool(rd)),
        ]

    last_err = None
    for name, ok in routes:
        if not ok:
            continue
        try:
            if name == "pump":
                return _exec_buy_pumpfun(mint, sol_amount, slippage_pct)
            if name == "moon":
                return _exec_buy_moonshot(mint, sol_amount, slippage_pct)
            if name == "jup":
                return _exec_buy_jupiter(mint, sol_amount, slippage_pct)
            if name == "ray":
                return _exec_buy_raydium(mint, sol_amount, slippage_pct)
        except Exception as e:
            log.warning("%s BUY falló: %s", name, e)
            last_err = e
            continue
    if last_err:
        raise last_err
    return False


def auto_sell(mint: str, pct: int, slippage_pct: int, mode: str = "price") -> bool:
    _sync_env_for_modules()

    pf = _pumpfun_active(mint)
    ms = _moonshot_supported(mint)
    rd = _raydium_has_pool(mint)

    routes = []
    if mode == "speed":
        routes = [
            ("pump", bool(pf)),
            ("moon", (pf is False and bool(ms))),
            ("ray", bool(rd)),
            ("jup", True),
        ]
    else:
        routes = [
            ("pump", bool(pf)),
            ("moon", (pf is False and bool(ms))),
            ("jup", True),
            ("ray", bool(rd)),
        ]

    last_err = None
    for name, ok in routes:
        if not ok:
            continue
        try:
            if name == "pump":
                return _exec_sell_pumpfun(mint, pct, slippage_pct)
            if name == "moon":
                return _exec_sell_moonshot(mint, pct, slippage_pct)
            if name == "jup":
                return _exec_sell_jupiter(mint, pct, slippage_pct)
            if name == "ray":
                return _exec_sell_raydium(mint, pct, slippage_pct)
        except Exception as e:
            log.warning("%s SELL falló: %s", name, e)
            last_err = e
            continue
    if last_err:
        raise last_err
    return False

# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router de swaps (Pump.fun → Moonshot → Jupiter → Raydium)")
    p.add_argument("--mint", required=True, help="Mint del token")
    p.add_argument("--sell", action="store_true", help="Vender en vez de comprar")
    p.add_argument("--sol", type=float, default=float(os.getenv("SOL_AMOUNT", "0.01")), help="SOL a usar en compra")
    p.add_argument("--pct", type=int, default=int(os.getenv("SELL_PERCENT", "100")), help="% a vender (1..100)")
    p.add_argument("--slippage", type=int, default=int(os.getenv("SLIPPAGE", "50")), help="Slippage en % (Jupiter/Pump/Raydium) o convertido a bps (Moonshot)")
    p.add_argument("--mode", choices=["price", "speed"], default=os.getenv("MODE", "price"), help="Priorizar mejor precio (price) o velocidad (speed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mint = args.mint
    slippage = max(0, int(args.slippage))
    mode = args.mode
    try:
        if args.sell:
            pct = min(100, max(1, int(args.pct)))
            ok = auto_sell(mint, pct, slippage, mode)
            log.info("SELL result: %s", ok)
            sys.exit(0 if ok else 2)
        else:
            sol_amount = max(0.0, float(args.sol))
            ok = auto_buy(mint, sol_amount, slippage, mode)
            log.info("BUY result: %s", ok)
            sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        log.warning("Interrumpido por el usuario")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        log.exception("Fallo inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
