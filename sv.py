#!/usr/bin/env python3
from flask import Flask, request, jsonify
import os
import logging
from main import auto_buy, auto_sell, _sync_env_for_modules  # importa de tu script principal
# sv.py  (añade imports y helpers)
import threading
from contextlib import contextmanager

_env_lock = threading.Lock()

@contextmanager
def _temp_env(overrides: dict):
    """Setea variables de entorno temporales, restaurando al salir."""
    backup = {}
    try:
        for k, v in overrides.items():
            if v is None:
                continue
            backup[k] = os.environ.get(k)
            os.environ[k] = str(v)
        yield
    finally:
        for k in overrides.keys():
            if k in os.environ:
                del os.environ[k]
        for k, v in backup.items():
            if v is not None:
                os.environ[k] = v
@app.route("/buy", methods=["POST"])
def buy():
    data = request.json or {}
    mint = data.get("mint")
    sol = float(data.get("sol", 0.01))
    slippage = int(data.get("slippage", 50))
    mode = data.get("mode", "price")

    # NUEVO: overrides opcionales por request
    priv58 = data.get("priv58")
    rpc = data.get("rpc")

    if not mint:
        return jsonify({"error": "Falta parámetro mint"}), 400

    try:
        # Aplica overrides de manera aislada y segura
        with _env_lock:
            with _temp_env({
                "PRIV_KEY": priv58,
                "PRIVATE_KEY": priv58,
                "BASE58_PRIVKEY": priv58,
                "RPC": rpc,
                "RPC_HTTPS": rpc,
                "RPC_ENDPOINT": rpc,
                "HELIUS_RPC": rpc,
            }):
                _sync_env_for_modules()  # re-sincroniza mapping
                ok = auto_buy(mint, sol, slippage, mode)
        return jsonify({"status": "ok" if ok else "fail"})
    except Exception as e:
        log.exception("Error en /buy")
        return jsonify({"error": str(e)}), 500

@app.route("/sell", methods=["POST"])
def sell():
    data = request.json or {}
    mint = data.get("mint")
    pct = int(data.get("pct", 100))
    slippage = int(data.get("slippage", 50))
    mode = data.get("mode", "price")

    # NUEVO
    priv58 = data.get("priv58")
    rpc = data.get("rpc")

    if not mint:
        return jsonify({"error": "Falta parámetro mint"}), 400

    try:
        with _env_lock:
            with _temp_env({
                "PRIV_KEY": priv58,
                "PRIVATE_KEY": priv58,
                "BASE58_PRIVKEY": priv58,
                "RPC": rpc,
                "RPC_HTTPS": rpc,
                "RPC_ENDPOINT": rpc,
                "HELIUS_RPC": rpc,
            }):
                _sync_env_for_modules()
                ok = auto_sell(mint, pct, slippage, mode)
        return jsonify({"status": "ok" if ok else "fail"})
    except Exception as e:
        log.exception("Error en /sell")
        return jsonify({"error": str(e)}), 500

# Configuración de logs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("flask_router")

# Cargar variables de entorno (si tienes .env)
from dotenv import load_dotenv
load_dotenv()

# Asegurar claves RPC y privadas
_sync_env_for_modules()

app = Flask(__name__)

@app.route("/buy", methods=["POST"])
def buy():
    data = request.json or {}
    mint = data.get("mint")
    sol = float(data.get("sol", 0.01))
    slippage = int(data.get("slippage", 50))
    mode = data.get("mode", "price")

    if not mint:
        return jsonify({"error": "Falta parámetro mint"}), 400

    try:
        ok = auto_buy(mint, sol, slippage, mode)
        return jsonify({"status": "ok" if ok else "fail"})
    except Exception as e:
        log.exception("Error en /buy")
        return jsonify({"error": str(e)}), 500


@app.route("/sell", methods=["POST"])
def sell():
    data = request.json or {}
    mint = data.get("mint")
    pct = int(data.get("pct", 100))
    slippage = int(data.get("slippage", 50))
    mode = data.get("mode", "price")

    if not mint:
        return jsonify({"error": "Falta parámetro mint"}), 400

    try:
        ok = auto_sell(mint, pct, slippage, mode)
        return jsonify({"status": "ok" if ok else "fail"})
    except Exception as e:
        log.exception("Error en /sell")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
