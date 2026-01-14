#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BOT DE TRADING: BREAKOUT + REENTRY - ESTRUCTURA MONOLÍTICA COMPLETA
================================================================================
Organizado según el diagrama de flujo:
1. Sincronizar con Bitget
2. Verificar TP/SL
3. Verificar Cierres
4. Escanear Mercado
5. Reoptimizar

Reglas de Oro Respaldadas:
1) NO MODIFICAR LA LÓGICA DE TRADING
2) NO PERDER NINGUNA FUNCIONALIDAD (Telegram, Charts, Logs, OptimizadorIA, etc.)
================================================================================
"""

# REGIÓN 1: IMPORTS Y CONFIGURACIÓN BASE
# ================================================================================
import requests
import time
import json
import os
import sys
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
import numpy as np
import math
import csv
import itertools
import statistics
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from io import BytesIO
from flask import Flask, request, jsonify
import threading
import logging
import uuid
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

# Configurar logging básico
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# REGIÓN 2: OPTIMIZADOR IA COMPLETO
# ================================================================================
class OptimizadorIA:
    """
    Optimizador basado en datos históricos de operaciones
    Evalúa configuraciones y encuentra los mejores parámetros
    """
    def __init__(self, log_path="operaciones_log.csv", min_samples=15):
        self.log_path = log_path
        self.min_samples = min_samples
        self.datos = self.cargar_datos()

    def cargar_datos(self):
        """Carga datos del archivo de log de operaciones"""
        datos = []
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        pnl = float(row.get('pnl_percent', 0))
                        angulo = float(row.get('angulo_tendencia', 0))
                        pearson = float(row.get('pearson', 0))
                        r2 = float(row.get('r2_score', 0))
                        ancho_relativo = float(row.get('ancho_canal_relativo', 0))
                        nivel_fuerza = int(row.get('nivel_fuerza', 1))
                        datos.append({
                            'pnl': pnl, 
                            'angulo': angulo, 
                            'pearson': pearson, 
                            'r2': r2,
                            'ancho_relativo': ancho_relativo,
                            'nivel_fuerza': nivel_fuerza
                        })
                    except Exception:
                        continue
        except FileNotFoundError:
            print("⚠ No se encontró operaciones_log.csv (optimizador)")
        return datos

    def evaluar_configuracion(self, trend_threshold, min_strength, entry_margin):
        """Evalúa una configuración específica de parámetros"""
        if not self.datos:
            return -99999
        filtradas = [
            op for op in self.datos
            if abs(op['angulo']) >= trend_threshold
            and abs(op['angulo']) >= min_strength
            and abs(op['pearson']) >= 0.4
            and op.get('nivel_fuerza', 1) >= 2
            and op.get('r2', 0) >= 0.4
        ]
        n = len(filtradas)
        if n < max(8, int(0.15 * len(self.datos))):
            return -10000 - n
        pnls = [op['pnl'] for op in filtradas]
        pnl_mean = statistics.mean(pnls) if filtradas else 0
        pnl_std = statistics.stdev(pnls) if len(pnls) > 1 else 0
        winrate = sum(1 for op in filtradas if op['pnl'] > 0) / n if n > 0 else 0
        score = (pnl_mean - 0.5 * pnl_std) * winrate * math.sqrt(n)
        ops_calidad = [op for op in filtradas if op.get('r2', 0) >= 0.6 and op.get('nivel_fuerza', 1) >= 3]
        if ops_calidad:
            score *= 1.2
        return score

    def buscar_mejores_parametros(self):
        """Busca la mejor combinación de parámetros"""
        if not self.datos or len(self.datos) < self.min_samples:
            print(f"ℹ️ No hay suficientes datos para optimizar (se requieren {self.min_samples}, hay {len(self.datos)})")
            return None
        mejor_score = -1e9
        mejores_param = None
        trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
        strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
        margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
        combos = list(itertools.product(trend_values, strength_values, margin_values))
        total = len(combos)
        print(f"🔎 Optimizador: probando {total} combinaciones...")
        for idx, (t, s, m) in enumerate(combos, start=1):
            score = self.evaluar_configuracion(t, s, m)
            if idx % 100 == 0 or idx == total:
                print(f"   · probado {idx}/{total} combos (mejor score actual: {mejor_score:.4f})")
            if score > mejor_score:
                mejor_score = score
                mejores_param = {
                    'trend_threshold_degrees': t,
                    'min_trend_strength_degrees': s,
                    'entry_margin': m,
                    'score': score,
                    'evaluated_samples': len(self.datos),
                    'total_combinations': total
                }
        if mejores_param:
            print("✅ Optimizador: mejores parámetros encontrados:", mejores_param)
            try:
                with open("mejores_parametros.json", "w", encoding='utf-8') as f:
                    json.dump(mejores_param, f, indent=2)
            except Exception as e:
                print("⚠ Error guardando mejores_parametros.json:", e)
        else:
            print("⚠ No se encontró una configuración mejor")
        return mejores_param

# REGIÓN 3: CLIENTE BITGET COMPLETO
# ================================================================================
class BitgetClient:
    """
    Cliente completo para interactuar con la API de Bitget Futures
    Implementa HMAC SHA256 para autenticación
    """
    def __init__(self, api_key, api_secret, passphrase, bot_instance=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self._bot_instance = bot_instance
        logger.info(f"Cliente Bitget FUTUROS inicializado con API Key: {api_key[:10]}...")

    def _generate_signature(self, timestamp, method, request_path, body=''):
        """Genera firma HMAC para autenticación"""
        if isinstance(body, str):
            body_str = body
        elif body:
            body_str = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        else:
            body_str = ''
        message = f'{timestamp}{method}{request_path}{body_str}'
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()

    def _get_headers(self, method, request_path, body=None):
        """Genera headers para requests"""
        timestamp = str(int(time.time() * 1000))
        sign = self._generate_signature(timestamp, method, request_path, body)
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': sign,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        return headers

    def verificar_credenciales(self):
        """Verifica que las credenciales sean válidas"""
        try:
            logger.info("Verificando credenciales Bitget FUTUROS...")
            if not self.api_key or not self.api_secret or not self.passphrase:
                logger.error("Credenciales incompletas")
                return False
            accounts = self.get_account_info()
            if accounts:
                logger.info("✓ Credenciales BITGET FUTUROS verificadas exitosamente")
                for account in accounts:
                    if account.get('marginCoin') == 'USDT':
                        available = float(account.get('available', 0))
                        logger.info(f"✓ Balance disponible FUTUROS: {available:.2f} USDT")
                return True
            else:
                logger.error("✗ No se pudo verificar credenciales BITGET FUTUROS")
                return False
        except Exception as e:
            logger.error(f"Error verificando credenciales BITGET FUTUROS: {e}")
            return False

    def get_account_info(self, product_type='USDT-FUTURES'):
        """Obtiene información de cuenta"""
        try:
            request_path = '/api/v2/mix/account/accounts'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            query_string = f"?productType={product_type}&marginCoin=USDT"
            full_request_path = request_path + query_string
            headers = self._get_headers('GET', full_request_path, '')
            response = requests.get(
                f"{self.base_url}{request_path}",
                headers=headers,
                params=params,
                timeout=10
            )
            logger.info(f"Respuesta cuenta FUTUROS - Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    return data.get('data', [])
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"Error API BITGET FUTUROS: {error_code} - {error_msg}")
                    if error_code == '40020' and product_type == 'USDT-FUTURES':
                        logger.info("Intentando con productType='USDT-MIX'...")
                        return self.get_account_info('USDT-MIX')
            return None
        except Exception as e:
            logger.error(f"Error en get_account_info BITGET FUTUROS: {e}")
            return None

    def get_symbol_info(self, symbol):
        """Obtiene información del símbolo FUTUROS"""
        try:
            request_path = '/api/v2/mix/market/contracts'
            params = {'productType': 'USDT-FUTURES'}
            query_string = f"?productType=USDT-FUTURES"
            full_request_path = request_path + query_string
            headers = self._get_headers('GET', full_request_path, '')
            response = requests.get(
                self.base_url + request_path,
                headers=headers,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    contracts = data.get('data', [])
                    for contract in contracts:
                        if contract.get('symbol') == symbol:
                            return contract
            params = {'productType': 'USDT-MIX'}
            response = requests.get(
                self.base_url + request_path,
                headers=headers,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    contracts = data.get('data', [])
                    for contract in contracts:
                        if contract.get('symbol') == symbol:
                            return contract
            return None
        except Exception as e:
            logger.error(f"Error obteniendo info del símbolo BITGET FUTUROS: {e}")
            return None

    def get_max_leverage(self, symbol, product_type='USDT-FUTURES'):
        """Obtiene el apalancamiento máximo permitido"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No se pudo obtener info de {symbol}, usando leverage 20x por defecto")
                return 20
            max_leverage = symbol_info.get('openMaxLeverage', 20)
            if not max_leverage or max_leverage < 1:
                max_leverage = 20
            logger.info(f"📊 {symbol}: Apalancamiento máximo permitido = {max_leverage}x")
            return int(max_leverage)
        except Exception as e:
            logger.error(f"Error obteniendo leverage máximo para {symbol}: {e}")
            return 20

    def place_tpsl_order(self, symbol, hold_side, trigger_price, order_type='stop_loss', 
                         stop_loss_price=None, take_profit_price=None, trade_direction=None):
        """Coloca orden de Stop Loss o Take Profit"""
        request_path = '/api/v2/mix/order/place-pos-tpsl'
        if trade_direction is None:
            trade_direction = 'LONG' if hold_side == 'long' else 'SHORT'
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            price_scale = symbol_info.get('priceScale', 4)
            logger.info(f"📋 {symbol}: priceScale = {price_scale}")
            precision_bitget = price_scale
        else:
            precision_bitget = 4
        body = {
            'symbol': symbol,
            'productType': 'USDT-FUTURES',
            'marginCoin': 'USDT',
            'holdSide': hold_side,
            'stopLossTriggerType': 'mark_price',
            'stopSurplusTriggerType': 'mark_price'
        }
        if order_type == 'stop_loss' and stop_loss_price:
            stop_loss_formatted = self.redondear_precio_manual(stop_loss_price, precision_bitget, symbol, trade_direction)
            body['stopLossTriggerPrice'] = stop_loss_formatted
            logger.info(f"🔧 SL para {symbol}: precio={stop_loss_price}, formatted={stop_loss_formatted}")
        elif order_type == 'take_profit' and take_profit_price:
            take_profit_formatted = self.redondear_precio_manual(take_profit_price, precision_bitget, symbol)
            body['stopSurplusTriggerPrice'] = take_profit_formatted
            logger.info(f"🔧 TP para {symbol}: precio={take_profit_price}, formatted={take_profit_formatted}")
        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        headers = self._get_headers('POST', request_path, body_json)
        response = requests.post(
            self.base_url + request_path,
            headers=headers,
            data=body_json,
            timeout=10
        )
        logger.info(f"📤 Respuesta TP/SL BITGET: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                logger.info(f"✅ {order_type.upper()} creado correctamente para {symbol}")
                return data.get('data')
        logger.error(f"❌ Error creando {order_type}: {response.text}")
        return None

    def get_order_status(self, order_id, symbol):
        """Verifica el estado de una orden específica"""
        try:
            request_path = '/api/v2/mix/order/detail'
            params = {'orderId': order_id, 'symbol': symbol, 'productType': 'USDT-FUTURES'}
            query_parts = [f"{key}={value}" for key, value in params.items()]
            query_string = "?" + "&".join(query_parts) if query_parts else ""
            full_request_path = request_path + query_string
            headers = self._get_headers('GET', full_request_path, '')
            response = requests.get(
                self.base_url + request_path,
                headers=headers,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    return data.get('data')
                else:
                    logger.warning(f"⚠️ Error consultando orden {order_id}: {data.get('msg')}")
            return None
        except Exception as e:
            logger.error(f"❌ Error verificando estado de orden {order_id}: {e}")
            return None

    def get_positions(self, symbol=None, product_type='USDT-FUTURES'):
        """Obtiene posiciones abiertas"""
        try:
            request_path = '/api/v2/mix/position/all-position'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            if symbol:
                params['symbol'] = symbol
            query_parts = [f"{key}={value}" for key, value in params.items()]
            query_string = "?" + "&".join(query_parts) if query_parts else ""
            full_request_path = request_path + query_string
            headers = self._get_headers('GET', full_request_path, '')
            response = requests.get(
                self.base_url + request_path,
                headers=headers,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    data_list = data.get('data', [])
                    if isinstance(data_list, list):
                        logger.info(f"🔍 get_positions: {len(data_list)} posiciones encontradas")
                        return data_list
            if product_type == 'USDT-FUTURES':
                return self.get_positions(symbol, 'USDT-MIX')
            return []
        except Exception as e:
            logger.error(f"Error obteniendo posiciones BITGET FUTUROS: {e}")
            return []

    def obtener_precision_precio(self, symbol):
        """Obtiene la precisión de precio para un símbolo"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                price_scale = symbol_info.get('priceScale', 4)
                logger.info(f"📋 {symbol}: priceScale = {price_scale}")
                return price_scale
            else:
                return 4
        except Exception as e:
            logger.error(f"Error obteniendo precisión de {symbol}: {e}")
            return 4

    def redondear_precio_manual(self, price, precision, symbol=None, trade_direction=None):
        """Redondea el precio con la precisión correcta"""
        try:
            price = float(price)
            if price == 0:
                return "0.0"
            if symbol:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    price_scale = symbol_info.get('priceScale', precision)
                else:
                    price_scale = precision
            else:
                price_scale = precision
            tick_size = 10 ** (-price_scale)
            precio_redondeado = round(price / tick_size) * tick_size
            import math
            if trade_direction and trade_direction in ['LONG', 'SHORT']:
                precio_redondeado = float(f"{precio_redondeado:.{price_scale}f}")
                if trade_direction == 'LONG':
                    if precio_redondeado >= price:
                        precio_redondeado = math.floor(price / tick_size) * tick_size
                elif trade_direction == 'SHORT':
                    if precio_redondeado <= price:
                        precio_redondeado = math.ceil(price / tick_size) * tick_size
            precio_formateado = f"{precio_redondeado:.{price_scale}f}"
            if float(precio_formateado) == 0.0 and price > 0:
                nueva_scale = price_scale + 4
                tick_size = 10 ** (-nueva_scale)
                precio_redondeado = round(price / tick_size) * tick_size
                precio_formateado = f"{precio_redondeado:.{nueva_scale}f}"
            return precio_formateado
        except Exception as e:
            logger.error(f"Error redondeando precio manualmente: {e}")
            return str(price)

    def set_leverage(self, symbol, leverage, hold_side='long'):
        """Configura apalancamiento"""
        try:
            request_path = '/api/v2/mix/account/set-leverage'
            body = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'leverage': str(leverage),
                'holdSide': hold_side
            }
            logger.info(f"Configurando leverage {leverage}x para {symbol} ({hold_side})")
            body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            headers = self._get_headers('POST', request_path, body_json)
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_json,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✓ Apalancamiento {leverage}x configurado en BITGET FUTUROS para {symbol}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error en set_leverage BITGET FUTUROS: {e}")
            return False

    def obtener_saldo_cuenta(self):
        """Obtiene el saldo actual"""
        try:
            accounts = self.get_account_info()
            if accounts:
                for account in accounts:
                    if account.get('marginCoin') == 'USDT':
                        balance_usdt = float(account.get('available', 0))
                        logger.info(f"💰 Saldo disponible USDT: ${balance_usdt:.2f}")
                        return balance_usdt
            return None
        except Exception as e:
            logger.error(f"Error obteniendo saldo de cuenta: {e}")
            return None

    def get_klines(self, symbol, interval='5m', limit=200):
        """Obtiene velas de BITGET FUTUROS"""
        try:
            interval_map = {
                '1m': '1m', '3m': '3m', '5m': '5m',
                '15m': '15m', '30m': '30m', '1h': '1H',
                '4h': '4H', '1d': '1D'
            }
            bitget_interval = interval_map.get(interval, '5m')
            request_path = f'/api/v2/mix/market/candles'
            params = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'granularity': bitget_interval,
                'limit': limit
            }
            response = requests.get(
                self.base_url + request_path,
                params=params,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    candles = data.get('data', [])
                    if candles:
                        return candles
            return None
        except Exception as e:
            logger.error(f"Error en get_klines: {e}")
            return None

    def place_order(self, symbol, side, size, order_type='market', posSide=None, 
                    is_hedged_account=False, stop_loss_price=None, take_profit_price=None, trade_direction=None):
        """Coloca orden de entrada"""
        request_path = '/api/v2/mix/order/place-order'
        if trade_direction is None:
            trade_direction = 'LONG' if side == 'buy' else 'SHORT'
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            price_scale = symbol_info.get('priceScale', 4)
            precision_bitget = price_scale
        else:
            precision_bitget = 4
        if stop_loss_price is not None:
            stop_loss_formatted = self.redondear_precio_manual(float(stop_loss_price), precision_bitget, symbol, trade_direction)
        else:
            stop_loss_formatted = None
        if take_profit_price is not None:
            take_profit_formatted = self.redondear_precio_manual(float(take_profit_price), precision_bitget, symbol)
        else:
            take_profit_formatted = None
        if is_hedged_account:
            body = {
                "symbol": symbol, "productType": "USDT-FUTURES",
                "marginMode": "isolated", "marginCoin": "USDT",
                "side": side, "orderType": "market", "size": str(size)
            }
        else:
            if not posSide:
                logger.error("❌ En modo unilateral, posSide es obligatorio")
                return None
            body = {
                "symbol": symbol, "productType": "USDT-FUTURES",
                "marginMode": "isolated", "marginCoin": "USDT",
                "side": side, "orderType": "market", "size": str(size),
                "posSide": posSide
            }
        if stop_loss_formatted is not None:
            body["presetStopLossPrice"] = str(stop_loss_formatted)
        if take_profit_formatted is not None:
            body["presetStopSurplusPrice"] = str(take_profit_formatted)
        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        headers = self._get_headers('POST', request_path, body_json)
        response = requests.post(
            self.base_url + request_path,
            headers=headers,
            data=body_json,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                logger.info(f"✅ Orden ejecutada ({side.upper()}) con TP/SL")
                return data.get('data')
        logger.error(f"❌ Error orden entrada: {response.text}")
        return None

    def close_position(self, symbol, hold_side, size=None, order_type='market'):
        """Cierra posición existente"""
        try:
            request_path = '/api/v2/mix/order/close-position'
            side = 'sell' if hold_side == 'long' else 'buy'
            body = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'holdSide': hold_side,
                'orderType': order_type,
                'size': str(size) if size else None
            }
            body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            headers = self._get_headers('POST', request_path, body_json)
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_json,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✅ Posición cerrada para {symbol}")
                    return data.get('data')
            logger.error(f"❌ Error cerrando posición: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error cerrando posición: {e}")
            return None

# REGIÓN 4: FUNCIONES DE TRADING CORE
# ================================================================================
def ejecutar_operacion_bitget(bitget_client, simbolo, tipo_operacion, capital_usd=None, leverage=None):
    """
    Ejecuta una operación completa en Bitget con gestión de capital inteligente
    """
    try:
        if not bitget_client:
            logger.error("❌ Cliente Bitget no disponible")
            return None
        
        logger.info(f"🚀 Ejecutando operación BITGET FUTUROS para {simbolo}")
        
        # Obtener precio actual
        klines = bitget_client.get_klines(simbolo, '1m', 1)
        if not klines:
            logger.error(f"❌ No se pudo obtener precio para {simbolo}")
            return None
        precio_actual = float(klines[0][4])
        logger.info(f"📊 Precio actual de {simbolo}: {precio_actual:.8f}")
        
        # Calcular capital a usar (3% del saldo actual)
        if capital_usd is None:
            balance = bitget_client.obtener_saldo_cuenta()
            if balance:
                capital_usd = balance * 0.03
                logger.info(f"💰 Usando 3% del saldo: ${capital_usd:.2f}")
            else:
                capital_usd = 10  # Fallback
                logger.warning(f"⚠️ No se pudo obtener balance, usando ${capital_usd} por defecto")
        
        # Obtener apalancamiento máximo
        if leverage is None:
            leverage = bitget_client.get_max_leverage(simbolo)
        leverage = min(leverage, 20)
        logger.info(f"⚡ Usando apalancamiento: {leverage}x")
        
        # Calcular tamaño de la orden
        valor_nocional = capital_usd * leverage
        cantidad_contratos = valor_nocional / precio_actual
        
        # Ajustar tamaño según reglas del símbolo
        reglas = bitget_client.obtener_reglas_simbolo(simbolo)
        cantidad_contratos = bitget_client.ajustar_tamaño_orden(simbolo, cantidad_contratos, reglas)
        
        if cantidad_contratos < reglas.get('min_trade_num', 1):
            logger.warning(f"⚠️ Tamaño de orden menor al mínimo permitido")
            return None
        
        logger.info(f"📊 Contratos a operar: {cantidad_contratos}")
        
        # Determinar side y posSide
        if tipo_operacion == "LONG":
            side = "buy"
            posSide = "long"
            hold_side = "long"
            sl_precio = precio_actual * (1 - 0.02)
            tp_precio = precio_actual * (1 + 0.04)
        else:
            side = "sell"
            posSide = "short"
            hold_side = "short"
            sl_precio = precio_actual * (1 + 0.02)
            tp_precio = precio_actual * (1 - 0.04)
        
        # Redondear precios con la precisión correcta
        sl_redondeado = float(bitget_client.redondear_precio_manual(sl_precio, None, simbolo, tipo_operacion))
        tp_redondeado = float(bitget_client.redondear_precio_manual(tp_precio, None, simbolo))
        
        logger.info(f"🎯 SL: {sl_redondeado:.8f}, TP: {tp_redondeado:.8f}")
        
        # Configurar apalancamiento
        bitget_client.set_leverage(simbolo, leverage, hold_side)
        
        # Colocar orden
        orden_entrada = bitget_client.place_order(
            simbolo, side, cantidad_contratos, 'market', posSide=posSide,
            is_hedged_account=False, stop_loss_price=sl_redondeado,
            take_profit_price=tp_redondeado, trade_direction=tipo_operacion
        )
        
        if not orden_entrada:
            logger.error(f"❌ Error colocando orden de entrada para {simbolo}")
            return None
        
        logger.info(f"✅ Orden de entrada ejecutada: {orden_entrada}")
        
        # Colocar SL y TP por separado como respaldo
        time.sleep(1)
        orden_sl = bitget_client.place_tpsl_order(
            simbolo, hold_side, sl_redondeado, 'stop_loss',
            stop_loss_price=sl_redondeado, trade_direction=tipo_operacion
        )
        orden_tp = bitget_client.place_tpsl_order(
            simbolo, hold_side, tp_redondeado, 'take_profit',
            take_profit_price=tp_redondeado
        )
        
        balance_actual = bitget_client.obtener_saldo_cuenta()
        
        return {
            'orden_entrada': orden_entrada,
            'orden_sl': orden_sl,
            'orden_tp': orden_tp,
            'simbolo': simbolo,
            'tipo_operacion': tipo_operacion,
            'precio_entrada': precio_actual,
            'stop_loss': sl_redondeado,
            'take_profit': tp_redondeado,
            'cantidad_contratos': cantidad_contratos,
            'capital_usado': capital_usd,
            'leverage': leverage,
            'saldo_cuenta': balance_actual
        }
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando operación para {simbolo}: {e}", exc_info=True)
        return None

# REGIÓN 5: INDICADORES TÉCNICOS AVANZADOS
# ================================================================================
def calcular_stochastic(cierres, maximos, minimos, period=14, k_period=3, d_period=3):
    """Calcula el oscilador Stochastic"""
    if len(cierres) < period:
        return 50, 50
    k_values = []
    for i in range(period-1, len(cierres)):
        highest_high = max(maximos[i-period+1:i+1])
        lowest_low = min(minimos[i-period+1:i+1])
        if highest_high == lowest_low:
            k = 50
        else:
            k = 100 * (cierres[i] - lowest_low) / (highest_high - lowest_low)
        k_values.append(k)
    if len(k_values) >= k_period:
        k_smoothed = []
        for i in range(k_period-1, len(k_values)):
            k_avg = sum(k_values[i-k_period+1:i+1]) / k_period
            k_smoothed.append(k_avg)
        if len(k_smoothed) >= d_period:
            d = sum(k_smoothed[-d_period:]) / d_period
            k_final = k_smoothed[-1]
            return k_final, d
    return 50, 50

def calcular_regresion_lineal(x, y):
    """Calcula regresión lineal simple"""
    if len(x) != len(y) or len(x) == 0:
        return None, None
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    denom = (n * sum_x2 - sum_x * sum_x)
    if denom == 0:
        pendiente = 0
    else:
        pendiente = (n * sum_xy - sum_x * sum_y) / denom
    intercepto = (sum_y - pendiente * sum_x) / n if n else 0
    return pendiente, intercepto

def calcular_pearson_y_angulo(x, y):
    """Calcula coeficiente de Pearson y ángulo de tendencia"""
    if len(x) != len(y) or len(x) < 2:
        return 0, 0
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    sum_y2 = np.sum(y * y)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if denominator == 0:
        return 0, 0
    pearson = numerator / denominator
    denom_pend = (n * sum_x2 - sum_x * sum_x)
    pendiente = (n * sum_xy - sum_x * sum_y) / denom_pend if denom_pend != 0 else 0
    if (max(y) - min(y)) != 0:
        angulo_radianes = math.atan(pendiente * len(x) / (max(y) - min(y)))
    else:
        angulo_radianes = 0
    angulo_grados = math.degrees(angulo_radianes)
    return pearson, angulo_grados

def clasificar_fuerza_tendencia(angulo_grados):
    """Clasifica la fuerza de la tendencia"""
    angulo_abs = abs(angulo_grados)
    if angulo_abs < 3:
        return "💔 Muy Débil", 1
    elif angulo_abs < 13:
        return "❤️‍🩹 Débil", 2
    elif angulo_abs < 27:
        return "💛 Moderada", 3
    elif angulo_abs < 45:
        return "💚 Fuerte", 4
    else:
        return "💙 Muy Fuerte", 5

def determinar_direccion_tendencia(angulo_grados, umbral_minimo=1):
    """Determina la dirección de la tendencia"""
    if abs(angulo_grados) < umbral_minimo:
        return "⚪ RANGO"
    elif angulo_grados > 0:
        return "🟢 ALCISTA"
    else:
        return "🔴 BAJISTA"

def calcular_r2(y_real, x, pendiente, intercepto):
    """Calcula el coeficiente R²"""
    if len(y_real) != len(x):
        return 0
    y_real = np.array(y_real)
    y_pred = pendiente * np.array(x) + intercepto
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - (ss_res / ss_tot)

def calcular_atr(datos, periodos=14):
    """Calcula el Average True Range"""
    high = datos['High']
    low = datos['Low']
    close = datos['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodos).mean()
    return atr.iloc[-1]

# REGIÓN 6: CLASE PRINCIPAL TRADINGBOT
# ================================================================================
class TradingBot:
    """
    Bot de trading principal con estrategia Breakout + Reentry
    Maneja múltiples símbolos y timeframes
    """
    def __init__(self, config):
        self.config = config
        self.operaciones_activas = {}
        self.senales_enviadas = set()
        self.esperando_reentry = {}
        self.breakouts_detectados = {}
        self.breakout_history = {}
        self.total_operaciones = 0
        self.archivo_log = self.config.get('log_path', 'operaciones_log.csv')
        self.estado_file = self.config.get('estado_file', 'estado_bot.json')
        self.ultima_optimizacion = datetime.now()
        self.operaciones_desde_optimizacion = 0
        self.mejores_parametros = {}
        self.config_optima_por_simbolo = {}
        self.bitget_client = None
        self.ejecutar_operaciones_automaticas = self.config.get('ejecutar_operaciones_automaticas', False)
        self.leverage_por_defecto = self.config.get('leverage_por_defecto', 20)
        self.inicializar_log()
        self.cargar_estado()
        self.inicializar_bitget()

    def inicializar_bitget(self):
        """Inicializa el cliente de Bitget si hay credenciales"""
        api_key = self.config.get('bitget_api_key')
        api_secret = self.config.get('bitget_secret_key')
        passphrase = self.config.get('bitget_passphrase')
        if api_key and api_secret and passphrase:
            self.bitget_client = BitgetClient(api_key, api_secret, passphrase, self)
            logger.info("✅ Cliente Bitget FUTUROS inicializado")
        else:
            logger.warning("⚠️ Credenciales de Bitget no encontradas - modo solo señales")

    def sincronizar_con_bitget(self):
        """Sincroniza el estado local con las posiciones reales en Bitget"""
        if not self.bitget_client:
            return
        try:
            logger.info("🔄 SINCRONIZANDO CON BITGET...")
            posiciones = self.bitget_client.get_positions()
            if posiciones:
                for pos in posiciones:
                    symbol = pos.get('symbol', '')
                    hold_side = pos.get('holdSide', '')
                    size = float(pos.get('holdVol', 0))
                    entry_price = float(pos.get('averageOpenPrice', 0))
                    if size > 0 and symbol:
                        self.operaciones_activas[symbol] = {
                            'tipo': 'LONG' if hold_side == 'long' else 'SHORT',
                            'precio_entrada': entry_price,
                            'take_profit': entry_price * 1.04 if hold_side == 'long' else entry_price * 0.96,
                            'stop_loss': entry_price * 0.98 if hold_side == 'long' else entry_price * 1.02,
                            'timestamp_entrada': datetime.now().isoformat(),
                            'operacion_ejecutada': True,
                            'operacion_manual_usuario': False,
                            'order_id_entrada': pos.get('positionId', '')
                        }
                logger.info(f"✅ Sincronizadas {len(posiciones)} posiciones")
            else:
                logger.info("ℹ️ No hay posiciones abiertas en Bitget")
        except Exception as e:
            logger.error(f"❌ Error sincronizando con Bitget: {e}")

    def verificar_y_recolocar_tp_sl(self):
        """Verifica y recoloca órdenes de TP/SL si es necesario"""
        if not self.bitget_client:
            return
        try:
            for simbolo, operacion in list(self.operaciones_activas.items()):
                if not operacion.get('operacion_ejecutada', False):
                    continue
                order_id_tp = operacion.get('order_id_tp')
                order_id_sl = operacion.get('order_id_sl')
                if order_id_tp and not self.bitget_client.verificar_orden_activa(order_id_tp, simbolo):
                    logger.info(f"🔄 TP no encontrado para {simbolo}, recolocando...")
                    hold_side = 'long' if operacion['tipo'] == 'LONG' else 'short'
                    self.bitget_client.place_tpsl_order(
                        simbolo, hold_side, operacion['take_profit'], 'take_profit',
                        take_profit_price=operacion['take_profit']
                    )
        except Exception as e:
            logger.error(f"❌ Error verificando/recapacitando TP/SL: {e}")

    def obtener_datos_mercado_config(self, symbol, timeframe, num_velas):
        """Obtiene datos del mercado con configuración específica"""
        try:
            datos = {'cierres': [], 'maximos': [], 'minimos': [], 'apertura': []}
            datos_bitget = None
            
            if self.bitget_client:
                klines = self.bitget_client.get_klines(symbol, timeframe, num_velas)
                if klines:
                    datos_bitget = klines
            
            if not datos_bitget:
                url = "https://api.binance.com/api/v3/klines"
                params = {'symbol': symbol, 'interval': timeframe, 'limit': num_velas}
                respuesta = requests.get(url, params=params, timeout=10)
                if respuesta.status_code == 200:
                    klines = respuesta.json()
                else:
                    return None
            else:
                klines = datos_bitget
            
            for kline in klines:
                datos['cierres'].append(float(kline[4]))
                datos['maximos'].append(float(kline[2]))
                datos['minimos'].append(float(kline[3]))
                datos['apertura'].append(float(kline[1]))
            
            datos['precio_actual'] = datos['cierres'][-1]
            datos['timestamp'] = klines[-1][0]
            return datos
        except Exception as e:
            return None

    def calcular_canal_regresion_config(self, datos, num_velas):
        """Calcula canal de regresión lineal con parámetros configurados"""
        try:
            cierres = datos['cierres']
            maximos = datos['maximos']
            minimos = datos['minimos']
            if len(cierres) < num_velas:
                return None
            cierres_recientes = cierres[-num_velas:]
            maximos_recientes = maximos[-num_velas:]
            minimos_recientes = minimos[-num_velas:]
            tiempos = list(range(len(cierres_recientes)))
            
            pendiente, intercepto = calcular_regresion_lineal(tiempos, cierres_recientes)
            pearson, angulo = calcular_pearson_y_angulo(tiempos, cierres_recientes)
            r2 = calcular_r2(cierres_recientes, tiempos, pendiente, intercepto)
            
            maximos_ajustados = []
            minimos_ajustados = []
            maximo_absoluto = max(maximos_recientes)
            minimo_absoluto = min(minimos_recientes)
            pendiente_resistencia = pendiente * 1.001
            pendiente_soporte = pendiente * 0.999
            resistencia = maximo_absoluto + abs(pendiente) * len(cierres_recientes)
            soporte = minimo_absoluto - abs(pendiente) * len(cierres_recientes)
            ancho_canal = resistencia - soporte
            ancho_canal_porcentual = (ancho_canal / datos['precio_actual']) * 100
            
            stoch_k, stoch_d = calcular_stochastic(cierres, maximos, minimos)
            direccion = determinar_direccion_tendencia(angulo)
            fuerza_texto, nivel_fuerza = clasificar_fuerza_tendencia(angulo)
            
            return {
                'pendiente': pendiente,
                'pendiente_resistencia': pendiente_resistencia,
                'pendiente_soporte': pendiente_soporte,
                'resistencia': resistencia,
                'soporte': soporte,
                'ancho_canal': ancho_canal,
                'ancho_canal_porcentual': ancho_canal_porcentual,
                'pearson': pearson,
                'angulo_tendencia': angulo,
                'r2_score': r2,
                'direccion': direccion,
                'fuerza_texto': fuerza_texto,
                'nivel_fuerza': nivel_fuerza,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            }
        except Exception as e:
            return None

    def detectar_breakout(self, simbolo, info_canal, datos_mercado):
        """Detecta patrones de breakout"""
        precio_actual = datos_mercado['precio_actual']
        resistencia = info_canal['resistencia']
        soporte = info_canal['soporte']
        precio_alto = max(datos_mercado['cierres'][-10:])
        precio_bajo = min(datos_mercado['cierres'][-10:])
        dentro_del_canal = soporte < precio_actual < resistencia
        
        if info_canal['nivel_fuerza'] < 2:
            return None
        
        if info_canal['angulo_tendencia'] > 0 and precio_actual >= resistencia and precio_alto > resistencia:
            return 'LONG'
        elif info_canal['angulo_tendencia'] < 0 and precio_actual <= soporte and precio_bajo < soporte:
            return 'SHORT'
        return None

    def detectar_reentry(self, simbolo, info_canal, datos_mercado):
        """Detecta oportunidades de reentrada"""
        precio_actual = datos_mercado['precio_actual']
        resistencia = info_canal['resistencia']
        soporte = info_canal['soporte']
        
        datos_previos = self.obtener_datos_mercado_config(simbolo, '1m', 20)
        if datos_previos and len(datos_previos['cierres']) >= 5:
            media_corta = sum(datos_previos['cierres'][-5:]) / 5
            media_larga = sum(datos_previos['cierres'][-20:]) / 20 if len(datos_previos['cierres']) >= 20 else media_corta
            tendencia_corta = media_corta - media_larga
            
            if info_canal['angulo_tendencia'] > 0 and tendencia_corta > 0 and info_canal['stoch_k'] < 80:
                if precio_actual > resistencia:
                    return 'LONG'
            elif info_canal['angulo_tendencia'] < 0 and tendencia_corta < 0 and info_canal['stoch_k'] > 20:
                if precio_actual < soporte:
                    return 'SHORT'
        return None

    def calcular_niveles_entrada(self, tipo_operacion, info_canal, precio_actual):
        """Calcula niveles de entrada, TP y SL"""
        if tipo_operacion == "LONG":
            precio_entrada = info_canal['resistencia'] * 1.001
            tp = precio_entrada * 1.004
            sl = info_canal['soporte'] * 0.999
        else:
            precio_entrada = info_canal['soporte'] * 0.999
            tp = precio_entrada * 0.996
            sl = info_canal['resistencia'] * 1.001
        return precio_entrada, tp, sl

    def buscar_configuracion_optima_simbolo(self, simbolo):
        """Busca la mejor configuración para un símbolo específico"""
        timeframes = self.config.get('timeframes', ['5m', '15m', '30m', '1h', '4h'])
        velas_options = self.config.get('velas_options', [80, 100, 120, 150, 200])
        
        for timeframe in timeframes:
            for num_velas in velas_options:
                datos = self.obtener_datos_mercado_config(simbolo, timeframe, num_velas)
                if not datos:
                    continue
                info_canal = self.calcular_canal_regresion_config(datos, num_velas)
                if info_canal:
                    ancho_pct = info_canal['ancho_canal_porcentual']
                    min_ancho = self.config.get('min_channel_width_percent', 4)
                    if ancho_pct >= min_ancho:
                        return {
                            'timeframe': timeframe,
                            'num_velas': num_velas,
                            'ancho_canal': ancho_pct,
                            'info_canal': info_canal
                        }
        return None

    def escanear_mercado(self):
        """Escanea el mercado completo en busca de oportunidades"""
        symbols = self.config.get('symbols', [])
        senales_encontradas = 0
        
        print(f"\n🔍 ESCANEANDO {len(symbols)} SÍMBOLOS...")
        for simbolo in symbols:
            try:
                if simbolo in self.operaciones_activas:
                    es_manual = self.operaciones_activas[simbolo].get('operacion_manual_usuario', False)
                    if es_manual:
                        print(f"   👤 {simbolo} - Operación manual detectada, omitiendo...")
                    continue
                
                config_optima = self.buscar_configuracion_optima_simbolo(simbolo)
                if not config_optima:
                    continue
                
                datos_mercado = self.obtener_datos_mercado_config(
                    simbolo, config_optima['timeframe'], config_optima['num_velas']
                )
                if not datos_mercado:
                    continue
                
                info_canal = self.calcular_canal_regresion_config(datos_mercado, config_optima['num_velas'])
                if not info_canal:
                    continue
                
                precio_actual = datos_mercado['precio_actual']
                
                if (info_canal['nivel_fuerza'] < 2 or
                    abs(info_canal['pearson']) < 0.4 or
                    info_canal['r2_score'] < 0.4):
                    continue
                
                if simbolo not in self.esperando_reentry:
                    tipo_breakout = self.detectar_breakout(simbolo, info_canal, datos_mercado)
                    if tipo_breakout:
                        self.esperando_reentry[simbolo] = {
                            'tipo': tipo_breakout,
                            'timestamp': datetime.now(),
                            'precio_breakout': precio_actual,
                            'config': config_optima
                        }
                        print(f"     🎯 {simbolo} - Breakout registrado, esperando reingreso...")
                        continue
                
                tipo_operacion = self.detectar_reentry(simbolo, info_canal, datos_mercado)
                if not tipo_operacion:
                    continue
                
                precio_entrada, tp, sl = self.calcular_niveles_entrada(
                    tipo_operacion, info_canal, datos_mercado['precio_actual']
                )
                if not precio_entrada or not tp or not sl:
                    continue
                
                if simbolo in self.operaciones_activas:
                    continue
                
                breakout_info = self.esperando_reentry[simbolo]
                self.generar_senal_operacion(
                    simbolo, tipo_operacion, precio_entrada, tp, sl, 
                    info_canal, datos_mercado, config_optima, breakout_info
                )
                senales_encontradas += 1
                self.breakout_history[simbolo] = datetime.now()
                del self.esperando_reentry[simbolo]
            except Exception as e:
                print(f"⚠️ Error analizando {simbolo}: {e}")
                continue
        
        if self.esperando_reentry:
            print(f"\n⏳ Esperando reingreso en {len(self.esperando_reentry)} símbolos:")
            for simbolo, info in self.esperando_reentry.items():
                tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                print(f"   • {simbolo} - {info['tipo']} - Esperando {tiempo_espera:.1f} min")
        
        if senales_encontradas > 0:
            print(f"✅ Se encontraron {senales_encontradas} señales de trading")
        else:
            print("❌ No se encontraron señales en este ciclo")
        return senales_encontradas

    def generar_senal_operacion(self, simbolo, tipo_operacion, precio_entrada, tp, sl,
                            info_canal, datos_mercado, config_optima, breakout_info=None):
        """Genera y envía señal de operación"""
        if simbolo in self.operaciones_activas:
            return
        
        if simbolo in self.senales_enviadas:
            return
        
        riesgo = abs(precio_entrada - sl)
        beneficio = abs(tp - precio_entrada)
        ratio_rr = beneficio / riesgo if riesgo > 0 else 0
        sl_percent = abs((sl - precio_entrada) / precio_entrada) * 100
        tp_percent = abs((tp - precio_entrada) / precio_entrada) * 100
        
        mensaje = f"""
🎯 <b>SEÑAL DE {tipo_operacion} - {simbolo}</b>
⏱️ <b>Configuración:</b> {config_optima['timeframe']} | {config_optima['num_velas']}v
📏 <b>Ancho Canal:</b> {info_canal['ancho_canal_porcentual']:.1f}%
💰 <b>Precio:</b> {datos_mercado['precio_actual']:.8f}
🎯 <b>Entrada:</b> {precio_entrada:.8f}
🛑 <b>SL:</b> {sl:.8f} ({sl_percent:.2f}%)
🎯 <b>TP:</b> {tp:.8f} ({tp_percent:.2f}%)
📊 <b>Ratio R/B:</b> {ratio_rr:.2f}:1
📈 <b>Tendencia:</b> {info_canal['direccion']} ({info_canal['angulo_tendencia']:.1f}°)
📊 <b>Pearson:</b> {info_canal['pearson']:.3f}
🎯 <b>R²:</b> {info_canal['r2_score']:.3f}
📊 <b>Stoch:</b> {info_canal['stoch_k']:.1f}/{info_canal['stoch_d']:.1f}
        """
        
        token = self.config.get('telegram_token')
        chat_ids = self.config.get('telegram_chat_ids', [])
        
        if token and chat_ids:
            try:
                print(f"     📊 Generando gráfico para {simbolo}...")
                buf = self.generar_grafico_profesional(simbolo, info_canal, datos_mercado, 
                                                      precio_entrada, tp, sl, tipo_operacion)
                if buf:
                    print(f"     📨 Enviando gráfico por Telegram...")
                    self.enviar_grafico_telegram(buf, token, chat_ids)
                    time.sleep(1)
                self._enviar_telegram_simple(mensaje, token, chat_ids)
                print(f"     ✅ Señal {tipo_operacion} para {simbolo} enviada")
            except Exception as e:
                print(f"     ❌ Error enviando señal: {e}")
        
        operacion_bitget = None
        if self.ejecutar_operaciones_automaticas and self.bitget_client:
            print(f"     🤖 Ejecutando operación automática en BITGET FUTUROS...")
            try:
                operacion_bitget = ejecutar_operacion_bitget(
                    bitget_client=self.bitget_client,
                    simbolo=simbolo,
                    tipo_operacion=tipo_operacion
                )
                if operacion_bitget:
                    print(f"     ✅ Operación ejecutada en BITGET FUTUROS para {simbolo}")
                    self._enviar_telegram_simple(f"🤖 Operación ejecutada para {simbolo}", token, chat_ids)
                    self.operaciones_activas[simbolo] = {
                        'tipo': tipo_operacion,
                        'precio_entrada': precio_entrada,
                        'take_profit': tp,
                        'stop_loss': sl,
                        'timestamp_entrada': datetime.now().isoformat(),
                        'angulo_tendencia': info_canal['angulo_tendencia'],
                        'pearson': info_canal['pearson'],
                        'r2_score': info_canal['r2_score'],
                        'nivel_fuerza': info_canal['nivel_fuerza'],
                        'timeframe_utilizado': config_optima['timeframe'],
                        'velas_utilizadas': config_optima['num_velas'],
                        'stoch_k': info_canal['stoch_k'],
                        'stoch_d': info_canal['stoch_d'],
                        'breakout_usado': breakout_info is not None,
                        'operacion_ejecutada': True,
                        'operacion_manual_usuario': False,
                        'order_id_entrada': operacion_bitget['orden_entrada'].get('orderId'),
                        'order_id_sl': operacion_bitget['orden_sl'].get('orderId') if operacion_bitget['orden_sl'] else None,
                        'order_id_tp': operacion_bitget['orden_tp'].get('orderId') if operacion_bitget['orden_tp'] else None,
                        'capital_usado': operacion_bitget['capital_usado'],
                        'leverage_usado': operacion_bitget['leverage']
                    }
                    self.guardar_estado()
            except Exception as e:
                print(f"     ⚠️ Error en ejecución automática: {e}")
        
        if not operacion_bitget:
            self.operaciones_activas[simbolo] = {
                'tipo': tipo_operacion,
                'precio_entrada': precio_entrada,
                'take_profit': tp,
                'stop_loss': sl,
                'timestamp_entrada': datetime.now().isoformat(),
                'angulo_tendencia': info_canal['angulo_tendencia'],
                'pearson': info_canal['pearson'],
                'r2_score': info_canal['r2_score'],
                'nivel_fuerza': info_canal['nivel_fuerza'],
                'timeframe_utilizado': config_optima['timeframe'],
                'velas_utilizadas': config_optima['num_velas'],
                'stoch_k': info_canal['stoch_k'],
                'stoch_d': info_canal['stoch_d'],
                'breakout_usado': breakout_info is not None,
                'operacion_ejecutada': False
            }
        
        self.senales_enviadas.add(simbolo)
        self.total_operaciones += 1

    def inicializar_log(self):
        """Inicializa el archivo de log"""
        if not os.path.exists(self.archivo_log):
            with open(self.archivo_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'tipo', 'precio_entrada',
                    'take_profit', 'stop_loss', 'precio_salida',
                    'resultado', 'pnl_percent', 'duracion_minutos',
                    'angulo_tendencia', 'pearson', 'r2_score',
                    'nivel_fuerza', 'timeframe_utilizado', 'velas_utilizadas',
                    'stoch_k', 'stoch_d', 'breakout_usado', 'operacion_ejecutada'
                ])

    def registrar_operacion(self, datos_operacion):
        """Registra una operación en el log"""
        with open(self.archivo_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datos_operacion['timestamp'],
                datos_operacion['symbol'],
                datos_operacion['tipo'],
                datos_operacion['precio_entrada'],
                datos_operacion['take_profit'],
                datos_operacion['stop_loss'],
                datos_operacion['precio_salida'],
                datos_operacion['resultado'],
                datos_operacion['pnl_percent'],
                datos_operacion['duracion_minutos'],
                datos_operacion['angulo_tendencia'],
                datos_operacion['pearson'],
                datos_operacion['r2_score'],
                datos_operacion['nivel_fuerza'],
                datos_operacion['timeframe_utilizado'],
                datos_operacion['velas_utilizadas'],
                datos_operacion['stoch_k'],
                datos_operacion['stoch_d'],
                datos_operacion['breakout_usado'],
                datos_operacion['operacion_ejecutada']
            ])

    def verificar_cierre_operaciones(self):
        """Verifica y cierra operaciones que alcanzaron TP/SL"""
        if not self.operaciones_activas:
            return []
        operaciones_cerradas = []
        for simbolo, operacion in list(self.operaciones_activas.items()):
            config_optima = self.config_optima_por_simbolo.get(simbolo)
            if not config_optima:
                continue
            if 'take_profit' not in operacion or 'stop_loss' not in operacion:
                continue
            datos = self.obtener_datos_mercado_config(simbolo, config_optima['timeframe'], config_optima['num_velas'])
            if not datos:
                continue
            precio_actual = datos['precio_actual']
            tp = operacion['take_profit']
            sl = operacion['stop_loss']
            tipo = operacion['tipo']
            resultado = None
            if tipo == "LONG":
                if precio_actual >= tp:
                    resultado = "TP"
                elif precio_actual <= sl:
                    resultado = "SL"
            else:
                if precio_actual <= tp:
                    resultado = "TP"
                elif precio_actual >= sl:
                    resultado = "SL"
            if resultado:
                if tipo == "LONG":
                    pnl_percent = ((precio_actual - operacion['precio_entrada']) / operacion['precio_entrada']) * 100
                else:
                    pnl_percent = ((operacion['precio_entrada'] - precio_actual) / operacion['precio_entrada']) * 100
                tiempo_entrada = datetime.fromisoformat(operacion['timestamp_entrada'])
                duracion_minutos = (datetime.now() - tiempo_entrada).total_seconds() / 60
                datos_operacion = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': simbolo,
                    'tipo': tipo,
                    'precio_entrada': operacion['precio_entrada'],
                    'take_profit': tp,
                    'stop_loss': sl,
                    'precio_salida': precio_actual,
                    'resultado': resultado,
                    'pnl_percent': pnl_percent,
                    'duracion_minutos': duracion_minutos,
                    'angulo_tendencia': operacion.get('angulo_tendencia', 0),
                    'pearson': operacion.get('pearson', 0),
                    'r2_score': operacion.get('r2_score', 0),
                    'nivel_fuerza': operacion.get('nivel_fuerza', 1),
                    'timeframe_utilizado': operacion.get('timeframe_utilizado', 'N/A'),
                    'velas_utilizadas': operacion.get('velas_utilizadas', 0),
                    'stoch_k': operacion.get('stoch_k', 0),
                    'stoch_d': operacion.get('stoch_d', 0),
                    'breakout_usado': operacion.get('breakout_usado', False),
                    'operacion_ejecutada': operacion.get('operacion_ejecutada', False)
                }
                token = self.config.get('telegram_token')
                chats = self.config.get('telegram_chat_ids', [])
                if token and chats:
                    try:
                        emoji = "🟢" if resultado == "TP" else "🔴"
                        mensaje = f"{emoji} <b>OPERACIÓN CERRADA - {simbolo}</b>\nResultado: {resultado}\nPnL: {pnl_percent:.2f}%"
                        self._enviar_telegram_simple(mensaje, token, chats)
                    except Exception:
                        pass
                self.registrar_operacion(datos_operacion)
                operaciones_cerradas.append(simbolo)
                del self.operaciones_activas[simbolo]
                if simbolo in self.senales_enviadas:
                    self.senales_enviadas.remove(simbolo)
                self.operaciones_desde_optimizacion += 1
                print(f"     📊 {simbolo} Operación {resultado} - PnL: {pnl_percent:.2f}%")
        return operaciones_cerradas

    def generar_grafico_profesional(self, simbolo, info_canal, datos_mercado, precio_entrada, tp, sl, tipo_operacion):
        """Genera gráfico profesional con mplfinance"""
        try:
            config_optima = self.config_optima_por_simbolo.get(simbolo)
            if not config_optima:
                return None
            
            if self.bitget_client:
                klines = self.bitget_client.get_klines(simbolo, config_optima['timeframe'], config_optima['num_velas'])
                if klines:
                    df_data = []
                    for kline in klines:
                        df_data.append({
                            'Date': pd.to_datetime(int(kline[0]), unit='ms'),
                            'Open': float(kline[1]),
                            'High': float(kline[2]),
                            'Low': float(kline[3]),
                            'Close': float(kline[4]),
                            'Volume': float(kline[5])
                        })
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                else:
                    return None
            else:
                return None
            
            tiempos_reg = list(range(len(df)))
            resistencia_values = []
            soporte_values = []
            for i, t in enumerate(tiempos_reg):
                resist = info_canal['pendiente_resistencia'] * t + (info_canal['resistencia'] - info_canal['pendiente_resistencia'] * tiempos_reg[-1])
                sop = info_canal['pendiente_soporte'] * t + (info_canal['soporte'] - info_canal['pendiente_soporte'] * tiempos_reg[-1])
                resistencia_values.append(resist)
                soporte_values.append(sop)
            df['Resistencia'] = resistencia_values
            df['Soporte'] = soporte_values
            
            period = 14
            k_period = 3
            d_period = 3
            stoch_k_values = []
            for i in range(len(df)):
                if i < period - 1:
                    stoch_k_values.append(50)
                else:
                    highest_high = df['High'].iloc[i-period+1:i+1].max()
                    lowest_low = df['Low'].iloc[i-period+1:i+1].min()
                    if highest_high == lowest_low:
                        k = 50
                    else:
                        k = 100 * (df['Close'].iloc[i] - lowest_low) / (highest_high - lowest_low)
                    stoch_k_values.append(k)
            k_smoothed = []
            for i in range(len(stoch_k_values)):
                if i < k_period - 1:
                    k_smoothed.append(stoch_k_values[i])
                else:
                    k_avg = sum(stoch_k_values[i-k_period+1:i+1]) / k_period
                    k_smoothed.append(k_avg)
            stoch_d_values = []
            for i in range(len(k_smoothed)):
                if i < d_period - 1:
                    stoch_d_values.append(k_smoothed[i])
                else:
                    d = sum(k_smoothed[i-d_period+1:i+1]) / d_period
                    stoch_d_values.append(d)
            df['Stoch_K'] = k_smoothed
            df['Stoch_D'] = stoch_d_values
            
            apds = [
                mpf.make_addplot(df['Resistencia'], color='#5444ff', linestyle='--', width=2, panel=0),
                mpf.make_addplot(df['Soporte'], color="#5444ff", linestyle='--', width=2, panel=0),
            ]
            if precio_entrada and tp and sl:
                entry_line = [precio_entrada] * len(df)
                tp_line = [tp] * len(df)
                sl_line = [sl] * len(df)
                apds.append(mpf.make_addplot(entry_line, color='#FFD700', linestyle='-', width=2, panel=0))
                apds.append(mpf.make_addplot(tp_line, color='#00FF00', linestyle='-', width=2, panel=0))
                apds.append(mpf.make_addplot(sl_line, color='#FF0000', linestyle='-', width=2, panel=0))
            apds.append(mpf.make_addplot(df['Stoch_K'], color='#00BFFF', width=1.5, panel=1, ylabel='Stochastic'))
            apds.append(mpf.make_addplot(df['Stoch_D'], color='#FF6347', width=1.5, panel=1))
            
            fig, axes = mpf.plot(df, type='candle', style='nightclouds',
                               title=f'{simbolo} | {tipo_operacion} | {config_optima["timeframe"]}',
                               ylabel='Precio',
                               addplot=apds,
                               volume=False,
                               returnfig=True,
                               figsize=(14, 10),
                               panel_ratios=(3, 1))
            axes[2].set_ylim([0, 100])
            axes[2].grid(True, alpha=0.3)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
            buf.seek(0)
            plt.close(fig)
            return buf
        except Exception as e:
            print(f"⚠️ Error generando gráfico: {e}")
            return None

    def enviar_grafico_telegram(self, buf, token, chat_ids):
        """Envía gráfico por Telegram"""
        if not buf or not token or not chat_ids:
            return False
        buf.seek(0)
        exito = False
        for chat_id in chat_ids:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            try:
                buf.seek(0)
                files = {'photo': ('grafico.png', buf.read(), 'image/png')}
                data = {'chat_id': chat_id}
                r = requests.post(url, files=files, data=data, timeout=120)
                if r.status_code == 200:
                    exito = True
            except Exception as e:
                print(f"     ❌ Error enviando gráfico: {e}")
        return exito

    def _enviar_telegram_simple(self, mensaje, token, chat_ids):
        """Envía mensaje simple a Telegram"""
        if not token or not chat_ids:
            return False
        resultados = []
        for chat_id in chat_ids:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {'chat_id': chat_id, 'text': mensaje, 'parse_mode': 'HTML'}
            try:
                r = requests.post(url, json=payload, timeout=10)
                resultados.append(r.status_code == 200)
            except Exception:
                resultados.append(False)
        return any(resultados)

    def reoptimizar_periodicamente(self):
        """Reoptimiza parámetros periódicamente"""
        try:
            horas_desde_opt = (datetime.now() - self.ultima_optimizacion).total_seconds() / 7200
            if self.operaciones_desde_optimizacion >= 8 or horas_desde_opt >= self.config.get('reevaluacion_horas', 24):
                print("🔄 Iniciando re-optimización automática...")
                ia = OptimizadorIA(log_path=self.log_path, min_samples=self.config.get('min_samples_optimizacion', 30))
                nuevos_parametros = ia.buscar_mejores_parametros()
                if nuevos_parametros:
                    self.actualizar_parametros(nuevos_parametros)
                    self.ultima_optimizacion = datetime.now()
                    self.operaciones_desde_optimizacion = 0
                    print("✅ Parámetros actualizados en tiempo real")
        except Exception as e:
            print(f"⚠ Error en re-optimización automática: {e}")

    def actualizar_parametros(self, nuevos_parametros):
        """Actualiza parámetros de configuración"""
        self.config['trend_threshold_degrees'] = nuevos_parametros.get('trend_threshold_degrees', 
                                                                        self.config.get('trend_threshold_degrees', 16))
        self.config['min_trend_strength_degrees'] = nuevos_parametros.get('min_trend_strength_degrees', 
                                                                           self.config.get('min_trend_strength_degrees', 16))
        self.config['entry_margin'] = nuevos_parametros.get('entry_margin', 
                                                             self.config.get('entry_margin', 0.001))

    def guardar_estado(self):
        """Guarda el estado del bot"""
        try:
            estado = {
                'operaciones_activas': {k: {kk: vv for kk, vv in v.items() if kk != 'operacion_manual_usuario' or not vv} 
                                       for k, v in self.operaciones_activas.items()},
                'senales_enviadas': list(self.senales_enviadas),
                'esperando_reentry': {k: {'tipo': v['tipo'], 'timestamp': v['timestamp'].isoformat()} 
                                     for k, v in self.esperando_reentry.items()},
                'total_operaciones': self.total_operaciones,
                'ultima_optimizacion': self.ultima_optimizacion.isoformat(),
                'operaciones_desde_optimizacion': self.operaciones_desde_optimizacion,
                'config': self.config
            }
            with open(self.estado_file, 'w') as f:
                json.dump(estado, f, indent=2)
        except Exception as e:
            print(f"⚠ Error guardando estado: {e}")

    def cargar_estado(self):
        """Carga el estado del bot"""
        try:
            if os.path.exists(self.estado_file):
                with open(self.estado_file, 'r') as f:
                    estado = json.load(f)
                self.operaciones_activas = estado.get('operaciones_activas', {})
                self.senales_enviadas = set(estado.get('senales_enviadas', []))
                self.total_operaciones = estado.get('total_operaciones', 0)
                self.ultima_optimizacion = datetime.fromisoformat(estado.get('ultima_optimizacion', datetime.now().isoformat()))
                self.operaciones_desde_optimizacion = estado.get('operaciones_desde_optimizacion', 0)
                print(f"✅ Estado cargado: {len(self.operaciones_activas)} operaciones activas")
        except Exception as e:
            print(f"⚠ Error cargando estado: {e}")

    def ejecutar_analisis(self):
        """Ejecuta análisis completo siguiendo el flujo del diagrama"""
        try:
            # FLUJO DEL DIAGRAMA - PASO 1: SINCRONIZAR CON BITGET
            if self.bitget_client:
                self.sincronizar_con_bitget()
            
            # FLUJO DEL DIAGRAMA - PASO 2: VERIFICAR TP/SL
            if self.bitget_client:
                self.verificar_y_recolocar_tp_sl()
            
            # FLUJO DEL DIAGRAMA - PASO 3: VERIFICAR CIERRES
            cierres = self.verificar_cierre_operaciones()
            if cierres:
                print(f"     📊 Operaciones cerradas: {', '.join(cierres)}")
            
            # FLUJO DEL DIAGRAMA - PASO 5: REEVALUACIÓN (Reoptimizar)
            if random.random() < 0.1:
                self.reoptimizar_periodicamente()
            
            # Guardar estado
            self.guardar_estado()
            
            # FLUJO DEL DIAGRAMA - PASO 4: ESCANEAR MERCADO
            return self.escanear_mercado()
            
        except Exception as e:
            logger.error(f"❌ Error en ejecutar_analisis: {e}")
            return 0

    def iniciar(self):
        """Inicia el bot"""
        print("\n" + "=" * 70)
        print("🤖 BOT DE TRADING - ESTRATEGIA BREAKOUT + REENTRY")
        print("🎯 PRIORIDAD: TIMEFRAMES CORTOS")
        print("💾 PERSISTENCIA: ACTIVADA")
        print("🏦 INTEGRACIÓN: BITGET FUTUROS API")
        print("=" * 70)
        print(f"💱 Símbolos: {len(self.config.get('symbols', []))} monedas")
        print(f"⏰ Timeframes: {', '.join(self.config.get('timeframes', []))}")
        print(f"📏 ANCHO MÍNIMO: {self.config.get('min_channel_width_percent', 4)}%")
        if self.bitget_client:
            print(f"🤖 BITGET FUTUROS: ✅ API Conectada")
            print(f"⚡ Apalancamiento: {self.leverage_por_defecto}x")
            if self.ejecutar_operaciones_automaticas:
                print(f"🤖 AUTO-TRADING: ✅ ACTIVADO (Dinero REAL)")
                print("⚠️  ADVERTENCIA: TRADING AUTOMÁTICO REAL ACTIVADO")
        else:
            print(f"🤖 BITGET FUTUROS: ❌ No configurado (solo señales)")
        print("=" * 70)
        
        # SINCRONIZACIÓN INICIAL
        if self.bitget_client:
            print("\n🔄 REALIZANDO SINCRONIZACIÓN INICIAL CON BITGET...")
            self.sincronizar_con_bitget()
            print("✅ Sincronización inicial completada")
        
        try:
            while True:
                nuevas_senales = self.ejecutar_analisis()
                minutos_espera = self.config.get('scan_interval_minutes', 6)
                print(f"\n✅ Análisis completado. Señales nuevas: {nuevas_senales}")
                print(f"⏳ Próximo análisis en {minutos_espera} minutos...")
                print("-" * 60)
                for minuto in range(minutos_espera):
                    time.sleep(60)
                    restantes = minutos_espera - (minuto + 1)
                    if restantes > 0 and restantes % 5 == 0:
                        print(f"   ⏰ {restantes} minutos restantes...")
        except KeyboardInterrupt:
            print("\n🛑 Bot detenido por el usuario")
            self.guardar_estado()

# REGIÓN 7: CONFIGURACIÓN
# ================================================================================
def crear_config_desde_entorno():
    """Configuración desde variables de entorno"""
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    telegram_chat_ids_str = os.environ.get('TELEGRAM_CHAT_ID', '1570204748')
    telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids_str.split(',') if cid.strip()]
    
    return {
        'min_channel_width_percent': 4.0,
        'trend_threshold_degrees': 16.0,
        'min_trend_strength_degrees': 16.0,
        'entry_margin': 0.001,
        'min_rr_ratio': 1.2,
        'scan_interval_minutes': 6,
        'timeframes': ['5m', '15m', '30m', '1h', '4h'],
        'velas_options': [80, 100, 120, 150, 200],
        'symbols': [
            'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'SHIBUSDT', 'POPCATUSDT',
            'CHILLGUYUSDT', 'PNUTUSDT', 'MEWUSDT', 'FARTCOINUSDT', 'DOGEUSDT',
            'VINEUSDT', 'HIPPOUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT',
            'ADAUSDT', 'ATOMUSDT', 'LINKUSDT', 'UNIUSDT',
            'SUSHIUSDT', 'CRVUSDT', 'SNXUSDT', 'SANDUSDT', 'MANAUSDT',
            'AXSUSDT', 'LRCUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT',
            'FILUSDT', 'SUIUSDT', 'AAVEUSDT', 'ENSUSDT',
            'LDOUSDT', 'POLUSDT', 'ALGOUSDT', 'QNTUSDT',
            '1INCHUSDT', 'CVCUSDT', 'STGUSDT', 'ENJUSDT', 'GALAUSDT',
            'MAGICUSDT', 'REZUSDT', 'BLURUSDT', 'HMSTRUSDT', 'BEATUSDT',
            'ZEREBROUSDT', 'ZENUSDT', 'CETUSUSDT', 'DRIFTUSDT', 'PHAUSDT',
            'API3USDT', 'ACHUSDT', 'SPELLUSDT', 'YGGUSDT',
            'GMXUSDT', 'C98USDT', 'XMRUSDT', 'DOTUSDT', 'BNBUSDT',
            'SOLUSDT', 'AVAXUSDT', 'VETUSDT', 'BCHUSDT', 'NEOUSDT',
            'TIAUSDT', 'TONUSDT', 'TRUMPUSDT', 'IPUSDT', 'TAOUSDT',
            'XPLUSDT', 'HOLOUSDT', 'MONUSDT', 'OGUSDT', 'MSTRUSDT',
            'VIRTUALUSDT', 'TLMUSDT', 'BOMEUSDT', 'KAITOUSDT', 'APEUSDT',
            'METUSDT', 'TUTUSDT'
        ],
        'telegram_token': os.environ.get('TELEGRAM_TOKEN'),
        'telegram_chat_ids': telegram_chat_ids,
        'auto_optimize': True,
        'min_samples_optimizacion': 15,
        'reevaluacion_horas': 6,
        'log_path': os.path.join(directorio_actual, 'operaciones_log.csv'),
        'estado_file': os.path.join(directorio_actual, 'estado_bot.json'),
        'bitget_api_key': os.environ.get('BITGET_API_KEY'),
        'bitget_api_secret': os.environ.get('BITGET_SECRET_KEY'),
        'bitget_passphrase': os.environ.get('BITGET_PASSPHRASE'),
        'webhook_url': os.environ.get('WEBHOOK_URL'),
        'ejecutar_operaciones_automaticas': os.environ.get('EJECUTAR_OPERACIONES_AUTOMATICAS', 'false').lower() == 'true',
        'leverage_por_defecto': min(int(os.environ.get('LEVERAGE_POR_DEFECTO', '20')), 20)
    }

# REGIÓN 8: SERVICIO WEB FLASK
# ================================================================================
app = Flask(__name__)

config = crear_config_desde_entorno()
bot = TradingBot(config)

def run_bot_loop():
    """Ejecuta el bot en un hilo separado"""
    logger.info("🤖 Iniciando hilo del bot...")
    while True:
        try:
            bot.ejecutar_analisis()
            time.sleep(bot.config.get('scan_interval_minutes', 1) * 60)
        except Exception as e:
            logger.error(f"❌ Error en el hilo del bot: {e}", exc_info=True)
            time.sleep(60)

bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
bot_thread.start()

@app.route('/')
def index():
    return "✅ Bot Breakout + Reentry con integración Bitget está en línea.", 200

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    if request.is_json:
        update = request.get_json()
        logger.info(f"📩 Update recibido: {json.dumps(update)}")
        return jsonify({"status": "ok"}), 200
    return jsonify({"error": "Request must be JSON"}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del bot"""
    try:
        status = {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "operaciones_activas": len(bot.operaciones_activas),
            "esperando_reentry": len(bot.esperando_reentry),
            "total_operaciones": bot.total_operaciones,
            "bitget_conectado": bot.bitget_client is not None,
            "auto_trading": bot.ejecutar_operaciones_automaticas
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_telegram_webhook():
    """Configura webhook de Telegram"""
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        logger.warning("⚠️ No hay token de Telegram configurado")
        return
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        render_url = os.environ.get('RENDER_EXTERNAL_URL')
        if render_url:
            webhook_url = f"{render_url}/webhook"
        else:
            logger.warning("⚠️ No hay URL de webhook configurada")
            return
    try:
        logger.info(f"🔗 Configurando webhook Telegram en: {webhook_url}")
        requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
        time.sleep(1)
        response = requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Webhook de Telegram configurado correctamente")
    except Exception as e:
        logger.error(f"❌ Error configurando webhook: {e}")

# REGIÓN 9: PUNTO DE ENTRADA PRINCIPAL
# ================================================================================
if __name__ == '__main__':
    logger.info("🚀 Iniciando aplicación Flask...")
    setup_telegram_webhook()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
