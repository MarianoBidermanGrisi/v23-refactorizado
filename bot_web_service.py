# bot_breakout_reentry.py
# VERSIÓN COMPLETA OPTIMIZADA - Sin código muerto
# Estrategia Breakout + Reentry para trading en Bitget Futures
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
import threading
import logging
from io import BytesIO

# Configuración de matplotlib y mplfinance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from flask import Flask, request, jsonify

# ============================================
# FUNCIONES DE LOGGING DETALLADO
# ============================================
def obtener_emoji_direccion(direccion):
    """Retorna emoji según dirección de tendencia"""
    return {
        'ALCISTA': '🟢',
        'BAJISTA': '🔴',
        'RANGO': '🟡'
    }.get(direccion, '⚪')

def obtener_emoji_fuerza(fuerza_texto):
    """Retorna emoji según fuerza de tendencia"""
    if 'Muy Fuerte' in fuerza_texto:
        return '💙'
    elif 'Fuerte' in fuerza_texto:
        return '💚'
    elif 'Moderada' in fuerza_texto:
        return '💛'
    elif 'Débil' in fuerza_texto:
        return '❤️‍🩹'
    else:
        return '⚪'

def obtener_emoji_stoch(stoch_estado):
    """Retorna emoji según estado del Stochastic"""
    return {
        'SOBRECOMPRA': '📈',
        'SOBREVENTA': '📉',
        'NEUTRO': '➖'
    }.get(stoch_estado, '➖')

def obtener_emoji_precio(posicion, direccion):
    """Retorna emoji según posición del precio"""
    if posicion == 'DENTRO':
        return '📍'
    elif posicion == 'FUERA':
        if direccion == 'ALCISTA':
            return '🔼 FUERA (arriba)'
        else:
            return '🔽 FUERA (abajo)'
    return '📍'

def log_detalle_simbolo(simbolo, timeframe, num_velas, direccion, angulo, fuerza_texto, 
                        ancho_canal_porcentual, stoch_k, stoch_d, stoch_estado, precio_actual, 
                        resistencia, soporte, posicion):
    """Genera línea de logging detallada para un símbolo"""
    emoji_direccion = obtener_emoji_direccion(direccion)
    emoji_fuerza = obtener_emoji_fuerza(fuerza_texto)
    emoji_stoch = obtener_emoji_stoch(stoch_estado)
    emoji_precio = obtener_emoji_precio(posicion, direccion)
    
    angulo_str = f"{angulo:.1f}" if abs(angulo) >= 1 else f"{angulo:.2f}"
    
    return f"📊 {simbolo} - {timeframe} - {num_velas}v | {emoji_direccion} {direccion} ({angulo_str}° - {emoji_fuerza} {fuerza_texto}) | Ancho: {ancho_canal_porcentual:.1f}% - Stoch: {stoch_k:.1f}/{stoch_d:.1f} {emoji_stoch} {stoch_estado} | Precio: {emoji_precio}"

def log_detalle_breakout(simbolo, tipo_breakout, precio_breakout, nivel_ruptura, direccion_ruptura):
    """Genera línea de logging detallada para un breakout"""
    if direccion_ruptura == 'ARRIBA':
        return f"🚀 {simbolo} - BREAKOUT {tipo_breakout}: {precio_breakout:.8f} ↑ Ruptura resistencia: {nivel_ruptura:.8f}"
    else:
        return f"🚀 {simbolo} - BREAKOUT {tipo_breakout}: {precio_breakout:.8f} ↓ Ruptura soporte: {nivel_ruptura:.8f}"

# Configurar logging básico
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# OPTIMIZADOR IA
# ---------------------------
class OptimizadorIA:
    def __init__(self, log_path="operaciones_log.csv", min_samples=15):
        self.log_path = log_path
        self.min_samples = min_samples
        self.datos = self.cargar_datos()

    def cargar_datos(self):
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
            print("No se encontró operaciones_log.csv (optimizador)")
        return datos

    def evaluar_configuracion(self, trend_threshold, min_strength, entry_margin):
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
        if not self.datos or len(self.datos) < self.min_samples:
            print(f"No hay suficientes datos para optimizar (se requieren {self.min_samples}, hay {len(self.datos)})")
            return None
        mejor_score = -1e9
        mejores_param = None
        trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
        strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
        margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
        combos = list(itertools.product(trend_values, strength_values, margin_values))
        total = len(combos)
        print(f"Optimizador: probando {total} combinaciones...")
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
            print("Optimizador: mejores parámetros encontrados:", mejores_param)
            try:
                with open("mejores_parametros.json", "w", encoding='utf-8') as f:
                    json.dump(mejores_param, f, indent=2)
            except Exception as e:
                print(f"Error guardando mejores_parametros.json: {e}")
        else:
            print("No se encontró una configuración mejor")
        return mejores_param

# ---------------------------
# BITGET CLIENT - INTEGRACIÓN COMPLETA CON API BITGET FUTUROS
# ---------------------------
class BitgetClient:
    def __init__(self, api_key, api_secret, passphrase, bot_instance=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self._bot_instance = bot_instance
        logger.info(f"Cliente Bitget FUTUROS inicializado con API Key: {api_key[:10]}...")

    def _generate_signature(self, timestamp, method, request_path, body=''):
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
        try:
            logger.info("Verificando credenciales Bitget FUTUROS...")
            
            if not self.api_key or not self.api_secret or not self.passphrase:
                logger.error("Credenciales incompletas")
                return False
            
            accounts = self.get_account_info()
            if accounts:
                logger.info("Credenciales BITGET FUTUROS verificadas exitosamente")
                for account in accounts:
                    if account.get('marginCoin') == 'USDT':
                        available = float(account.get('available', 0))
                        logger.info(f"Balance disponible FUTUROS: {available:.2f} USDT")
                return True
            else:
                logger.error("No se pudo verificar credenciales BITGET FUTUROS")
                return False
                
        except Exception as e:
            logger.error(f"Error verificando credenciales BITGET FUTUROS: {e}")
            return False

    def get_account_info(self, product_type='USDT-FUTURES'):
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
            else:
                logger.error(f"Error HTTP BITGET FUTUROS: {response.status_code} - {response.text}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error en get_account_info BITGET FUTUROS: {e}")
            return None

    def get_symbol_info(self, symbol):
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
            query_string = f"?productType=USDT-MIX"
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
            
            return None
        except Exception as e:
            logger.error(f"Error obteniendo info del símbolo BITGET FUTUROS: {e}")
            return None

    def place_tpsl_order(self, symbol, hold_side, trigger_price, order_type='stop_loss', stop_loss_price=None, take_profit_price=None):
        request_path = '/api/v2/mix/order/place-pos-tpsl'
        
        precision_adaptada = self.obtener_precision_adaptada(trigger_price, symbol)
        trigger_price_formatted = self.redondear_precio_manual(trigger_price, precision_adaptada, symbol)
        
        body = {
            'symbol': symbol,
            'productType': 'USDT-FUTURES',
            'marginCoin': 'USDT',
            'holdSide': hold_side,
            'orderType': 'market',
            'triggerType': 'mark_price',
            'triggerPrice': trigger_price_formatted,
            'stopLossTriggerType': 'mark_price',
            'stopSurplusTriggerType': 'mark_price'
        }
        
        if order_type == 'stop_loss' and stop_loss_price:
            precision_sl = self.obtener_precision_adaptada(stop_loss_price, symbol)
            stop_loss_formatted = self.redondear_precio_manual(stop_loss_price, precision_sl, symbol)
            body['stopLossTriggerPrice'] = stop_loss_formatted
            logger.info(f"SL para {symbol}: precio={stop_loss_price}, precision={precision_sl}, formatted={stop_loss_formatted}")
        elif order_type == 'take_profit' and take_profit_price:
            precision_tp = self.obtener_precision_adaptada(take_profit_price, symbol)
            take_profit_formatted = self.redondear_precio_manual(take_profit_price, precision_tp, symbol)
            body['stopSurplusTriggerPrice'] = take_profit_formatted
            logger.info(f"TP para {symbol}: precio={take_profit_price}, precision={precision_tp}, formatted={take_profit_formatted}")
        
        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        headers = self._get_headers('POST', request_path, body_json)
        
        logger.info(f"Enviando orden {order_type} para {symbol}: {body}")
        
        response = requests.post(
            self.base_url + request_path,
            headers=headers,
            data=body_json,
            timeout=10
        )
        
        logger.info(f"Respuesta TP/SL BITGET: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                logger.info(f"{order_type.upper()} creado correctamente para {symbol}")
                return data.get('data')
            else:
                if data.get('code') == '40017':
                    logger.error(f"Error 40017 en {order_type}: {data.get('msg')}")
                    logger.error(f"Body enviado: {body}")
                if data.get('code') == '40034':
                    logger.error(f"Error 40034 en {order_type}: {data.get('msg')}")
                    logger.error(f"Body enviado: {body}")
        
        logger.error(f"Error creando {order_type}: {response.text}")
        return None

    def get_position_mode(self, symbol, product_type='USDT-FUTURES'):
        try:
            request_path = '/api/v2/mix/account/account'
            params = {'productType': product_type, 'marginCoin': 'USDT', 'symbol': symbol}
            
            query_parts = []
            for key, value in params.items():
                query_parts.append(f"{key}={value}")
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
                    account_data = data.get('data', {})
                    logger.info(f"Modo de cuenta para {symbol}: {account_data}")
                    return account_data
            
            return None
        except Exception as e:
            logger.error(f"Error obteniendo modo de posición: {e}")
            return None

    def set_hedged_mode(self, symbol, hedged_mode=True):
        try:
            request_path = '/api/v2/mix/account/set-position-mode'
            body = {
                'symbol': symbol,
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'holdSide': 'long' if hedged_mode else 'short',
                'positionMode': 'hedged' if hedged_mode else 'single'
            }
            
            logger.info(f"Configurando modo {'hedged' if hedged_mode else 'unilateral'} para {symbol}")
            
            body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            headers = self._get_headers('POST', request_path, body_json)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_json,
                timeout=10
            )
            
            logger.info(f"Respuesta set_position_mode BITGET: {response.status_code} - {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"Modo {'hedged' if hedged_mode else 'unilateral'} configurado para {symbol}")
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error configurando modo de posición: {e}")
            return False

    def place_order(self, symbol, side, size, order_type='market', posSide=None, is_hedged_account=False, 
                    stop_loss_price=None, take_profit_price=None):
        request_path = '/api/v2/mix/order/place-order'

        if stop_loss_price is not None:
            precision_sl = self.obtener_precision_adaptada(float(stop_loss_price), symbol)
            stop_loss_formatted = self.redondear_precio_manual(float(stop_loss_price), precision_sl, symbol)
        else:
            stop_loss_formatted = None
            
        if take_profit_price is not None:
            precision_tp = self.obtener_precision_adaptada(float(take_profit_price), symbol)
            take_profit_formatted = self.redondear_precio_manual(float(take_profit_price), precision_tp, symbol)
        else:
            take_profit_formatted = None

        if is_hedged_account:
            body = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "side": side,
                "orderType": "market",
                "size": str(size)
            }
            logger.info(f"Orden en MODO HEDGE: side={side}, size={size} (sin posSide)")
        else:
            if not posSide:
                logger.error("En modo unilateral, posSide es obligatorio ('long' o 'short')")
                return None
            body = {
                "symbol": symbol,
                "productType": "USDT-FUTURES",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "side": side,
                "orderType": "market",
                "size": str(size),
                "posSide": posSide
            }
            logger.info(f"Orden en MODO UNILATERAL: side={side}, posSide={posSide}, size={size}")

        if stop_loss_formatted is not None:
            body["presetStopLossPrice"] = str(stop_loss_formatted)
            logger.info(f"SL integrado: presetStopLossPrice={stop_loss_formatted}")
        
        if take_profit_formatted is not None:
            body["presetStopSurplusPrice"] = str(take_profit_formatted)
            logger.info(f"TP integrado: presetStopSurplusPrice={take_profit_formatted}")

        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)

        headers = self._get_headers('POST', request_path, body_json)

        logger.info(f"Enviando orden con TP/SL integrados: {body}")

        response = requests.post(
            self.base_url + request_path,
            headers=headers,
            data=body_json,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                logger.info(f"Orden ejecutada ({side.upper()}) con TP/SL en modo {'HEDGE' if is_hedged_account else 'UNILATERAL'}")
                return data.get('data')
            else:
                if data.get('code') == '40774':
                    logger.error(f"Error 40774: La cuenta está en modo {'HEDGE' if not is_hedged_account else 'UNILATERAL'} pero la orden espera el otro modo")
                    logger.error(f"Solución: Verificar configuración de modo de posición en Bitget")
                if data.get('code') == '40034':
                    logger.error(f"Error 40034: {data.get('msg')}")

        logger.error(f"Error orden entrada: {response.text}")
        return None

    def obtener_precision_precio(self, symbol):
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                price_scale = symbol_info.get('priceScale', 4)
                logger.info(f"{symbol}: priceScale = {price_scale}")
                return price_scale
            else:
                logger.warning(f"No se pudo obtener info de {symbol}, usando 4 decimales por defecto")
                return 4
        except Exception as e:
            logger.error(f"Error obteniendo precisión de {symbol}: {e}")
            return 4

    def redondear_precio_precision(self, price, symbol):
        try:
            precision = self.obtener_precision_precio(symbol)
            precio_redondeado = round(float(price), precision)
            logger.info(f"{symbol}: {price} -> {precio_redondeado} (precisión: {precision} decimales)")
            return precio_redondeado
        except Exception as e:
            logger.error(f"Error redondeando precio para {symbol}: {e}")
            return float(price)

    def obtener_precision_adaptada(self, price, symbol):
        try:
            price = float(price)
            
            if price < 1:
                if price < 0.00001:
                    return 12
                elif price < 0.0001:
                    return 10
                elif price < 0.001:
                    return 8
                elif price < 0.01:
                    return 7
                elif price < 0.1:
                    return 6
                elif price < 1:
                    return 5
            else:
                return 4
                
        except Exception as e:
            logger.error(f"Error calculando precisión adaptativa: {e}")
            return 8

    def redondear_precio_manual(self, price, precision, symbol=None):
        try:
            price = float(price)
            if price == 0:
                return "0.0"
            
            if symbol:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    price_scale = symbol_info.get('priceScale', 4)
                    tick_size = 10 ** (-price_scale)
                    
                    precio_redondeado = round(price / tick_size) * tick_size
                    precio_formateado = f"{precio_redondeado:.{price_scale}f}"
                    
                    if float(precio_formateado) == 0.0 and price > 0:
                        nueva_scale = price_scale + 4
                        tick_size = 10 ** (-nueva_scale)
                        precio_redondeado = round(price / tick_size) * tick_size
                        precio_formateado = f"{precio_redondeado:.{nueva_scale}f}"
                    
                    logger.info(f"{symbol}: precio={price}, priceScale={price_scale}, tick={tick_size}, resultado={precio_formateado}")
                    return precio_formateado
            
            tick_size = 10 ** (-precision)
            precio_redondeado = round(price / tick_size) * tick_size
            precio_formateado = f"{precio_redondeado:.{precision}f}"
            
            if float(precio_formateado) == 0.0 and price > 0:
                nueva_precision = precision + 4
                return self.redondear_precio_manual(price, nueva_precision, symbol)
            
            return precio_formateado
        except Exception as e:
            logger.error(f"Error redondeando precio manualmente: {e}")
            return str(price)

    def redondear_a_price_step(self, price, symbol):
        try:
            precision = self.obtener_precision_precio(symbol)
            price_step = 10 ** (-precision)
            
            precio_redondeado = round(price / price_step) * price_step
            
            return float(f"{precio_redondeado:.{precision}f}")
        except Exception as e:
            logger.error(f"Error redondeando a priceStep para {symbol}: {e}")
            return float(f"{price:.4f}")

    def set_leverage(self, symbol, leverage, hold_side='long'):
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
            
            logger.info(f"Respuesta leverage BITGET: {response.status_code} - {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"Apalancamiento {leverage}x configurado en BITGET FUTUROS para {symbol}")
                    return True
                else:
                    logger.error(f"Error configurando leverage BITGET FUTUROS: {data.get('code')} - {data.get('msg')}")
            else:
                logger.error(f"Error HTTP configurando leverage BITGET FUTUROS: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"Error en set_leverage BITGET FUTUROS: {e}")
            return False

    def get_positions(self, symbol=None, product_type='USDT-FUTURES'):
        try:
            request_path = '/api/v2/mix/position/all-position'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            if symbol:
                params['symbol'] = symbol
            
            query_parts = []
            for key, value in params.items():
                query_parts.append(f"{key}={value}")
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
                    return data.get('data', [])
            
            if product_type == 'USDT-FUTURES':
                return self.get_positions(symbol, 'USDT-MIX')
            
            return []
        except Exception as e:
            logger.error(f"Error obteniendo posiciones BITGET FUTUROS: {e}")
            return []

    def obtener_reglas_simbolo(self, symbol):
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No se pudo obtener info de {symbol}, usando valores por defecto")
                return {
                    'size_scale': 0,
                    'quantity_scale': 0,
                    'min_trade_num': 1,
                    'size_multiplier': 1,
                    'delivery_mode': 0
                }
            
            reglas = {
                'size_scale': int(symbol_info.get('sizeScale', 0)),
                'quantity_scale': int(symbol_info.get('quantityScale', 0)),
                'min_trade_num': float(symbol_info.get('minTradeNum', 1)),
                'size_multiplier': float(symbol_info.get('sizeMultiplier', 1)),
                'delivery_mode': symbol_info.get('deliveryMode', 0)
            }
            
            logger.info(f"Reglas de {symbol}:")
            logger.info(f"  - sizeScale: {reglas['size_scale']}")
            logger.info(f"  - quantityScale: {reglas['quantity_scale']}")
            logger.info(f"  - minTradeNum: {reglas['min_trade_num']}")
            logger.info(f"  - sizeMultiplier: {reglas['size_multiplier']}")
            
            return reglas
            
        except Exception as e:
            logger.error(f"Error obteniendo reglas de {symbol}: {e}")
            return {
                'size_scale': 0,
                'quantity_scale': 0,
                'min_trade_num': 1,
                'size_multiplier': 1,
                'delivery_mode': 0
            }
    
    def ajustar_tamaño_orden(self, symbol, cantidad_contratos, reglas):
        try:
            size_scale = reglas['size_scale']
            quantity_scale = reglas['quantity_scale']
            min_trade_num = reglas['min_trade_num']
            size_multiplier = reglas['size_multiplier']
            
            escala_actual = quantity_scale if quantity_scale > 0 else size_scale
            
            if escala_actual == 0:
                cantidad_contratos = round(cantidad_contratos)
                logger.info(f"{symbol}: ajustado a entero = {cantidad_contratos}")
            elif escala_actual == 1:
                cantidad_contratos = round(cantidad_contratos, 1)
                logger.info(f"{symbol}: ajustado a 1 decimal = {cantidad_contratos}")
            elif escala_actual == 2:
                cantidad_contratos = round(cantidad_contratos, 2)
                logger.info(f"{symbol}: ajustado a 2 decimales = {cantidad_contratos}")
            else:
                cantidad_contratos = round(cantidad_contratos, escala_actual)
                logger.info(f"{symbol}: ajustado a {escala_actual} decimales = {cantidad_contratos}")
            
            if size_multiplier > 1:
                cantidad_contratos = round(cantidad_contratos / size_multiplier) * size_multiplier
                logger.info(f"{symbol}: aplicado multiplicador {size_multiplier}x = {cantidad_contratos}")
            
            if cantidad_contratos < min_trade_num:
                cantidad_contratos = min_trade_num
                logger.info(f"{symbol}: ajustado a mínimo = {min_trade_num}")
            
            if escala_actual == 0:
                if min_trade_num < 1 and min_trade_num > 0:
                    cantidad_contratos = max(1, int(round(cantidad_contratos)))
                    logger.info(f"{symbol}: caso especial - min decimal pero requiere entero = {cantidad_contratos}")
                else:
                    cantidad_contratos = int(round(cantidad_contratos))
                logger.info(f"Final {symbol}: {cantidad_contratos} (entero)")
            else:
                cantidad_contratos = round(cantidad_contratos, escala_actual)
                logger.info(f"Final {symbol}: {cantidad_contratos} ({escala_actual} decimales)")
            
            return cantidad_contratos
            
        except Exception as e:
            logger.error(f"Error ajustando tamaño para {symbol}: {e}")
            return int(round(cantidad_contratos))
    
    def obtener_saldo_cuenta(self):
        try:
            accounts = self.get_account_info()
            if accounts:
                for account in accounts:
                    if account.get('marginCoin') == 'USDT':
                        balance_usdt = float(account.get('available', 0))
                        logger.info(f"Saldo disponible USDT: ${balance_usdt:.2f}")
                        return balance_usdt
            logger.warning("No se pudo obtener saldo de la cuenta")
            return None
        except Exception as e:
            logger.error(f"Error obteniendo saldo de cuenta: {e}")
            return None

    def get_klines(self, symbol, interval='5m', limit=200):
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
                    return candles
                else:
                    params['productType'] = 'USDT-MIX'
                    response = requests.get(
                        self.base_url + request_path,
                        params=params,
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('code') == '00000':
                            candles = data.get('data', [])
                            return candles
            return None
        except Exception as e:
            logger.error(f"Error en get_klines BITGET FUTUROS: {e}")
            return None

# ---------------------------
# FUNCIONES DE OPERACIONES BITGET FUTUROS
# ---------------------------
def ejecutar_operacion_bitget(bitget_client, simbolo, tipo_operacion, capital_usd=None, leverage=20):
    logger.info(f"EJECUTANDO OPERACIÓN REAL EN BITGET FUTUROS")
    logger.info(f"Símbolo: {simbolo}")
    logger.info(f"Tipo: {tipo_operacion}")
    
    try:
        saldo_cuenta = bitget_client.obtener_saldo_cuenta()
        if not saldo_cuenta or saldo_cuenta < 10:
            logger.error(f"Saldo insuficiente o no disponible: ${saldo_cuenta if saldo_cuenta else 0:.2f}")
            return None
        
        margin_usdt_objetivo = round(saldo_cuenta * 0.03, 2)
        
        logger.info(f"Saldo actual cuenta: ${saldo_cuenta:.2f}")
        logger.info(f"3% del saldo actual (MARGIN USDT objetivo): ${margin_usdt_objetivo:.2f}")
        logger.info(f"Apalancamiento: {leverage}x")
        logger.info(f"Esta operación usará hasta 3% del saldo actual: ${margin_usdt_objetivo:.2f}")
        logger.info(f"Saldo restante después de esta operación: ${saldo_cuenta - margin_usdt_objetivo:.2f}")
        
        symbol_info = bitget_client.get_symbol_info(simbolo)
        if not symbol_info:
            logger.error(f"No se pudo obtener info de {simbolo} en BITGET FUTUROS")
            return None
        
        hold_side = 'long' if tipo_operacion == 'LONG' else 'short'
        
        logger.info(f"Configurando apalancamiento {leverage}x para {simbolo} ({hold_side})")
        leverage_ok = bitget_client.set_leverage(simbolo, leverage, hold_side)
        
        if not leverage_ok:
            logger.warning("No se pudo configurar apalancamiento, continuando...")
        else:
            logger.info("Apalancamiento configurado exitosamente")
            
        time.sleep(1)
        
        klines = bitget_client.get_klines(simbolo, '1m', 1)
        if not klines or len(klines) == 0:
            logger.error(f"No se pudo obtener precio de {simbolo} en BITGET FUTUROS")
            return None
        
        klines.reverse()
        precio_actual = float(klines[0][4])
        
        logger.info(f"Precio actual: {precio_actual:.8f}")
        
        reglas = bitget_client.obtener_reglas_simbolo(simbolo)
        
        valor_nocional_objetivo = margin_usdt_objetivo * leverage
        logger.info(f"MARGIN USDT (3% del saldo): ${margin_usdt_objetivo:.2f}")
        logger.info(f"Valor nocional objetivo (MARGIN x LEVERAGE): ${valor_nocional_objetivo:.2f}")
        logger.info(f"Objetivo: Todas las operaciones tendrán hasta ${margin_usdt_objetivo:.2f} como MARGIN USDT")
        
        cantidad_contratos = valor_nocional_objetivo / precio_actual
        
        cantidad_contratos = bitget_client.ajustar_tamaño_orden(simbolo, cantidad_contratos, reglas)
        
        valor_nocional_real = cantidad_contratos * precio_actual
        margin_real = valor_nocional_real / leverage
        
        logger.info(f"Cantidad ajustada: {cantidad_contratos} contratos")
        logger.info(f"Valor nocional real: ${valor_nocional_real:.2f}")
        logger.info(f"MARGIN USDT real: ${margin_real:.2f}")
        logger.info(f"Size scale: {reglas['size_scale']}")
        logger.info(f"Quantity scale: {reglas['quantity_scale']}")
        logger.info(f"Min trade num: {reglas['min_trade_num']}")
        logger.info(f"Size multiplier: {reglas['size_multiplier']}")
        
        max_margin_permitido = saldo_cuenta * 0.95
        if margin_real > max_margin_permitido:
            logger.warning(f"MARGIN USDT real (${margin_real:.2f}) excede el máximo permitido (${max_margin_permitido:.2f})")
            logger.warning(f"Calculando tamaño máximo permitido según saldo disponible...")
            
            max_valor_nocional = max_margin_permitido * leverage
            cantidad_maxima = max_valor_nocional / precio_actual
            
            cantidad_maxima = bitget_client.ajustar_tamaño_orden(simbolo, cantidad_maxima, reglas)
            
            if cantidad_maxima < cantidad_contratos:
                cantidad_contratos = cantidad_maxima
                valor_nocional_real = cantidad_contratos * precio_actual
                margin_real = valor_nocional_real / leverage
                
                logger.info(f"Cantidad reducida al máximo permitido: {cantidad_contratos} contratos")
                logger.info(f"Nuevo valor nocional real: ${valor_nocional_real:.2f}")
                logger.info(f"Nuevo MARGIN USDT real: ${margin_real:.2f}")
        
        if tipo_operacion == "LONG":
            sl = precio_actual * (1 - 0.015)
            tp = precio_actual * (1 + 0.018)
        else:
            sl = precio_actual * (1 + 0.015)
            tp = precio_actual * (1 - 0.018)
        
        sl = bitget_client.redondear_a_price_step(sl, simbolo)
        tp = bitget_client.redondear_a_price_step(tp, simbolo)
        
        logger.info(f"Precio actual: {precio_actual:.8f}")
        logger.info(f"Stop Loss: {sl:.8f}")
        logger.info(f"Take Profit: {tp:.8f}")
        
        risk_reward = abs(tp - precio_actual) / abs(precio_actual - sl)
        logger.info(f"Risk/Reward ratio: {risk_reward:.2f}")
        
        side = 'buy' if tipo_operacion == 'LONG' else 'sell'
        
        is_hedged = False
        
        orden_entrada = bitget_client.place_order(
            symbol=simbolo,
            side=side,
            size=cantidad_contratos,
            posSide=hold_side,
            is_hedged_account=is_hedged,
            stop_loss_price=sl,
            take_profit_price=tp
        )
        
        if orden_entrada:
            orden_sl = bitget_client.place_tpsl_order(simbolo, hold_side, sl, 'stop_loss', stop_loss_price=sl)
            orden_tp = bitget_client.place_tpsl_order(simbolo, hold_side, tp, 'take_profit', take_profit_price=tp)
            
            logger.info(f"OPERACIÓN EXITOSA EN BITGET FUTUROS")
            logger.info(f"Símbolo: {simbolo}")
            logger.info(f"Tipo: {tipo_operacion}")
            logger.info(f"Capital USDT (MARGIN): ${margin_usdt_objetivo:.2f}")
            logger.info(f"Saldo después: ${saldo_cuenta - margin_usdt_objetivo:.2f}")
            logger.info(f"Contratos: {cantidad_contratos}")
            logger.info(f"Precio entrada: {precio_actual:.8f}")
            logger.info(f"Stop Loss: {sl:.8f}")
            logger.info(f"Take Profit: {tp:.8f}")
            logger.info(f"ID Orden Entrada: {orden_entrada.get('orderId', 'N/A')}")
            
            if orden_sl:
                logger.info(f"ID Orden SL: {orden_sl.get('orderId', 'N/A')}")
            if orden_tp:
                logger.info(f"ID Orden TP: {orden_tp.get('orderId', 'N/A')}")
            
            return {
                'orden_entrada': orden_entrada,
                'orden_sl': orden_sl,
                'orden_tp': orden_tp,
                'precio_entrada': precio_actual,
                'stop_loss': sl,
                'take_profit': tp,
                'capital_usado': margin_real,
                'leverage': leverage,
                'saldo_cuenta': saldo_cuenta
            }
        else:
            logger.error(f"ERROR: No se pudo ejecutar la orden de entrada para {simbolo}")
            return None
            
    except Exception as e:
        logger.error(f"Error ejecutando operación en BITGET FUTUROS: {e}")
        return None

# ---------------------------
# FUNCIONES DE ANÁLISIS DE MERCADO
# ---------------------------
def obtener_datos_binance(simbolo, intervalo='5m', limite=200):
    try:
        url = "https://api.binance.com/api/v3/klines"
        parametros = {
            'symbol': simbolo,
            'interval': intervalo,
            'limit': limite
        }
        respuesta = requests.get(url, params=parametros, timeout=10)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            cierres = [float(d[4]) for d in datos]
            maximos = [float(d[2]) for d in datos]
            minimos = [float(d[3]) for d in datos]
            precios = [float(d[4]) for d in datos]
            return {
                'cierres': cierres,
                'maximos': maximos,
                'minimos': minimos,
                'precios': precios,
                'precio_actual': float(datos[-1][4])
            }
        else:
            return None
    except Exception as e:
        print(f"Error obteniendo datos de Binance: {e}")
        return None

def obtener_datos_bitget(simbolo, intervalo='5m', limite=200):
    try:
        url = "https://api.bitget.com/api/v2/spot/market/candles"
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m',
            '15m': '15m', '30m': '30m', '1h': '1H',
            '4h': '4H', '1d': '1D'
        }
        bitget_interval = interval_map.get(intervalo, '5m')
        params = {
            'symbol': simbolo,
            'granularity': bitget_interval,
            'limit': limite
        }
        respuesta = requests.get(url, params=params, timeout=10)
        if respuesta.status_code == 200:
            data = respuesta.json()
            if data.get('code') == '00000':
                candles = data.get('data', [])
                cierres = [float(c[4]) for c in candles]
                maximos = [float(c[2]) for c in candles]
                minimos = [float(c[3]) for c in candles]
                return {
                    'cierres': cierres,
                    'maximos': maximos,
                    'minimos': minimos,
                    'precios': cierres,
                    'precio_actual': cierres[-1] if cierres else 0
                }
        return None
    except Exception as e:
        print(f"Error obteniendo datos de Bitget: {e}")
        return None

def calcular_soporte_resistencia(datos, num_velas=80):
    cierres = datos['cierres'][-num_velas:]
    maximos = datos['maximos'][-num_velas:]
    minimos = datos['minimos'][-num_velas:]
    
    if len(cierres) < 10:
        return None
    
    resistencia = max(maximos)
    soporte = min(minimos)
    
    return {
        'resistencia': resistencia,
        'soporte': soporte,
        'ancho_canal': resistencia - soporte,
        'ancho_canal_porcentual': ((resistencia - soporte) / soporte) * 100 if soporte > 0 else 0
    }

def calcular_lineas_tendencia(datos, num_velas=80):
    cierres = datos['cierres'][-num_velas:]
    precios = datos['precios'][-num_velas:]
    
    if len(cierres) < 10:
        return None
    
    x = np.arange(len(cierres))
    
    # Calcular línea de tendencia central (regresión lineal)
    pendiente, intercepto = np.polyfit(x, cierres, 1)
    
    # Calcular rango del precio para el canal
    precio_max = np.max(cierres)
    precio_min = np.min(cierres)
    rango = precio_max - precio_min
    
    # La línea de resistencia pasa por el precio máximo
    # La línea de soporte pasa por el precio mínimo
    # Ambas líneas tienen la MISMA pendiente que la tendencia central
    
    # Calcular interceptos para líneas paralelas
    # Resistencia: pasa por precio_max
    intercepto_resistencia = precio_max - pendiente * (len(cierres) - 1)
    
    # Soporte: pasa por precio_min
    intercepto_soporte = precio_min - pendiente * (len(cierres) - 1)
    
    # Calcular valores actuales (en el punto final)
    x_final = len(cierres) - 1
    resistencia_actual = pendiente * x_final + intercepto_resistencia
    soporte_actual = pendiente * x_final + intercepto_soporte
    
    # Determinar dirección de la tendencia
    if abs(pendiente) < 1e-10:
        direccion = "RANGO"
    elif pendiente > 0:
        direccion = "ALCISTA"
    else:
        direccion = "BAJISTA"
    
    return {
        'pendiente': pendiente,
        'intercepto': intercepto,
        'intercepto_resistencia': intercepto_resistencia,
        'intercepto_soporte': intercepto_soporte,
        'resistencia': precio_max,
        'soporte': precio_min,
        'resistencia_actual': resistencia_actual,
        'soporte_actual': soporte_actual,
        'ancho_canal': rango,
        'ancho_canal_porcentual': (rango / precio_min) * 100 if precio_min > 0 else 0,
        'pendiente_resistencia': pendiente,
        'pendiente_soporte': pendiente,
        'direccion': direccion,
        'resistencia_values': pendiente * x + intercepto_resistencia,
        'soporte_values': pendiente * x + intercepto_soporte
    }

def calcular_angulo_tendencia(datos, num_velas=80):
    cierres = datos['cierres'][-num_velas:]
    
    if len(cierres) < 10:
        return None
    
    x = np.arange(len(cierres))
    pendiente, intercepto = np.polyfit(x, cierres, 1)
    
    diferencia_precios = np.max(cierres) - np.min(cierres)
    diferencia_x = len(cierres)
    
    if diferencia_x == 0:
        return None
    
    pendiente_normalizada = diferencia_precios / diferencia_x if diferencia_x > 0 else 0
    
    angulo_rad = math.atan(pendiente_normalizada * diferencia_x / (cierres[-1] if cierres[-1] > 0 else 1))
    angulo_deg = math.degrees(angulo_rad)
    
    return angulo_deg

def calcular_coeficiente_pearson(datos, num_velas=80):
    cierres = datos['cierres'][-num_velas:]
    
    if len(cierres) < 2:
        return 0
    
    x = np.arange(len(cierres))
    y = np.array(cierres)
    
    media_x = np.mean(x)
    media_y = np.mean(y)
    
    numerador = np.sum((x - media_x) * (y - media_y))
    denominador = np.sqrt(np.sum((x - media_x)**2) * np.sum((y - media_y)**2))
    
    if denominador == 0:
        return 0
    
    return numerador / denominador

def detectar_breakout(info_canal, precio_actual, datos=None):
    """
    Detecta si el precio ha hecho breakout del canal.
    
    Un breakout ALCISTA ocurre cuando:
    - La tendencia es ALCISTA (pendiente > 0)
    - El precio rompe por encima de la resistencia dinámica
    
    Un breakout BAJISTA ocurre cuando:
    - La tendencia es BAJISTA (pendiente < 0)  
    - El precio rompe por debajo del soporte dinámico
    
    Returns: {
        'tipo': 'ALCISTA' | 'BAJISTA',
        'precio_breakout': precio al que rompió,
        'nivel_ruptura': nivel técnico que rompió,
        ' direccion_ruptura': 'ARRIBA' | 'ABAJO'
    }
    """
    if not info_canal:
        return None
    
    resistencia = info_canal.get('resistencia_actual', info_canal.get('resistencia', precio_actual))
    soporte = info_canal.get('soporte_actual', info_canal.get('soporte', precio_actual))
    pendiente = info_canal.get('pendiente', 0)
    
    # Determinar dirección de la tendencia
    if abs(pendiente) < 1e-10:
        # Sin tendencia clara, usar lógica tradicional
        if precio_actual >= resistencia:
            return {
                'tipo': 'ALCISTA',
                'precio_breakout': precio_actual,
                'nivel_ruptura': resistencia,
                'direccion_ruptura': 'ARRIBA'
            }
        elif precio_actual <= soporte:
            return {
                'tipo': 'BAJISTA',
                'precio_breakout': precio_actual,
                'nivel_ruptura': soporte,
                'direccion_ruptura': 'ABAJO'
            }
        return None
    
    # Tendencia ALCISTA: solo contar breakout ALCISTA (rompiendo resistencia)
    if pendiente > 0:
        if precio_actual >= resistencia:
            return {
                'tipo': 'ALCISTA',
                'precio_breakout': precio_actual,
                'nivel_ruptura': resistencia,
                'direccion_ruptura': 'ARRIBA'
            }
    
    # Tendencia BAJISTA: solo contar breakout BAJISTA (rompiendo soporte)
    elif pendiente < 0:
        if precio_actual <= soporte:
            return {
                'tipo': 'BAJISTA',
                'precio_breakout': precio_actual,
                'nivel_ruptura': soporte,
                'direccion_ruptura': 'ABAJO'
            }
    
    return None

def calcular_estado_stochastic(datos, period=14, k_period=3, d_period=3):
    cierres = datos['cierres']
    maximos = datos['maximos']
    minimos = datos['minimos']
    
    if len(cierres) < period:
        return 50, 50, "NEUTRAL"
    
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
            
            if k_final < 20:
                estado = "SOBREVENTA"
            elif k_final > 80:
                estado = "SOBRECOMPRA"
            else:
                estado = "NEUTRAL"
            
            return k_final, d, estado
    
    return 50, 50, "NEUTRAL"

# ---------------------------
# TRADING BOT PRINCIPAL
# ---------------------------
class TradingBot:
    def __init__(self, config=None):
        self.config = config or {}
        self.config_optima_por_simbolo = {}
        self.operaciones_activas = {}
        self.senales_enviadas = set()
        self.breakouts_detectados = {}
        self.esperando_reentry = {}
        self.breakout_history = {}
        self.total_operaciones = 0
        self.operaciones_desde_optimizacion = 0
        self.ultima_optimizacion = datetime.now()
        self.ultimo_grafico_enviado = {}
        self.ultimo_mensaje_estado = {}
        
        self.bitget_client = None
        self.ejecutar_operaciones_automaticas = self.config.get('ejecutar_operaciones_automaticas', False)
        self.leverage_por_defecto = self.config.get('leverage_por_defecto', 20)
        self.archivo_log = self.config.get('log_path', 'operaciones_log.csv')
        self.archivo_estado = self.config.get('estado_file', 'estado_bot.json')
        
        self.inicializar_log()
        self.cargar_estado()
        
        api_key = self.config.get('bitget_api_key')
        api_secret = self.config.get('bitget_api_secret')
        passphrase = self.config.get('bitget_passphrase')
        
        if api_key and api_secret and passphrase:
            self.bitget_client = BitgetClient(api_key, api_secret, passphrase, self)
            credenciales_ok = self.bitget_client.verificar_credenciales()
            if credenciales_ok:
                print("BITGET FUTUROS: API conectada y verificada")
            else:
                print("BITGET FUTUROS: Error en credenciales")
        else:
            print("BITGET FUTUROS: No configurado")
        
        self.config_optima_por_simbolo = self.cargar_config_optima()

    def obtener_datos_mercado_config(self, simbolo, timeframe, num_velas):
        if self.bitget_client:
            klines = self.bitget_client.get_klines(simbolo, timeframe, num_velas)
            if klines:
                cierres = [float(k[4]) for k in klines]
                maximos = [float(k[2]) for k in klines]
                minimos = [float(k[3]) for k in klines]
                return {
                    'cierres': cierres,
                    'maximos': maximos,
                    'minimos': minimos,
                    'precios': cierres,
                    'precio_actual': cierres[-1] if cierres else 0
                }
        
        datos_binance = obtener_datos_binance(simbolo, timeframe, num_velas)
        if datos_binance:
            return datos_binance
        
        datos_bitget = obtener_datos_bitget(simbolo, timeframe, num_velas)
        if datos_bitget:
            return datos_bitget
        
        return None

    def cargar_config_optima(self):
        try:
            if os.path.exists("mejores_parametros.json"):
                with open("mejores_parametros.json", 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error cargando configuración óptima: {e}")
        return {}

    def guardar_estado(self):
        estado = {
            'operaciones_activas': self.operaciones_activas,
            'config_optima_por_simbolo': self.config_optima_por_simbolo,
            'breakouts_detectados': {k: {'tipo': v['tipo'], 'timestamp': v['timestamp'].isoformat() if isinstance(v['timestamp'], datetime) else v['timestamp']} for k, v in self.breakouts_detectados.items()},
            'esperando_reentry': self.esperando_reentry,
            'senales_enviadas': list(self.senales_enviadas),
            'total_operaciones': self.total_operaciones,
            'operaciones_desde_optimizacion': self.operaciones_desde_optimizacion,
            'ultima_optimizacion': self.ultima_optimizacion.isoformat() if isinstance(self.ultima_optimizacion, datetime) else self.ultima_optimizacion
        }
        try:
            with open(self.archivo_estado, 'w') as f:
                json.dump(estado, f, indent=2)
        except Exception as e:
            print(f"Error guardando estado: {e}")

    def cargar_estado(self):
        try:
            if os.path.exists(self.archivo_estado):
                with open(self.archivo_estado, 'r') as f:
                    estado = json.load(f)
                    self.operaciones_activas = estado.get('operaciones_activas', {})
                    self.breakouts_detectados = {k: {'tipo': v['tipo'], 'timestamp': datetime.fromisoformat(v['timestamp']) if isinstance(v['timestamp'], str) else v['timestamp']} for k, v in estado.get('breakouts_detectados', {}).items()}
                    self.esperando_reentry = estado.get('esperando_reentry', {})
                    self.senales_enviadas = set(estado.get('senales_enviadas', []))
                    self.total_operaciones = estado.get('total_operaciones', 0)
                    self.operaciones_desde_optimizacion = estado.get('operaciones_desde_optimizacion', 0)
                    self.ultima_optimizacion = datetime.fromisoformat(estado.get('ultima_optimizacion', datetime.now().isoformat()))
                    print(f"Estado cargado: {len(self.operaciones_activas)} operaciones activas")
        except Exception as e:
            print(f"Error cargando estado: {e}")

    def sincronizar_con_bitget(self):
        if not self.bitget_client:
            return
        
        posiciones = self.bitget_client.get_positions()
        if posiciones is None:
            return
        
        simbolos_con_posicion = set()
        for posicion in posiciones:
            symbol = posicion.get('symbol')
            if symbol:
                simbolos_con_posicion.add(symbol)
        
        simbolos_a_eliminar = []
        for simbolo, operacion in self.operaciones_activas.items():
            if simbolo not in simbolos_con_posicion:
                if not operacion.get('operacion_manual_usuario', False):
                    if operacion.get('operacion_ejecutada', False):
                        simbolos_a_eliminar.append(simbolo)
        
        for simbolo in simbolos_a_eliminar:
            del self.operaciones_activas[simbolo]
            if simbolo in self.senales_enviadas:
                self.senales_enviadas.remove(simbolo)
            print(f"Símbolo {simbolo} removido de operaciones_activas (no encontrado en exchange)")
        
        self.guardar_estado()

    def verificar_y_recolocar_tp_sl(self):
        if not self.bitget_client:
            return
        
        for simbolo, operacion in list(self.operaciones_activas.items()):
            if not operacion.get('operacion_ejecutada', False):
                continue
            
            orden_id_tp = operacion.get('order_id_tp')
            orden_id_sl = operacion.get('order_id_sl')
            
            if orden_id_tp is None and orden_id_sl is None:
                continue
            
            print(f"Verificando TP/SL para {simbolo}...")
            
            orden_tp_existe = False
            orden_sl_existe = False
            
            posiciones = self.bitget_client.get_positions(simbolo)
            if posiciones:
                for pos in posiciones:
                    position_activate = pos.get('positionActivate', '')
                    if position_activate == 'tp' or position_activate == 'sl':
                        if orden_id_tp and position_activate == 'tp':
                            orden_tp_existe = True
                        if orden_id_sl and position_activate == 'sl':
                            orden_sl_existe = True
            
            if not orden_tp_existe and orden_id_tp:
                print(f"TP no encontrado para {simbolo}, recolocando...")
                try:
                    resultado = self.bitget_client.place_tpsl_order(
                        symbol=simbolo,
                        hold_side='long' if operacion['tipo'] == 'LONG' else 'short',
                        trigger_price=operacion['take_profit'],
                        order_type='take_profit',
                        take_profit_price=operacion['take_profit']
                    )
                    if resultado:
                        print(f"TP recolocado exitosamente para {simbolo}")
                except Exception as e:
                    print(f"Error recolocando TP para {simbolo}: {e}")
            
            if not orden_sl_existe and orden_id_sl:
                print(f"SL no encontrado para {simbolo}, recolocando...")
                try:
                    resultado = self.bitget_client.place_tpsl_order(
                        symbol=simbolo,
                        hold_side='long' if operacion['tipo'] == 'LONG' else 'short',
                        trigger_price=operacion['stop_loss'],
                        order_type='stop_loss',
                        stop_loss_price=operacion['stop_loss']
                    )
                    if resultado:
                        print(f"SL recolocado exitosamente para {simbolo}")
                except Exception as e:
                    print(f"Error recolocando SL para {simbolo}: {e}")

    def calcular_nivel_entrada(self, info_canal, tipo_operacion, margen=0.001):
        if not info_canal:
            return None
        
        resistencia = info_canal.get('resistencia_actual', info_canal.get('resistencia', 0))
        soporte = info_canal.get('soporte_actual', info_canal.get('soporte', 0))
        ancho_canal = resistencia - soporte
        
        if tipo_operacion == "LONG":
            entrada = resistencia + (ancho_canal * margen)
        else:
            entrada = soporte - (ancho_canal * margen)
        
        return entrada

    def calcular_stop_loss_take_profit(self, entrada, tipo_operacion, ancho_canal):
        if tipo_operacion == "LONG":
            sl = entrada - (ancho_canal * 0.5)
            tp = entrada + (ancho_canal * 0.6)
        else:
            sl = entrada + (ancho_canal * 0.5)
            tp = entrada - (ancho_canal * 0.6)
        
        return sl, tp

    def validar_senal(self, info_canal, stoch_k, stoch_d, angulo, pearson, r2):
        try:
            min_channel_width_percent = self.config.get('min_channel_width_percent', 4)
            if info_canal['ancho_canal_porcentual'] < min_channel_width_percent:
                return False, f"Canal estrecho ({info_canal['ancho_canal_porcentual']:.1f}% < {min_channel_width_percent}%)"
            
            entrada = self.calcular_nivel_entrada(info_canal, "LONG")
            if not entrada:
                return False, "No se pudo calcular entrada"
            
            sl, tp = self.calcular_stop_loss_take_profit(entrada, "LONG", info_canal['ancho_canal'])
            if sl and tp:
                riesgo = abs(entrada - sl)
                beneficio = abs(tp - entrada)
                ratio_rr = beneficio / riesgo if riesgo > 0 else 0
                min_rr_ratio = self.config.get('min_rr_ratio', 1.2)
                if ratio_rr < min_rr_ratio:
                    return False, f"Risk/Reward bajo ({ratio_rr:.2f} < {min_rr_ratio})"
            
            if abs(angulo) < self.config.get('trend_threshold_degrees', 16):
                return False, f"Ángulo débil ({angulo:.1f}° < {self.config.get('trend_threshold_degrees', 16)}°)"
            
            fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo)
            if nivel_fuerza < 3:
                return False, f"Fuerza débil ({fuerza_texto})"
            
            if abs(pearson) < 0.4:
                return False, f"Pearson bajo ({pearson:.3f} < 0.4)"
            
            if r2 < 0.4:
                return False, f"R² bajo ({r2:.3f} < 0.4)"
            
            return True, "Válida"
        except Exception as e:
            return False, f"Error validando: {e}"

    def escanear_mercado(self):
        symbols = self.config.get('symbols', [])
        timeframes = self.config.get('timeframes', [])
        velas_options = self.config.get('velas_options', [80, 100, 120, 150, 200])
        
        if not symbols:
            print("No hay símbolos configurados")
            return 0
        
        senales_encontradas = 0
        angulo_senales_encontradas = 0
        
        print(f"\nESCANEANDO MERCADO: {len(symbols)} símbolos en {len(timeframes)} timeframes")
        print(f"Configuración: ángulo>={self.config.get('trend_threshold_degrees', 16)}°, fuerza>=MODERADA, Pearson>=0.4, R²>=0.4, canal>={self.config.get('min_channel_width_percent', 4)}%")
        
        # Pre-obtener datos base para Stochastic una vez por símbolo
        datos_base = {}
        for sim in symbols:
            try:
                dm = self.obtener_datos_mercado_config(sim, '5m', 80)
                if dm:
                    datos_base[sim] = dm
            except:
                pass
        
        for simbolo in symbols:
            if simbolo in self.senales_enviadas:
                continue
            
            try:
                datos_mercado = datos_base.get(simbolo)
                if not datos_mercado:
                    continue
                
                precio_actual = datos_mercado['precio_actual']
                stoch_k, stoch_d, stoch_estado = calcular_estado_stochastic(datos_mercado, 14, 3, 3)
                
                for timeframe in timeframes:
                    for num_velas in velas_options:
                        datos = self.obtener_datos_mercado_config(simbolo, timeframe, num_velas)
                        if not datos:
                            continue
                        
                        precio_actual = datos['precio_actual']
                        
                        info_canal = calcular_lineas_tendencia(datos, num_velas)
                        if not info_canal:
                            continue
                        
                        # calcular_soporte_resistencia ya no es necesario ya que calcular_lineas_tendencia incluye todos los valores
                        
                        ancho_canal_pct = info_canal.get('ancho_canal_porcentual', 0)
                        min_ancho = self.config.get('min_channel_width_percent', 4)
                        
                        angulo = calcular_angulo_tendencia(datos, num_velas)
                        if angulo is None:
                            continue
                        
                        pearson = calcular_coeficiente_pearson(datos, num_velas)
                        
                        cierres = datos['cierres']
                        x = np.arange(len(cierres))
                        pendiente, intercepto = np.polyfit(x, cierres, 1)
                        r2 = self.calcular_r2(cierres, x, pendiente, intercepto)
                        
                        info_canal['angulo_tendencia'] = angulo
                        info_canal['coeficiente_pearson'] = pearson
                        info_canal['r2_score'] = r2
                        info_canal['stoch_k'] = stoch_k
                        info_canal['stoch_d'] = stoch_d
                        
                        fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo)
                        info_canal['fuerza_texto'] = fuerza_texto
                        info_canal['nivel_fuerza'] = nivel_fuerza
                        direccion = self.determinar_direccion_tendencia(angulo)
                        info_canal['direccion'] = direccion
                        
                        # Determinar posición del precio
                        resistencia = info_canal.get('resistencia_actual', info_canal.get('resistencia', precio_actual))
                        soporte = info_canal.get('soporte_actual', info_canal.get('soporte', precio_actual))
                        
                        if precio_actual >= resistencia:
                            posicion = 'FUERA'
                        elif precio_actual <= soporte:
                            posicion = 'FUERA'
                        else:
                            posicion = 'DENTRO'
                        
                        # LOGGING DETALLADO: mostrar todos los símbolos con canal válido
                        if abs(angulo) >= self.config.get('trend_threshold_degrees', 16) / 2:
                            linea_log = log_detalle_simbolo(
                                simbolo, timeframe, num_velas, direccion, angulo,
                                fuerza_texto, ancho_canal_pct, stoch_k, stoch_d,
                                stoch_estado, precio_actual, resistencia, soporte, posicion
                            )
                            print(linea_log)
                        
                        # Verificar ancho mínimo del canal
                        if ancho_canal_pct < min_ancho:
                            continue
                        
                        valida, razon = self.validar_senal(info_canal, stoch_k, stoch_d, angulo, pearson, r2)
                        
                        if not valida:
                            if "Ángulo débil" in razon or "Fuerza débil" in razon:
                                angulo_senales_encontradas += 1
                            continue
                        
                        entrada = self.calcular_nivel_entrada(info_canal, "LONG")
                        sl, tp = self.calcular_stop_loss_take_profit(entrada, "LONG", info_canal['ancho_canal'])
                        
                        config_optima = {'timeframe': timeframe, 'num_velas': num_velas}
                        self.config_optima_por_simbolo[simbolo] = config_optima
                        
                        tipo_operacion = "LONG" if angulo > 0 else "SHORT"
                        
                        # DETECTAR BREAKOUT
                        breakout_info = detectar_breakout(info_canal, precio_actual)
                        if breakout_info:
                            breakout_info['timestamp'] = datetime.now()
                            
                            # Logging detallado del breakout
                            linea_breakout = log_detalle_breakout(
                                simbolo, breakout_info['tipo'], 
                                breakout_info['precio_breakout'], 
                                breakout_info['nivel_ruptura'],
                                breakout_info.get('direccion_ruptura', 'ARRIBA')
                            )
                            print(f"   {linea_breakout}")
                            
                            # Verificar cooldown de breakout
                            if simbolo in self.breakout_history:
                                tiempo_desde_ultimo = (datetime.now() - self.breakout_history[simbolo]).total_seconds() / 60
                                if tiempo_desde_ultimo < 120:
                                    print(f"      ⏭️ Saltando {simbolo} (último breakout hace {tiempo_desde_ultimo:.1f} min)")
                                    continue
                            
                            # ENVIAR NOTIFICACIÓN DE BREAKOUT A TELEGRAM
                            token = self.config.get('telegram_token')
                            chat_ids = self.config.get('telegram_chat_ids', [])
                            if token and chat_ids:
                                try:
                                    # Generar gráfico de breakout
                                    print(f"      📊 Generando gráfico de breakout para {simbolo}...")
                                    buf = self.generar_grafico_profesional(simbolo, info_canal, datos_mercado, 
                                                                          None, None, None, None)
                                    if buf:
                                        print(f"      📤 Enviando alerta de breakout por Telegram...")
                                        self.enviar_grafico_telegram(buf, token, chat_ids)
                                        time.sleep(1)
                                    
                                    # Mensaje de texto detallado
                                    direccion_emoji = "⬆️" if breakout_info.get('direccion_ruptura') == 'ARRIBA' else "⬇️"
                                    msg_breakout = f"🚀 <b>BREAKOUT DETECTADO</b> {direccion_emoji}\n\n"
                                    msg_breakout += f"Símbolo: <b>{simbolo}</b>\n"
                                    msg_breakout += f"Tipo: <b>{breakout_info['tipo']}</b>\n"
                                    msg_breakout += f"Timeframe: {timeframe}\n"
                                    msg_breakout += f"Precio Actual: {precio_actual:.8f}\n"
                                    msg_breakout += f"Nivel Ruptura: {breakout_info['nivel_ruptura']:.8f}\n"
                                    msg_breakout += f"Tendencia: {direccion} ({angulo:.1f}°)\n"
                                    msg_breakout += f"Fuerza: {fuerza_texto}\n"
                                    msg_breakout += f"Ancho Canal: {ancho_canal_pct:.1f}%\n"
                                    msg_breakout += f"Stochastic: {stoch_k:.1f}/{stoch_d:.1f} ({stoch_estado})\n"
                                    msg_breakout += f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    msg_breakout += f"\n⏳ <i>Esperando reentry para confirmar operación...</i>"
                                    
                                    self._enviar_telegram_simple(msg_breakout, token, chat_ids)
                                    print(f"      ✅ Alerta de breakout enviada para {simbolo}")
                                except Exception as e:
                                    print(f"      ⚠️ Error enviando notificación de breakout: {e}")
                            
                            print(f"   🎯 {simbolo} - Breakout {breakout_info['tipo']} registrado, esperando reingreso...")
                            
                            self.breakouts_detectados[simbolo] = {
                                'tipo': breakout_info['tipo'],
                                'timestamp': datetime.now(),
                                'precio_breakout': breakout_info['precio_breakout'],
                                'nivel_ruptura': breakout_info['nivel_ruptura']
                            }
                            self.breakout_history[simbolo] = datetime.now()
                            continue
                        
                        if simbolo in self.breakouts_detectados:
                            tiempo_desde_breakout = (datetime.now() - self.breakouts_detectados[simbolo]['timestamp']).total_seconds() / 60
                            periodo_reentry = 15
                            timeout_reentry = 60
                            
                            tipo_breakout = self.breakouts_detectados[simbolo]['tipo']
                            nivel_ruptura = self.breakouts_detectados[simbolo]['nivel_ruptura']
                            
                            if tiempo_desde_breakout > timeout_reentry:
                                print(f"   ⏰ {simbolo} - Timeout reentry ({tiempo_desde_breakout:.1f} min > {timeout_reentry} min)")
                                del self.breakouts_detectados[simbolo]
                                if simbolo in self.esperando_reentry:
                                    del self.esperando_reentry[simbolo]
                                continue
                            
                            if tiempo_desde_breakout >= periodo_reentry:
                                print(f"   🔄 {simbolo} - PERIODO REENTRY ({tiempo_desde_breakout:.1f} min >= {periodo_reentry} min)")
                                
                                precio_retorno = nivel_ruptura
                                
                                if tipo_breakout == "ALCISTA" and precio_actual <= precio_retorno:
                                    print(f"   ✅ {simbolo} - REENTRY LONG confirmado (precio {precio_actual:.8f} <= {precio_retorno:.8f})")
                                    
                                    if simbolo in self.esperando_reentry:
                                        tiempo_espera = (datetime.now() - self.esperando_reentry[simbolo]['timestamp']).total_seconds() / 60
                                        if tiempo_espera < 10:
                                            print(f"      ⏭️ Saltando {simbolo} (esperando {tiempo_espera:.1f} min)")
                                            continue
                                    
                                    es_valido_stoch = tipo_operacion == "LONG" and stoch_k > 20
                                    
                                    if es_valido_stoch:
                                        self.esperando_reentry[simbolo] = {
                                            'tipo': 'LONG',
                                            'timestamp': datetime.now(),
                                            'breakout_info': self.breakouts_detectados[simbolo]
                                        }
                                        
                                        self.generar_senal_operacion(
                                            simbolo, "LONG", entrada, tp, sl,
                                            info_canal, datos_mercado, config_optima,
                                            self.breakouts_detectados[simbolo]
                                        )
                                        senales_encontradas += 1
                                        self.breakout_history[simbolo] = datetime.now()
                                        del self.breakouts_detectados[simbolo]
                                    else:
                                        print(f"   ❌ {simbolo} - Stochastic no confirma ({stoch_k:.1f} <= 20)")
                                
                                elif tipo_breakout == "BAJISTA" and precio_actual >= precio_retorno:
                                    print(f"   ✅ {simbolo} - REENTRY SHORT confirmado (precio {precio_actual:.8f} >= {precio_retorno:.8f})")
                                    
                                    if simbolo in self.esperando_reentry:
                                        tiempo_espera = (datetime.now() - self.esperando_reentry[simbolo]['timestamp']).total_seconds() / 60
                                        if tiempo_espera < 10:
                                            print(f"      ⏭️ Saltando {simbolo} (esperando {tiempo_espera:.1f} min)")
                                            continue
                                    
                                    es_valido_stoch = tipo_operacion == "SHORT" and stoch_k < 80
                                    
                                    if es_valido_stoch:
                                        sl_short, tp_short = self.calcular_stop_loss_take_profit(entrada, "SHORT", info_canal['ancho_canal'])
                                        
                                        self.esperando_reentry[simbolo] = {
                                            'tipo': 'SHORT',
                                            'timestamp': datetime.now(),
                                            'breakout_info': self.breakouts_detectados[simbolo]
                                        }
                                        
                                        self.generar_senal_operacion(
                                            simbolo, "SHORT", entrada, tp_short, sl_short,
                                            info_canal, datos_mercado, config_optima,
                                            self.breakouts_detectados[simbolo]
                                        )
                                        senales_encontradas += 1
                                        self.breakout_history[simbolo] = datetime.now()
                                        del self.breakouts_detectados[simbolo]
                                    else:
                                        print(f"   ❌ {simbolo} - Stochastic no confirma ({stoch_k:.1f} >= 80)")
                                else:
                                    print(f"   ⏳ {simbolo} - Esperando precio de retorno ({precio_retorno:.8f}, actual: {precio_actual:.8f})")
                            else:
                                print(f"   ⏳ {simbolo} - Esperando periodo reentry ({tiempo_desde_breakout:.1f}/{periodo_reentry} min)")
                        else:
                            pass
            
            except Exception as e:
                print(f"Error analizando {simbolo}: {e}")
                continue
        
        if self.esperando_reentry:
            print(f"\nEsperando reingreso en {len(self.esperando_reentry)} símbolos:")
            for simbolo, info in self.esperando_reentry.items():
                tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                print(f"   · {simbolo} - {info['tipo']} - Esperando {tiempo_espera:.1f} min")
        
        if self.breakouts_detectados:
            print(f"\nBreakouts detectados recientemente:")
            for simbolo, info in self.breakouts_detectados.items():
                tiempo_desde_deteccion = (datetime.now() - info['timestamp']).total_seconds() / 60
                print(f"   · {simbolo} - {info['tipo']} - Hace {tiempo_desde_deteccion:.1f} min")
        
        if senales_encontradas > 0:
            print(f"Se encontraron {senales_encontradas} señales de trading")
        else:
            print("No se encontraron señales en este ciclo")
        
        return senales_encontradas

    def generar_senal_operacion(self, simbolo, tipo_operacion, precio_entrada, tp, sl,
                            info_canal, datos_mercado, config_optima, breakout_info=None):
        if simbolo in self.operaciones_activas:
            es_manual = self.operaciones_activas[simbolo].get('operacion_manual_usuario', False)
            if es_manual:
                print(f"    {simbolo} - Operación manual detectada, omitiendo señal")
            else:
                print(f"    {simbolo} - Operación automática activa, omitiendo señal")
            return
        
        if simbolo in self.senales_enviadas:
            return
        
        if precio_entrada is None or tp is None or sl is None:
            print(f"    Niveles inválidos para {simbolo}, omitiendo señal")
            return
        
        riesgo = abs(precio_entrada - sl)
        beneficio = abs(tp - precio_entrada)
        ratio_rr = beneficio / riesgo if riesgo > 0 else 0
        sl_percent = abs((sl - precio_entrada) / precio_entrada) * 100
        tp_percent = abs((tp - precio_entrada) / precio_entrada) * 100
        stoch_estado = "SOBREVENTA" if tipo_operacion == "LONG" else "SOBRECOMPRA"
        breakout_texto = ""
        
        if breakout_info:
            tiempo_breakout = (datetime.now() - breakout_info['timestamp']).total_seconds() / 60
            breakout_texto = f"""
BREAKOUT + REENTRY DETECTADO:
Tiempo desde breakout: {tiempo_breakout:.1f} minutos
Precio breakout: {breakout_info['precio_breakout']:.8f}
"""
        
        mensaje = f"""
SEÑAL DE {tipo_operacion} - {simbolo}
{breakout_texto}
Configuración óptima:
Timeframe: {config_optima['timeframe']}
Velas: {config_optima['num_velas']}
Ancho Canal: {info_canal['ancho_canal_porcentual']:.1f}%
Precio Actual: {datos_mercado['precio_actual']:.8f}
Entrada: {precio_entrada:.8f}
Stop Loss: {sl:.8f}
Take Profit: {tp:.8f}
Ratio R/B: {ratio_rr:.2f}:1
SL: {sl_percent:.2f}%
TP: {tp_percent:.2f}%
Riesgo: {riesgo:.8f}
Beneficio Objetivo: {beneficio:.8f}
Tendencia: {info_canal['direccion']}
Fuerza: {info_canal['fuerza_texto']}
Ángulo: {info_canal['angulo_tendencia']:.1f}°
Pearson: {info_canal['coeficiente_pearson']:.3f}
R² Score: {info_canal['r2_score']:.3f}
Stochástico: {stoch_estado}
Stoch K: {info_canal['stoch_k']:.1f}
Stoch D: {info_canal['stoch_d']:.1f}
Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Estrategia: BREAKOUT + REENTRY con confirmación Stochastic
        """
        
        token = self.config.get('telegram_token')
        chat_ids = self.config.get('telegram_chat_ids', [])
        if token and chat_ids:
            try:
                print(f"     Generando gráfico para {simbolo}...")
                buf = self.generar_grafico_profesional(simbolo, info_canal, datos_mercado, 
                                                      precio_entrada, tp, sl, tipo_operacion)
                if buf:
                    print(f"     Enviando gráfico por Telegram...")
                    self.enviar_grafico_telegram(buf, token, chat_ids)
                    time.sleep(1)
                self._enviar_telegram_simple(mensaje, token, chat_ids)
                print(f"     Señal {tipo_operacion} para {simbolo} enviada")
            except Exception as e:
                print(f"     Error enviando señal: {e}")
        
        operacion_bitget = None
        
        if self.ejecutar_operaciones_automaticas and self.bitget_client:
            print(f"     Ejecutando operación automática en BITGET FUTUROS...")
            try:
                operacion_bitget = ejecutar_operacion_bitget(
                    bitget_client=self.bitget_client,
                    simbolo=simbolo,
                    tipo_operacion=tipo_operacion,
                    capital_usd=None,
                    leverage=self.leverage_por_defecto
                )
                if operacion_bitget:
                    print(f"     Operación ejecutada en BITGET FUTUROS para {simbolo}")
                    mensaje_confirmacion = f"""
OPERACIÓN AUTOMÁTICA EJECUTADA - {simbolo}
Status: EJECUTADA EN BITGET FUTUROS
Tipo: {tipo_operacion}
MARGIN USDT: ${operacion_bitget.get('capital_usado', 0):.2f} (3% del saldo actual)
Saldo Total: ${operacion_bitget.get('saldo_cuenta', 0):.2f}
Saldo Restante: ${operacion_bitget.get('saldo_cuenta', 0) - operacion_bitget.get('capital_usado', 0):.2f}
Valor Nocional: ${operacion_bitget.get('capital_usado', 0) * operacion_bitget.get('leverage', 1):.2f}
Apalancamiento: {operacion_bitget.get('leverage', self.leverage_por_defecto)}x
Entrada: {operacion_bitget['precio_entrada']:.8f}
Stop Loss: {operacion_bitget['stop_loss']:.8f}
Take Profit: {operacion_bitget['take_profit']:.8f}
ID Orden: {operacion_bitget['orden_entrada'].get('orderId', 'N/A')}
Sistema: Cada operación usa 3% del saldo actual (saldo disminuye)
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    self._enviar_telegram_simple(mensaje_confirmacion, token, chat_ids)
                    
                    self.operaciones_activas[simbolo] = {
                        'tipo': tipo_operacion,
                        'precio_entrada': precio_entrada,
                        'take_profit': tp,
                        'stop_loss': sl,
                        'timestamp_entrada': datetime.now().isoformat(),
                        'angulo_tendencia': info_canal['angulo_tendencia'],
                        'pearson': info_canal['coeficiente_pearson'],
                        'r2_score': info_canal['r2_score'],
                        'ancho_canal_relativo': info_canal['ancho_canal'] / precio_entrada,
                        'ancho_canal_porcentual': info_canal['ancho_canal_porcentual'],
                        'nivel_fuerza': info_canal['nivel_fuerza'],
                        'timeframe_utilizado': config_optima['timeframe'],
                        'velas_utilizadas': config_optima['num_velas'],
                        'stoch_k': info_canal['stoch_k'],
                        'stoch_d': info_canal['stoch_d'],
                        'breakout_usado': breakout_info is not None,
                        'operacion_ejecutada': True,
                        'order_id_entrada': operacion_bitget['orden_entrada'].get('orderId'),
                        'order_id_sl': operacion_bitget['orden_sl'].get('orderId') if operacion_bitget['orden_sl'] else None,
                        'order_id_tp': operacion_bitget['orden_tp'].get('orderId') if operacion_bitget['orden_tp'] else None,
                        'capital_usado': operacion_bitget['capital_usado'],
                        'valor_nocional': operacion_bitget['capital_usado'] * operacion_bitget['leverage'],
                        'margin_usdt_real': operacion_bitget['capital_usado'],
                        'leverage_usado': operacion_bitget['leverage']
                    }
                    
                    self.guardar_estado()
                    
                else:
                    print(f"     Error ejecutando operación en BITGET FUTUROS para {simbolo}")
                    print(f"     Operación NO agregada a operaciones_activas (falló ejecución)")
                    
            except Exception as e:
                print(f"     Error en ejecución automática: {e}")
                print(f"     Operación NO agregada a operaciones_activas (excepción: {e})")
        
        if not operacion_bitget:
            self.operaciones_activas[simbolo] = {
                'tipo': tipo_operacion,
                'precio_entrada': precio_entrada,
                'take_profit': tp,
                'stop_loss': sl,
                'timestamp_entrada': datetime.now().isoformat(),
                'angulo_tendencia': info_canal['angulo_tendencia'],
                'pearson': info_canal['coeficiente_pearson'],
                'r2_score': info_canal['r2_score'],
                'ancho_canal_relativo': info_canal['ancho_canal'] / precio_entrada,
                'ancho_canal_porcentual': info_canal['ancho_canal_porcentual'],
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
        if not os.path.exists(self.archivo_log):
            with open(self.archivo_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'tipo', 'precio_entrada',
                    'take_profit', 'stop_loss', 'precio_salida',
                    'resultado', 'pnl_percent', 'duracion_minutos',
                    'angulo_tendencia', 'pearson', 'r2_score',
                    'ancho_canal_relativo', 'ancho_canal_porcentual',
                    'nivel_fuerza', 'timeframe_utilizado', 'velas_utilizadas',
                    'stoch_k', 'stoch_d', 'breakout_usado', 'operacion_ejecutada'
                ])

    def registrar_operacion(self, datos_operacion):
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
                datos_operacion.get('ancho_canal_relativo', 0),
                datos_operacion.get('ancho_canal_porcentual', 0),
                datos_operacion.get('nivel_fuerza', 1),
                datos_operacion.get('timeframe_utilizado', 'N/A'),
                datos_operacion.get('velas_utilizadas', 0),
                datos_operacion.get('stoch_k', 0),
                datos_operacion.get('stoch_d', 0),
                datos_operacion.get('breakout_usado', False),
                datos_operacion.get('operacion_ejecutada', False)
            ])

    def verificar_cierre_operaciones(self):
        if not self.operaciones_activas:
            return []
        
        operaciones_cerradas = []
        for simbolo, operacion in list(self.operaciones_activas.items()):
            config_optima = self.config_optima_por_simbolo.get(simbolo)
            if not config_optima:
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
                    'ancho_canal_relativo': operacion.get('ancho_canal_relativo', 0),
                    'ancho_canal_porcentual': operacion.get('ancho_canal_porcentual', 0),
                    'nivel_fuerza': operacion.get('nivel_fuerza', 1),
                    'timeframe_utilizado': operacion.get('timeframe_utilizado', 'N/A'),
                    'velas_utilizadas': operacion.get('velas_utilizadas', 0),
                    'stoch_k': operacion.get('stoch_k', 0),
                    'stoch_d': operacion.get('stoch_d', 0),
                    'breakout_usado': operacion.get('breakout_usado', False),
                    'operacion_ejecutada': operacion.get('operacion_ejecutada', False)
                }
                
                mensaje_cierre = self.generar_mensaje_cierre(datos_operacion)
                token = self.config.get('telegram_token')
                chats = self.config.get('telegram_chat_ids', [])
                if token and chats:
                    try:
                        self._enviar_telegram_simple(mensaje_cierre, token, chats)
                    except Exception:
                        pass
                
                self.registrar_operacion(datos_operacion)
                operaciones_cerradas.append(simbolo)
                del self.operaciones_activas[simbolo]
                
                if simbolo in self.senales_enviadas:
                    self.senales_enviadas.remove(simbolo)
                
                self.operaciones_desde_optimizacion += 1
                print(f"     {simbolo} Operación {resultado} - PnL: {pnl_percent:.2f}%")
        
        return operaciones_cerradas

    def generar_mensaje_cierre(self, datos_operacion):
        emoji = "🟢" if datos_operacion['resultado'] == "TP" else "🔴"
        color_emoji = "✅" if datos_operacion['resultado'] == "TP" else "❌"
        
        if datos_operacion['tipo'] == 'LONG':
            pnl_absoluto = datos_operacion['precio_salida'] - datos_operacion['precio_entrada']
        else:
            pnl_absoluto = datos_operacion['precio_entrada'] - datos_operacion['precio_salida']
        
        breakout_usado = "Sí" if datos_operacion.get('breakout_usado', False) else "No"
        operacion_ejecutada = "Sí" if datos_operacion.get('operacion_ejecutada', False) else "No"
        
        mensaje = f"""
OPERACIÓN CERRADA - {datos_operacion['symbol']}
{color_emoji} RESULTADO: {datos_operacion['resultado']}
Tipo: {datos_operacion['tipo']}
Entrada: {datos_operacion['precio_entrada']:.8f}
Salida: {datos_operacion['precio_salida']:.8f}
PnL Absoluto: {pnl_absoluto:.8f}
PnL %: {datos_operacion['pnl_percent']:.2f}%
Duración: {datos_operacion['duracion_minutos']:.1f} minutos
Breakout+Reentry: {breakout_usado}
Operación BITGET FUTUROS: {operacion_ejecutada}
Ángulo: {datos_operacion['angulo_tendencia']:.1f}°
Pearson: {datos_operacion['pearson']:.3f}
R²: {datos_operacion['r2_score']:.3f}
Ancho: {datos_operacion.get('ancho_canal_porcentual', 0):.1f}%
TF: {datos_operacion.get('timeframe_utilizado', 'N/A')}
Velas: {datos_operacion.get('velas_utilizadas', 0)}
{datos_operacion['timestamp']}
        """
        return mensaje

    def calcular_stochastic(self, datos_mercado, period=14, k_period=3, d_period=3):
        if len(datos_mercado['cierres']) < period:
            return 50, 50
        
        cierres = datos_mercado['cierres']
        maximos = datos_mercado['maximos']
        minimos = datos_mercado['minimos']
        
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

    def calcular_regresion_lineal(self, x, y):
        if len(x) != len(y) or len(x) == 0:
            return None
        
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

    def calcular_pearson_y_angulo(self, x, y):
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
        
        angulo_radianes = math.atan(pendiente * len(x) / (max(y) - min(y)) if (max(y) - min(y)) != 0 else 0)
        angulo_grados = math.degrees(angulo_radianes)
        
        return pearson, angulo_grados

    def clasificar_fuerza_tendencia(self, angulo_grados):
        angulo_abs = abs(angulo_grados)
        if angulo_abs < 3:
            return "Muy Débil", 1
        elif angulo_abs < 13:
            return "Débil", 2
        elif angulo_abs < 27:
            return "Moderada", 3
        elif angulo_abs < 45:
            return "Fuerte", 4
        else:
            return "Muy Fuerte", 5

    def determinar_direccion_tendencia(self, angulo_grados, umbral_minimo=1):
        if abs(angulo_grados) < umbral_minimo:
            return "RANGO"
        elif angulo_grados > 0:
            return "ALCISTA"
        else:
            return "BAJISTA"

    def calcular_r2(self, y_real, x, pendiente, intercepto):
        if len(y_real) != len(x):
            return 0
        
        y_real = np.array(y_real)
        y_pred = pendiente * np.array(x) + intercepto
        
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)

    def generar_grafico_profesional(self, simbolo, info_canal, datos_mercado, precio_entrada, tp, sl, tipo_operacion):
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
                    url = "https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': simbolo,
                        'interval': config_optima['timeframe'],
                        'limit': config_optima['num_velas']
                    }
                    respuesta = requests.get(url, params=params, timeout=10)
                    klines = respuesta.json()
                    df_data = []
                    for kline in klines:
                        df_data.append({
                            'Date': pd.to_datetime(kline[0], unit='ms'),
                            'Open': float(kline[1]),
                            'High': float(kline[2]),
                            'Low': float(kline[3]),
                            'Close': float(kline[4]),
                            'Volume': float(kline[5])
                        })
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
            else:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': simbolo,
                    'interval': config_optima['timeframe'],
                    'limit': config_optima['num_velas']
                }
                respuesta = requests.get(url, params=params, timeout=10)
                klines = respuesta.json()
                df_data = []
                for kline in klines:
                    df_data.append({
                        'Date': pd.to_datetime(kline[0], unit='ms'),
                        'Open': float(kline[1]),
                        'High': float(kline[2]),
                        'Low': float(kline[3]),
                        'Close': float(kline[4]),
                        'Volume': float(kline[5])
                    })
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
            
            tiempos_reg = list(range(len(df)))
            
            # Usar los interceptos corregidos para líneas paralelas
            if 'intercepto_resistencia' in info_canal and 'intercepto_soporte' in info_canal:
                resistencia_values = [info_canal['pendiente_resistencia'] * t + info_canal['intercepto_resistencia'] for t in tiempos_reg]
                soporte_values = [info_canal['pendiente_soporte'] * t + info_canal['intercepto_soporte'] for t in tiempos_reg]
            else:
                # Fallback al cálculo antiguo (menos preciso)
                resistencia_values = []
                soporte_values = []
                for i, t in enumerate(tiempos_reg):
                    resist = info_canal['pendiente_resistencia'] * t + \
                            (info_canal['resistencia'] - info_canal['pendiente_resistencia'] * tiempos_reg[-1])
                    sop = info_canal['pendiente_soporte'] * t + \
                         (info_canal['soporte'] - info_canal['pendiente_soporte'] * tiempos_reg[-1])
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
            
            overbought = [80] * len(df)
            oversold = [20] * len(df)
            apds.append(mpf.make_addplot(overbought, color="#E7E4E4", linestyle='--', width=0.8, panel=1, alpha=0.5))
            apds.append(mpf.make_addplot(oversold, color="#E9E4E4", linestyle='--', width=0.8, panel=1, alpha=0.5))
            
            fig, axes = mpf.plot(df, type='candle', style='nightclouds',
                               title=f'{simbolo} | {tipo_operacion} | {config_optima["timeframe"]} | BITGET FUTUROS + Breakout+Reentry',
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
            print(f"Error generando gráfico: {e}")
            return None

    def enviar_grafico_telegram(self, buf, token, chat_ids):
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
                print(f"     Error enviando gráfico: {e}")
        return exito

    def _enviar_telegram_simple(self, mensaje, token, chat_ids):
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
        try:
            horas_desde_opt = (datetime.now() - self.ultima_optimizacion).total_seconds() / 7200
            if self.operaciones_desde_optimizacion >= 8 or horas_desde_opt >= self.config.get('reevaluacion_horas', 24):
                print("Iniciando re-optimización automática...")
                ia = OptimizadorIA(log_path=self.log_path, min_samples=self.config.get('min_samples_optimizacion', 30))
                nuevos_parametros = ia.buscar_mejores_parametros()
                if nuevos_parametros:
                    self.actualizar_parametros(nuevos_parametros)
                    self.ultima_optimizacion = datetime.now()
                    self.operaciones_desde_optimizacion = 0
                    print("Parámetros actualizados en tiempo real")
        except Exception as e:
            print(f"Error en re-optimización automática: {e}")

    def actualizar_parametros(self, nuevos_parametros):
        self.config['trend_threshold_degrees'] = nuevos_parametros.get('trend_threshold_degrees', 
                                                                        self.config.get('trend_threshold_degrees', 16))
        self.config['min_trend_strength_degrees'] = nuevos_parametros.get('min_trend_strength_degrees', 
                                                                           self.config.get('min_trend_strength_degrees', 16))
        self.config['entry_margin'] = nuevos_parametros.get('entry_margin', 
                                                             self.config.get('entry_margin', 0.001))

    def ejecutar_analisis(self):
        try:
            if self.bitget_client:
                self.sincronizar_con_bitget()
            
            if self.bitget_client:
                self.verificar_y_recolocar_tp_sl()
            
            if random.random() < 0.1:
                self.reoptimizar_periodicamente()
            
            cierres = self.verificar_cierre_operaciones()
            if cierres:
                print(f"     Operaciones cerradas: {', '.join(cierres)}")
            
            self.guardar_estado()
            
            return self.escanear_mercado()
            
        except Exception as e:
            logger.error(f"Error en ejecutar_analisis: {e}")
            try:
                self.guardar_estado()
            except:
                pass
            return 0

    def mostrar_resumen_operaciones(self):
        print(f"\nRESUMEN OPERACIONES:")
        print(f"   Activas: {len(self.operaciones_activas)}")
        print(f"   Esperando reentry: {len(self.esperando_reentry)}")
        print(f"   Total ejecutadas: {self.total_operaciones}")
        
        if self.bitget_client:
            print(f"   BITGET FUTUROS: Conectado")
            if self.ejecutar_operaciones_automaticas:
                print(f"   AUTO-TRADING: ACTIVADO (Dinero REAL)")
            else:
                print(f"   AUTO-TRADING: Solo señales")
        else:
            print(f"   BITGET FUTUROS: No configurado")
        
        if self.operaciones_activas:
            for simbolo, op in self.operaciones_activas.items():
                estado = "LONG" if op['tipo'] == 'LONG' else "SHORT"
                ancho_canal = op.get('ancho_canal_porcentual', 0)
                timeframe = op.get('timeframe_utilizado', 'N/A')
                velas = op.get('velas_utilizadas', 0)
                breakout = "🚀" if op.get('breakout_usado', False) else ""
                ejecutada = "🤖" if op.get('operacion_ejecutada', False) else ""
                manual = "👤" if op.get('operacion_manual_usuario', False) else ""
                print(f"   · {simbolo} {estado} {breakout} {ejecutada} {manual} - {timeframe} - {velas}v - Ancho: {ancho_canal:.1f}%")

    def iniciar(self):
        print("\n" + "=" * 70)
        print("BOT DE TRADING - ESTRATEGIA BREAKOUT + REENTRY")
        print("PRIORIDAD: TIMEFRAMES CORTOS (1m > 3m > 5m > 15m > 30m)")
        print("PERSISTENCIA: ACTIVADA")
        print("RE-EVALUACIÓN: CADA 2 HORAS")
        print("INTEGRACIÓN: BITGET FUTUROS API (Dinero REAL)")
        print("=" * 70)
        print(f"Símbolos: {len(self.config.get('symbols', []))} monedas")
        print(f"Timeframes: {', '.join(self.config.get('timeframes', []))}")
        print(f"Velas: {self.config.get('velas_options', [])}")
        print(f"ANCHO MÍNIMO: {self.config.get('min_channel_width_percent', 4)}%")
        print(f"Estrategia: 1) Detectar Breakout -> 2) Esperar Reentry -> 3) Confirmar con Stoch")
        
        if self.bitget_client:
            print(f"BITGET FUTUROS: API Conectada")
            print(f"Apalancamiento: {self.leverage_por_defecto}x")
            print(f"MARGIN USDT: 3% del saldo actual (se recalcula para CADA operación)")
            print(f"Sistema: El saldo disminuye progresivamente con cada operación")
            
            if self.ejecutar_operaciones_automaticas:
                print(f"AUTO-TRADING: ACTIVADO (Operaciones REALES con dinero)")
                print("ADVERTENCIA: TRADING AUTOMÁTICO REAL ACTIVADO")
                print("   El bot ejecutará operaciones REALES en Bitget Futures")
                print("   Cada operación usará 3% del saldo actual (el saldo disminuye)")
                print("   Usa con cuidado y solo con capital que puedas perder")
                confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
                if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
                    print("Operación cancelada")
                    return
            else:
                print(f"AUTO-TRADING: Solo señales (Paper Trading)")
        else:
            print(f"BITGET FUTUROS: No configurado (solo señales)")
        
        print("=" * 70)
        print("\nINICIANDO BOT...")
        
        if self.bitget_client:
            print("\nREALIZANDO SINCRONIZACIÓN INICIAL CON BITGET...")
            self.sincronizar_con_bitget()
            print("Sincronización inicial completada")
        
        try:
            while True:
                nuevas_senales = self.ejecutar_analisis()
                self.mostrar_resumen_operaciones()
                minutos_espera = self.config.get('scan_interval_minutes', 1)
                print(f"\nAnálisis completado. Señales nuevas: {nuevas_senales}")
                print(f"Próximo análisis en {minutos_espera} minutos...")
                print("-" * 60)
                for minuto in range(minutos_espera):
                    time.sleep(60)
                    restantes = minutos_espera - (minuto + 1)
                    if restantes > 0 and restantes % 5 == 0:
                        print(f"   {restantes} minutos restantes...")
        except KeyboardInterrupt:
            print("\nBot detenido por el usuario")
            print("Guardando estado final...")
            self.guardar_estado()
            print("¡Hasta pronto!")
        except Exception as e:
            print(f"\nError en el bot: {e}")
            print("Intentando guardar estado...")
            try:
                self.guardar_estado()
            except:
                pass

# ---------------------------
# CONFIGURACIÓN CON CREDENCIALES REALES DE BITGET FUTUROS
# ---------------------------
def crear_config_completa():
    BITGET_API_KEY = 'bg_0e9c732f2ed08d90c986a7fd9a4cdedd'
    BITGET_SECRET_KEY = '52582b11761d83bce4e4475182b1510617081dd4e56051e787178a2a06a5bd3b'
    BITGET_PASSPHRASE = 'Rasputino977'
    
    TELEGRAM_TOKEN = '8406173543:AAFIuYlFd3jtAF1Q6SNntUGn1PopgkZ7S0k'
    TELEGRAM_CHAT_ID = '2108159591'
    
    if 'RENDER' in os.environ:
        print("Leyendo configuración desde variables de entorno (Render.com)...")
        
        if os.environ.get('BITGET_API_KEY'):
            BITGET_API_KEY = os.environ.get('BITGET_API_KEY')
            print("Usando BITGET_API_KEY desde variable de entorno")
        if os.environ.get('BITGET_SECRET_KEY'):
            BITGET_SECRET_KEY = os.environ.get('BITGET_SECRET_KEY')
            print("Usando BITGET_SECRET_KEY desde variable de entorno")
        if os.environ.get('BITGET_PASSPHRASE'):
            BITGET_PASSPHRASE = os.environ.get('BITGET_PASSPHRASE')
            print("Usando BITGET_PASSPHRASE desde variable de entorno")
        if os.environ.get('TELEGRAM_TOKEN'):
            TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
            print("Usando TELEGRAM_TOKEN desde variable de entorno")
        if os.environ.get('TELEGRAM_CHAT_ID'):
            TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
            print("Usando TELEGRAM_CHAT_ID desde variable de entorno")
        
        ejecutar_automaticas = True
        if os.environ.get('EJECUTAR_OPERACIONES_AUTOMATICAS'):
            ejecutar_automaticas = os.environ.get('EJECUTAR_OPERACIONES_AUTOMATICAS').lower() == 'true'
            print(f"EJECUTAR_OPERACIONES_AUTOMATICAS: {ejecutar_automaticas}")
        
        leverage = 20
        if os.environ.get('LEVERAGE_POR_DEFECTO'):
            leverage = int(os.environ.get('LEVERAGE_POR_DEFECTO'))
            print(f"LEVERAGE_POR_DEFECTO: {leverage}")
        
        capital_por_operacion = 0.03
        if os.environ.get('CAPITAL_POR_OPERACION'):
            try:
                capital_por_operacion = float(os.environ.get('CAPITAL_POR_OPERACION'))
                print(f"CAPITAL_POR_OPERACION: {capital_por_operacion}")
            except ValueError:
                print("CAPITAL_POR_OPERACION inválido, usando valor por defecto 3%")
    else:
        ejecutar_automaticas = True
        leverage = 20
        capital_por_operacion = 0.03
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'min_channel_width_percent': 4.0,
        'trend_threshold_degrees': 16.0,
        'min_trend_strength_degrees': 16.0,
        'entry_margin': 0.001,
        'min_rr_ratio': 1.2,
        'scan_interval_minutes': 8,
        'timeframes': ['5m', '15m', '30m', '1h'],
        'velas_options': [80, 100, 120, 150, 200],
        'symbols': [
            'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'SHIBUSDT', 'POPCATUSDT',
            'CHILLGUYUSDT', 'PNUTUSDT', 'MEWUSDT', 'FARTCOINUSDT', 'DOGEUSDT',
            'VINEUSDT', 'HIPPOUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT',
            'ADAUSDT', 'ATOMUSDT', 'ETCUSDT', 'LINKUSDT', 'UNIUSDT',
            'SUSHIUSDT', 'CRVUSDT', 'SNXUSDT', 'SANDUSDT', 'MANAUSDT',
            'AXSUSDT', 'LRCUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT',
            'FILUSDT', 'SUIUSDT', 'AAVEUSDT', 'COMPUSDT', 'ENSUSDT',
            'LDOUSDT', 'RENDERUSDT', 'POLUSDT', 'ALGOUSDT', 'QNTUSDT',
            '1INCHUSDT', 'CVCUSDT', 'STGUSDT', 'ENJUSDT', 'GALAUSDT',
            'MAGICUSDT', 'REZUSDT', 'BLURUSDT', 'HMSTRUSDT', 'BEATUSDT',
            'ZEREBROUSDT', 'ZENUSDT', 'CETUSUSDT', 'DRIFTUSDT', 'PHAUSDT',
            'API3USDT', 'ACHUSDT', 'SPELLUSDT', 'ILVUSDT', 'YGGUSDT',
            'GMXUSDT', 'C98USDT', 'BALUSDT','XMRUSDT','AAVEUSDT','DOTUSDT',
            'BNBUSDT','SOLUSDT','AVAXUSDT','VETUSDT','ICPUSDT','FILUSDT',
            'BCHUSDT','NEOUSDT','TIAUSDT','TONUSDT','NMRUSDT','TRUMPUSDT'
        ],
        'telegram_token': TELEGRAM_TOKEN,
        'telegram_chat_ids': [TELEGRAM_CHAT_ID],
        'auto_optimize': True,
        'min_samples_optimizacion': 30,
        'reevaluacion_horas': 24,
        'log_path': os.path.join(directorio_actual, 'operaciones_log_v23_real.csv'),
        'estado_file': os.path.join(directorio_actual, 'estado_bot_v23_real.json'),
        'bitget_api_key': BITGET_API_KEY,
        'bitget_api_secret': BITGET_SECRET_KEY,
        'bitget_passphrase': BITGET_PASSPHRASE,
        'ejecutar_operaciones_automaticas': ejecutar_automaticas,
        'leverage_por_defecto': leverage
    }

# ---------------------------
# FLASK APP PARA EJECUCIÓN LOCAL
# ---------------------------
app = Flask(__name__)

config = crear_config_completa()
bot = TradingBot(config)

def run_bot_loop():
    while True:
        try:
            bot.ejecutar_analisis()
            time.sleep(bot.config.get('scan_interval_minutes', 2) * 60)
        except Exception as e:
            print(f"Error en el hilo del bot: {e}", file=sys.stderr)
            time.sleep(60)

bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
bot_thread.start()

@app.route('/')
def index():
    return "Bot Breakout + Reentry con integración Bitget FUTUROS está en línea.", 200

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    if request.is_json:
        update = request.get_json()
        print(f"Update recibido: {json.dumps(update)}", file=sys.stdout)
        return jsonify({"status": "ok"}), 200
    return jsonify({"error": "Request must be JSON"}), 400

@app.route('/health', methods=['GET'])
def health_check():
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
        print(f"Error en health check: {e}", file=sys.stderr)
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_telegram_webhook():
    token = config.get('telegram_token')
    if not token:
        print("No hay token de Telegram configurado", file=sys.stdout)
        return
    
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        render_url = os.environ.get('RENDER_EXTERNAL_URL')
        if render_url:
            webhook_url = f"{render_url}/webhook"
        else:
            print("No hay URL de webhook configurada y no se encontró RENDER_EXTERNAL_URL", file=sys.stdout)
            return
    
    try:
        print(f"Configurando webhook Telegram en: {webhook_url}", file=sys.stdout)
        requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
        time.sleep(1)
        response = requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}", timeout=10)
        
        if response.status_code == 200:
            print("Webhook de Telegram configurado correctamente", file=sys.stdout)
        else:
            print(f"Error configurando webhook: {response.status_code} - {response.text}", file=sys.stderr)
    except Exception as e:
        print(f"Error configurando webhook: {e}", file=sys.stderr)

def ejecutar_bot_directamente():
    print("="*70)
    print("BOT BREAKOUT+REENTRY COMPLETO - BITGET FUTUROS")
    print("Características completas:")
    print("   1. Estrategia Breakout+Reentry con confirmación Stochastic")
    print("   2. Gráficos profesionales con mplfinance")
    print("   3. Conexión REAL a Bitget Futures (Dinero REAL)")
    print("   4. Trading automático REAL con SL/TP")
    print("   5. Optimizador IA automático")
    print("   6. Alertas Telegram con gráficos")
    print("   7. Persistencia de estado")
    print("="*70)
    
    bot.iniciar()

if __name__ == '__main__':
    render_external_url = os.environ.get('RENDER_EXTERNAL_URL')
    render_port = os.environ.get('PORT')
    
    if render_external_url or render_port:
        print("Detectando entorno Render.com", file=sys.stdout)
        if render_external_url:
            print(f"   URL externa: {render_external_url}", file=sys.stdout)
        if render_port:
            print(f"   Puerto: {render_port}", file=sys.stdout)
        
        setup_telegram_webhook()
        
        port = int(render_port) if render_port else 5000
        print(f"Iniciando servidor Flask en puerto {port}...", file=sys.stdout)
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Modo ejecución local detectado", file=sys.stdout)
        ejecutar_bot_directamente()
