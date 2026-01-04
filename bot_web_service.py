# abre ordenes con TP/SL integrado - Version muy estable
# Adaptación para Render del bot Breakout + Reentry con correcciones Bitget
# CORRECCIÓN: Eliminados warnings de matplotlib para emojis
# ACTUALIZACIÓN: Integración completa de módulos TP/SL
# CORRECCIÓN: Error en place_plan_order (hold_side no definido)

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
import warnings

# Importar configuración de mínimos de Bitget
try:
    from config.bitget_config import get_minimum_size, get_recommended_leverage, get_price_precision
    BITGET_CONFIG_AVAILABLE = True
except ImportError:
    BITGET_CONFIG_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from io import BytesIO
from flask import Flask, request, jsonify
import threading
import logging

# Configuración mejorada de matplotlib para manejar emojis correctamente
def configurar_matplotlib():
    """Configura matplotlib para manejar emojis correctamente"""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*font.*')
    warnings.filterwarnings('ignore', message='.*font.*warning.*')
    
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    plt.rcParams['axes.facecolor'] = '#1a1a1a'
    matplotlib.rcParams['savefig.dpi'] = 100
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.facecolor'] = '#1a1a1a'
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

configurar_matplotlib()
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.*')
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*font.*')

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_debug.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ==================== MÓDULOS TP/SL INTEGRADOS ====================
# Clase BitgetTPSLClient - Gestión especializada de Take Profit y Stop Loss

class BitgetTPSLClient:
    """
    Cliente especializado en gestión de Take Profit (TP) y Stop Loss (SL)
    para Bitget Futures.
    
    Proporciona métodos para:
    - Colocar órdenes TP/SL independientes
    - Colocar órdenes de entrada con TP/SL integrados
    - Calcular precisión de precios adaptativa
    - Redondear precios según requisitos del exchange
    """
    
    def __init__(self, api_key, api_secret, passphrase, base_url="https://api.bitget.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = base_url
        logger.info("Cliente TP/SL Bitget inicializado")
    
    def _generate_signature(self, timestamp, method, request_path, body=''):
        """Generar firma HMAC-SHA256 para autenticación"""
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
        """Generar headers de autenticación"""
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
    
    def get_symbol_info(self, symbol, product_type='USDT-FUTURES'):
        """Obtener información del símbolo Futures"""
        try:
            request_path = '/api/v2/mix/market/contracts'
            params = {'productType': product_type}
            
            query_string = f"?productType={product_type}"
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
            logger.error(f"Error obteniendo info del símbolo: {e}")
            return None
    
    def obtener_precision_precio(self, symbol):
        """Obtiene la precisión de precio para un símbolo específico"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                price_scale = symbol_info.get('priceScale', 4)
                logger.info(f"📋 {symbol}: priceScale = {price_scale}")
                return price_scale
            else:
                logger.warning(f"No se pudo obtener info de {symbol}, usando 4 decimales")
                return 4
        except Exception as e:
            logger.error(f"Error obteniendo precisión de {symbol}: {e}")
            return 4
    
    def obtener_precision_adaptada(self, price, symbol):
        """
        Obtiene la precisión adaptativa basada en el precio.
        Evita redondeo a cero en memecoins con precios muy pequeños
        como SHIBUSDT, PEPE, ENS, XLM, PHA, etc.
        """
        try:
            price = float(price)
            
            if price < 1:
                if price < 0.00001:
                    return 12  # PEPE, SHIB
                elif price < 0.0001:
                    return 10  # Memecoins extremos
                elif price < 0.001:
                    return 8   # Memecoins y precios muy pequeños
                elif price < 0.01:
                    return 7   # Precios como ENS (~0.008)
                elif price < 0.1:
                    return 6   # Precios como PHA (~0.1)
                elif price < 1:
                    return 5   # Precios como XLM (~0.2)
            else:
                return 4
                
        except Exception as e:
            logger.error(f"Error calculando precisión adaptativa: {e}")
            return 8
    
    def redondear_precio_manual(self, price, precision, symbol=None):
        """
        Redondea el precio con una precisión específica.
        IMPORTANTE: Para la API de Bitget, el precio debe ser un múltiplo
        del priceStep del símbolo.
        """
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
                    
                    logger.info(f"🔢 {symbol}: precio={price}, priceScale={price_scale}, resultado={precio_formateado}")
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
    
    def place_tpsl_order(self, symbol, hold_side, trigger_price, order_type='stop_loss',
                         stop_loss_price=None, take_profit_price=None,
                         product_type='USDT-FUTURES', margin_coin='USDT'):
        """
        Coloca orden de Stop Loss o Take Profit en Bitget Futuros
        """
        request_path = '/api/v2/mix/order/place-pos-tpsl'
        
        precision_adaptada = self.obtener_precision_adaptada(trigger_price, symbol)
        trigger_price_formatted = self.redondear_precio_manual(trigger_price, precision_adaptada, symbol)
        
        body = {
            'symbol': symbol,
            'productType': product_type,
            'marginCoin': margin_coin,
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
            logger.info(f"🔧 SL para {symbol}: precio={stop_loss_price}, formatted={stop_loss_formatted}")
        elif order_type == 'take_profit' and take_profit_price:
            precision_tp = self.obtener_precision_adaptada(take_profit_price, symbol)
            take_profit_formatted = self.redondear_precio_manual(take_profit_price, precision_tp, symbol)
            body['stopSurplusTriggerPrice'] = take_profit_formatted
            logger.info(f"🔧 TP para {symbol}: precio={take_profit_price}, formatted={take_profit_formatted}")
        
        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        headers = self._get_headers('POST', request_path, body_json)
        
        logger.info(f"📤 Enviando orden {order_type} para {symbol}: {body}")
        
        response = requests.post(
            self.base_url + request_path,
            headers=headers,
            data=body_json,
            timeout=10
        )
        
        logger.info(f"📤 Respuesta TP/SL BITGET: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                logger.info(f"✅ {order_type.upper()} creado correctamente para {symbol}")
                return data.get('data')
            else:
                if data.get('code') == '40017':
                    logger.error(f"❌ Error 40017 en {order_type}: {data.get('msg')}")
                if data.get('code') == '40034':
                    logger.error(f"❌ Error 40034 en {order_type}: {data.get('msg')}")
        
        logger.error(f"❌ Error creando {order_type}: {response.text}")
        return None
    
    def place_order_with_tpsl(self, symbol, side, size, posSide=None, is_hedged_account=False,
                               stop_loss_price=None, take_profit_price=None,
                               order_type='market', product_type='USDT-FUTURES'):
        """
        Coloca orden de entrada con TP/SL integrados
        """
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
                "productType": product_type,
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "side": side,
                "orderType": order_type,
                "size": str(size)
            }
            logger.info(f"📤 Orden en MODO HEDGE: side={side}, size={size}")
        else:
            if not posSide:
                logger.error("❌ En modo unilateral, posSide es obligatorio ('long' o 'short')")
                return None
            body = {
                "symbol": symbol,
                "productType": product_type,
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "side": side,
                "orderType": order_type,
                "size": str(size),
                "posSide": posSide
            }
            logger.info(f"📤 Orden en MODO UNILATERAL: side={side}, posSide={posSide}, size={size}")
        
        if stop_loss_formatted is not None:
            body["presetStopLossPrice"] = str(stop_loss_formatted)
            logger.info(f"🔧 SL integrado: presetStopLossPrice={stop_loss_formatted}")
        
        if take_profit_formatted is not None:
            body["presetStopSurplusPrice"] = str(take_profit_formatted)
            logger.info(f"🔧 TP integrado: presetStopSurplusPrice={take_profit_formatted}")
        
        body_json = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        headers = self._get_headers('POST', request_path, body_json)
        
        logger.info(f"📤 Enviando orden con TP/SL integrados: {body}")
        
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
            else:
                if data.get('code') == '40774':
                    logger.error(f"❌ Error 40774: Modo de posición conflicto")
                if data.get('code') == '40034':
                    logger.error(f"❌ Error 40034: {data.get('msg')}")
        
        logger.error(f"❌ Error orden entrada: {response.text}")
        return None


def calcular_tp_sl(precio_entrada, tipo_operacion, porcentaje_tp=5.0, porcentaje_sl=2.5):
    """
    Calcular niveles de Take Profit y Stop Loss
    
    Args:
        precio_entrada: Precio de entrada de la operación
        tipo_operacion: 'LONG' o 'SHORT'
        porcentaje_tp: Porcentaje para Take Profit
        porcentaje_sl: Porcentaje para Stop Loss
    
    Returns:
        tuple: (take_profit, stop_loss)
    """
    if tipo_operacion == 'LONG':
        take_profit = precio_entrada * (1 + porcentaje_tp / 100)
        stop_loss = precio_entrada * (1 - porcentaje_sl / 100)
    else:
        take_profit = precio_entrada * (1 - porcentaje_tp / 100)
        stop_loss = precio_entrada * (1 + porcentaje_sl / 100)
    
    return take_profit, stop_loss


def calcular_riesgo_reward(precio_entrada, take_profit, stop_loss, tipo_operacion):
    """Calcular el ratio Riesgo/Recompensa (R/R)"""
    riesgo = abs(precio_entrada - stop_loss)
    beneficio = abs(take_profit - precio_entrada)
    
    if riesgo == 0:
        return 0
    
    return beneficio / riesgo


# ==================== FIN MÓDULOS TP/SL ====================


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
            logger.warning("No se encontró operaciones_log.csv (optimizador)")
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
            logger.info(f"No hay suficientes datos para optimizar (se requieren {self.min_samples}, hay {len(self.datos)})")
            return None
        mejor_score = -1e9
        mejores_param = None
        trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
        strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
        margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
        combos = list(itertools.product(trend_values, strength_values, margin_values))
        total = len(combos)
        logger.info(f"Optimizador: probando {total} combinaciones...")
        for idx, (t, s, m) in enumerate(combos, start=1):
            score = self.evaluar_configuracion(t, s, m)
            if idx % 100 == 0 or idx == total:
                logger.info(f"   · probado {idx}/{total} combos (mejor score actual: {mejor_score:.4f})")
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
            logger.info("Optimizador: mejores parámetros encontrados:", mejores_param)
            try:
                with open("mejores_parametros.json", "w", encoding='utf-8') as f:
                    json.dump(mejores_param, f, indent=2)
            except Exception as e:
                logger.error("Error guardando mejores_parametros.json:", e)
        else:
            logger.warning("No se encontró una configuración mejor")
        return mejores_param


class BitgetClient:
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self.product_type = "USDT-FUTURES"  
        logger.info(f"Cliente Bitget inicializado con API Key: {api_key[:10]}...")

    def _normalize_body(self, body):
        """Normaliza el body a string JSON sin espacios para la firma"""
        if body is None or body == '':
            return ''
        
        if isinstance(body, dict):
            return json.dumps(body, separators=(',', ':'))
        
        try:
            parsed = json.loads(body)
            return json.dumps(parsed, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            return body

    def _generate_signature(self, timestamp, method, request_path, body=''):
        """Generar firma HMAC-SHA256 para Bitget V2"""
        try:
            body_str = self._normalize_body(body)
            message = timestamp + method.upper() + request_path + body_str
            
            if not self.api_secret:
                logger.error("API Secret está vacía")
                return None
                
            mac = hmac.new(
                bytes(self.api_secret, 'utf-8'),
                bytes(message, 'utf-8'),
                digestmod=hashlib.sha256
            )
            
            signature = base64.b64encode(mac.digest()).decode()
            return signature
            
        except Exception as e:
            logger.error(f"Error generando firma: {e}")
            return None

    def _get_headers(self, method, request_path, body=''):
        """Obtener headers con firma para Bitget V2"""
        try:
            timestamp = str(int(time.time() * 1000))
            body_str = self._normalize_body(body)
            sign = self._generate_signature(timestamp, method, request_path, body_str)
            
            if not sign:
                logger.error("No se pudo generar la firma")
                return None
            
            headers = {
                'Content-Type': 'application/json',
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': sign,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase,
                'locale': 'en-US'
            }
            
            return headers
            
        except Exception as e:
            logger.error(f"Error creando headers: {e}")
            return None

    def verificar_credenciales(self):
        """Verificar que las credenciales sean válidas"""
        try:
            logger.info("Verificando credenciales Bitget...")
            
            if not self.api_key or not self.api_secret or not self.passphrase:
                logger.error("Credenciales incompletas")
                return False
            
            for product_type in ['USDT-FUTURES']:
                logger.info(f"Probando productType: {product_type}")
                
                try:
                    test_response = self.get_account_info(product_type)
                    if test_response is not None:
                        logger.info(f"Credenciales verificadas con productType: {product_type}")
                        self.product_type = product_type
                        
                        accounts = self.get_account_info()
                        if accounts:
                            for account in accounts:
                                if account.get('marginCoin') == 'USDT':
                                    available = float(account.get('available', 0))
                                    logger.info(f"Balance disponible: {available:.2f} USDT")
                        
                        return True
                        
                except Exception as e:
                    logger.warning(f"Error probando productType {product_type}: {e}")
                    continue
            
            logger.error("No se pudo verificar credenciales con ningún productType")
            return False
                
        except Exception as e:
            logger.error(f"Error verificando credenciales: {e}")
            return False

    def get_account_info(self, product_type=None):
        """Obtener información de cuenta Bitget V2"""
        try:
            if product_type is None:
                product_type = self.product_type
            
            request_path = '/api/v2/mix/account/accounts'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            
            query_string = f"?productType={product_type}&marginCoin=USDT"
            full_request_path = request_path + query_string
            
            headers = self._get_headers('GET', full_request_path, '')
            
            if not headers:
                logger.error("No se pudieron generar headers para la solicitud")
                return None
            
            response = requests.get(
                f"{self.base_url}{request_path}",
                headers=headers,
                params=params,
                timeout=10
            )
            
            logger.info(f"Respuesta cuenta - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    return data.get('data', [])
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"Error API Bitget: {error_code} - {error_msg}")
                    
                    if error_code == '40020':
                        logger.info(f"Error 40020 con productType {product_type}, intentando alternativo...")
                        return None
            else:
                logger.error(f"Error HTTP: {response.status_code} - {response.text}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error en get_account_info: {e}")
            return None

    def get_symbol_info(self, symbol):
        """Obtener información del símbolo"""
        try:
            request_path = '/api/v2/mix/market/contracts'
            params = {'productType': self.product_type}
            
            query_string = f"?productType={self.product_type}"
            full_request_path = request_path + query_string
            
            headers = self._get_headers('GET', full_request_path, '')
            
            if not headers:
                logger.error("No se pudieron generar headers")
                return None
            
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
                            logger.info(f"Informacion de simbolo obtenida para {symbol}")
                            return contract
            
            logger.warning(f"No se encontro informacion para {symbol} en {self.product_type}")
            return None
        except Exception as e:
            logger.error(f"Error obteniendo info del simbolo: {e}")
            return None

    def place_order(self, symbol, side, order_type, size, price=None, 
                    client_order_id=None, time_in_force='normal'):
        """Colocar orden de mercado o limite"""
        try:
            logger.info(f"Colocando orden: {symbol} {side} {size} {order_type}")
            
            request_path = '/api/v2/mix/order/place-order'
            body = {
                'symbol': symbol,
                'productType': self.product_type,
                'marginMode': 'isolated',
                'marginCoin': 'USDT',
                'side': side,
                'orderType': order_type,
                'size': str(size),
                'timeInForce': time_in_force
            }
            
            if price:
                body['price'] = str(price)
            if client_order_id:
                body['clientOrderId'] = client_order_id
            
            body_str = self._normalize_body(body)
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("No se pudieron generar headers para la orden")
                return None
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"Respuesta orden - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"Orden placed successfully: {data.get('data', {})}")
                    return data.get('data', {})
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"Error en orden Bitget: {error_code} - {error_msg}")
                    return None
            else:
                logger.error(f"Error HTTP en orden: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def place_plan_order(self, symbol, side, trigger_price, order_type, size, 
                         price=None, plan_type='loss_plan', trigger_type='mark_price', hold_side=None):
        """
        Colocar orden de plan (TP/SL) - CORREGIDO con hold_side como parametro
        """
        try:
            logger.info(f"Colocando orden plan: {symbol} {side} TP/SL en {trigger_price}")
            
            price_precision = self.obtener_precision_precio(symbol)
            trigger_price_formatted = str(round(float(trigger_price), price_precision))
            
            request_path = '/api/v2/mix/order/place-tpsl-order'
            body = {
                'symbol': symbol,
                'productType': self.product_type,
                'marginMode': 'isolated',
                'marginCoin': 'USDT',
                'side': side,
                'orderType': order_type,
                'triggerPrice': trigger_price_formatted,
                'size': str(size),
                'planType': plan_type,
                'triggerType': trigger_type,
                'holdSide': hold_side  # CORREGIDO: ahora hold_side se pasa como parametro
            }
            
            if price:
                body['executePrice'] = str(price)
            
            body_str = self._normalize_body(body)
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("No se pudieron generar headers para plan order")
                return None
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"Respuesta plan order - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"Plan order placed successfully")
                    return data.get('data', {})
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"Error en plan order Bitget: {error_code} - {error_msg}")
                    return None
            else:
                logger.error(f"Error HTTP en plan order: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error placing plan order: {e}")
            return None

    def set_leverage(self, symbol, leverage, hold_side='long'):
        """Configurar apalancamiento en BITGET"""
        try:
            logger.info(f"Configurando leverage {leverage}x para {symbol} ({hold_side})")
            
            request_path = '/api/v2/mix/account/set-leverage'
            body = {
                'symbol': symbol,
                'productType': self.product_type,
                'marginCoin': 'USDT',
                'leverage': str(leverage),
                'holdSide': hold_side
            }
            
            body_str = self._normalize_body(body)
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("No se pudieron generar headers para leverage")
                return False
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"Respuesta leverage - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"Apalancamiento {leverage}x configurado exitosamente")
                    return True
                else:
                    logger.error(f"Error configurando leverage: {data.get('code')} - {data.get('msg')}")
            else:
                logger.error(f"Error HTTP configurando leverage: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"Error en set_leverage: {e}")
            return False

    def get_positions(self, symbol=None, product_type=None):
        """Obtener posiciones abiertas"""
        try:
            if product_type is None:
                product_type = self.product_type
                
            request_path = '/api/v2/mix/position/all-position'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            if symbol:
                params['symbol'] = symbol
            
            query_string = f"?productType={product_type}&marginCoin=USDT"
            if symbol:
                query_string += f"&symbol={symbol}"
            full_request_path = request_path + query_string
            
            headers = self._get_headers('GET', full_request_path, '')
            
            if not headers:
                return []
            
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
            
            return []
        except Exception as e:
            logger.error(f"Error obteniendo posiciones: {e}")
            return []

    def obtener_saldo_cuenta(self):
        """Obtiene el saldo actual de la cuenta"""
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
        """Obtener velas (datos de mercado)"""
        try:
            interval_map = {
                '5m': '5m',
                '15m': '15m', '30m': '30m', '1h': '1H','4h': '4H', '1d': '1D'
            }
            bitget_interval = interval_map.get(interval, '5m')
            request_path = f'/api/v2/mix/market/candles'
            params = {
                'symbol': symbol,
                'productType': self.product_type,
                'granularity': bitget_interval,
                'limit': limit
            }
            
            headers = self._get_headers('GET', request_path, '')
            
            if not headers:
                return None
            
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
            return None
        except Exception as e:
            logger.error(f"Error en get_klines: {e}")
            return None

    def obtener_precision_precio(self, symbol):
        """Obtiene la precision de precio para un simbolo especifico"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                price_scale = symbol_info.get('priceScale', 4)
                logger.info(f"{symbol}: priceScale = {price_scale}")
                return price_scale
            else:
                logger.warning(f"No se pudo obtener info de {symbol}, usando 4 decimales")
                return 4
        except Exception as e:
            logger.error(f"Error obteniendo precision de {symbol}: {e}")
            return 4

    def obtener_reglas_simbolo(self, symbol):
        """Obtiene las reglas especificas de tamano para un simbolo"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No se pudo obtener info de {symbol}")
                
                if BITGET_CONFIG_AVAILABLE:
                    default_min_trade = get_minimum_size(symbol)
                    logger.info(f"Usando configuracion centralizada para {symbol}: {default_min_trade}")
                else:
                    default_min_trade = 0.001
                    if 'BTC' in symbol:
                        default_min_trade = 0.001
                    elif 'ETH' in symbol:
                        default_min_trade = 0.01
                
                return {
                    'size_scale': 0,
                    'quantity_scale': 0,
                    'min_trade_num': default_min_trade,
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
            
            return reglas
            
        except Exception as e:
            logger.error(f"Error obteniendo reglas de {symbol}: {e}")
            default_min_trade = 0.001
            if 'BTC' in symbol:
                default_min_trade = 0.001
            elif 'ETH' in symbol:
                default_min_trade = 0.01
                
            return {
                'size_scale': 0,
                'quantity_scale': 0,
                'min_trade_num': default_min_trade,
                'size_multiplier': 1,
                'delivery_mode': 0
            }

    def ajustar_tamano_orden(self, symbol, cantidad_contratos, reglas):
        """Ajusta el tamano de la orden segun las reglas del simbolo"""
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
                logger.info(f"{symbol}: ajustado a minimo = {min_trade_num}")
            
            if escala_actual == 0:
                if min_trade_num < 1 and min_trade_num > 0:
                    cantidad_contratos = max(1, int(round(cantidad_contratos)))
                    logger.info(f"{symbol}: caso especial - min decimal pero requiere entero = {cantidad_contratos}")
                else:
                    cantidad_contratos = int(round(cantidad_contratos))
                logger.info(f"{symbol} final: {cantidad_contratos} (entero)")
            else:
                cantidad_contratos = round(cantidad_contratos, escala_actual)
                logger.info(f"{symbol} final: {cantidad_contratos} ({escala_actual} decimales)")
            
            return cantidad_contratos
            
        except Exception as e:
            logger.error(f"Error ajustando tamano para {symbol}: {e}")
            return int(round(cantidad_contratos))


def ejecutar_operacion_bitget(bitget_client, simbolo, tipo_operacion, capital_usd, leverage=10):
    """
    Ejecutar una operacion completa en Bitget (posicion + TP/SL) - INTEGRADA CON MODULOS TP/SL
    """
    logger.info("EJECUTANDO OPERACION REAL EN BITGET")
    logger.info(f"Símbolo: {simbolo}")
    logger.info(f"Tipo: {tipo_operacion}")
    logger.info(f"Apalancamiento: {leverage}x")
    logger.info(f"Capital: ${capital_usd}")
    
    try:
        hold_side = 'long' if tipo_operacion == 'LONG' else 'short'
        leverage = min(leverage, 10)
        
        leverage_ok = bitget_client.set_leverage(simbolo, leverage, hold_side)
        if not leverage_ok:
            logger.error("Error configurando apalancamiento")
            logger.info("Intentando con apalancamiento 5x...")
            leverage_ok = bitget_client.set_leverage(simbolo, 5, hold_side)
            if not leverage_ok:
                logger.error("No se pudo configurar apalancamiento")
                return None
            leverage = 5
        
        time.sleep(1)
        
        klines = bitget_client.get_klines(simbolo, '1m', 1)
        if not klines or len(klines) == 0:
            logger.error(f"No se pudo obtener precio de {simbolo}")
            try:
                url = "https://api.binance.com/api/v3/ticker/price"
                params = {'symbol': simbolo}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    precio_actual = float(data['price'])
                    logger.info(f"Precio actual (de Binance): {precio_actual:.8f}")
                else:
                    logger.error("No se pudo obtener precio de ninguna fuente")
                    return None
            except Exception as e:
                logger.error(f"Error obteniendo precio de Binance: {e}")
                return None
        else:
            klines.reverse()
            precio_actual = float(klines[0][4])
            logger.info(f"Precio actual: {precio_actual:.8f}")
        
        symbol_info = bitget_client.get_symbol_info(simbolo)
        if not symbol_info:
            logger.warning(f"No se pudo obtener info de {simbolo}")
            size_multiplier = 1
            
            if BITGET_CONFIG_AVAILABLE:
                min_trade_num = get_minimum_size(simbolo)
                price_place = get_price_precision(simbolo)
                logger.info(f"Usando configuracion centralizada para {simbolo}: min={min_trade_num}, prec={price_place}")
            else:
                min_trade_num = 0.001
                if 'BTC' in simbolo:
                    min_trade_num = 0.001
                elif 'ETH' in simbolo:
                    min_trade_num = 0.01
                price_place = 8
        else:
            size_multiplier = float(symbol_info.get('sizeMultiplier', 1))
            min_trade_num = float(symbol_info.get('minTradeNum', 0.001))
            price_place = int(symbol_info.get('pricePlace', 8))
        
        cantidad_usd = capital_usd * leverage
        cantidad_contratos = cantidad_usd / precio_actual
        cantidad_contratos = round(cantidad_contratos / size_multiplier) * size_multiplier
        
        if cantidad_contratos < min_trade_num:
            logger.warning(f"Cantidad calculada ({cantidad_contratos}) menor al mínimo ({min_trade_num})")
            cantidad_contratos = min_trade_num
            logger.info(f"Cantidad ajustada al mínimo permitido: {cantidad_contratos}")
        
        cantidad_contratos = round(cantidad_contratos, 8)
        
        logger.info(f"Cantidad: {cantidad_contratos} contratos")
        logger.info(f"Valor nocional: ${cantidad_contratos * precio_actual:.2f}")
        
        # USAR MODULO TP/SL INTEGRADO
        take_profit, stop_loss = calcular_tp_sl(precio_actual, tipo_operacion, 5.0, 2.5)
        stop_loss = round(stop_loss, price_place)
        take_profit = round(take_profit, price_place)
        
        logger.info(f"Stop Loss: {stop_loss:.8f}")
        logger.info(f"Take Profit: {take_profit:.8f}")
        
        side = 'buy' if tipo_operacion == 'LONG' else 'sell'
        orden_entrada = bitget_client.place_order(
            symbol=simbolo,
            side=side,
            order_type='market',
            size=cantidad_contratos
        )
        
        if not orden_entrada:
            logger.error("Error abriendo posicion")
            return None
        
        logger.info("Posicion abierta exitosamente")
        time.sleep(1)
        
        # USAR MODULO TP/SL INTEGRADO - place_plan_order CORREGIDO
        sl_side = 'sell' if tipo_operacion == 'LONG' else 'buy'
        orden_sl = bitget_client.place_plan_order(
            symbol=simbolo,
            side=sl_side,
            trigger_price=stop_loss,
            order_type='market',
            size=cantidad_contratos,
            plan_type='loss_plan',
            hold_side=hold_side  # CORREGIDO: ahora se pasa correctamente
        )
        
        if orden_sl:
            logger.info(f"Stop Loss configurado en: {stop_loss:.8f}")
        else:
            logger.warning("Error configurando Stop Loss")
        
        time.sleep(0.5)
        
        orden_tp = bitget_client.place_plan_order(
            symbol=simbolo,
            side=sl_side,
            trigger_price=take_profit,
            order_type='market',
            size=cantidad_contratos,
            plan_type='profit_plan',
            hold_side=hold_side  # CORREGIDO: ahora se pasa correctamente
        )
        
        if orden_tp:
            logger.info(f"Take Profit configurado en: {take_profit:.8f}")
        else:
            logger.warning("Error configurando Take Profit")
        
        operacion_data = {
            'orden_entrada': orden_entrada,
            'orden_sl': orden_sl,
            'orden_tp': orden_tp,
            'cantidad_contratos': cantidad_contratos,
            'precio_entrada': precio_actual,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'leverage': leverage,
            'capital_usado': capital_usd,
            'tipo': tipo_operacion,
            'timestamp_entrada': datetime.now().isoformat(),
            'symbol': simbolo
        }
        
        logger.info("OPERACION EJECUTADA EXITOSAMENTE")
        logger.info(f"ID Orden: {orden_entrada.get('orderId', 'N/A')}")
        logger.info(f"Contratos: {cantidad_contratos}")
        logger.info(f"Entrada: {precio_actual:.8f}")
        logger.info(f"SL: {stop_loss:.8f} (-2%)")
        logger.info(f"TP: {take_profit:.8f} (+5%)")
        logger.info(f"Ratio Riesgo/Beneficio: {calcular_riesgo_reward(precio_actual, take_profit, stop_loss, tipo_operacion):.2f}")
        
        return operacion_data
        
    except Exception as e:
        logger.error(f"Error en ejecutar_operacion_bitget: {e}", exc_info=True)
        return None


class TradingBot:
    def __init__(self, config):
        self.config = config
        self.symbols = config.get('symbols', [])
        self.timeframes = config.get('timeframes', [])
        self.velas_options = config.get('velas_options', [])
        
        self.operaciones_activas = {}
        self.senales_enviadas = set()
        self.esperando_reentry = {}
        self.breakouts_detectados = {}
        self.breakout_history = {}
        
        self.total_operaciones = 0
        self.operaciones_desde_optimizacion = 0
        self.ultima_optimizacion = datetime.now()
        
        self.bitget_client = None
        self.ejecutar_operaciones_automaticas = config.get('ejecutar_operaciones_automaticas', False)
        self.capital_por_operacion = config.get('capital_por_operacion', 4)
        self.leverage_por_defecto = min(config.get('leverage_por_defecto', 10), 10)
        
        self.log_path = config.get('log_path', 'operaciones_log_v23.csv')
        self.archivo_log = self.log_path
        self.estado_file = config.get('estado_file', 'estado_bot_v23.json')
        
        self.config_optima_por_simbolo = self.cargar_config_optima()
        self.inicializar_log()
        
        if config.get('bitget_api_key') and config.get('bitget_api_secret') and config.get('bitget_passphrase'):
            self.bitget_client = BitgetClient(
                config['bitget_api_key'],
                config['bitget_api_secret'],
                config['bitget_passphrase']
            )
            if self.bitget_client.verificar_credenciales():
                logger.info("Bitget API connectado y verificado")
            else:
                logger.warning("Error al verificar credenciales de Bitget")
                self.bitget_client = None
        
        self.cargar_estado()

    def guardar_config_optima(self):
        try:
            with open('config_optima.json', 'w', encoding='utf-8') as f:
                json.dump(self.config_optima_por_simbolo, f, indent=2)
            logger.info(f"Configuración óptima guardada para {len(self.config_optima_por_simbolo)} símbolos")
        except Exception as e:
            logger.error(f"Error guardando config_optima: {e}")

    def cargar_config_optima(self):
        if os.path.exists('config_optima.json'):
            try:
                with open('config_optima.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"Configuración óptima cargada para {len(config)} símbolos")
                    return config
            except Exception as e:
                logger.error(f"Error cargando config_optima: {e}")
        return {}

    def guardar_estado(self):
        estado = {
            'operaciones_activas': {k: {kk: str(vv) if isinstance(vv, (datetime,)) else vv for kk, vv in v.items()} for k, v in self.operaciones_activas.items()},
            'senales_enviadas': list(self.senales_enviadas),
            'esperando_reentry': {k: {'tipo': v['tipo'], 'timestamp': v['timestamp'].isoformat() if isinstance(v['timestamp'], datetime) else v['timestamp'], 'precio_breakout': v['precio_breakout'], 'config': v['config']} for k, v in self.esperando_reentry.items()},
            'breakouts_detectados': {k: {'tipo': v['tipo'], 'timestamp': v['timestamp'].isoformat() if isinstance(v['timestamp'], datetime) else v['timestamp'], 'precio_breakout': v['precio_breakout']} for k, v in self.breakouts_detectados.items()},
            'total_operaciones': self.total_operaciones,
            'operaciones_desde_optimizacion': self.operaciones_desde_optimizacion,
            'ultima_optimizacion': self.ultima_optimizacion.isoformat() if isinstance(self.ultima_optimizacion, datetime) else self.ultima_optimizacion
        }
        try:
            with open(self.estado_file, 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2, default=str)
            logger.debug(f"Estado guardado: {len(self.operaciones_activas)} operaciones activas")
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")

    def cargar_estado(self):
        if os.path.exists(self.estado_file):
            try:
                with open(self.estado_file, 'r', encoding='utf-8') as f:
                    estado = json.load(f)
                    self.total_operaciones = estado.get('total_operaciones', 0)
                    self.operaciones_desde_optimizacion = estado.get('operaciones_desde_optimizacion', 0)
                    try:
                        self.ultima_optimizacion = datetime.fromisoformat(estado.get('ultima_optimizacion', datetime.now().isoformat()))
                    except (ValueError, TypeError):
                        self.ultima_optimizacion = datetime.now()
                    logger.info(f"Estado cargado: {len(estado.get('operaciones_activas', {}))} operaciones activas")
            except Exception as e:
                logger.error(f"Error cargando estado: {e}")

    def obtener_datos_mercado(self, symbol, timeframe='5m', limit=200):
        """Obtener datos de mercado de Binance (backup de Bitget)"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            respuesta = requests.get(url, params=params, timeout=10)
            
            if respuesta.status_code == 200:
                datos = respuesta.json()
                cierres = [float(k[4]) for k in datos]
                maximos = [float(k[2]) for k in datos]
                minimos = [float(k[3]) for k in datos]
                return {
                    'cierres': cierres,
                    'maximos': maximos,
                    'minimos': minimos,
                    'precio_actual': cierres[-1] if cierres else 0
                }
            else:
                logger.error(f"Error obteniendo datos de Binance para {symbol}: {respuesta.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error en obtener_datos_mercado (Binance): {e}")
            return None

    def obtener_datos_mercado_config(self, symbol, timeframe='5m', limit=200):
        """Obtener datos de mercado con fallback a Binance"""
        try:
            if self.bitget_client:
                klines = self.bitget_client.get_klines(symbol, timeframe, limit)
                if klines:
                    cierres = [float(k[4]) for k in klines]
                    maximos = [float(k[2]) for k in klines]
                    minimos = [float(k[3]) for k in klines]
                    return {
                        'cierres': cierres,
                        'maximos': maximos,
                        'minimos': minimos,
                        'precio_actual': cierres[-1] if cierres else 0
                    }
            
            return self.obtener_datos_mercado(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Error en obtener_datos_mercado_config: {e}")
            return None

    def calcular_canal_regresion(self, cierres, maximos, minimos, num_velas=80):
        """Calcular canal de regresion lineal"""
        try:
            if len(cierres) < num_velas:
                return None
            
            cierres_slice = cierres[-num_velas:]
            maximos_slice = maximos[-num_velas:]
            minimos_slice = minimos[-num_velas:]
            
            x = np.arange(len(cierres_slice))
            
            pendiente_cierre, intercepto_cierre = self.calcular_regresion_lineal(x, cierres_slice)
            pendiente_max, intercepto_max = self.calcular_regresion_lineal(x, maximos_slice)
            pendiente_min, intercepto_min = self.calcular_regresion_lineal(x, minimos_slice)
            
            if pendiente_cierre is None:
                return None
            
            precio_final = pendiente_cierre * (len(cierres_slice) - 1) + intercepto_cierre
            resistencia_final = pendiente_max * (len(cierres_slice) - 1) + intercepto_max
            soporte_final = pendiente_min * (len(cierres_slice) - 1) + intercepto_min
            
            precio_actual = cierres_slice[-1]
            
            ancho_absoluto = resistencia_final - soporte_final
            ancho_porcentual = (ancho_absoluto / precio_actual) * 100 if precio_actual > 0 else 0
            
            pearson, angulo = self.calcular_pearson_y_angulo(x, cierres_slice)
            r2 = self.calcular_r2(cierres_slice, x, pendiente_cierre, intercepto_cierre)
            fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo)
            direccion = self.determinar_direccion_tendencia(angulo)
            
            stoch_k, stoch_d = self.calcular_stochastic({'cierres': cierres, 'maximos': maximos, 'minimos': minimos})
            
            return {
                'pendiente_resistencia': pendiente_max,
                'pendiente_soporte': pendiente_min,
                'pendiente_cierre': pendiente_cierre,
                'resistencia': resistencia_final,
                'soporte': soporte_final,
                'precio_actual': precio_actual,
                'ancho_canal': ancho_absoluto,
                'ancho_canal_porcentual': ancho_porcentual,
                'coeficiente_pearson': pearson,
                'angulo_tendencia': angulo,
                'r2_score': r2,
                'fuerza_texto': fuerza_texto,
                'nivel_fuerza': nivel_fuerza,
                'direccion': direccion,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            }
        except Exception as e:
            logger.error(f"Error en calcular_canal_regresion: {e}")
            return None

    def calcular_canal_regresion_config(self, datos, num_velas=80):
        """Calcular canal de regresion con config"""
        return self.calcular_canal_regresion(
            datos['cierres'],
            datos['maximos'],
            datos['minimos'],
            num_velas
        )

    def detectar_breakout(self, symbol, info_canal, datos):
        """Detectar breakout del canal"""
        try:
            precio_actual = datos['precio_actual']
            resistencia = info_canal['resistencia']
            soporte = info_canal['soporte']
            
            umbral_breakout = self.config.get('entry_margin', 0.001)
            ancho_canal = info_canal['ancho_canal']
            precio_actual_val = datos['precio_actual']
            
            if precio_actual > resistencia * (1 + umbral_breakout) and info_canal['direccion'] == '🟢 ALCISTA':
                distancia_breakout = (precio_actual - resistencia) / precio_actual_val * 100
                if distancia_breakout < 5:
                    return 'LONG'
            
            elif precio_actual < soporte * (1 - umbral_breakout) and info_canal['direccion'] == '🔴 BAJISTA':
                distancia_breakout = (soporte - precio_actual) / precio_actual_val * 100
                if distancia_breakout < 5:
                    return 'SHORT'
            
            return None
        except Exception as e:
            logger.error(f"Error detectando breakout: {e}")
            return None

    def detectar_reentry(self, symbol, info_canal, datos):
        """Detectar reentry al canal"""
        try:
            precio_actual = datos['precio_actual']
            resistencia = info_canal['resistencia']
            soporte = info_canal['soporte']
            
            if symbol not in self.esperando_reentry:
                return None
            
            info_reentry = self.esperando_reentry[symbol]
            tipo_breakout = info_reentry['tipo']
            
            margen_reentry = self.config.get('entry_margin', 0.001)
            
            if tipo_breakout == 'LONG':
                precio_reentry = resistencia * (1 + margen_reentry)
                if precio_actual <= resistencia and precio_actual >= precio_reentry * 0.99:
                    if info_canal['stoch_k'] <= 30:
                        return 'LONG'
            
            elif tipo_breakout == 'SHORT':
                precio_reentry = soporte * (1 - margen_reentry)
                if precio_actual >= soporte and precio_actual <= precio_reentry * 1.01:
                    if info_canal['stoch_k'] >= 70:
                        return 'SHORT'
            
            return None
        except Exception as e:
            logger.error(f"Error detectando reentry: {e}")
            return None

    def calcular_niveles_entrada(self, tipo_operacion, info_canal, precio_actual):
        """Calcular niveles de entrada, TP y SL"""
        try:
            resistencia = info_canal['resistencia']
            soporte = info_canal['soporte']
            
            if tipo_operacion == 'LONG':
                precio_entrada = resistencia * (1 + self.config.get('entry_margin', 0.001))
                stop_loss = soporte
                take_profit = precio_entrada + (precio_entrada - stop_loss) * 2
            
            else:
                precio_entrada = soporte * (1 - self.config.get('entry_margin', 0.001))
                stop_loss = resistencia
                take_profit = precio_entrada - (stop_loss - precio_entrada) * 2
            
            precio_entrada = round(precio_entrada, 8)
            stop_loss = round(stop_loss, 8)
            take_profit = round(take_profit, 8)
            
            riesgo = abs(precio_entrada - stop_loss)
            beneficio = abs(take_profit - precio_entrada)
            
            if riesgo > 0:
                ratio_rr = beneficio / riesgo
                if ratio_rr < self.config.get('min_rr_ratio', 1.2):
                    return None, None, None
            
            return precio_entrada, take_profit, stop_loss
        except Exception as e:
            logger.error(f"Error calculando niveles de entrada: {e}")
            return None, None, None

    def escanear_mercado(self):
        """Escanear mercado en busca de señales"""
        senales_encontradas = 0
        
        for simbolo in self.symbols:
            try:
                mejor_config = None
                datos_mercado = None
                info_canal = None
                
                for tf in self.timeframes:
                    for num_velas in self.velas_options:
                        datos = self.obtener_datos_mercado_config(simbolo, tf, num_velas)
                        if not datos:
                            continue
                        
                        canal = self.calcular_canal_regresion_config(datos, num_velas)
                        if not canal:
                            continue
                        
                        ancho_pct = canal['ancho_canal_porcentual']
                        min_ancho = self.config.get('min_channel_width_percent', 4)
                        
                        if (ancho_pct >= min_ancho and 
                            canal['nivel_fuerza'] >= 2 and 
                            abs(canal['coeficiente_pearson']) >= 0.4 and 
                            canal['r2_score'] >= 0.4):
                            
                            if (mejor_config is None or 
                                (tf in ['5m', '15m'] and canal['nivel_fuerza'] > self.config_optima_por_simbolo.get(simbolo, {}).get('fuerza', 0))):
                                mejor_config = {'timeframe': tf, 'num_velas': num_velas, 'fuerza': canal['nivel_fuerza']}
                                datos_mercado = datos
                                info_canal = canal
                
                if not datos_mercado:
                    continue
                
                config_optima = self.config_optima_por_simbolo.get(simbolo, mejor_config)
                
                if simbolo not in self.esperando_reentry:
                    tipo_breakout = self.detectar_breakout(simbolo, info_canal, datos_mercado)
                    if tipo_breakout:
                        self.esperando_reentry[simbolo] = {
                            'tipo': tipo_breakout,
                            'timestamp': datetime.now(),
                            'precio_breakout': datos_mercado['precio_actual'],
                            'config': config_optima
                        }
                        self.breakouts_detectados[simbolo] = {
                            'tipo': tipo_breakout,
                            'timestamp': datetime.now(),
                            'precio_breakout': datos_mercado['precio_actual']
                        }
                        logger.info(f"Breakout detectado en {simbolo}, esperando reentry...")
                        continue
                
                tipo_operacion = self.detectar_reentry(simbolo, info_canal, datos_mercado)
                if not tipo_operacion:
                    continue
                
                precio_entrada, tp, sl = self.calcular_niveles_entrada(tipo_operacion, info_canal, datos_mercado['precio_actual'])
                
                if not precio_entrada or not tp or not sl:
                    continue
                
                if simbolo in self.breakout_history:
                    ultimo_breakout = self.breakout_history[simbolo]
                    tiempo_desde_ultimo = (datetime.now() - ultimo_breakout).total_seconds() / 3600
                    if tiempo_desde_ultimo < 2:
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
                logger.error(f"Error analizando {simbolo}: {e}")
                continue
        
        if self.esperando_reentry:
            logger.info(f"Esperando reentry en {len(self.esperando_reentry)} simbolos:")
        
        if senales_encontradas > 0:
            logger.info(f"Se encontraron {senales_encontradas} senales de trading")
        else:
            logger.info("No se encontraron senales en este ciclo")
        
        return senales_encontradas

    def generar_senal_operacion(self, simbolo, tipo_operacion, precio_entrada, tp, sl,
                            info_canal, datos_mercado, config_optima, breakout_info=None):
        """Genera y envia senal de operacion"""
        if simbolo in self.senales_enviadas:
            return
        
        if precio_entrada is None or tp is None or sl is None:
            logger.warning(f"Niveles invalidos para {simbolo}, omitiendo senal")
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
Configuracion optima:
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
Angulo: {info_canal['angulo_tendencia']:.1f}°
Pearson: {info_canal['coeficiente_pearson']:.3f}
R² Score: {info_canal['r2_score']:.3f}
Stochastico: {stoch_estado}
Stoch K: {info_canal['stoch_k']:.1f}
Stoch D: {info_canal['stoch_d']:.1f}
Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Estrategia: BREAKOUT + REENTRY con confirmacion Stochastic
        """
        
        token = self.config.get('telegram_token')
        chat_ids = self.config.get('telegram_chat_ids', [])
        
        if token and chat_ids:
            try:
                logger.info(f"Generando grafico para {simbolo}...")
                buf = self.generar_grafico_profesional(simbolo, info_canal, datos_mercado, 
                                                      precio_entrada, tp, sl, tipo_operacion)
                if buf:
                    logger.info("Enviando grafico por Telegram...")
                    self.enviar_grafico_telegram(buf, token, chat_ids)
                    time.sleep(1)
                
                self._enviar_telegram_simple(mensaje, token, chat_ids)
                logger.info(f"Señal {tipo_operacion} para {simbolo} enviada")
            except Exception as e:
                logger.error(f"Error enviando senal: {e}")
        
        if self.ejecutar_operaciones_automaticas and self.bitget_client:
            logger.info(f"Ejecutando operacion automatica en Bitget...")
            try:
                operacion_bitget = ejecutar_operacion_bitget(
                    bitget_client=self.bitget_client,
                    simbolo=simbolo,
                    tipo_operacion=tipo_operacion,
                    capital_usd=self.capital_por_operacion,
                    leverage=self.leverage_por_defecto
                )
                
                if operacion_bitget:
                    logger.info(f"Operacion ejecutada en Bitget para {simbolo}")
                    mensaje_confirmacion = f"""
OPERACION AUTOMATICA EJECUTADA - {simbolo}
Status: EJECUTADA EN BITGET
Tipo: {tipo_operacion}
Capital: ${self.capital_por_operacion}
Apalancamiento: {self.leverage_por_defecto}x
Entrada: {operacion_bitget['precio_entrada']:.8f}
Stop Loss: {operacion_bitget['stop_loss']:.8f}
Take Profit: {operacion_bitget['take_profit']:.8f}
ID Orden: {operacion_bitget['orden_entrada'].get('orderId', 'N/A')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    self._enviar_telegram_simple(mensaje_confirmacion, token, chat_ids)
                else:
                    logger.error(f"Error ejecutando operacion en Bitget para {simbolo}")
            except Exception as e:
                logger.error(f"Error en ejecucion automatica: {e}")
        
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
            'operacion_ejecutada': self.ejecutar_operaciones_automaticas and self.bitget_client is not None
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
        try:
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
            logger.info(f"Operacion registrada en log: {datos_operacion['symbol']}")
        except Exception as e:
            logger.error(f"Error registrando operacion en log: {e}")

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
                    except Exception as e:
                        logger.error(f"Error enviando mensaje de cierre: {e}")
                
                self.registrar_operacion(datos_operacion)
                operaciones_cerradas.append(simbolo)
                del self.operaciones_activas[simbolo]
                
                if simbolo in self.senales_enviadas:
                    self.senales_enviadas.remove(simbolo)
                
                self.operaciones_desde_optimizacion += 1
                logger.info(f"{simbolo} Operacion {resultado} - PnL: {pnl_percent:.2f}%")
        
        return operaciones_cerradas

    def generar_mensaje_cierre(self, datos_operacion):
        emoji = "🟢" if datos_operacion['resultado'] == "TP" else "🔴"
        color_emoji = "✅" if datos_operacion['resultado'] == "TP" else "❌"
        
        if datos_operacion['tipo'] == 'LONG':
            pnl_absoluto = datos_operacion['precio_salida'] - datos_operacion['precio_entrada']
        else:
            pnl_absoluto = datos_operacion['precio_entrada'] - datos_operacion['precio_salida']
        
        breakout_usado = "Si" if datos_operacion.get('breakout_usado', False) else "No"
        operacion_ejecutada = "Si" if datos_operacion.get('operacion_ejecutada', False) else "No"
        
        mensaje = f"""
{emoji} OPERACION CERRADA - {datos_operacion['symbol']}
{color_emoji} RESULTADO: {datos_operacion['resultado']}
Tipo: {datos_operacion['tipo']}
Entrada: {datos_operacion['precio_entrada']:.8f}
Salida: {datos_operacion['precio_salida']:.8f}
PnL Absoluto: {pnl_absoluto:.8f}
PnL %: {datos_operacion['pnl_percent']:.2f}%
Duracion: {datos_operacion['duracion_minutos']:.1f} minutos
Breakout+Reentry: {breakout_usado}
Operacion Bitget: {operacion_ejecutada}
Angulo: {datos_operacion['angulo_tendencia']:.1f}°
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
            return "Muy Debil", 1
        elif angulo_abs < 13:
            return "Debil", 2
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
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
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
                               title=f'{simbolo} | {tipo_operacion} | {config_optima["timeframe"]} | Bitget + Breakout+Reentry',
                               ylabel='Precio',
                               addplot=apds,
                               volume=False,
                               returnfig=True,
                               figsize=(14, 10),
                               panel_ratios=(3, 1))
            
            axes[2].set_ylim([0, 100])
            axes[2].grid(True, alpha=0.3)
            
            buf = BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
            buf.seek(0)
            plt.close(fig)
            
            return buf
        except Exception as e:
            logger.error(f"Error generando grafico: {e}")
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
                else:
                    logger.error(f"Error enviando grafico a {chat_id}: {r.status_code}")
            except Exception as e:
                logger.error(f"Error enviando grafico: {e}")
        
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
                if r.status_code != 200:
                    logger.error(f"Error enviando mensaje a {chat_id}: {r.status_code}")
            except Exception as e:
                logger.error(f"Error enviando mensaje: {e}")
                resultados.append(False)
        
        return any(resultados)

    def reoptimizar_periodicamente(self):
        try:
            horas_desde_opt = (datetime.now() - self.ultima_optimizacion).total_seconds() / 7200
            
            if self.operaciones_desde_optimizacion >= 8 or horas_desde_opt >= self.config.get('reevaluacion_horas', 24):
                logger.info("Iniciando re-optimizacion automatica...")
                ia = OptimizadorIA(log_path=self.log_path, min_samples=self.config.get('min_samples_optimizacion', 30))
                nuevos_parametros = ia.buscar_mejores_parametros()
                
                if nuevos_parametros:
                    self.actualizar_parametros(nuevos_parametros)
                    self.ultima_optimizacion = datetime.now()
                    self.operaciones_desde_optimizacion = 0
                    logger.info("Parametros actualizados en tiempo real")
        except Exception as e:
            logger.error(f"Error en re-optimizacion automatica: {e}")

    def actualizar_parametros(self, nuevos_parametros):
        self.config['trend_threshold_degrees'] = nuevos_parametros.get('trend_threshold_degrees', 
                                                                        self.config.get('trend_threshold_degrees', 16))
        self.config['min_trend_strength_degrees'] = nuevos_parametros.get('min_trend_strength_degrees', 
                                                                           self.config.get('min_trend_strength_degrees', 16))
        self.config['entry_margin'] = nuevos_parametros.get('entry_margin', 
                                                             self.config.get('entry_margin', 0.001))

    def ejecutar_analisis(self):
        if random.random() < 0.1:
            self.reoptimizar_periodicamente()
        
        cierres = self.verificar_cierre_operaciones()
        if cierres:
            logger.info(f"Operaciones cerradas: {', '.join(cierres)}")
        
        self.guardar_estado()
        return self.escanear_mercado()

    def mostrar_resumen_operaciones(self):
        logger.info(f"RESUMEN OPERACIONES:")
        logger.info(f"   Activas: {len(self.operaciones_activas)}")
        logger.info(f"   Esperando reentry: {len(self.esperando_reentry)}")
        logger.info(f"   Total ejecutadas: {self.total_operaciones}")
        
        if self.bitget_client:
            logger.info(f"   Bitget: Conectado")
        else:
            logger.info(f"   Bitget: No configurado")
        
        if self.operaciones_activas:
            for simbolo, op in self.operaciones_activas.items():
                estado = "LONG" if op['tipo'] == 'LONG' else "SHORT"
                ancho_canal = op.get('ancho_canal_porcentual', 0)
                timeframe = op.get('timeframe_utilizado', 'N/A')
                velas = op.get('velas_utilizadas', 0)
                breakout = "B" if op.get('breakout_usado', False) else ""
                ejecutada = "A" if op.get('operacion_ejecutada', False) else ""
                logger.info(f"   • {simbolo} {estado} {breakout} {ejecutada} - {timeframe} - {velas}v - Ancho: {ancho_canal:.1f}%")

    def iniciar(self):
        logger.info("=" * 70)
        logger.info("BOT DE TRADING - ESTRATEGIA BREAKOUT + REENTRY")
        logger.info("PRIORIDAD: TIMEFRAMES CORTOS (1m > 3m > 5m > 15m > 30m)")
        logger.info("PERSISTENCIA: ACTIVADA")
        logger.info("REEVALUACION: CADA 2 HORAS")
        logger.info("INTEGRACION: BITGET API")
        logger.info("=" * 70)
        logger.info(f"Simbolos: {len(self.config.get('symbols', []))} monedas")
        logger.info(f"Timeframes: {', '.join(self.config.get('timeframes', []))}")
        logger.info(f"Velas: {self.config.get('velas_options', [])}")
        logger.info(f"ANCHO MINIMO: {self.config.get('min_channel_width_percent', 4)}%")
        logger.info(f"Estrategia: 1) Detectar Breakout -> 2) Esperar Reentry -> 3) Confirmar con Stoch")
        
        if self.bitget_client:
            logger.info(f"BITGET: API Conectada")
            logger.info(f"Apalancamiento: {self.leverage_por_defecto}x")
            logger.info(f"Capital por operacion: ${self.capital_por_operacion}")
            
            if self.ejecutar_operaciones_automaticas:
                logger.info(f"AUTO-TRADING: ACTIVADO")
            else:
                logger.info(f"AUTO-TRADING: Solo senales")
        else:
            logger.info(f"BITGET: No configurado (solo senales)")
        
        logger.info("=" * 70)
        logger.info("INICIANDO BOT...")
        
        try:
            while True:
                nuevas_senales = self.ejecutar_analisis()
                self.mostrar_resumen_operaciones()
                
                minutos_espera = self.config.get('scan_interval_minutes', 1)
                logger.info(f"Analisis completado. Senales nuevas: {nuevas_senales}")
                logger.info(f"Proximo analisis en {minutos_espera} minutos...")
                logger.info("-" * 60)
                
                for minuto in range(minutos_espera):
                    time.sleep(60)
                    restantes = minutos_espera - (minuto + 1)
                    if restantes > 0 and restantes % 5 == 0:
                        logger.info(f"{restantes} minutos restantes...")
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
            logger.info("Guardando estado final...")
            self.guardar_estado()
            logger.info("Hasta pronto!")
        except Exception as e:
            logger.error(f"Error en el bot: {e}")
            logger.info("Intentando guardar estado...")
            try:
                self.guardar_estado()
            except Exception as e2:
                logger.error(f"Error guardando estado final: {e2}")


def crear_config_desde_entorno():
    """Configuracion desde variables de entorno"""
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
        'timeframes': ['5m', '15m', '30m', '1h','4h'],
        'velas_options': [80, 100, 120, 150, 200],
        'symbols': [
            'XMRUSDT','AAVEUSDT','DOTUSDT','LINKUSDT',
            'BNBUSDT','XRPUSDT','SOLUSDT','AVAXUSDT',
            'DOGEUSDT','LTCUSDT','ATOMUSDT','XLMUSDT',
            'ALGOUSDT','VETUSDT','ICPUSDT','FILUSDT',
            'BCHUSDT','NEOUSDT','TRXUSDT','XTZUSDT',
            'SUSHIUSDT','COMPUSDT','PEPEUSDT','ETCUSDT',
            'SNXUSDT','RENDERUSDT','1INCHUSDT','UNIUSDT',
            'ZILUSDT','HOTUSDT','ENJUSDT','HYPEUSDT',
            'BEATUSDT','PIPPINUSDT','ADAUSDT','ASTERUSDT',
            'ENAUSDT','TAOUSDT','LUNCUSDT','WLDUSDT',
            'WIFUSDT','APTUSDT','HBARUSDT','CRVUSDT',
            'LUNAUSDT','TIAUSDT','ARBUSDT','ONDOUSDT',
            'FOLKSUSDT','BRETTUSDT','TRUMPUSDT',
            'INJUSDT','ZECUSDT','NOTUSDT','SHIBUSDT',
            'LDOUSDT','KASUSDT','STRKUSDT','DYDXUSDT',
            'SEIUSDT','TONUSDT','NMRUSDT'
        ],
        'telegram_token': os.environ.get('TELEGRAM_TOKEN'),
        'telegram_chat_ids': telegram_chat_ids,
        'auto_optimize': True,
        'min_samples_optimizacion': 30,
        'reevaluacion_horas': 24,
        'log_path': os.path.join(directorio_actual, 'operaciones_log_v23.csv'),
        'estado_file': os.path.join(directorio_actual, 'estado_bot_v23.json'),
        'bitget_api_key': os.environ.get('BITGET_API_KEY'),
        'bitget_api_secret': os.environ.get('BITGET_SECRET_KEY'),
        'bitget_passphrase': os.environ.get('BITGET_PASSPHRASE'),
        'webhook_url': os.environ.get('WEBHOOK_URL'),
        'ejecutar_operaciones_automaticas': os.environ.get('EJECUTAR_OPERACIONES_AUTOMATICAS', 'false').lower() == 'true',
        'capital_por_operacion': float(os.environ.get('CAPITAL_POR_OPERACION', '4')),
        'leverage_por_defecto': min(int(os.environ.get('LEVERAGE_POR_DEFECTO', '10')), 10)
    }


app = Flask(__name__)

config = crear_config_desde_entorno()
bot = TradingBot(config)

def run_bot_loop():
    """Ejecuta el bot en un hilo separado"""
    logger.info("Iniciando hilo del bot...")
    while True:
        try:
            bot.ejecutar_analisis()
            time.sleep(bot.config.get('scan_interval_minutes', 1) * 60)
        except Exception as e:
            logger.error(f"Error en el hilo del bot: {e}", exc_info=True)
            time.sleep(60)

bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
bot_thread.start()

@app.route('/')
def index():
    return "Bot Breakout + Reentry con integracion Bitget esta en linea.", 200

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    if request.is_json:
        update = request.get_json()
        logger.info(f"Update recibido: {json.dumps(update)}")
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
        logger.error(f"Error en health check: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_telegram_webhook():
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        logger.warning("No hay token de Telegram configurado")
        return
    
    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        render_url = os.environ.get('RENDER_EXTERNAL_URL')
        if render_url:
            webhook_url = f"{render_url}/webhook"
        else:
            logger.warning("No hay URL de webhook configurada")
            return
    
    try:
        logger.info(f"Configurando webhook Telegram en: {webhook_url}")
        requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
        time.sleep(1)
        response = requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}", timeout=10)
        
        if response.status_code == 200:
            logger.info("Webhook de Telegram configurado correctamente")
        else:
            logger.error(f"Error configurando webhook: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error configurando webhook: {e}")

if __name__ == '__main__':
    logger.info("Iniciando aplicacion Flask...")
    setup_telegram_webhook()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
