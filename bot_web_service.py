# bot_web_service.py
# Adaptación para Render del bot Breakout + Reentry con correcciones Bitget
# CORRECCIÓN: Eliminados warnings de matplotlib para emojis
# ACTUALIZACIÓN: Configuración de mínimos de Bitget 2025
# INTEGRACIÓN: Cliente Bitget mejorado con precisión adaptativa para memecoins
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
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ config/bitget_config.py no disponible, usando valores por defecto")
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from io import BytesIO
from flask import Flask, request, jsonify
import threading
import logging

# Configuración mejorada de matplotlib para evitar warnings de emojis
def configurar_matplotlib():
    """Configura matplotlib para manejar emojis correctamente"""
    # Suprimir warnings de fuentes de manera más agresiva
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*font.*')
    warnings.filterwarnings('ignore', message='.*font.*warning.*')
    
    # Configurar fuentes con soporte para emojis y caracteres especiales
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    plt.rcParams['axes.facecolor'] = '#1a1a1a'
    
    # Configurar parámetros adicionales para evitar warnings
    matplotlib.rcParams['savefig.dpi'] = 100
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.facecolor'] = '#1a1a1a'
    
    # Configurar el logger de matplotlib para suprimir warnings
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Configurar matplotlib al inicio
configurar_matplotlib()

# Suprimir todos los warnings de matplotlib de forma global
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

tpsl_manager = create_tpsl_manager(
    api_key=os.environ.get('BITGET_API_KEY', ''),
    secret_key=os.environ.get('BITGET_SECRET_KEY', ''),
    passphrase=os.environ.get('BITGET_API_PASSPHRASE', '')
)
# ---------------------------
# [INICIO DEL CÓDIGO DEL BOT NUEVO]
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
            logger.warning("⚠ No se encontró operaciones_log.csv (optimizador)")
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
            logger.info(f"ℹ️ No hay suficientes datos para optimizar (se requieren {self.min_samples}, hay {len(self.datos)})")
            return None
        mejor_score = -1e9
        mejores_param = None
        trend_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
        strength_values = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
        margin_values = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01]
        combos = list(itertools.product(trend_values, strength_values, margin_values))
        total = len(combos)
        logger.info(f"🔎 Optimizador: probando {total} combinaciones...")
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
            logger.info("✅ Optimizador: mejores parámetros encontrados:", mejores_param)
            try:
                with open("mejores_parametros.json", "w", encoding='utf-8') as f:
                    json.dump(mejores_param, f, indent=2)
            except Exception as e:
                logger.error("⚠ Error guardando mejores_parametros.json:", e)
        else:
            logger.warning("⚠ No se encontró una configuración mejor")
        return mejores_param

# ---------------------------
# BITGET CLIENT - INTEGRACIÓN COMPLETA CON API BITGET CORREGIDA
# ---------------------------
class BitgetClient:
    """
    Cliente completo para interactuar con la API de Bitget Futures.
    Maneja autenticación, ejecución de órdenes, gestión de TP/SL y más.
    """
    
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self.product_type = "USDT-FUTURES"
        logger.info(f"Cliente Bitget inicializado con API Key: {api_key[:10]}...")

    def _generate_signature(self, timestamp, method, request_path, body=''):
        """
        Generar firma HMAC-SHA256 para autenticación.
        
        Args:
            timestamp: Timestamp en milisegundos
            method: Método HTTP (GET/POST)
            request_path: Ruta de la solicitud
            body: Cuerpo de la solicitud (dict o string JSON)
        
        Returns:
            str: Firma codificada en Base64
        """
        # SI body ya es string (JSON), NO lo tocamos
        if isinstance(body, str):
            body_str = body
        # SI es dict, lo convertimos UNA sola vez
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
        """
        Generar headers de autenticación para la API de Bitget.
        
        Args:
            method: Método HTTP
            request_path: Ruta de la solicitud
            body: Cuerpo de la solicitud (opcional)
        
        Returns:
            dict: Headers con firma y credenciales
        """
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
        """Verificar que las credenciales sean válidas - CORREGIDO CON MÚLTIPLES PRODUCT TYPES"""
        try:
            logger.info("🔐 Verificando credenciales Bitget...")
            
            if not self.api_key or not self.api_secret or not self.passphrase:
                logger.error("❌ Credenciales incompletas")
                return False
            
            product_types = ['USDT-FUTURES']
            
            for product_type in product_types:
                logger.info(f"🔍 Probando productType: {product_type}")
                
                try:
                    test_response = self.get_account_info(product_type)
                    if test_response is not None:
                        logger.info(f"✅ Credenciales verificadas exitosamente con productType: {product_type}")
                        self.product_type = product_type
                        
                        accounts = self.get_account_info()
                        if accounts:
                            for account in accounts:
                                if account.get('marginCoin') == 'USDT':
                                    available = float(account.get('available', 0))
                                    logger.info(f"💰 Balance disponible: {available:.2f} USDT")
                        
                        return True
                    else:
                        logger.warning(f"⚠️ ProductType {product_type} no funcionó")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error probando productType {product_type}: {e}")
                    continue
            
            logger.error("❌ No se pudo verificar credenciales con ningún productType")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error verificando credenciales: {e}")
            return False

    def get_account_info(self, product_type=None):
        """Obtener información de cuenta Bitget V2 - CORREGIDO"""
        try:
            if product_type is None:
                product_type = self.product_type
            
            request_path = '/api/v2/mix/account/accounts'
            params = {'productType': product_type, 'marginCoin': 'USDT'}
            
            query_string = f"?productType={product_type}&marginCoin=USDT"
            full_request_path = request_path + query_string
            
            headers = self._get_headers('GET', full_request_path, '')
            
            if not headers:
                logger.error("❌ No se pudieron generar headers para la solicitud")
                return None
            
            response = requests.get(
                f"{self.base_url}{request_path}",
                headers=headers,
                params=params,
                timeout=10
            )
            
            logger.info(f"📊 Respuesta cuenta - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    return data.get('data', [])
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"❌ Error API Bitget: {error_code} - {error_msg}")
                    
                    # Fallback a USDT-MIX
                    if error_code == '40020' and product_type == 'USDT-FUTURES':
                        logger.info(f"🔄 Error 40020 con productType {product_type}, intentando con USDT-MIX...")
                        return self.get_account_info('USDT-MIX')
            else:
                logger.error(f"❌ Error HTTP: {response.status_code} - {response.text}")
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en get_account_info: {e}")
            return None

    def get_symbol_info(self, symbol):
        """Obtener información del símbolo - CORREGIDO"""
        try:
            request_path = '/api/v2/mix/market/contracts'
            params = {'productType': self.product_type}
            
            query_string = f"?productType={self.product_type}"
            full_request_path = request_path + query_string
            
            headers = self._get_headers('GET', full_request_path, '')
            
            if not headers:
                logger.error("❌ No se pudieron generar headers")
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
                            logger.info(f"✅ Información de símbolo obtenida para {symbol}")
                            return contract
            
            logger.warning(f"⚠️ No se encontró información para {symbol} en {self.product_type}")
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo info del símbolo: {e}")
            return None

    def place_order(self, symbol, side, order_type, size, price=None, 
                    client_order_id=None, time_in_force='normal',
                    posSide=None, is_hedged_account=False,
                    stop_loss_price=None, take_profit_price=None):
        """
        Colocar orden de mercado o límite - CORREGIDO con TP/SL integrados
        
        Args:
            symbol: Símbolo de trading
            side: 'buy' o 'sell'
            order_type: Tipo de orden ('market' o 'limit')
            size: Tamaño de la orden
            price: Precio para órdenes límite
            posSide: 'long' o 'short' (para modo unilateral)
            is_hedged_account: Si la cuenta está en modo cobertura
            stop_loss_price: Precio de Stop Loss integrado
            take_profit_price: Precio de Take Profit integrado
        """
        try:
            logger.info(f"📤 Colocando orden: {symbol} {side} {size} {order_type}")
            
            request_path = '/api/v2/mix/order/place-order'
            
            # Diferenciar entre modo HEDGE y UNILATERAL
            if is_hedged_account:
                body = {
                    'symbol': symbol,
                    'productType': self.product_type,
                    'marginMode': 'isolated',
                    'marginCoin': 'USDT',
                    'side': side,
                    'orderType': order_type,
                    'size': str(size)
                }
                logger.info(f"📤 Orden en MODO HEDGE: side={side}, size={size} (sin posSide)")
            else:
                if not posSide:
                    logger.warning("⚠️ En modo unilateral, posSide no especificado, usando 'long' por defecto")
                    posSide = 'long'
                body = {
                    'symbol': symbol,
                    'productType': self.product_type,
                    'marginMode': 'isolated',
                    'marginCoin': 'USDT',
                    'side': side,
                    'orderType': order_type,
                    'size': str(size),
                    'posSide': posSide
                }
                logger.info(f"📤 Orden en MODO UNILATERAL: side={side}, posSide={posSide}, size={size}")
            
            if price:
                body['price'] = str(price)
            if client_order_id:
                body['clientOrderId'] = client_order_id
            
            # AGREGAR TP/SL DIRECTAMENTE EN LA ORDEN (nuevo)
            if stop_loss_price is not None:
                precision_sl = self.obtener_precision_adaptada(float(stop_loss_price), symbol)
                stop_loss_formatted = self.redondear_precio_manual(float(stop_loss_price), precision_sl, symbol)
                body["presetStopLossPrice"] = str(stop_loss_formatted)
                logger.info(f"🔧 SL integrado: presetStopLossPrice={stop_loss_formatted}")
            
            if take_profit_price is not None:
                precision_tp = self.obtener_precision_adaptada(float(take_profit_price), symbol)
                take_profit_formatted = self.redondear_precio_manual(float(take_profit_price), precision_tp, symbol)
                body["presetStopSurplusPrice"] = str(take_profit_formatted)
                logger.info(f"🔧 TP integrado: presetStopSurplusPrice={take_profit_formatted}")
            
            logger.debug(f"📦 Body de orden: {body}")
            
            # Normalizar body para usar en la solicitud
            body_str = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("❌ No se pudieron generar headers para la orden")
                return None
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"📥 Respuesta orden - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✅ Orden placed successfully: {data.get('data', {})}")
                    return data.get('data', {})
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"❌ Error en orden Bitget: {error_code} - {error_msg}")
                    logger.error(f"❌ Body enviado: {body}")
                    
                    if error_code == '40774':
                        logger.error(f"❌ Error 40774: Modo de posición conflictivo")
                    if error_code == '40034':
                        logger.error(f"❌ Error 40034: {error_msg}")
                    return None
            else:
                logger.error(f"❌ Error HTTP en orden: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")
            return None

    def place_tpsl_order(self, symbol, hold_side, trigger_price, order_type='stop_loss', 
                         stop_loss_price=None, take_profit_price=None):
        """
        Coloca orden de Stop Loss o Take Profit separada en Bitget Futuros.
        
        Args:
            symbol: Símbolo (ej: 'CRVUSDT')
            hold_side: 'long' o 'short'
            trigger_price: Precio de activación
            order_type: 'stop_loss' o 'take_profit'
            stop_loss_price: Precio de stop loss (opcional)
            take_profit_price: Precio de take profit (opcional)
        
        Returns:
            dict: Datos de la orden creada o None si hay error
        """
        try:
            request_path = '/api/v2/mix/order/place-pos-tpsl'
            
            # Usar precisión dinámica basada en el precio
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
                logger.info(f"🔧 SL para {symbol}: precio={stop_loss_price}, precision={precision_sl}, formatted={stop_loss_formatted}")
            elif order_type == 'take_profit' and take_profit_price:
                precision_tp = self.obtener_precision_adaptada(take_profit_price, symbol)
                take_profit_formatted = self.redondear_precio_manual(take_profit_price, precision_tp, symbol)
                body['stopSurplusTriggerPrice'] = take_profit_formatted
                logger.info(f"🔧 TP para {symbol}: precio={take_profit_price}, precision={precision_tp}, formatted={take_profit_formatted}")
            
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
                        logger.error(f"💡 Body enviado: {body}")
                    if data.get('code') == '40034':
                        logger.error(f"❌ Error 40034 en {order_type}: {data.get('msg')}")
                        logger.error(f"💡 Body enviado: {body}")
            
            logger.error(f"❌ Error creando {order_type}: {response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Error en place_tpsl_order: {e}")
            return None

    def place_plan_order(self, symbol, side, trigger_price, order_type, size, 
                         price=None, plan_type='loss_plan', trigger_type='mark_price',
                         hold_side='long'):
        """
        Colocar orden de plan (TP/SL) - CORREGIDO para API Bitget V2
        
        Args:
            symbol: Símbolo de trading (ej. 'BTCUSDT')
            side: 'buy' o 'sell'
            trigger_price: Precio de activación
            order_type: Tipo de orden ('market' o 'limit')
            size: Tamaño de la orden
            price: Precio de ejecución (para órdenes límite)
            plan_type: Tipo de plan ('profit_plan' para TP, 'loss_plan' para SL)
            trigger_type: Tipo de trigger ('mark_price' o 'latest_price')
            hold_side: 'long' o 'short'
        """
        try:
            logger.info(f"📤 Colocando orden plan: {symbol} {side} TP/SL en {trigger_price}")
            
            # Obtener la precisión correcta del símbolo para formatear precios
            price_precision = self.obtener_precision_precio(symbol)
            
            # Redondear triggerPrice según la precisión del símbolo
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
                'holdSide': hold_side  # CORREGIDO: Ahora hold_side se recibe como parámetro
            }
            
            if price:
                body['executePrice'] = str(price)
            
            logger.debug(f"📦 Body de plan order: {body}")
            
            # Normalizar body para usar en la solicitud
            body_str = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("❌ No se pudieron generar headers para plan order")
                return None
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"📥 Respuesta plan order - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✅ Plan order placed successfully")
                    return data.get('data', {})
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code', 'Unknown')
                    logger.error(f"❌ Error en plan order Bitget: {error_code} - {error_msg}")
                    return None
            else:
                logger.error(f"❌ Error HTTP en plan order: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Error placing plan order: {e}")
            return None

    def set_leverage(self, symbol, leverage, hold_side='long'):
        """Configurar apalancamiento en BITGET"""
        try:
            logger.info(f"⚡ Configurando leverage {leverage}x para {symbol} ({hold_side})")
            
            request_path = '/api/v2/mix/account/set-leverage'
            body = {
                'symbol': symbol,
                'productType': self.product_type,
                'marginCoin': 'USDT',
                'leverage': str(leverage),
                'holdSide': hold_side
            }
            
            # Normalizar body para usar en la solicitud
            body_str = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
            
            headers = self._get_headers('POST', request_path, body_str)
            
            if not headers:
                logger.error("❌ No se pudieron generar headers para leverage")
                return False
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body_str,
                timeout=10
            )
            
            logger.info(f"📥 Respuesta leverage - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✅ Apalancamiento {leverage}x configurado exitosamente")
                    return True
                else:
                    logger.error(f"❌ Error configurando leverage: {data.get('code')} - {data.get('msg')}")
                    
                    if data.get('code') == '45110':
                        logger.warning(f"⚠️ Apalancamiento {leverage}x muy alto, intentando con 5x")
                        return self.set_leverage(symbol, 5, hold_side)
                    
                    if data.get('code') == '40020':
                        logger.error(f"❌ Error 40020: ProductType {self.product_type} incorrecto para leverage")
                        return False
            else:
                logger.error(f"❌ Error HTTP configurando leverage: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"❌ Error en set_leverage: {e}")
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
            
            if product_type == 'USDT-FUTURES':
                return self.get_positions(symbol, 'USDT-MIX')
            
            return []
        except Exception as e:
            logger.error(f"❌ Error obteniendo posiciones: {e}")
            return []

    def obtener_saldo_cuenta(self):
        """Obtiene el saldo actual de la cuenta"""
        try:
            accounts = self.get_account_info()
            if accounts:
                for account in accounts:
                    if account.get('marginCoin') == 'USDT':
                        balance_usdt = float(account.get('available', 0))
                        logger.info(f"💰 Saldo disponible USDT: ${balance_usdt:.2f}")
                        return balance_usdt
            logger.warning("⚠️ No se pudo obtener saldo de la cuenta")
            return None
        except Exception as e:
            logger.error(f"❌ Error obteniendo saldo de cuenta: {e}")
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
            
            params['productType'] = 'USDT-MIX'
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
            logger.error(f"❌ Error en get_klines: {e}")
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
                logger.warning(f"No se pudo obtener info de {symbol}, usando 4 decimales por defecto")
                return 4
        except Exception as e:
            logger.error(f"Error obteniendo precisión de {symbol}: {e}")
            return 4

    def obtener_precision_adaptada(self, price, symbol):
        """
        Obtiene la precisión adaptativa basada en el precio para evitar redondeo a cero.
        
        Para símbolos con precios muy pequeños (SHIBUSDT, PEPE, ENS, XLM, etc.), la precisión
        de priceScale no es suficiente. Este método calcula la precisión necesaria
        para mantener al menos 4-6 dígitos significativos.
        
        Args:
            price: Precio a evaluar
            symbol: Símbolo de trading
        
        Returns:
            int: Número de decimales a usar
        """
        try:
            price = float(price)
            
            if price < 1:
                if price < 0.00001:
                    return 12  # Para PEPE, SHIB y similares
                elif price < 0.0001:
                    return 10  # Para memecoins extremos
                elif price < 0.001:
                    return 8   # Para memecoins y precios muy pequeños
                elif price < 0.01:
                    return 7   # Para precios como ENS (~0.008)
                elif price < 0.1:
                    return 6   # Para precios como PHA (~0.1)
                elif price < 1:
                    return 5   # Para precios como XLM (~0.2)
            else:
                return 4
                
        except Exception as e:
            logger.error(f"Error calculando precisión adaptativa: {e}")
            return 8

    def redondear_precio_manual(self, price, precision, symbol=None):
        """
        Redondea el precio con una precisión específica, asegurando que sea un múltiplo válido.
        
        Args:
            price: Precio a redondear
            precision: Número de decimales
            symbol: Símbolo opcional para obtener priceStep real del exchange
        
        Returns:
            str: Precio redondeado como string
        """
        try:
            price = float(price)
            if price == 0:
                return "0.0"
            
            # Si tenemos el símbolo, intentar obtener el priceStep real del exchange
            if symbol:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    price_scale = symbol_info.get('priceScale', 4)
                    tick_size = 10 ** (-price_scale)
                    
                    # Redondear matemáticamente al múltiplo más cercano del tick_size
                    precio_redondeado = round(price / tick_size) * tick_size
                    
                    # Usar formato para evitar errores de punto flotante
                    precio_formateado = f"{precio_redondeado:.{price_scale}f}"
                    
                    # Verificar que no sea cero
                    if float(precio_formateado) == 0.0 and price > 0:
                        nueva_scale = price_scale + 4
                        tick_size = 10 ** (-nueva_scale)
                        precio_redondeado = round(price / tick_size) * tick_size
                        precio_formateado = f"{precio_redondeado:.{nueva_scale}f}"
                    
                    logger.info(f"🔢 {symbol}: precio={price}, priceScale={price_scale}, resultado={precio_formateado}")
                    return precio_formateado
            
            # Fallback: usar la precisión proporcionada
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
        """
        Redondea el precio al priceStep correcto del símbolo.
        
        Args:
            price: Precio a redondear
            symbol: Símbolo de trading
        
        Returns:
            float: Precio redondeado
        """
        try:
            precision = self.obtener_precision_precio(symbol)
            price_step = 10 ** (-precision)
            precio_redondeado = round(price / price_step) * price_step
            return float(f"{precio_redondeado:.{precision}f}")
        except Exception as e:
            logger.error(f"Error redondeando a priceStep para {symbol}: {e}")
            return float(f"{price:.4f}")

    def redondear_precio_precision(self, price, symbol):
        """Redondea el precio a la precisión correcta para el símbolo"""
        try:
            precision = self.obtener_precision_precio(symbol)
            precio_redondeado = round(float(price), precision)
            logger.info(f"🔢 {symbol}: {price} → {precio_redondeado} (precisión: {precision} decimales)")
            return precio_redondeado
        except Exception as e:
            logger.error(f"Error redondeando precio para {symbol}: {e}")
            return float(price)

    def obtener_reglas_simbolo(self, symbol):
        """Obtiene las reglas específicas de tamaño para un símbolo"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"No se pudo obtener info de {symbol}, usando valores por defecto")
                
                # Usar configuración centralizada si está disponible
                if BITGET_CONFIG_AVAILABLE:
                    default_min_trade = get_minimum_size(symbol)
                    logger.info(f"📋 Usando configuración centralizada para {symbol}: {default_min_trade}")
                else:
                    # Fallback a valores por defecto
                    default_min_trade = 0.001
                    if 'BTC' in symbol:
                        default_min_trade = 0.001  # BTC/USDT: 0.001 BTC
                    elif 'ETH' in symbol:
                        default_min_trade = 0.01   # ETH/USDT: 0.01 ETH
                
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
            
            logger.info(f"📋 Reglas de {symbol}:")
            logger.info(f"  - sizeScale: {reglas['size_scale']}")
            logger.info(f"  - quantityScale: {reglas['quantity_scale']}")
            logger.info(f"  - minTradeNum: {reglas['min_trade_num']}")
            logger.info(f"  - sizeMultiplier: {reglas['size_multiplier']}")
            
            return reglas
            
        except Exception as e:
            logger.error(f"Error obteniendo reglas de {symbol}: {e}")
            # Valores por defecto actualizados según mínimos de Bitget 2025
            default_min_trade = 0.001
            if 'BTC' in symbol:
                default_min_trade = 0.001  # BTC/USDT: 0.001 BTC
            elif 'ETH' in symbol:
                default_min_trade = 0.01   # ETH/USDT: 0.01 ETH
                
            return {
                'size_scale': 0,
                'quantity_scale': 0,
                'min_trade_num': default_min_trade,
                'size_multiplier': 1,
                'delivery_mode': 0
            }

    def ajustar_tamaño_orden(self, symbol, cantidad_contratos, reglas):
        """Ajusta el tamaño de la orden según las reglas del símbolo"""
        try:
            size_scale = reglas['size_scale']
            quantity_scale = reglas['quantity_scale']
            min_trade_num = reglas['min_trade_num']
            size_multiplier = reglas['size_multiplier']
            
            # Determinar la escala a usar (prioridad: quantityScale > sizeScale)
            escala_actual = quantity_scale if quantity_scale > 0 else size_scale
            
            # Ajustar según la escala
            if escala_actual == 0:
                # Requiere entero
                cantidad_contratos = round(cantidad_contratos)
                logger.info(f"🔢 {symbol}: ajustado a entero = {cantidad_contratos}")
            elif escala_actual == 1:
                # 1 decimal permitido
                cantidad_contratos = round(cantidad_contratos, 1)
                logger.info(f"🔢 {symbol}: ajustado a 1 decimal = {cantidad_contratos}")
            elif escala_actual == 2:
                # 2 decimales permitidos
                cantidad_contratos = round(cantidad_contratos, 2)
                logger.info(f"🔢 {symbol}: ajustado a 2 decimales = {cantidad_contratos}")
            else:
                # Otros casos
                cantidad_contratos = round(cantidad_contratos, escala_actual)
                logger.info(f"🔢 {symbol}: ajustado a {escala_actual} decimales = {cantidad_contratos}")
            
            # Aplicar multiplicador si existe
            if size_multiplier > 1:
                cantidad_contratos = round(cantidad_contratos / size_multiplier) * size_multiplier
                logger.info(f"🔢 {symbol}: aplicado multiplicador {size_multiplier}x = {cantidad_contratos}")
            
            # Verificar mínimo
            if cantidad_contratos < min_trade_num:
                cantidad_contratos = min_trade_num
                logger.info(f"⚠️ {symbol}: ajustado a mínimo = {min_trade_num}")
            
            # Validación final
            if escala_actual == 0:
                if min_trade_num < 1 and min_trade_num > 0:
                    cantidad_contratos = max(1, int(round(cantidad_contratos)))
                    logger.info(f"🔢 {symbol}: caso especial - min decimal pero requiere entero = {cantidad_contratos}")
                else:
                    cantidad_contratos = int(round(cantidad_contratos))
                logger.info(f"✅ {symbol} final: {cantidad_contratos} (entero)")
            else:
                cantidad_contratos = round(cantidad_contratos, escala_actual)
                logger.info(f"✅ {symbol} final: {cantidad_contratos} ({escala_actual} decimales)")
            
            return cantidad_contratos
            
        except Exception as e:
            logger.error(f"Error ajustando tamaño para {symbol}: {e}")
            return int(round(cantidad_contratos))

    def set_hedged_mode(self, symbol, hedged_mode=True):
        """
        Configurar el modo de posición (hedge/unilateral) para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            hedged_mode: True para modo cobertura, False para unilateral
        
        Returns:
            bool: True si se configuró correctamente
        """
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
            
            logger.info(f"Respuesta set_position_mode: {response.status_code} - {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    logger.info(f"✓ Modo configurado para {symbol}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error configurando modo de posición: {e}")
            return False

# ---------------------------
# FUNCIONES DE OPERACIONES BITGET CORREGIDAS
# ---------------------------
def ejecutar_operacion_bitget(bitget_client, simbolo, tipo_operacion, capital_usd, leverage=10):
    """
    Ejecutar una operación completa en Bitget (posición + TP/SL) - CORREGIDO
    """
    
    logger.info("🚀 EJECUTANDO OPERACIÓN REAL EN BITGET")
    logger.info(f"📈 Símbolo: {simbolo}")
    logger.info(f"🎯 Tipo: {tipo_operacion}")
    logger.info(f"⚡ Apalancamiento: {leverage}x")
    logger.info(f"💰 Capital: ${capital_usd}")
    
    try:
        hold_side = 'long' if tipo_operacion == 'LONG' else 'short'
        leverage = min(leverage, 10)
        
        leverage_ok = bitget_client.set_leverage(simbolo, leverage, hold_side)
        if not leverage_ok:
            logger.error("❌ Error configurando apalancamiento")
            logger.info("🔄 Intentando con apalancamiento 5x...")
            leverage_ok = bitget_client.set_leverage(simbolo, 5, hold_side)
            if not leverage_ok:
                logger.error("❌ No se pudo configurar apalancamiento después de múltiples intentos")
                return None
            leverage = 5
        
        time.sleep(1)
        
        klines = bitget_client.get_klines(simbolo, '1m', 1)
        if not klines or len(klines) == 0:
            logger.error(f"❌ No se pudo obtener precio de {simbolo}")
            try:
                url = "https://api.binance.com/api/v3/ticker/price"
                params = {'symbol': simbolo}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    precio_actual = float(data['price'])
                    logger.info(f"💰 Precio actual (de Binance): {precio_actual:.8f}")
                else:
                    logger.error("❌ No se pudo obtener precio de ninguna fuente")
                    return None
            except Exception as e:
                logger.error(f"❌ Error obteniendo precio de Binance: {e}")
                return None
        else:
            klines.reverse()
            precio_actual = float(klines[0][4])
            logger.info(f"💰 Precio actual: {precio_actual:.8f}")
        
        symbol_info = bitget_client.get_symbol_info(simbolo)
        if not symbol_info:
            logger.warning(f"⚠️ No se pudo obtener info de {simbolo} de Bitget, usando valores por defecto")
            size_multiplier = 1
            
            # Usar configuración centralizada si está disponible
            if BITGET_CONFIG_AVAILABLE:
                min_trade_num = get_minimum_size(simbolo)
                price_place = get_price_precision(simbolo)
                logger.info(f"📋 Usando configuración centralizada para {simbolo}: min={min_trade_num}, prec={price_place}")
            else:
                # Fallback a valores por defecto
                min_trade_num = 0.001  # Por defecto para la mayoría de símbolos
                if 'BTC' in simbolo:
                    min_trade_num = 0.001  # BTC/USDT: 0.001 BTC
                elif 'ETH' in simbolo:
                    min_trade_num = 0.01   # ETH/USDT: 0.01 ETH
                price_place = 8
        else:
            size_multiplier = float(symbol_info.get('sizeMultiplier', 1))
            min_trade_num = float(symbol_info.get('minTradeNum', 0.001))
            price_place = int(symbol_info.get('pricePlace', 8))
        
        cantidad_usd = capital_usd * leverage
        cantidad_contratos = cantidad_usd / precio_actual
        cantidad_contratos = round(cantidad_contratos / size_multiplier) * size_multiplier
        
        # Validación final: asegurar que cumple con los mínimos de Bitget
        if cantidad_contratos < min_trade_num:
            logger.warning(f"⚠️ Cantidad calculada ({cantidad_contratos}) menor al mínimo ({min_trade_num}) - ajustando")
            cantidad_contratos = min_trade_num
            logger.info(f"✅ Cantidad ajustada al mínimo permitido: {cantidad_contratos}")
        
        cantidad_contratos = round(cantidad_contratos, 8)
        
        logger.info(f"📊 Cantidad: {cantidad_contratos} contratos")
        logger.info(f"💵 Valor nocional: ${cantidad_contratos * precio_actual:.2f}")
        
        if tipo_operacion == "LONG":
            sl_porcentaje = 0.02
            tp_porcentaje = 0.04
            stop_loss = precio_actual * (1 - sl_porcentaje)
            take_profit = precio_actual * (1 + tp_porcentaje)
        else:
            sl_porcentaje = 0.02
            tp_porcentaje = 0.04
            stop_loss = precio_actual * (1 + sl_porcentaje)
            take_profit = precio_actual * (1 - tp_porcentaje)
        
        stop_loss = round(stop_loss, price_place)
        take_profit = round(take_profit, price_place)
        
        logger.info(f"🛑 Stop Loss: {stop_loss:.8f}")
        logger.info(f"🎯 Take Profit: {take_profit:.8f}")
        
        side = 'buy' if tipo_operacion == 'LONG' else 'sell'
        
        # Usar el nuevo método place_order con TP/SL integrados
        orden_entrada = bitget_client.place_order(
            symbol=simbolo,
            side=side,
            order_type='market',
            size=cantidad_contratos,
            posSide=hold_side,
            is_hedged_account=False,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit
        )
        
        if not orden_entrada:
            logger.error("❌ Error abriendo posición")
            return None
        
        logger.info(f"✅ Posición abierta exitosamente")
        time.sleep(1)
        
        # Colocar órdenes de TP/SL separadas como backup
        sl_side = 'sell' if tipo_operacion == 'LONG' else 'buy'
        
        orden_sl = bitget_client.place_tpsl_order(
            symbol=simbolo,
            hold_side=hold_side,
            trigger_price=stop_loss,
            order_type='stop_loss',
            stop_loss_price=stop_loss
        )
        
        if orden_sl:
            logger.info(f"✅ Stop Loss separado configurado en: {stop_loss:.8f}")
        else:
            logger.warning("⚠️ Error configurando Stop Loss separado")
        
        time.sleep(0.5)
        
        orden_tp = bitget_client.place_tpsl_order(
            symbol=simbolo,
            hold_side=hold_side,
            trigger_price=take_profit,
            order_type='take_profit',
            take_profit_price=take_profit
        )
        
        if orden_tp:
            logger.info(f"✅ Take Profit separado configurado en: {take_profit:.8f}")
        else:
            logger.warning("⚠️ Error configurando Take Profit separado")
        
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
        
        logger.info("✅ OPERACIÓN EJECUTADA EXITOSAMENTE")
        logger.info(f"📋 ID Orden: {orden_entrada.get('orderId', 'N/A')}")
        logger.info(f"📊 Contratos: {cantidad_contratos}")
        logger.info(f"💰 Entrada: {precio_actual:.8f}")
        logger.info(f"🛑 SL: {stop_loss:.8f} (-2%)")
        logger.info(f"🎯 TP: {take_profit:.8f} (+4%)")
        logger.info(f"⚡ Leverage: {leverage}x")
        logger.info(f"💵 Capital usado: ${capital_usd}")
        
        return operacion_data
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando operación en Bitget: {e}")
        import traceback
        logger.error("Traceback completo:")
        logger.error(traceback.format_exc())
        return None

# ---------------------------
# CLASE PRINCIPAL DEL BOT DE TRADING
# ---------------------------

class TradingBot:
    def __init__(self, config):
        self.config = config
        self.bitget_client = None
        self.operaciones_activas = {}
        self.operaciones_cerradas = []
        self.senales_enviadas = set()
        self.total_operaciones = 0
        self.operaciones_desde_optimizacion = 0
        self.ultima_optimizacion = datetime.now()
        self.esperando_reentry = {}
        self.breakouts_detectados = {}
        self.breakout_history = {}
        self.ultimo_reporte_semanal = None
        self.dias_sin_operar = 0
        self.bitget_client = self._init_bitget_client()
        self.ejecutar_operaciones_automaticas = config.get('ejecutar_operaciones_automaticas', False)
        self.capital_por_operacion = config.get('capital_por_operacion', 4)
        self.leverage_por_defecto = config.get('leverage_por_defecto', 10)
        self.archivo_log = config.get('log_path', 'operaciones_log.csv')
        self.inicializar_log()
        self.cargar_estado()
        self.config_optima_por_simbolo = self.obtener_configuracion_optima()
        
        logger.info(f"🤖 Bot inicializado - Auto-trading: {'ACTIVADO' if self.ejecutar_operaciones_automaticas else 'DESACTIVADO'}")
        logger.info(f"💰 Capital por operación: ${self.capital_por_operacion}")
        logger.info(f"⚡ Leverage por defecto: {self.leverage_por_defecto}x")

    def _init_bitget_client(self):
        """Inicializar cliente de Bitget con credenciales de entorno"""
        api_key = self.config.get('bitget_api_key') or os.environ.get('BITGET_API_KEY')
        api_secret = self.config.get('bitget_api_secret') or os.environ.get('BITGET_SECRET_KEY')
        passphrase = self.config.get('bitget_passphrase') or os.environ.get('BITGET_PASSPHRASE')
        
        if api_key and api_secret and passphrase:
            logger.info("🔑 Credenciales de Bitget encontradas en variables de entorno")
            return BitgetClient(api_key, api_secret, passphrase)
        else:
            logger.warning("⚠️ No se encontraron credenciales de Bitget")
            return None

    def obtener_datos_mercado_config(self, simbolo, timeframe, num_velas):
        """Obtener datos de mercado con configuración de velas y timeframe"""
        try:
            datos_binance = obtener_datos_binance(simbolo, timeframe, num_velas)
            if datos_binance:
                return datos_binance
            
            if self.bitget_client:
                datos_bitget = self.bitget_client.get_klines(simbolo, timeframe, num_velas)
                if datos_bitget:
                    return convertir_klines_bitget(datos_bitget)
            
            return None
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado para {simbolo}: {e}")
            return None

    def calcular_canal_regresion_config(self, datos_mercado, num_velas=None):
        """Calcular canal de regresión lineal con configuración específica"""
        try:
            cierres = datos_mercado['cierres']
            tiempos = list(range(len(cierres)))
            
            x = np.array(tiempos)
            y = np.array(cierres)
            
            pendiente, intercepto = self.calcular_regresion_lineal(x, y)
            if pendiente is None:
                return None
            
            predicciones = pendiente * x + intercepto
            residuos = y - predicciones
            
            resistencia = max(predicciones + residuos)
            soporte = min(predicciones + residuos)
            
            precio_actual = cierres[-1]
            ancho_pips = (resistencia - soporte) / precio_actual * 100
            
            pearson, angulo_grados = self.calcular_pearson_y_angulo(x, cierres)
            
            fuerza_texto, nivel_fuerza = self.clasificar_fuerza_tendencia(angulo_grados)
            direccion = self.determinar_direccion_tendencia(angulo_grados)
            
            r2 = self.calcular_r2(cierres, x, pendiente, intercepto)
            
            stoch_k, stoch_d = self.calcular_stochastic(datos_mercado)
            
            info = {
                'pendiente': pendiente,
                'intercepto': intercepto,
                'resistencia': resistencia,
                'soporte': soporte,
                'pendiente_resistencia': pendiente,
                'pendiente_soporte': pendiente,
                'ancho_canal': resistencia - soporte,
                'ancho_canal_porcentual': ancho_pips,
                'coeficiente_pearson': abs(pearson),
                'angulo_tendencia': abs(angulo_grados),
                'fuerza_texto': fuerza_texto,
                'nivel_fuerza': nivel_fuerza,
                'direccion': direccion,
                'r2_score': r2,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'precio_actual': datos_mercado['precio_actual']
            }
            
            return info
        except Exception as e:
            logger.error(f"Error calculando canal de regresión: {e}")
            return None

    def detectar_breakout(self, simbolo, info_canal, datos):
        """Detectar breakout en resistencia o soporte"""
        try:
            precio_actual = datos['precio_actual']
            resistencia = info_canal['resistencia']
            soporte = info_canal['soporte']
            
            if info_canal['nivel_fuerza'] < 3:
                return None
            
            precio_resistencia = precio_actual - resistencia
            precio_soporte = precio_actual - soporte
            
            if abs(precio_resistencia) < abs(precio_soporte):
                if info_canal['direccion'] == "🟢 ALCISTA":
                    if precio_actual > resistencia:
                        return "LONG"
            else:
                if info_canal['direccion'] == "🔴 BAJISTA":
                    if precio_actual < soporte:
                        return "SHORT"
            
            return None
        except Exception as e:
            logger.error(f"Error detectando breakout: {e}")
            return None

    def detectar_reentry(self, simbolo, info_canal, datos):
        """Detectar reentry después de breakout"""
        try:
            precio_actual = datos['precio_actual']
            resistencia = info_canal['resistencia']
            soporte = info_canal['soporte']
            
            if simbolo not in self.esperando_reentry:
                return None
            
            info_breakout = self.esperando_reentry[simbolo]
            tipo_breakout = info_breakout['tipo']
            
            margen = self.config.get('entry_margin', 0.001)
            
            if tipo_breakout == "LONG":
                if info_canal['stoch_k'] < 30:
                    if precio_actual >= resistencia and precio_actual <= resistencia * (1 + margen):
                        return "LONG"
            else:
                if info_canal['stoch_k'] > 70:
                    if precio_actual <= soporte and precio_actual >= soporte * (1 - margen):
                        return "SHORT"
            
            return None
        except Exception as e:
            logger.error(f"Error detectando reentry: {e}")
            return None

    def calcular_niveles_entrada(self, tipo_operacion, info_canal, precio_actual):
        """Calcular niveles de entrada, TP y SL"""
        try:
            sl_porcentaje = 0.02
            tp_porcentaje = 0.04
            
            if tipo_operacion == "LONG":
                precio_entrada = precio_actual
                stop_loss = precio_actual * (1 - sl_porcentaje)
                take_profit = precio_actual * (1 + tp_porcentaje)
            else:
                precio_entrada = precio_actual
                stop_loss = precio_actual * (1 + sl_porcentaje)
                take_profit = precio_actual * (1 - tp_porcentaje)
            
            return precio_entrada, take_profit, stop_loss
        except Exception as e:
            logger.error(f"Error calculando niveles de entrada: {e}")
            return None, None, None

    def guardar_estado(self):
        """Guardar estado del bot"""
        try:
            estado = {
                'operaciones_activas': self.operaciones_activas,
                'senales_enviadas': list(self.senales_enviadas),
                'total_operaciones': self.total_operaciones,
                'operaciones_desde_optimizacion': self.operaciones_desde_optimizacion,
                'ultima_optimizacion': self.ultima_optimizacion.isoformat(),
                'esperando_reentry': {k: {
                    'tipo': v['tipo'],
                    'timestamp': v['timestamp'].isoformat(),
                    'precio_breakout': v['precio_breakout'],
                    'config': v['config']
                } for k, v in self.esperando_reentry.items()},
                'breakouts_detectados': {k: {
                    'tipo': v['tipo'],
                    'timestamp': v['timestamp'].isoformat(),
                    'precio_breakout': v['precio_breakout']
                } for k, v in self.breakouts_detectados.items()}
            }
            
            with open(self.config.get('estado_file', 'estado_bot.json'), 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2, default=str)
            
            logger.info("💾 Estado del bot guardado")
        except Exception as e:
            logger.error(f"❌ Error guardando estado: {e}")

    def cargar_estado(self):
        """Cargar estado anterior del bot"""
        try:
            estado_file = self.config.get('estado_file', 'estado_bot.json')
            if os.path.exists(estado_file):
                with open(estado_file, 'r', encoding='utf-8') as f:
                    estado = json.load(f)
                
                self.operaciones_activas = estado.get('operaciones_activas', {})
                self.senales_enviadas = set(estado.get('senales_enviadas', []))
                self.total_operaciones = estado.get('total_operaciones', 0)
                self.operaciones_desde_optimizacion = estado.get('operaciones_desde_optimizacion', 0)
                self.ultima_optimizacion = datetime.fromisoformat(estado.get('ultima_optimizacion', datetime.now().isoformat()))
                self.esperando_reentry = {k: {
                    'tipo': v['tipo'],
                    'timestamp': datetime.fromisoformat(v['timestamp']),
                    'precio_breakout': v['precio_breakout'],
                    'config': v['config']
                } for k, v in estado.get('esperando_reentry', {}).items()}
                self.breakouts_detectados = {k: {
                    'tipo': v['tipo'],
                    'timestamp': datetime.fromisoformat(v['timestamp']),
                    'precio_breakout': v['precio_breakout']
                } for k, v in estado.get('breakouts_detectados', {}).items()}
                
                logger.info(f"✅ Estado cargado: {len(self.operaciones_activas)} operaciones activas, {len(self.senales_enviadas)} señales enviadas")
        except Exception as e:
            logger.error(f"❌ Error cargando estado: {e}")

    def obtener_configuracion_optima(self):
        """Obtener configuración óptima desde archivo JSON"""
        try:
            if os.path.exists("mejores_parametros.json"):
                with open("mejores_parametros.json", 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                config_optima = {
                    'trend_threshold_degrees': params.get('trend_threshold_degrees', 16.0),
                    'min_trend_strength_degrees': params.get('min_trend_strength_degrees', 16.0),
                    'entry_margin': params.get('entry_margin', 0.001)
                }
                
                logger.info(f"📊 Parámetros óptimos cargados: {config_optima}")
                return config_optima
        except Exception as e:
            logger.error(f"Error cargando parámetros óptimos: {e}")
        
        return None

    def enviar_alerta_breakout(self, simbolo, tipo_breakout, info_canal, datos, config):
        """Enviar alerta de breakout detectado"""
        try:
            token = self.config.get('telegram_token')
            chat_ids = self.config.get('telegram_chat_ids', [])
            
            if token and chat_ids:
                mensaje = f"""
🚨 <b>BREAKOUT DETECTADO - {simbolo}</b>
📈 Tipo: {tipo_breakout}
💰 Precio: {datos['precio_actual']:.8f}
📊 Canal: {info_canal['ancho_canal_porcentual']:.1f}%
📏 Ángulo: {info_canal['angulo_tendencia']:.1f}°
⏰ Timeframe: {config.get('timeframe', 'N/A')}
                """
                
                self._enviar_telegram_simple(mensaje, token, chat_ids)
        except Exception as e:
            logger.error(f"Error enviando alerta de breakout: {e}")

    def escanear_mercado(self):
        """Escanear mercado en busca de oportunidades"""
        simbolos = self.config.get('symbols', [])
        timeframes = self.config.get('timeframes', ['5m', '15m', '30m', '1h'])
        velas_options = self.config.get('velas_options', [80, 100, 120, 150, 200])
        
        logger.info(f"\n🔍 Escaneando mercado: {len(simbolos)} símbolos en {len(timeframes)} timeframes...")
        
        senales_encontradas = 0
        
        for simbolo in simbolos:
            if simbolo in self.senales_enviadas:
                continue
            
            try:
                datos_mercado = self.obtener_datos_mercado_config(simbolo, '5m', 80)
                if not datos_mercado:
                    logger.warning(f"   ❌ {simbolo} - Error obteniendo datos")
                    continue
                
                info_canal = self.calcular_canal_regresion_config(datos_mercado, 80)
                if not info_canal:
                    logger.warning(f"   ❌ {simbolo} - Error calculando canal")
                    continue
                
                if info_canal['ancho_canal_porcentual'] < self.config.get('min_channel_width_percent', 4):
                    continue
                
                estado_stoch = ""
                if info_canal['stoch_k'] <= 30:
                    estado_stoch = "📉 OVERSOLD"
                elif info_canal['stoch_k'] >= 70:
                    estado_stoch = "📈 OVERBOUGHT"
                else:
                    estado_stoch = "➖ NEUTRO"
                
                precio_actual = datos_mercado['precio_actual']
                resistencia = info_canal['resistencia']
                soporte = info_canal['soporte']
                
                if precio_actual > resistencia:
                    posicion = "🔼 FUERA (arriba)"
                elif precio_actual < soporte:
                    posicion = "🔽 FUERA (abajo)"
                else:
                    posicion = "📍 DENTRO"
                
                logger.info(
                    f"📊 {simbolo} - 5m - 80v | "
                    f"{info_canal['direccion']} ({info_canal['angulo_tendencia']:.1f}° - {info_canal['fuerza_texto']}) | "
                    f"Ancho: {info_canal['ancho_canal_porcentual']:.1f}% - Stoch: {info_canal['stoch_k']:.1f}/{info_canal['stoch_d']:.1f} {estado_stoch} | "
                    f"Precio: {posicion}"
                )
                
                if (info_canal['nivel_fuerza'] < 2 or 
                    abs(info_canal['coeficiente_pearson']) < 0.4 or 
                    info_canal['r2_score'] < 0.4):
                    continue
                
                if simbolo not in self.esperando_reentry:
                    tipo_breakout = self.detectar_breakout(simbolo, info_canal, datos_mercado)
                    if tipo_breakout:
                        self.esperando_reentry[simbolo] = {
                            'tipo': tipo_breakout,
                            'timestamp': datetime.now(),
                            'precio_breakout': precio_actual,
                            'config': {'timeframe': '5m', 'num_velas': 80}
                        }
                        self.breakouts_detectados[simbolo] = {
                            'tipo': tipo_breakout,
                            'timestamp': datetime.now(),
                            'precio_breakout': precio_actual
                        }
                        logger.info(f"     🎯 {simbolo} - Breakout registrado, esperando reingreso...")
                        self.enviar_alerta_breakout(simbolo, tipo_breakout, info_canal, datos_mercado, {'timeframe': '5m'})
                        continue
                
                tipo_operacion = self.detectar_reentry(simbolo, info_canal, datos_mercado)
                if not tipo_operacion:
                    continue
                
                precio_entrada, tp, sl = self.calcular_niveles_entrada(
                    tipo_operacion, info_canal, datos_mercado['precio_actual']
                )
                
                if not precio_entrada or not tp or not sl:
                    continue
                
                if simbolo in self.breakout_history:
                    ultimo_breakout = self.breakout_history[simbolo]
                    tiempo_desde_ultimo = (datetime.now() - ultimo_breakout).total_seconds() / 3600
                    if tiempo_desde_ultimo < 2:
                        logger.info(f"   ⏳ {simbolo} - Señal reciente, omitiendo...")
                        continue
                
                breakout_info = self.esperando_reentry[simbolo]
                self.generar_senal_operacion(
                    simbolo, tipo_operacion, precio_entrada, tp, sl, 
                    info_canal, datos_mercado, {'timeframe': '5m', 'num_velas': 80}, breakout_info
                )
                
                senales_encontradas += 1
                self.breakout_history[simbolo] = datetime.now()
                del self.esperando_reentry[simbolo]
                
            except Exception as e:
                logger.error(f"⚠️ Error analizando {simbolo}: {e}")
                continue
        
        if self.esperando_reentry:
            logger.info(f"\n⏳ Esperando reingreso en {len(self.esperando_reentry)} símbolos:")
            for simbolo, info in self.esperando_reentry.items():
                tiempo_espera = (datetime.now() - info['timestamp']).total_seconds() / 60
                logger.info(f"   • {simbolo} - {info['tipo']} - Esperando {tiempo_espera:.1f} min")
        
        if self.breakouts_detectados:
            logger.info(f"\n⏰ Breakouts detectados recientemente:")
            for simbolo, info in self.breakouts_detectados.items():
                tiempo_desde_deteccion = (datetime.now() - info['timestamp']).total_seconds() / 60
                logger.info(f"   • {simbolo} - {info['tipo']} - Hace {tiempo_desde_deteccion:.1f} min")
        
        if senales_encontradas > 0:
            logger.info(f"✅ Se encontraron {senales_encontradas} señales de trading")
        else:
            logger.info("❌ No se encontraron señales en este ciclo")
        
        return senales_encontradas

    def generar_senal_operacion(self, simbolo, tipo_operacion, precio_entrada, tp, sl,
                            info_canal, datos_mercado, config_optima, breakout_info=None):
        """Genera y envía señal de operación con info de breakout"""
        if simbolo in self.senales_enviadas:
            return
        
        if precio_entrada is None or tp is None or sl is None:
            logger.warning(f"    ❌ Niveles inválidos para {simbolo}, omitiendo señal")
            return
        
        riesgo = abs(precio_entrada - sl)
        beneficio = abs(tp - precio_entrada)
        ratio_rr = beneficio / riesgo if riesgo > 0 else 0
        
        sl_percent = abs((sl - precio_entrada) / precio_entrada) * 100
        tp_percent = abs((tp - precio_entrada) / precio_entrada) * 100
        
        stoch_estado = "📉 SOBREVENTA" if tipo_operacion == "LONG" else "📈 SOBRECOMPRA"
        breakout_texto = ""
        
        if breakout_info:
            tiempo_breakout = (datetime.now() - breakout_info['timestamp']).total_seconds() / 60
            breakout_texto = f"""
🚀 <b>BREAKOUT + REENTRY DETECTADO:</b>
⏰ Tiempo desde breakout: {tiempo_breakout:.1f} minutos
💰 Precio breakout: {breakout_info['precio_breakout']:.8f}
"""
        
        mensaje = f"""
🎯 <b>SEÑAL DE {tipo_operacion} - {simbolo}</b>
{breakout_texto}
⏱️ <b>Configuración:</b>
📊 Timeframe: {config_optima['timeframe']}
🕯️ Velas: {config_optima.get('num_velas', 80)}
📏 Ancho Canal: {info_canal['ancho_canal_porcentual']:.1f}% ⭐
💰 <b>Precio Actual:</b> {datos_mercado['precio_actual']:.8f}
🎯 <b>Entrada:</b> {precio_entrada:.8f}
🛑 <b>Stop Loss:</b> {sl:.8f}
🎯 <b>Take Profit:</b> {tp:.8f}
📊 <b>Ratio R/B:</b> {ratio_rr:.2f}:1
🎯 <b>SL:</b> {sl_percent:.2f}%
🎯 <b>TP:</b> {tp_percent:.2f}%
💰 <b>Riesgo:</b> {riesgo:.8f}
🎯 <b>Beneficio Objetivo:</b> {beneficio:.8f}
📈 <b>Tendencia:</b> {info_canal['direccion']}
💪 <b>Fuerza:</b> {info_canal['fuerza_texto']}
📏 <b>Ángulo:</b> {info_canal['angulo_tendencia']:.1f}°
📊 <b>Pearson:</b> {info_canal['coeficiente_pearson']:.3f}
🎯 <b>R² Score:</b> {info_canal['r2_score']:.3f}
🎰 <b>Stochástico:</b> {stoch_estado}
📊 <b>Stoch K:</b> {info_canal['stoch_k']:.1f}
📈 <b>Stoch D:</b> {info_canal['stoch_d']:.1f}
⏰ <b>Hora:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💡 <b>Estrategia:</b> BREAKOUT + REENTRY con confirmación Stochastic
        """
        
        token = self.config.get('telegram_token')
        chat_ids = self.config.get('telegram_chat_ids', [])
        
        if token and chat_ids:
            try:
                logger.info(f"     📊 Generando gráfico para {simbolo}...")
                buf = self.generar_grafico_profesional(simbolo, info_canal, datos_mercado, 
                                                      precio_entrada, tp, sl, tipo_operacion)
                if buf:
                    logger.info(f"     📨 Enviando gráfico por Telegram...")
                    self.enviar_grafico_telegram(buf, token, chat_ids)
                    time.sleep(1)
                
                self._enviar_telegram_simple(mensaje, token, chat_ids)
                logger.info(f"     ✅ Señal {tipo_operacion} para {simbolo} enviada")
            except Exception as e:
                logger.error(f"     ❌ Error enviando señal: {e}")
        
        if self.ejecutar_operaciones_automaticas and self.bitget_client:
            logger.info(f"     🤖 Ejecutando operación automática en Bitget...")
            try:
                operacion_bitget = ejecutar_operacion_bitget(
                    bitget_client=self.bitget_client,
                    simbolo=simbolo,
                    tipo_operacion=tipo_operacion,
                    capital_usd=self.capital_por_operacion,
                    leverage=self.leverage_por_defecto
                )
                
                if operacion_bitget:
                    logger.info(f"     ✅ Operación ejecutada en Bitget para {simbolo}")
                    mensaje_confirmacion = f"""
🤖 <b>OPERACIÓN AUTOMÁTICA EJECUTADA - {simbolo}</b>
✅ <b>Status:</b> EJECUTADA EN BITGET
📊 <b>Tipo:</b> {tipo_operacion}
💰 <b>Capital:</b> ${self.capital_por_operacion}
⚡ <b>Apalancamiento:</b> {self.leverage_por_defecto}x
🎯 <b>Entrada:</b> {operacion_bitget['precio_entrada']:.8f}
🛑 <b>Stop Loss:</b> {operacion_bitget['stop_loss']:.8f}
🎯 <b>Take Profit:</b> {operacion_bitget['take_profit']:.8f}
📋 <b>ID Orden:</b> {operacion_bitget['orden_entrada'].get('orderId', 'N/A')}
⏰ <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    self._enviar_telegram_simple(mensaje_confirmacion, token, chat_ids)
                else:
                    logger.error(f"     ❌ Error ejecutando operación en Bitget para {simbolo}")
            except Exception as e:
                logger.error(f"     ⚠️ Error en ejecución automática: {e}")
        
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
            'velas_utilizadas': config_optima.get('num_velas', 80),
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
            logger.info(f"📝 Operación registrada en log: {datos_operacion['symbol']}")
        except Exception as e:
            logger.error(f"❌ Error registrando operación en log: {e}")

    def filtrar_operaciones_ultima_semana(self):
        """Filtra operaciones de los últimos 7 días"""
        if not os.path.exists(self.archivo_log):
            return []
        try:
            ops_recientes = []
            fecha_limite = datetime.now() - timedelta(days=7)
            with open(self.archivo_log, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        if timestamp >= fecha_limite:
                            ops_recientes.append({
                                'timestamp': timestamp,
                                'symbol': row['symbol'],
                                'resultado': row['resultado'],
                                'pnl_percent': float(row['pnl_percent']),
                                'tipo': row['tipo'],
                                'breakout_usado': row.get('breakout_usado', 'False') == 'True',
                                'operacion_ejecutada': row.get('operacion_ejecutada', 'False') == 'True'
                            })
                    except Exception:
                        continue
            return ops_recientes
        except Exception as e:
            logger.error(f"⚠️ Error filtrando operaciones: {e}")
            return []

    def contar_breakouts_semana(self):
        """Cuenta breakouts detectados en la última semana"""
        ops = self.filtrar_operaciones_ultima_semana()
        breakouts = sum(1 for op in ops if op.get('breakout_usado', False))
        return breakouts

    def generar_reporte_semanal(self):
        """Genera reporte automático cada semana"""
        ops_ultima_semana = self.filtrar_operaciones_ultima_semana()
        if not ops_ultima_semana:
            return None
        
        total_ops = len(ops_ultima_semana)
        wins = sum(1 for op in ops_ultima_semana if op['resultado'] == 'TP')
        losses = sum(1 for op in ops_ultima_semana if op['resultado'] == 'SL')
        winrate = (wins/total_ops*100) if total_ops > 0 else 0
        pnl_total = sum(op['pnl_percent'] for op in ops_ultima_semana)
        
        mejor_op = max(ops_ultima_semana, key=lambda x: x['pnl_percent'])
        peor_op = min(ops_ultima_semana, key=lambda x: x['pnl_percent'])
        
        ganancias = [op['pnl_percent'] for op in ops_ultima_semana if op['pnl_percent'] > 0]
        perdidas = [abs(op['pnl_percent']) for op in ops_ultima_semana if op['pnl_percent'] < 0]
        
        avg_ganancia = sum(ganancias)/len(ganancias) if ganancias else 0
        avg_perdida = sum(perdidas)/len(perdidas) if perdidas else 0
        
        racha_actual = 0
        for op in reversed(ops_ultima_semana):
            if op['resultado'] == 'TP':
                racha_actual += 1
            else:
                break
        
        ops_automaticas = sum(1 for op in ops_ultima_semana if op.get('operacion_ejecutada', False))
        
        emoji_resultado = "🟢" if pnl_total > 0 else "🔴" if pnl_total < 0 else "⚪"
        
        mensaje = f"""
━━━━━━━━━━━━━━━━━━━━
📊 <b>REPORTE SEMANAL</b>
━━━━━━━━━━━━━━━━━━━━
📅 {datetime.now().strftime('%d/%m/%Y')} | Últimos 7 días
<b>RENDIMIENTO GENERAL</b>
{emoji_resultado} PnL Total: <b>{pnl_total:+.2f}%</b>
📈 Win Rate: <b>{winrate:.1f}%</b>
✅ Ganadas: {wins} | ❌ Perdidas: {losses}
<b>ESTADÍSTICAS</b>
📊 Operaciones: {total_ops}
🤖 Automáticas: {ops_automaticas}
💰 Ganancia Promedio: +{avg_ganancia:.2f}%
📉 Pérdida Promedio: -{avg_perdida:.2f}%
🔥 Racha actual: {racha_actual} wins
<b>DESTACADOS</b>
🏆 Mejor: {mejor_op['symbol']} ({mejor_op['tipo']})
   → {mejor_op['pnl_percent']:+.2f}%
⚠️ Peor: {peor_op['symbol']} ({peor_op['tipo']})
   → {peor_op['pnl_percent']:+.2f}%
━━━━━━━━━━━━━━━━━━━━
🤖 Bot automático 24/7
⚡ Estrategia: Breakout + Reentry
💎 Integración: Bitget API
💻 Acceso Premium: @TuUsuario
    """
        return mensaje

    def enviar_reporte_semanal(self):
        """Envía el reporte semanal por Telegram"""
        mensaje = self.generar_reporte_semanal()
        if not mensaje:
            logger.info("ℹ️ No hay datos suficientes para generar reporte")
            return False
        
        token = self.config.get('telegram_token')
        chat_ids = self.config.get('telegram_chat_ids', [])
        
        if token and chat_ids:
            try:
                self._enviar_telegram_simple(mensaje, token, chat_ids)
                logger.info("✅ Reporte semanal enviado correctamente")
                return True
            except Exception as e:
                logger.error(f"❌ Error enviando reporte: {e}")
                return False
        return False

    def verificar_envio_reporte_automatico(self):
        """Verifica si debe enviar el reporte semanal (cada lunes a las 9:00)"""
        ahora = datetime.now()
        if ahora.weekday() == 0 and 9 <= ahora.hour < 10:
            archivo_control = "ultimo_reporte.txt"
            try:
                if os.path.exists(archivo_control):
                    with open(archivo_control, 'r') as f:
                        ultima_fecha = f.read().strip()
                        if ultima_fecha == ahora.strftime('%Y-%m-%d'):
                            return False
                
                if self.enviar_reporte_semanal():
                    with open(archivo_control, 'w') as f:
                        f.write(ahora.strftime('%Y-%m-%d'))
                    return True
            except Exception as e:
                logger.error(f"⚠️ Error en envío automático: {e}")
        return False

    def verificar_cierre_operaciones(self):
        if not self.operaciones_activas:
            return []
        
        operaciones_cerradas = []
        
        for simbolo, operacion in list(self.operaciones_activas.items()):
            config_optima = self.config_optima_por_simbolo or {'timeframe': '5m', 'num_velas': 80}
            if not config_optima:
                continue
            
            datos = self.obtener_datos_mercado_config(simbolo, config_optima['timeframe'], config_optima.get('num_velas', 80))
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
                logger.info(f"     📊 {simbolo} Operación {resultado} - PnL: {pnl_percent:.2f}%")
        
        return operaciones_cerradas

    # =============================================================================
    # FUNCIONES DE SINCRONIZACIÓN CON BITGET (Integración mejorada)
    # =============================================================================

    def sincronizar_con_bitget(self):
        """
        Sincronizar estado local con posiciones reales en Bitget.
        Elimina operaciones del estado local que ya no existen en la exchange.
        """
        if not self.bitget_client:
            return
        
        try:
            posiciones_bitget = self.bitget_client.get_positions()
            
            if posiciones_bitget is None:
                logger.warning("⚠️ No se pudo obtener posiciones de Bitget")
                return
            
            posiciones_por_simbolo = {}
            for pos in posiciones_bitget:
                symbol = pos.get('symbol')
                if symbol:
                    posiciones_por_simbolo[symbol] = pos
            
            # Verificar operaciones activas locales
            simbolos_a_eliminar = []
            for simbolo, op_local in self.operaciones_activas.items():
                if simbolo not in posiciones_por_simbolo:
                    logger.info(f"🔄 {simbolo}: Posición no encontrada en Bitget (cerrada manualmente)")
                    simbolos_a_eliminar.append(simbolo)
            
            for simbolo in simbolos_a_eliminar:
                if simbolo in self.operaciones_activas:
                    del self.operaciones_activas[simbolo]
                    if simbolo in self.senales_enviadas:
                        self.senales_enviadas.remove(simbolo)
            
            if simbolos_a_eliminar:
                logger.info(f"✅ Sincronización: {len(simbolos_a_eliminar)} operaciones actualizadas")
                self.guardar_estado()
                
        except Exception as e:
            logger.error(f"❌ Error sincronizando con Bitget: {e}")

    def verificar_y_recolocar_tp_sl(self):
        """
        Verificar y recolocar TP/SL si es necesario.
        Útil cuando las órdenes de protección se han cancelado o no se crearon correctamente.
        """
        if not self.bitget_client:
            return
        
        try:
            for simbolo, op_local in list(self.operaciones_activas.items()):
                if not op_local.get('operacion_ejecutada', False):
                    continue
                
                order_id_tp = op_local.get('order_id_tp')
                order_id_sl = op_local.get('order_id_sl')
                
                # Recolocar si no existen órdenes de protección
                if not order_id_tp or not order_id_sl:
                    logger.info(f"🔄 Recolocando TP/SL para {simbolo}...")
                    
                    take_profit = op_local['take_profit']
                    stop_loss = op_local['stop_loss']
                    tipo = op_local['tipo']
                    
                    klines = self.bitget_client.get_klines(simbolo, '1m', 1)
                    if not klines:
                        continue
                    
                    precio_actual = float(klines[0][4])
                    hold_side = 'long' if tipo == 'LONG' else 'short'
                    
                    # Recolocar Stop Loss
                    if stop_loss and not order_id_sl:
                        precision_sl = self.bitget_client.obtener_precision_adaptada(stop_loss, simbolo)
                        precio_sl_formatted = self.bitget_client.redondear_precio_manual(
                            stop_loss, precision_sl, simbolo
                        )
                        
                        orden_sl = self.bitget_client.place_tpsl_order(
                            simbolo, hold_side,
                            trigger_price=float(precio_sl_formatted),
                            order_type='stop_loss',
                            stop_loss_price=float(precio_sl_formatted)
                        )
                        
                        if orden_sl:
                            op_local['order_id_sl'] = orden_sl.get('orderId')
                            logger.info(f"✅ SL recolocado para {simbolo}")
                        else:
                            logger.warning(f"⚠️ No se pudo recolocar SL para {simbolo}")
                    
                    # Recolocar Take Profit
                    if take_profit and not order_id_tp:
                        precision_tp = self.bitget_client.obtener_precision_adaptada(take_profit, simbolo)
                        precio_tp_formatted = self.bitget_client.redondear_precio_manual(
                            take_profit, precision_tp, simbolo
                        )
                        
                        orden_tp = self.bitget_client.place_tpsl_order(
                            simbolo, hold_side,
                            trigger_price=float(precio_tp_formatted),
                            order_type='take_profit',
                            take_profit_price=float(precio_tp_formatted)
                        )
                        
                        if orden_tp:
                            op_local['order_id_tp'] = orden_tp.get('orderId')
                            logger.info(f"✅ TP recolocado para {simbolo}")
                        else:
                            logger.warning(f"⚠️ No se pudo recolocar TP para {simbolo}")
            
            if self.operaciones_activas:
                self.guardar_estado()
        
        except Exception as e:
            logger.error(f"❌ Error verificando/recuperando TP/SL: {e}")

    def generar_mensaje_cierre(self, datos_operacion):
        emoji = "🟢" if datos_operacion['resultado'] == "TP" else "🔴"
        color_emoji = "✅" if datos_operacion['resultado'] == "TP" else "❌"
        
        if datos_operacion['tipo'] == 'LONG':
            pnl_absoluto = datos_operacion['precio_salida'] - datos_operacion['precio_entrada']
        else:
            pnl_absoluto = datos_operacion['precio_entrada'] - datos_operacion['precio_salida']
        
        breakout_usado = "🚀 Sí" if datos_operacion.get('breakout_usado', False) else "❌ No"
        operacion_ejecutada = "🤖 Sí" if datos_operacion.get('operacion_ejecutada', False) else "❌ No"
        
        mensaje = f"""
{emoji} <b>OPERACIÓN CERRADA - {datos_operacion['symbol']}</b>
{color_emoji} <b>RESULTADO: {datos_operacion['resultado']}</b>
📊 Tipo: {datos_operacion['tipo']}
💰 Entrada: {datos_operacion['precio_entrada']:.8f}
🎯 Salida: {datos_operacion['precio_salida']:.8f}
💵 PnL Absoluto: {pnl_absoluto:.8f}
📈 PnL %: {datos_operacion['pnl_percent']:.2f}%
⏰ Duración: {datos_operacion['duracion_minutos']:.1f} minutos
🚀 Breakout+Reentry: {breakout_usado}
🤖 Operación Bitget: {operacion_ejecutada}
📏 Ángulo: {datos_operacion['angulo_tendencia']:.1f}°
📊 Pearson: {datos_operacion['pearson']:.3f}
🎯 R²: {datos_operacion['r2_score']:.3f}
📏 Ancho: {datos_operacion.get('ancho_canal_porcentual', 0):.1f}%
⏱️ TF: {datos_operacion.get('timeframe_utilizado', 'N/A')}
🕯️ Velas: {datos_operacion.get('velas_utilizadas', 0)}
🕒 {datos_operacion['timestamp']}
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
            return "💔 Muy Débil", 1
        elif angulo_abs < 13:
            return "❤️‍🩹 Débil", 2
        elif angulo_abs < 27:
            return "💛 Moderada", 3
        elif angulo_abs < 45:
            return "💚 Fuerte", 4
        else:
            return "💙 Muy Fuerte", 5

    def determinar_direccion_tendencia(self, angulo_grados, umbral_minimo=1):
        if abs(angulo_grados) < umbral_minimo:
            return "⚪ RANGO"
        elif angulo_grados > 0:
            return "🟢 ALCISTA"
        else:
            return "🔴 BAJISTA"

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
            config_optima = self.config_optima_por_simbolo or {'timeframe': '5m', 'num_velas': 80}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': simbolo,
                'interval': config_optima.get('timeframe', '5m'),
                'limit': config_optima.get('num_velas', 80)
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
                               title=f'{simbolo} | {tipo_operacion} | {config_optima.get("timeframe", "5m")} | Bitget + Breakout+Reentry',
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
            logger.error(f"⚠️ Error generando gráfico: {e}")
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
                    logger.error(f"     ❌ Error enviando gráfico a {chat_id}: {r.status_code}")
            except Exception as e:
                logger.error(f"     ❌ Error enviando gráfico: {e}")
        
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
                    logger.error(f"     ❌ Error enviando mensaje a {chat_id}: {r.status_code}")
            except Exception as e:
                logger.error(f"     ❌ Error enviando mensaje: {e}")
                resultados.append(False)
        
        return any(resultados)

    def reoptimizar_periodicamente(self):
        try:
            horas_desde_opt = (datetime.now() - self.ultima_optimizacion).total_seconds() / 7200
            
            if self.operaciones_desde_optimizacion >= 8 or horas_desde_opt >= self.config.get('reevaluacion_horas', 24):
                logger.info("🔄 Iniciando re-optimización automática...")
                ia = OptimizadorIA(log_path=self.log_path, min_samples=self.config.get('min_samples_optimizacion', 30))
                nuevos_parametros = ia.buscar_mejores_parametros()
                
                if nuevos_parametros:
                    self.actualizar_parametros(nuevos_parametros)
                    self.ultima_optimizacion = datetime.now()
                    self.operaciones_desde_optimizacion = 0
                    logger.info("✅ Parámetros actualizados en tiempo real")
        except Exception as e:
            logger.error(f"⚠ Error en re-optimización automática: {e}")

    def actualizar_parametros(self, nuevos_parametros):
        self.config['trend_threshold_degrees'] = nuevos_parametros.get('trend_threshold_degrees', 
                                                                        self.config.get('trend_threshold_degrees', 16))
        self.config['min_trend_strength_degrees'] = nuevos_parametros.get('min_trend_strength_degrees', 
                                                                           self.config.get('min_trend_strength_degrees', 16))
        self.config['entry_margin'] = nuevos_parametros.get('entry_margin', 
                                                             self.config.get('entry_margin', 0.001))

    def ejecutar_analisis(self):
        # Verificación periódica de Bitget (10% de probabilidad)
        if random.random() < 0.1:
            self.reoptimizar_periodicamente()
            self.verificar_envio_reporte_automatico()
        
        # Sincronizar con Bitget para mantener consistencia
        if self.bitget_client and random.random() < 0.05:
            self.sincronizar_con_bitget()
            self.verificar_y_recolocar_tp_sl()
        
        cierres = self.verificar_cierre_operaciones()
        if cierres:
            logger.info(f"     📊 Operaciones cerradas: {', '.join(cierres)}")
        
        self.guardar_estado()
        return self.escanear_mercado()

    def mostrar_resumen_operaciones(self):
        logger.info(f"\n📊 RESUMEN OPERACIONES:")
        logger.info(f"   Activas: {len(self.operaciones_activas)}")
        logger.info(f"   Esperando reentry: {len(self.esperando_reentry)}")
        logger.info(f"   Total ejecutadas: {self.total_operaciones}")
        
        if self.bitget_client:
            logger.info(f"   🤖 Bitget: ✅ Conectado")
        else:
            logger.info(f"   🤖 Bitget: ❌ No configurado")
        
        if self.operaciones_activas:
            for simbolo, op in self.operaciones_activas.items():
                estado = "🟢 LONG" if op['tipo'] == 'LONG' else "🔴 SHORT"
                ancho_canal = op.get('ancho_canal_porcentual', 0)
                timeframe = op.get('timeframe_utilizado', 'N/A')
                velas = op.get('velas_utilizadas', 0)
                breakout = "🚀" if op.get('breakout_usado', False) else ""
                ejecutada = "🤖" if op.get('operacion_ejecutada', False) else ""
                logger.info(f"   • {simbolo} {estado} {breakout} {ejecutada} - {timeframe} - {velas}v - Ancho: {ancho_canal:.1f}%")

    def iniciar(self):
        logger.info("\n" + "=" * 70)
        logger.info("🤖 BOT DE TRADING - ESTRATEGIA BREAKOUT + REENTRY")
        logger.info("🎯 PRIORIDAD: TIMEFRAMES CORTOS (1m > 3m > 5m > 15m > 30m)")
        logger.info("💾 PERSISTENCIA: ACTIVADA")
        logger.info("🔄 REEVALUACIÓN: CADA 2 HORAS")
        logger.info("🏦 INTEGRACIÓN: BITGET API")
        logger.info("=" * 70)
        logger.info(f"💱 Símbolos: {len(self.config.get('symbols', []))} monedas")
        logger.info(f"⏰ Timeframes: {', '.join(self.config.get('timeframes', []))}")
        logger.info(f"🕯️ Velas: {self.config.get('velas_options', [])}")
        logger.info(f"📏 ANCHO MÍNIMO: {self.config.get('min_channel_width_percent', 4)}%")
        logger.info(f"🚀 Estrategia: 1) Detectar Breakout → 2) Esperar Reentry → 3) Confirmar con Stoch")
        
        if self.bitget_client:
            logger.info(f"🤖 BITGET: ✅ API Conectada")
            logger.info(f"⚡ Apalancamiento: {self.leverage_por_defecto}x")
            logger.info(f"💰 Capital por operación: ${self.capital_por_operacion}")
            
            if self.ejecutar_operaciones_automaticas:
                logger.info(f"🤖 AUTO-TRADING: ✅ ACTIVADO")
            else:
                logger.info(f"🤖 AUTO-TRADING: ❌ Solo señales")
        else:
            logger.info(f"🤖 BITGET: ❌ No configurado (solo señales)")
        
        logger.info("=" * 70)
        logger.info("\n🚀 INICIANDO BOT...")
        
        try:
            while True:
                nuevas_senales = self.ejecutar_analisis()
                self.mostrar_resumen_operaciones()
                
                minutos_espera = self.config.get('scan_interval_minutes', 1)
                logger.info(f"\n✅ Análisis completado. Señales nuevas: {nuevas_senales}")
                logger.info(f"⏳ Próximo análisis en {minutos_espera} minutos...")
                logger.info("-" * 60)
                
                for minuto in range(minutos_espera):
                    time.sleep(60)
                    restantes = minutos_espera - (minuto + 1)
                    if restantes > 0 and restantes % 5 == 0:
                        logger.info(f"   ⏰ {restantes} minutos restantes...")
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot detenido por el usuario")
            logger.info("💾 Guardando estado final...")
            self.guardar_estado()
            logger.info("👋 ¡Hasta pronto!")
        except Exception as e:
            logger.error(f"\n❌ Error en el bot: {e}")
            logger.info("💾 Intentando guardar estado...")
            try:
                self.guardar_estado()
            except Exception as e2:
                logger.error(f"❌ Error guardando estado final: {e2}")

# ---------------------------
# CONFIGURACIÓN DE CREDENCIALES Y RENDER.COM
# ---------------------------

def obtener_credenciales_bitget():
    """
    Obtiene las credenciales de Bitget desde variables de entorno.
    Detecta automáticamente si está ejecutándose en Render.com.
    
    Variables de entorno esperadas:
        - BITGET_API_KEY
        - BITGET_SECRET_KEY
        - BITGET_PASSPHRASE
        - EJECUTAR_OPERACIONES_AUTOMATICAS (opcional)
        - LEVERAGE_POR_DEFECTO (opcional)
        - CAPITAL_POR_OPERACION (opcional)
    
    Returns:
        dict: Diccionario con las credenciales y configuraciones
    """
    # Valores por defecto
    api_key = os.environ.get('BITGET_API_KEY', '')
    api_secret = os.environ.get('BITGET_SECRET_KEY', '')
    passphrase = os.environ.get('BITGET_PASSPHRASE', '')
    ejecutar_automaticas = os.environ.get('EJECUTAR_OPERACIONES_AUTOMATICAS', 'false').lower() == 'true'
    leverage = int(os.environ.get('LEVERAGE_POR_DEFECTO', '10'))
    capital_por_operacion = float(os.environ.get('CAPITAL_POR_OPERACION', '4'))
    
    # Verificar si estamos en Render.com
    en_render = 'RENDER' in os.environ
    
    if en_render:
        logger.info("📝 Modo Render.com detectado - Variables de entorno activas")
        
        if api_key:
            logger.info("📝 BITGET_API_KEY cargada correctamente")
        if api_secret:
            logger.info("📝 BITGET_SECRET_KEY cargada correctamente")
        if passphrase:
            logger.info("📝 BITGET_PASSPHRASE cargada correctamente")
            
        logger.info(f"📝 EJECUTAR_OPERACIONES_AUTOMATICAS: {ejecutar_automaticas}")
        logger.info(f"📝 LEVERAGE_POR_DEFECTO: {leverage}x")
        logger.info(f"📝 CAPITAL_POR_OPERACION: ${capital_por_operacion}")
    else:
        logger.info("📝 Modo local detectado")
    
    return {
        'api_key': api_key,
        'api_secret': api_secret,
        'passphrase': passphrase,
        'ejecutar_operaciones_automaticas': ejecutar_automaticas,
        'leverage': leverage,
        'capital_por_operacion': capital_por_operacion,
        'en_render': en_render
    }


def crear_cliente_bitget(api_key=None, api_secret=None, passphrase=None):
    """
    Crea una instancia del cliente de Bitget con las credenciales proporcionadas
    o desde variables de entorno (Render.com).
    
    Args:
        api_key: API Key (opcional)
        api_secret: Secret Key (opcional)
        passphrase: Passphrase (opcional)
    
    Returns:
        BitgetClient: Instancia del cliente o None si faltan credenciales
    """
    # Si no se proporcionan credenciales, intentar desde variables de entorno
    if not api_key or not api_secret or not passphrase:
        credenciales = obtener_credenciales_bitget()
        api_key = api_key or credenciales['api_key']
        api_secret = api_secret or credenciales['api_secret']
        passphrase = passphrase or credenciales['passphrase']
    
    # Verificar que tenemos todas las credenciales
    if not api_key or not api_secret or not passphrase:
        logger.error("❌ Faltan credenciales de Bitget")
        logger.error("   Configure las variables de entorno en Render.com:")
        logger.error("   - BITGET_API_KEY")
        logger.error("   - BITGET_SECRET_KEY")
        logger.error("   - BITGET_PASSPHRASE")
        return None
    
    # Crear cliente
    client = BitgetClient(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase
    )
    
    return client


# ---------------------------
# CONFIGURACIÓN SIMPLE
# ---------------------------

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
        'timeframes': ['5m', '15m', '30m', '1h'],
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

# ---------------------------
# FLASK APP Y RENDER
# ---------------------------

app = Flask(__name__)

# Crear bot con configuración desde entorno
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

# Iniciar hilo del bot
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

# Configuración automática del webhook
def setup_telegram_webhook():
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
        # Eliminar webhook anterior
        requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
        time.sleep(1)
        # Configurar nuevo webhook
        response = requests.get(f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Webhook de Telegram configurado correctamente")
        else:
            logger.error(f"❌ Error configurando webhook: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Error configurando webhook: {e}")

if __name__ == '__main__':
    logger.info("🚀 Iniciando aplicación Flask...")
    setup_telegram_webhook()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


# =============================================================================
# CONFIGURACIÓN EN RENDER.COM - GUÍA RÁPIDA
# =============================================================================
"""
CONFIGURACIÓN EN RENDER.COM:
============================

1. Ve a tu dashboard de Render.com
2. Selecciona tu servicio/web service
3. Ve a la sección 'Environment Variables'
4. Agrega las siguientes variables:

   Variable                        | Valor
   -----------------------------------------
   BITGET_API_KEY                  | tu_api_key_de_bitget
   BITGET_SECRET_KEY               | tu_secret_key_de_bitget
   BITGET_PASSPHRASE               | tu_passphrase_de_bitget
   TELEGRAM_TOKEN                  | tu_token_de_telegram
   TELEGRAM_CHAT_ID                | tu_chat_id_de_telegram
   EJECUTAR_OPERACIONES_AUTOMATICAS| true (o false)
   LEVERAGE_POR_DEFECTO            | 10 (o tu preferencia)
   CAPITAL_POR_OPERACION           | 4 (o tu preferencia)
   PORT                            | 5000

PARA EJECUTAR OPERACIONES REALES:
=================================
1. Configurar variables de entorno en Render.com
2. EJECUTAR_OPERACIONES_AUTOMATICAS=true
3. CAPITAL_POR_OPERACION=4 (ajustar según preferencia)
4. LEVERAGE_POR_DEFECTO=10 (máximo recomendado)

DETECCIÓN AUTOMÁTICA DE RENDER.COM:
===================================
El bot detecta automáticamente si está ejecutándose en Render.com
verificando la variable de entorno 'RENDER' en el sistema.
"""
