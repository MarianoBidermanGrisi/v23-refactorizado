"""
API Clients para conexiones externas.
Maneja todas las comunicaciones con Binance, Telegram y Render.
"""

import requests
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO

from ..config.settings import config, Constants

class BinanceAPIClient:
    """Cliente para la API de Binance"""
    
    def __init__(self):
        self.base_url = config.binance_api_base
        self.klines_endpoint = config.binance_klines_endpoint
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def obtener_datos_klines(self, symbol: str, interval: str, limit: int) -> Optional[List]:
        """
        Obtiene datos de klines de Binance
        
        Args:
            symbol: Símbolo de trading (ej. BTCUSDT)
            interval: Intervalo de tiempo (ej. 5m, 1h)
            limit: Número de velas a obtener
            
        Returns:
            Lista de klines o None si hay error
        """
        try:
            url = f"{self.base_url}{self.klines_endpoint}"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            self.logger.debug(f"Obteniendo datos de {symbol} - {interval} - {limit} velas")
            respuesta = self.session.get(url, params=params, timeout=10)
            respuesta.raise_for_status()
            
            datos = respuesta.json()
            
            if not isinstance(datos, list) or len(datos) == 0:
                self.logger.warning(f"Respuesta vacía para {symbol}")
                return None
                
            self.logger.debug(f"✅ Datos obtenidos correctamente para {symbol}")
            return datos
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout obteniendo datos de {symbol}")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Error de conexión con Binance para {symbol}")
            return None
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de {symbol}: {e}")
            return None
    
    def verificar_conexion(self) -> bool:
        """Verifica la conexión con Binance"""
        try:
            url = f"{self.base_url}/api/v3/ping"
            respuesta = self.session.get(url, timeout=5)
            return respuesta.status_code == 200
        except Exception as e:
            self.logger.error(f"Error verificando conexión Binance: {e}")
            return False

class TelegramAPIClient:
    """Cliente para la API de Telegram"""
    
    def __init__(self):
        self.base_url = config.telegram_api_base
        self.token = config.telegram_token
        self.logger = logging.getLogger(__name__)
        
    def enviar_mensaje(self, chat_ids: List[str], mensaje: str, parse_mode: str = 'HTML') -> bool:
        """
        Envía mensaje por Telegram
        
        Args:
            chat_ids: Lista de IDs de chat
            mensaje: Mensaje a enviar
            parse_mode: Modo de parseo (HTML, Markdown)
            
        Returns:
            True si se envió exitosamente a al menos un chat
        """
        if not self.token or not chat_ids:
            self.logger.warning("Token de Telegram o chat IDs no configurados")
            return False
            
        try:
            resultados = []
            for chat_id in chat_ids:
                url = f"{self.base_url}/bot{self.token}/sendMessage"
                payload = {
                    'chat_id': chat_id,
                    'text': mensaje,
                    'parse_mode': parse_mode
                }
                
                try:
                    respuesta = requests.post(url, json=payload, timeout=10)
                    if respuesta.status_code == 200:
                        self.logger.debug(f"✅ Mensaje enviado a chat {chat_id}")
                        resultados.append(True)
                    else:
                        self.logger.warning(f"❌ Error enviando a chat {chat_id}: {respuesta.status_code}")
                        resultados.append(False)
                except Exception as e:
                    self.logger.error(f"❌ Error enviando mensaje a {chat_id}: {e}")
                    resultados.append(False)
                    
            return any(resultados)
            
        except Exception as e:
            self.logger.error(f"❌ Error general enviando mensajes: {e}")
            return False
    
    def enviar_grafico(self, chat_ids: List[str], imagen: BytesIO, caption: str = None) -> bool:
        """
        Envía gráfico por Telegram
        
        Args:
            chat_ids: Lista de IDs de chat
            imagen: Buffer de imagen
            caption: Título opcional
            
        Returns:
            True si se envió exitosamente
        """
        if not self.token or not chat_ids:
            self.logger.warning("Token de Telegram o chat IDs no configurados")
            return False
            
        try:
            exito = False
            for chat_id in chat_ids:
                url = f"{self.base_url}/bot{self.token}/sendPhoto"
                
                try:
                    imagen.seek(0)
                    files = {'photo': ('grafico.png', imagen.read(), 'image/png')}
                    data = {'chat_id': chat_id}
                    
                    if caption:
                        data['caption'] = caption
                        
                    respuesta = requests.post(url, files=files, data=data, timeout=120)
                    
                    if respuesta.status_code == 200:
                        self.logger.debug(f"✅ Gráfico enviado a chat {chat_id}")
                        exito = True
                    else:
                        self.logger.warning(f"❌ Error enviando gráfico a chat {chat_id}: {respuesta.status_code}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error enviando gráfico a {chat_id}: {e}")
                    
            return exito
            
        except Exception as e:
            self.logger.error(f"❌ Error general enviando gráficos: {e}")
            return False
    
    def configurar_webhook(self, webhook_url: str) -> bool:
        """
        Configura webhook de Telegram
        
        Args:
            webhook_url: URL del webhook
            
        Returns:
            True si se configuró exitosamente
        """
        if not self.token:
            self.logger.error("Token de Telegram no configurado")
            return False
            
        try:
            # Eliminar webhook anterior
            url_delete = f"{self.base_url}/bot{self.token}/deleteWebhook"
            requests.get(url_delete, timeout=10)
            
            # Configurar nuevo webhook
            url_set = f"{self.base_url}/bot{self.token}/setWebhook"
            params = {'url': webhook_url}
            respuesta = requests.get(url_set, params=params, timeout=10)
            
            if respuesta.status_code == 200:
                self.logger.info(f"✅ Webhook configurado: {webhook_url}")
                return True
            else:
                self.logger.error(f"❌ Error configurando webhook: {respuesta.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error configurando webhook: {e}")
            return False

class RenderHealthClient:
    """Cliente para el health check de Render"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def verificar_health(self) -> Dict[str, Any]:
        """
        Verifica el estado de salud del sistema
        
        Returns:
            Diccionario con estado de salud
        """
        try:
            # Verificar conexión con Binance
            binance_client = BinanceAPIClient()
            binance_ok = binance_client.verificar_conexion()
            
            # Verificar conexión con Telegram
            telegram_client = TelegramAPIClient()
            telegram_ok = bool(config.telegram_token)
            
            # Estado general
            estado_salud = {
                'status': 'ok' if binance_ok and telegram_ok else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'binance': 'ok' if binance_ok else 'error',
                    'telegram': 'ok' if telegram_ok else 'error',
                    'render': 'ok'
                },
                'config': {
                    'symbols_count': len(config.symbols),
                    'timeframes': config.timeframes,
                    'auto_optimize': config.auto_optimize
                }
            }
            
            return estado_salud
            
        except Exception as e:
            self.logger.error(f"Error verificando health: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

# Instancias globales de clientes
binance_client = BinanceAPIClient()
telegram_client = TelegramAPIClient()
render_health_client = RenderHealthClient()

print("🔌 API Clients inicializados correctamente")
