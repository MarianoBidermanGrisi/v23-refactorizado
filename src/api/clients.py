"""
Clientes de API con logs detallados.
Maneja integraciones con Binance, Telegram y Render con logs extensivos.
MEJORADO CON LOGS EXTENSIVOS PARA MAYOR VISIBILIDAD
"""
import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
from ..config.settings import config

class BinanceAPIClient:
    """Cliente de Binance API con logs detallados"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = config.binance_api_base
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': config.binance_api_key,
            'Content-Type': 'application/json'
        })
        
        # Estadísticas de uso
        self.stats = {
            'requests_totales': 0,
            'requests_exitosos': 0,
            'requests_fallidos': 0,
            'tiempo_total_requests': 0.0,
            'datos_klines_obtenidos': 0,
            'errores_por_endpoint': {}
        }
        
        self.logger.info("💰 [BINANCE] Cliente de Binance API inicializado")
        self.logger.info(f"   • Base URL: {self.base_url}")
        self.logger.info(f"   • API Key configurada: {'✅' if config.binance_api_key else '❌'}")

    def _hacer_request(self, endpoint: str, params: Dict[str, Any] = None, method: str = 'GET') -> Optional[Dict[Any, Any]]:
        """
        Realiza request a la API con logs detallados
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la request
            method: Método HTTP
        Returns:
            Respuesta de la API o None
        """
        try:
            timestamp_inicio = datetime.now()
            url = f"{self.base_url}{endpoint}"
            
            self.stats['requests_totales'] += 1
            
            self.logger.debug(f"🌐 [BINANCE] Request: {method} {url}")
            if params:
                self.logger.debug(f"   • Parámetros: {params}")
            
            # Realizar request
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=params, timeout=10)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            timestamp_fin = datetime.now()
            tiempo_request = (timestamp_fin - timestamp_inicio).total_seconds()
            self.stats['tiempo_total_requests'] += tiempo_request
            
            self.logger.debug(f"⏱️ [BINANCE] Tiempo response: {tiempo_request:.3f}s")
            self.logger.debug(f"📊 [BINANCE] Status code: {response.status_code}")
            
            if response.status_code == 200:
                self.stats['requests_exitosos'] += 1
                data = response.json()
                
                self.logger.debug(f"✅ [BINANCE] Request exitoso")
                return data
            else:
                self.stats['requests_fallidos'] += 1
                error_key = f"{endpoint}_{response.status_code}"
                self.stats['errores_por_endpoint'][error_key] = self.stats['errores_por_endpoint'].get(error_key, 0) + 1
                
                self.logger.error(f"❌ [BINANCE] Error en request: {response.status_code}")
                self.logger.error(f"   • Respuesta: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.stats['requests_fallidos'] += 1
            self.logger.error(f"⏰ [BINANCE] Timeout en request a {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            self.stats['requests_fallidos'] += 1
            self.logger.error(f"🔌 [BINANCE] Error de conexión a {endpoint}")
            return None
        except Exception as e:
            self.stats['requests_fallidos'] += 1
            self.logger.error(f"❌ [BINANCE] Error inesperado en request: {e}")
            return None

    def obtener_datos_klines(self, symbol: str, interval: str, limit: int) -> Optional[List[List]]:
        """
        Obtiene datos de klines con logs detallados
        Args:
            symbol: Símbolo de trading
            interval: Intervalo de tiempo
            limit: Número de klines
        Returns:
            Lista de klines o None
        """
        try:
            timestamp_inicio = datetime.now()
            
            self.logger.debug(f"🕯️ [BINANCE] Obteniendo klines:")
            self.logger.debug(f"   • Símbolo: {symbol}")
            self.logger.debug(f"   • Intervalo: {interval}")
            self.logger.debug(f"   • Límite: {limit}")
            
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            
            data = self._hacer_request('/api/v3/klines', params)
            
            if data:
                timestamp_fin = datetime.now()
                tiempo_obtencion = (timestamp_fin - timestamp_inicio).total_seconds()
                
                self.stats['datos_klines_obtenidos'] += 1
                
                # Log de validación de datos
                if data:
                    primera_vela = data[0]
                    ultima_vela = data[-1]
                    
                    self.logger.info(f"✅ [BINANCE] {symbol} {interval}:")
                    self.logger.info(f"   • Klines obtenidos: {len(data)}")
                    self.logger.info(f"   • Primera vela: {datetime.fromtimestamp(primera_vela[0]/1000).strftime('%Y-%m-%d %H:%M')}")
                    self.logger.info(f"   • Última vela: {datetime.fromtimestamp(ultima_vela[0]/1000).strftime('%Y-%m-%d %H:%M')}")
                    self.logger.info(f"   • Precio actual: {float(ultima_vela[4]):.8f}")
                    self.logger.info(f"   • Tiempo obtención: {tiempo_obtencion:.3f}s")
                
                return data
            else:
                self.logger.error(f"❌ [BINANCE] No se pudieron obtener klines para {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [BINANCE] Error obteniendo klines: {e}")
            return None

    def verificar_conexion(self) -> bool:
        """
        Verifica la conexión con Binance con logs detallados
        Returns:
            True si la conexión es exitosa
        """
        try:
            self.logger.debug("🔍 [BINANCE] Verificando conexión...")
            
            data = self._hacer_request('/api/v3/ping')
            
            if data is not None:
                self.logger.info("✅ [BINANCE] Conexión exitosa")
                return True
            else:
                self.logger.error("❌ [BINANCE] Conexión fallida")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [BINANCE] Error verificando conexión: {e}")
            return False

    def obtener_estadisticas_api(self) -> Dict[str, Any]:
        """Obtiene estadísticas de uso de la API"""
        stats = self.stats.copy()
        stats.update({
            'tasa_exito': (self.stats['requests_exitosos'] / self.stats['requests_totales'] * 100) if self.stats['requests_totales'] > 0 else 0,
            'tiempo_promedio_request': (self.stats['tiempo_total_requests'] / self.stats['requests_totales']) if self.stats['requests_totales'] > 0 else 0
        })
        return stats

class TelegramAPIClient:
    """Cliente de Telegram API con logs detallados"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = config.telegram_api_base
        self.token = config.telegram_token
        self.session = requests.Session()
        
        # Estadísticas de uso
        self.stats = {
            'mensajes_enviados': 0,
            'graficos_enviados': 0,
            'errores_envio': 0,
            'tiempo_total_envio': 0.0,
            'chat_ids_atendidos': set()
        }
        
        self.logger.info("📱 [TELEGRAM] Cliente de Telegram API inicializado")
        self.logger.info(f"   • Base URL: {self.base_url}")
        self.logger.info(f"   • Token configurado: {'✅' if self.token else '❌'}")

    def _hacer_request(self, method: str, data: Dict[str, Any] = None) -> Optional[Dict[Any, Any]]:
        """
        Realiza request a la API de Telegram con logs detallados
        Args:
            method: Método de la API
            data: Datos del request
        Returns:
            Respuesta de la API o None
        """
        try:
            url = f"{self.base_url}/bot{self.token}/{method}"
            
            self.logger.debug(f"📤 [TELEGRAM] Envío: {method}")
            if data:
                # Log sin el token por seguridad
                data_sin_token = {k: v for k, v in data.items() if k != 'caption'}
                self.logger.debug(f"   • Datos: {data_sin_token}")
            
            response = self.session.post(url, data=data, timeout=30)
            
            self.logger.debug(f"📊 [TELEGRAM] Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    self.logger.debug(f"✅ [TELEGRAM] {method} exitoso")
                    return result
                else:
                    self.logger.error(f"❌ [TELEGRAM] {method} falló: {result.get('description', 'Error desconocido')}")
                    return None
            else:
                self.logger.error(f"❌ [TELEGRAM] Error HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error en request {method}: {e}")
            return None

    def enviar_mensaje(self, chat_ids: List[str], mensaje: str, parse_mode: str = 'Markdown') -> bool:
        """
        Envía mensaje por Telegram con logs detallados
        Args:
            chat_ids: Lista de chat IDs
            mensaje: Mensaje a enviar
            parse_mode: Modo de parseo
        Returns:
            True si el envío es exitoso
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo envío")
                return False
            
            timestamp_inicio = datetime.now()
            exitosos = 0
            fallidos = 0
            
            self.logger.debug(f"💬 [TELEGRAM] Enviando mensaje a {len(chat_ids)} chats")
            self.logger.debug(f"   • Longitud mensaje: {len(mensaje)} caracteres")
            self.logger.debug(f"   • Parse mode: {parse_mode}")
            
            for chat_id in chat_ids:
                try:
                    self.stats['chat_ids_atendidos'].add(chat_id)
                    
                    data = {
                        'chat_id': chat_id,
                        'text': mensaje,
                        'parse_mode': parse_mode,
                        'disable_web_page_preview': True
                    }
                    
                    result = self._hacer_request('sendMessage', data)
                    
                    if result:
                        exitosos += 1
                        self.logger.debug(f"   ✅ Chat {chat_id}: Mensaje enviado")
                    else:
                        fallidos += 1
                        self.logger.error(f"   ❌ Chat {chat_id}: Error enviando mensaje")
                        
                    # Pausa para evitar rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    fallidos += 1
                    self.logger.error(f"   ❌ Chat {chat_id}: Error inesperado: {e}")
            
            timestamp_fin = datetime.now()
            tiempo_envio = (timestamp_fin - timestamp_inicio).total_seconds()
            self.stats['tiempo_total_envio'] += tiempo_envio
            
            if exitosos > 0:
                self.stats['mensajes_enviados'] += exitosos
                self.logger.info(f"📤 [TELEGRAM] Mensaje enviado: {exitosos}/{len(chat_ids)} chats")
                self.logger.info(f"   • Tiempo total: {tiempo_envio:.3f}s")
                return True
            else:
                self.stats['errores_envio'] += fallidos
                self.logger.error(f"❌ [TELEGRAM] Fallo al enviar mensaje a todos los chats")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error enviando mensaje: {e}")
            self.stats['errores_envio'] += 1
            return False

    def enviar_grafico(self, chat_ids: List[str], imagen: Union[BytesIO, bytes], caption: str = None) -> bool:
        """
        Envía gráfico por Telegram con logs detallados
        Args:
            chat_ids: Lista de chat IDs
            imagen: Imagen a enviar
            caption: Leyenda de la imagen
        Returns:
            True si el envío es exitoso
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo envío de gráfico")
                return False
            
            timestamp_inicio = datetime.now()
            exitosos = 0
            fallidos = 0
            
            self.logger.debug(f"🖼️ [TELEGRAM] Enviando gráfico a {len(chat_ids)} chats")
            if caption:
                self.logger.debug(f"   • Caption: {len(caption)} caracteres")
            
            # Preparar archivo
            if isinstance(imagen, BytesIO):
                imagen.seek(0)
                files = {'photo': ('chart.png', imagen, 'image/png')}
            else:
                files = {'photo': ('chart.png', imagen, 'image/png')}
            
            for chat_id in chat_ids:
                try:
                    self.stats['chat_ids_atendidos'].add(chat_id)
                    
                    data = {'chat_id': chat_id}
                    if caption:
                        data['caption'] = caption
                        data['parse_mode'] = 'Markdown'
                    
                    # Crear una copia de los archivos para cada chat
                    files_copy = {}
                    for key, value in files.items():
                        if hasattr(value[1], 'read'):
                            value[1].seek(0)
                        files_copy[key] = value
                    
                    result = self._hacer_request('sendPhoto', {**data, **files_copy})
                    
                    if result:
                        exitosos += 1
                        self.logger.debug(f"   ✅ Chat {chat_id}: Gráfico enviado")
                    else:
                        fallidos += 1
                        self.logger.error(f"   ❌ Chat {chat_id}: Error enviando gráfico")
                        
                    # Pausa para evitar rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    fallidos += 1
                    self.logger.error(f"   ❌ Chat {chat_id}: Error inesperado: {e}")
            
            timestamp_fin = datetime.now()
            tiempo_envio = (timestamp_fin - timestamp_inicio).total_seconds()
            self.stats['tiempo_total_envio'] += tiempo_envio
            
            if exitosos > 0:
                self.stats['graficos_enviados'] += exitosos
                self.logger.info(f"📤 [TELEGRAM] Gráfico enviado: {exitosos}/{len(chat_ids)} chats")
                self.logger.info(f"   • Tiempo total: {tiempo_envio:.3f}s")
                return True
            else:
                self.stats['errores_envio'] += fallidos
                self.logger.error(f"❌ [TELEGRAM] Fallo al enviar gráfico a todos los chats")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error enviando gráfico: {e}")
            self.stats['errores_envio'] += 1
            return False

    def configurar_webhook(self, webhook_url: str) -> bool:
        """
        Configura webhook de Telegram con logs detallados
        Args:
            webhook_url: URL del webhook
        Returns:
            True si la configuración es exitosa
        """
        try:
            if not self.token:
                self.logger.warning("⚠️ [TELEGRAM] Token no configurado, omitiendo configuración de webhook")
                return False
            
            self.logger.debug(f"🔗 [TELEGRAM] Configurando webhook: {webhook_url}")
            
            data = {
                'url': webhook_url,
                'drop_pending_updates': True
            }
            
            result = self._hacer_request('setWebhook', data)
            
            if result:
                self.logger.info("✅ [TELEGRAM] Webhook configurado exitosamente")
                return True
            else:
                self.logger.error("❌ [TELEGRAM] Error configurando webhook")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [TELEGRAM] Error configurando webhook: {e}")
            return False

    def obtener_estadisticas_api(self) -> Dict[str, Any]:
        """Obtiene estadísticas de uso de la API"""
        stats = self.stats.copy()
        stats.update({
            'chat_ids_unicos': len(self.stats['chat_ids_atendidos']),
            'total_envios': self.stats['mensajes_enviados'] + self.stats['graficos_enviados'],
            'tasa_exito': ((self.stats['mensajes_enviados'] + self.stats['graficos_enviados']) / 
                          max(1, self.stats['mensajes_enviados'] + self.stats['graficos_enviados'] + self.stats['errores_envio']) * 100)
        })
        return stats

class RenderHealthClient:
    """Cliente de health check para Render con logs detallados"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.binance_client = binance_client
        self.telegram_client = telegram_client
        
        # Estadísticas de health checks
        self.stats = {
            'health_checks_realizados': 0,
            'health_checks_exitosos': 0,
            'health_checks_fallidos': 0,
            'servicios_monitoreados': ['binance', 'telegram'],
            'tiempo_promedio_check': 0.0
        }
        
        self.logger.info("🏥 [HEALTH] Cliente de health check inicializado")
        self.logger.info(f"   • Servicios monitoreados: {self.stats['servicios_monitoreados']}")

    def verificar_health(self) -> Dict[str, Any]:
        """
        Verifica la salud del sistema con logs detallados
        Returns:
            Diccionario con estado de salud
        """
        try:
            timestamp_inicio = datetime.now()
            
            self.logger.debug("🔍 [HEALTH] Iniciando verificación de salud del sistema...")
            
            health_status = {
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'servicios': {},
                'configuracion': {
                    'symbols_count': len(config.symbols) if config.symbols else 0,
                    'timeframes': config.timeframes,
                    'auto_optimize': config.auto_optimize
                }
            }
            
            # Verificar Binance
            self.logger.debug("🔍 [HEALTH] Verificando Binance...")
            try:
                binance_ok = self.binance_client.verificar_conexion()
                health_status['servicios']['binance'] = {
                    'status': 'ok' if binance_ok else 'error',
                    'response_time': 'N/A',  # Se podría medir específicamente
                    'ultimo_check': datetime.now().isoformat()
                }
                
                if binance_ok:
                    self.logger.debug("   ✅ Binance: OK")
                else:
                    self.logger.error("   ❌ Binance: ERROR")
                    health_status['status'] = 'degraded'
                    
            except Exception as e:
                health_status['servicios']['binance'] = {
                    'status': 'error',
                    'error': str(e),
                    'ultimo_check': datetime.now().isoformat()
                }
                health_status['status'] = 'degraded'
                self.logger.error(f"   ❌ Binance: ERROR - {e}")
            
            # Verificar Telegram
            self.logger.debug("🔍 [HEALTH] Verificando Telegram...")
            try:
                telegram_token_ok = bool(config.telegram_token)
                health_status['servicios']['telegram'] = {
                    'status': 'ok' if telegram_token_ok else 'warning',
                    'configurado': telegram_token_ok,
                    'ultimo_check': datetime.now().isoformat()
                }
                
                if telegram_token_ok:
                    self.logger.debug("   ✅ Telegram: Configurado")
                else:
                    self.logger.warning("   ⚠️ Telegram: No configurado")
                    if health_status['status'] == 'ok':
                        health_status['status'] = 'degraded'
                        
            except Exception as e:
                health_status['servicios']['telegram'] = {
                    'status': 'error',
                    'error': str(e),
                    'ultimo_check': datetime.now().isoformat()
                }
                health_status['status'] = 'degraded'
                self.logger.error(f"   ❌ Telegram: ERROR - {e}")
            
            # Verificar configuración crítica
            self.logger.debug("🔍 [HEALTH] Verificando configuración...")
            config_issues = []
            
            if not config.symbols:
                config_issues.append("No hay símbolos configurados")
            
            if not config.timeframes:
                config_issues.append("No hay timeframes configurados")
            
            health_status['configuracion']['issues'] = config_issues
            
            if config_issues:
                health_status['status'] = 'degraded'
                self.logger.warning(f"   ⚠️ Configuración: {len(config_issues)} problemas encontrados")
                for issue in config_issues:
                    self.logger.warning(f"      • {issue}")
            else:
                self.logger.debug("   ✅ Configuración: OK")
            
            # Determinar status final
            servicios_con_error = sum(1 for servicio in health_status['servicios'].values() 
                                    if servicio.get('status') == 'error')
            servicios_degraded = sum(1 for servicio in health_status['servicios'].values() 
                                   if servicio.get('status') == 'warning')
            
            if servicios_con_error > 0:
                health_status['status'] = 'error'
            elif servicios_degraded > 0 or config_issues:
                health_status['status'] = 'degraded'
            
            timestamp_fin = datetime.now()
            tiempo_check = (timestamp_fin - timestamp_inicio).total_seconds()
            
            # Actualizar estadísticas
            self.stats['health_checks_realizados'] += 1
            if health_status['status'] == 'ok':
                self.stats['health_checks_exitosos'] += 1
            else:
                self.stats['health_checks_fallidos'] += 1
            
            # Calcular tiempo promedio
            total_checks = self.stats['health_checks_realizados']
            if total_checks > 1:
                self.stats['tiempo_promedio_check'] = ((self.stats['tiempo_promedio_check'] * (total_checks - 1)) + tiempo_check) / total_checks
            else:
                self.stats['tiempo_promedio_check'] = tiempo_check
            
            # Log de resultado
            status_emoji = {
                'ok': '✅',
                'degraded': '⚠️',
                'error': '❌'
            }
            
            self.logger.info(f"{status_emoji.get(health_status['status'], '❓')} [HEALTH] Verificación completada:")
            self.logger.info(f"   • Status general: {health_status['status']}")
            self.logger.info(f"   • Tiempo verificación: {tiempo_check:.3f}s")
            self.logger.info(f"   • Servicios OK: {len([s for s in health_status['servicios'].values() if s.get('status') == 'ok'])}/{len(health_status['servicios'])}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ [HEALTH] Error en verificación de salud: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'servicios': {},
                'configuracion': {}
            }

    def obtener_estadisticas_health(self) -> Dict[str, Any]:
        """Obtiene estadísticas de health checks"""
        stats = self.stats.copy()
        stats.update({
            'tasa_exito': (self.stats['health_checks_exitosos'] / self.stats['health_checks_realizados'] * 100) if self.stats['health_checks_realizados'] > 0 else 0
        })
        return stats

# Instancias globales de los clientes
binance_client = BinanceAPIClient()
telegram_client = TelegramAPIClient()
render_health_client = RenderHealthClient()

print("🌐 Clientes de API con logs detallados cargados correctamente")
