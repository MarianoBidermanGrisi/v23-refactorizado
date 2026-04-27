"""
=============================================================
  RENDER + BITGET CORE
  Base completa para desplegar en Render.com y operar en Bitget
=============================================================

ESTRUCTURA:
  1.  Imports y logging
  2.  Configuración desde variables de entorno (Render)
  3.  Parámetros de operación
  4.  Inicialización del exchange Bitget (ccxt)
  5.  Utilidades: Telegram, precisión decimal, memoria, balance, cooldown
  6.  Apertura de operaciones en Bitget  ← función principal
  7.  [STUB] escanear_mercado()          ← AQUÍ VA LA ESTRATEGIA
  8.  Flask App lista para Render.com
       /         → ping
       /health   → estado del bot
       /webhook  → updates de Telegram
  9.  Configuración automática del webhook de Telegram
  10. Main (arranque en Render)

VARIABLES DE ENTORNO requeridas en Render:
  BITGET_API_KEY        → tu API key de Bitget
  BITGET_SECRET_KEY     → tu secret de Bitget
  BITGET_PASSPHRASE     → tu passphrase de Bitget
  TELEGRAM_TOKEN        → token del bot de Telegram
  TELEGRAM_CHAT_ID      → uno o varios IDs separados por coma
  RENDER_EXTERNAL_URL   → Render la asigna automáticamente
  PORT                  → Render la asigna automáticamente
=============================================================
"""

# ==============================================================
#  1. IMPORTS Y LOGGING
# ==============================================================
import ccxt
import sys
import pandas as pd
import time
import requests
import os
import json
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from flask import Flask, request, jsonify
import threading
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================
#  2. CONFIGURACIÓN DESDE VARIABLES DE ENTORNO (RENDER)
# ==============================================================
def crear_config_desde_entorno():
    """Lee las credenciales desde las env vars de Render."""
    telegram_chat_ids_str = os.environ.get('TELEGRAM_CHAT_ID', '')
    telegram_chat_ids = [cid.strip() for cid in telegram_chat_ids_str.split(',') if cid.strip()]
    return {
        'bitget_api_key':    os.environ.get('BITGET_API_KEY'),
        'bitget_api_secret': os.environ.get('BITGET_SECRET_KEY'),
        'bitget_passphrase': os.environ.get('BITGET_PASSPHRASE'),
        'telegram_token':    os.environ.get('TELEGRAM_TOKEN'),
        'telegram_chat_ids': telegram_chat_ids,
    }


# ==============================================================
#  3. PARÁMETROS DE OPERACIÓN
# ==============================================================
MARGEN_USDT      = 1        # Margen por operación en USDT
PALANCA_ESTRICTA = 10       # Apalancamiento fijo (x10)
STOP_FIJO        = 0.016    # 1.6% stop loss fijo
COOLDOWN_OPERACION = 180    # Segundos mínimos entre operaciones
MEMORIA_FILE     = 'memoria_bot.json'


# ==============================================================
#  4. INICIALIZAR EXCHANGE BITGET (ccxt)
# ==============================================================
config = crear_config_desde_entorno()

exchange = ccxt.bitget({
    'apiKey':          config['bitget_api_key'],
    'secret':          config['bitget_api_secret'],
    'password':        config['bitget_passphrase'],
    'options':         {'defaultType': 'swap'},
    'enableRateLimit': True,
})


# ==============================================================
#  5. UTILIDADES
# ==============================================================

def enviar_telegram(msg, config=None, foto_buf=None):
    """
    Envía un mensaje de texto o foto a Telegram.
    Soporta múltiples chat IDs separados por coma en TELEGRAM_CHAT_ID.
    """
    try:
        if config is None:
            config = crear_config_desde_entorno()
        token    = config.get('telegram_token')
        chat_ids = config.get('telegram_chat_ids', [])

        if not token or not chat_ids:
            logger.warning("⚠️ Credenciales Telegram no configuradas. Mensaje no enviado.")
            return False

        for chat_id in chat_ids:
            try:
                if foto_buf:
                    url = f"https://api.telegram.org/bot{token}/sendPhoto"
                    foto_buf.seek(0)
                    files = {'photo': ('grafico.png', foto_buf.read(), 'image/png')}
                    data = {'chat_id': chat_id, 'caption': msg, 'parse_mode': 'Markdown'}
                    r = requests.post(url, files=files, data=data, timeout=30)
                else:
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    r = requests.post(
                        url,
                        data={'chat_id': chat_id, 'text': msg, 'parse_mode': 'Markdown'},
                        timeout=10
                    )
                if r.status_code == 200:
                    logger.info(f"✅ Telegram enviado a {chat_id}")
                else:
                    logger.warning(f"⚠️ Telegram error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Error enviando a {chat_id}: {e}")
                continue
        return True
    except Exception as e:
        logger.error(f"❌ Error crítico en enviar_telegram: {e}")
        return False


def generar_grafico_operacion(symbol, side, df, entrada, sl, tp):
    """Genera gráfico de imagen usando mplfinance con las líneas de operación."""
    try:
        df_plot = df.copy()
        df_plot['ts'] = pd.to_datetime(df_plot['ts'], unit='ms')
        df_plot.set_index('ts', inplace=True)
        df_plot.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume'}, inplace=True)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_plot[col] = df_plot[col].astype(float)

        apds = []
        if entrada and tp and sl:
            entry_line = [entrada] * len(df_plot)
            tp_line = [tp] * len(df_plot)
            sl_line = [sl] * len(df_plot)
            apds.append(mpf.make_addplot(entry_line, color='#FFD700', linestyle='-', width=1.5, panel=0))
            apds.append(mpf.make_addplot(tp_line, color='#00FF00', linestyle='-', width=1.5, panel=0))
            apds.append(mpf.make_addplot(sl_line, color='#FF0000', linestyle='-', width=1.5, panel=0))

        if 'supertrend' in df_plot.columns:
            st_line = df_plot['supertrend'].astype(float).tolist()
            apds.append(mpf.make_addplot(st_line, color='#00FFFF', linestyle='--', width=2.0, panel=0))

        title_str = f'{symbol} | {side.upper()} | ENTRADA: {entrada}'
        buf = BytesIO()
        fig, axes = mpf.plot(
            df_plot, 
            type='candle', 
            style='nightclouds',
            title=title_str,
            addplot=apds,
            volume=False,
            returnfig=True,
            figsize=(10, 6)
        )
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"❌ Error generando gráfico: {e}")
        return None


def a_decimal_estricto(numero, precision_raw):
    """
    Redondea 'numero' según la precisión del mercado (ROUND_DOWN).
    Acepta precisión como float (ej. 0.01) o int (ej. 2 decimales).
    """
    if numero is None:
        return None
    if isinstance(precision_raw, float):
        precision_str = format(precision_raw, 'f').rstrip('0')
        decimales = len(precision_str.split('.')[1]) if '.' in precision_str else 0
    else:
        decimales = int(precision_raw)
    valor = Decimal(str(numero)).quantize(
        Decimal(str(10 ** -decimales)), rounding=ROUND_DOWN
    )
    return str(valor)


def cargar_memoria():
    """Carga el estado persistente del bot desde disco."""
    if os.path.exists(MEMORIA_FILE):
        with open(MEMORIA_FILE, 'r') as f:
            return json.load(f)
    return {"operaciones_activas": [], "ultima_operacion_time": 0}


def guardar_memoria(datos):
    """Persiste el estado del bot en disco."""
    with open(MEMORIA_FILE, 'w') as f:
        json.dump(datos, f, indent=4)


def obtener_balance_real():
    """Retorna el saldo USDT disponible en Bitget Futuros."""
    try:
        balance = exchange.fetch_balance()
        for item in balance['info']:
            if item['marginCoin'] == 'USDT':
                return float(item['available'])
    except Exception:
        return 0.0
    return 0.0


def verificar_cooldown(memoria):
    """
    Devuelve True si ya pasaron COOLDOWN_OPERACION segundos
    desde la última operación ejecutada.
    """
    tiempo_desde_ultima = time.time() - memoria.get('ultima_operacion_time', 0)
    if tiempo_desde_ultima < COOLDOWN_OPERACION:
        restante = int(COOLDOWN_OPERACION - tiempo_desde_ultima)
        logger.info(f"⏳ Cooldown activo: {restante}s restantes")
        return False
    return True


# ==============================================================
#  6. APERTURA DE OPERACIONES EN BITGET
# ==============================================================

def _cerrar_posicion_emergencia(symbol, side, cant_tokens):
    """
    Cierra una posición abierta de forma inmediata (helper interno).
    Se usa cuando la verificación post-apertura falla.
    """
    try:
        cerrar_params = {
            'marginCoin': 'USDT',
            'marginMode': 'isolated',
            'reduceOnly': True,
        }
        cerrar_params.pop('posSide', None)
        close_side = 'sell' if side == 'buy' else 'buy'
        exchange.create_order(symbol, 'market', close_side, float(cant_tokens), params=cerrar_params)
        logger.info(f"   ✅ Posición cerrada de emergencia: {symbol}")
    except Exception as e:
        logger.warning(f"   ⚠️ Error al cerrar {symbol}: {e}")

def cerrar_operacion_estrategia(symbol, razon):
    """Cierre inmediato por pérdida de fundamento técnico."""
    try:
        posiciones = exchange.fetch_positions()
        posicion = next((p for p in posiciones 
                        if p['symbol'] == symbol and float(p.get('contracts', 0)) > 0), None)
        
        if not posicion:
            return
            
        cantidad = abs(float(posicion.get('contracts', 0)))
        lado_actual = posicion.get('side', '').lower()
        cerrar_side = 'sell' if lado_actual in ['buy', 'long'] else 'buy'
        
        params = {
            'marginCoin': 'USDT',
            'marginMode': 'isolated',
            'reduceOnly': True,
        }
        params.pop('posSide', None)
        
        exchange.create_order(symbol, 'market', cerrar_side, float(cantidad), params=params)
        
        # Actualizar memoria
        memoria = cargar_memoria()
        if symbol in memoria.get('operaciones_activas', []):
            memoria['operaciones_activas'].remove(symbol)
        if 'sl_dinamicos' in memoria and symbol in memoria['sl_dinamicos']:
            del memoria['sl_dinamicos'][symbol]
        guardar_memoria(memoria)
        
        logger.info(f"✅ {symbol}: CERRADO por {razon}")
        
        msg = f"⚠️ *CIERRE JERÁRQUICO*\nPar: `{symbol}`\nRazón: {razon}\n_Operación cerrada para proteger capital_"
        enviar_telegram(msg)
        
    except Exception as e:
        logger.error(f"❌ Error cerrando {symbol} por estrategia: {e}")

def actualizar_sl_dinamico(symbol, nuevo_sl, lado, nivel="Ajuste"):
    """Guarda un SL dinámico en memoria (Trailing) si es más estricto que el anterior."""
    memoria = cargar_memoria()
    if 'sl_dinamicos' not in memoria:
        memoria['sl_dinamicos'] = {}
        
    sl_actual = memoria['sl_dinamicos'].get(symbol)
    actualizado = False
    
    if lado == 'buy':
        if sl_actual is None or nuevo_sl > sl_actual:
            memoria['sl_dinamicos'][symbol] = nuevo_sl
            logger.info(f"🛡️ {symbol}: SL Dinámico (Long) subido a {nuevo_sl:.6f}")
            actualizado = True
    else:
        if sl_actual is None or nuevo_sl < sl_actual:
            memoria['sl_dinamicos'][symbol] = nuevo_sl
            logger.info(f"🛡️ {symbol}: SL Dinámico (Short) bajado a {nuevo_sl:.6f}")
            actualizado = True
            
    if actualizado:
        guardar_memoria(memoria)
        msg = f"🛡️ *NUEVO STOP LOSS ({nivel})*\nPar: `{symbol}`\nLado: `{lado.upper()}`\nSL Dinámico: `{nuevo_sl:.6f}`\n_Capital protegido_"
        enviar_telegram(msg)

def obtener_sl_dinamico(symbol):
    """Obtiene el SL dinámico de la memoria."""
    memoria = cargar_memoria()
    return memoria.get('sl_dinamicos', {}).get(symbol)


def abrir_operacion(symbol, side, entrada, df, memoria, tendencia, fuerza):
    """
    Ejecuta una operación de futuros perpetuos en Bitget.

    Flujo completo:
      1. Configura apalancamiento (PALANCA_ESTRICTA x)
      2. Calcula cantidad de tokens para usar exactamente MARGEN_USDT
         (dos intentos de cálculo para maximizar precisión)
      3. Verifica que el margen calculado esté dentro de tolerancia (±0.04 USDT)
      4. Calcula SL (STOP_FIJO %) y TP (rango × 0.24)
      5. Ejecuta la orden de mercado con SL/TP preset en Bitget
      6. Verifica la posición en el exchange post-apertura
      7. Cierra de emergencia si apalancamiento o margen no coinciden
      8. Notifica por Telegram y actualiza la memoria del bot

    Parámetros:
      symbol    : Par de trading ej. 'BTC/USDT:USDT'
      side      : 'buy' (long) o 'sell' (short)
      entrada   : Precio de entrada (float)
      df        : DataFrame OHLCV — mínimo 50 velas,
                  columnas: ['ts','open','high','low','close','vol']
      memoria   : Dict cargado con cargar_memoria()
      tendencia : String con tendencia detectada (ej. 'ALCISTA')
      fuerza    : Fuerza de señal (int, rango 0-7)
    """
    try:
        logger.info(f"\n{'='*55}")
        logger.info(f"🚀 SEÑAL CONFIRMADA: {side.upper()} en {symbol}")
        logger.info(f"   📊 Tendencia: {tendencia} | Fuerza señal: {fuerza}/7")
        logger.info(f"   ✨ REGLA DE ORO: {MARGEN_USDT} USDT x{PALANCA_ESTRICTA}")

        # ── PASO 1: Configurar apalancamiento ─────────────────────
        try:
            exchange.set_leverage(PALANCA_ESTRICTA, symbol, params={'marginCoin': 'USDT'})
            logger.info(f"   ✅ Apalancamiento {PALANCA_ESTRICTA}x configurado")
        except Exception as e:
            logger.error(f"❌ RECHAZADA: {symbol} no acepta x{PALANCA_ESTRICTA}. Error: {e}")
            return

        # ── PASO 2: Cargar mercado y calcular tokens ───────────────
        exchange.load_markets()
        market           = exchange.market(symbol)
        precision_amount = market['precision']['amount']

        # Intento 1 — cálculo directo
        cant_tokens_base = (MARGEN_USDT * PALANCA_ESTRICTA) / entrada
        cant_tokens      = a_decimal_estricto(cant_tokens_base, precision_amount)
        valor_pos_1      = float(cant_tokens) * entrada
        margen_real_1    = valor_pos_1 / PALANCA_ESTRICTA

        # Intento 2 — ROUND_HALF_UP si hay desvío
        if abs(margen_real_1 - MARGEN_USDT) > 0.000001:
            valor_objetivo  = MARGEN_USDT * PALANCA_ESTRICTA
            if isinstance(precision_amount, float):
                precision_str = format(precision_amount, 'f').rstrip('0')
                decimales = len(precision_str.split('.')[1]) if '.' in precision_str else 0
            else:
                decimales = int(precision_amount)
                
            cant_tokens_alt = Decimal(str(valor_objetivo)).quantize(
                Decimal(str(10 ** -decimales)), rounding=ROUND_HALF_UP
            )
            valor_pos_alt   = float(cant_tokens_alt) * entrada
            margen_real_alt = valor_pos_alt / PALANCA_ESTRICTA

            logger.info(f"   📐 Intento 1: {cant_tokens} tokens → {margen_real_1:.6f} USDT")
            logger.info(f"   📐 Intento 2: {cant_tokens_alt} tokens → {margen_real_alt:.6f} USDT")

            if abs(margen_real_alt - MARGEN_USDT) < abs(margen_real_1 - MARGEN_USDT):
                cant_tokens = str(cant_tokens_alt)
                margen_real = margen_real_alt
            else:
                margen_real = margen_real_1
        else:
            margen_real = margen_real_1

        logger.info(f"   📐 Tokens: {cant_tokens} | Precio entrada: {entrada} USDT")
        logger.info(f"   📐 Valor posición: {float(cant_tokens) * entrada:.4f} USDT")
        logger.info(f"   📐 Margen calculado: {margen_real:.6f} USDT")

        # ── PASO 3: Verificación previa de margen ─────────────────
        TOLERANCIA_MAX = 0.04
        diferencia = abs(margen_real - MARGEN_USDT)

        if diferencia > TOLERANCIA_MAX:
            logger.error(
                f"❌ RECHAZADA: margen {margen_real:.6f} USDT "
                f"— diferencia {diferencia:.6f} supera {TOLERANCIA_MAX}"
            )
            return
        if diferencia > 0.000001:
            logger.warning(f"⚠️ Margen {margen_real:.6f} USDT (dif: {diferencia:.6f}) — dentro de tolerancia, continuando...")
        else:
            logger.info(f"   ✅ REGLA DE ORO CUMPLIDA: {margen_real:.6f} USDT exacto")

        # ── PASO 4: Calcular SL y TP (Ratio 2:1 Estricto) ───────────────
        # SL fijo configurado (1.6%). TP exigido al doble del riesgo (3.2%)
        distancia_sl = entrada * STOP_FIJO
        distancia_tp = distancia_sl * 2
        
        sl = entrada - distancia_sl if side == 'buy' else entrada + distancia_sl
        tp = entrada + distancia_tp if side == 'buy' else entrada - distancia_tp

        sl_str = a_decimal_estricto(sl, market['precision']['price'])
        tp_str = a_decimal_estricto(tp, market['precision']['price'])

        logger.info(f"   🎯 SL: {sl_str} | TP: {tp_str}")

        # ── PASO 5: Ejecutar orden en Bitget ──────────────────────
        params = {
            'marginCoin':             'USDT',
            'marginMode':             'isolated',
            'presetStopLossPrice':    sl_str,
            'presetStopSurplusPrice': tp_str,
        }
        params.pop('posSide', None)

        logger.info(f"   🚀 Enviando orden {side.upper()} al exchange...")
        exchange.create_order(symbol, 'market', side, float(cant_tokens), params=params)

        # ── PASO 6: Verificación post-apertura ────────────────────
        time.sleep(1)
        posiciones = exchange.fetch_positions()
        posicion_encontrada = next(
            (p for p in posiciones
             if p['symbol'] == symbol and float(p.get('contracts', 0)) > 0),
            None
        )

        if posicion_encontrada is None:
            logger.warning(f"⚠️ No se encontró posición abierta para {symbol} — puede estar pendiente")
            return

        margen_verificado         = float(posicion_encontrada.get('initialMargin', 0))
        apalancamiento_verificado = float(posicion_encontrada.get('leverage', 0))

        logger.info(f"   🔍 Exchange reporta: margen={margen_verificado:.6f} USDT | palanca={apalancamiento_verificado}x")

        TOLERANCIA_POST = 0.04

        # ── PASO 7: Cierre de emergencia si algo no cuadra ────────
        if apalancamiento_verificado != PALANCA_ESTRICTA:
            logger.error(
                f"❌ APALANCAMIENTO INCORRECTO: {apalancamiento_verificado}x ≠ {PALANCA_ESTRICTA}x — cerrando"
            )
            _cerrar_posicion_emergencia(symbol, side, cant_tokens)
            enviar_telegram(
                f"❌ *OPERACIÓN RECHAZADA POR EXCHANGE*\n"
                f"Par: `{symbol}`\n"
                f"El exchange no respetó la REGLA DE ORO\n"
                f"Apalancamiento: `{apalancamiento_verificado}x` (esperado: {PALANCA_ESTRICTA}x)\n"
                f"_Posición cancelada automáticamente_"
            )
            return

        if abs(margen_verificado - MARGEN_USDT) >= TOLERANCIA_POST:
            logger.error(
                f"❌ MARGEN FUERA DE TOLERANCIA: {margen_verificado:.4f} USDT — cerrando"
            )
            _cerrar_posicion_emergencia(symbol, side, cant_tokens)
            enviar_telegram(
                f"❌ *MARGEN EXCEDE TOLERANCIA*\n"
                f"Par: `{symbol}`\n"
                f"Margen real: `{margen_verificado:.4f} USDT` (tolerancia: ±{TOLERANCIA_POST})\n"
                f"Apalancamiento: `{apalancamiento_verificado}x`\n"
                f"_Posición cancelada automáticamente_"
            )
            return

        # ── PASO 8: Éxito — guardar estado y notificar ────────────
        logger.info(f"   ✅ VERIFICACIÓN OK: {margen_verificado:.6f} USDT x{apalancamiento_verificado}")

        memoria['operaciones_activas'].append(symbol)
        memoria['ultima_operacion_time'] = time.time()
        guardar_memoria(memoria)

        # Generar Gráfico Visual
        foto_buf = generar_grafico_operacion(symbol, side, df, entrada, sl, tp)

        msg_exito = (
            f"🔥 *REGLA DE ORO EXACTA* ✅\n"
            f"Par: `{symbol}`\n"
            f"Lado: `{side.upper()}`\n"
            f"Margen verificado: `{margen_verificado:.6f} USDT` (x{apalancamiento_verificado})\n"
            f"Tendencia 15m: `{tendencia}`\n"
            f"Fuerza señal: `{fuerza}/7`\n"
            f"SL: `{sl_str}` | TP: `{tp_str}`\n"
            f"_Posición abierta exitosamente_ ✅"
        )
        enviar_telegram(msg=msg_exito, foto_buf=foto_buf)

        logger.info(f"✅ {side.upper()} abierto en {symbol}")
        logger.info(f"   💰 SL: {sl_str} | TP: {tp_str}")
        logger.info(f"   🎯 {margen_verificado:.6f} USDT x{apalancamiento_verificado}")
        logger.info(f"{'='*55}\n")

    except ccxt.InsufficientFunds as e:
        logger.warning(f"⚠️ FONDOS INSUFICIENTES para abrir {side.upper()} en {symbol}. Margin USDT requerido: {MARGEN_USDT}. (Error: {e})")
    except Exception as e:
        logger.error(f"⚠️ Error inesperado en abrir_operacion [{symbol}]: {e}", exc_info=True)


# ==============================================================
#  7. ESTRATEGIA TDF — Trend Duration Forecast
# ==============================================================
try:
    from estrategia_tdf import escanear_mercado
    logger.info("✅ Estrategia TDF cargada correctamente desde estrategia_tdf.py")
except ImportError:
    logger.warning("⚠️ estrategia_tdf.py no encontrado — usando modo stub")

    def escanear_mercado():
        """
        STUB de respaldo. Activa la estrategia real copiando
        estrategia_tdf.py y tdf_indicator.py junto a este archivo.
        """
        memoria = cargar_memoria()
        saldo   = obtener_balance_real()
        logger.info(f"[STUB] escanear_mercado() — saldo: {saldo:.2f} USDT | "
                    f"activas: {len(memoria['operaciones_activas'])}")
        logger.info("[STUB] Copia estrategia_tdf.py y tdf_indicator.py para activar la estrategia TDF.")


# ==============================================================
#  8. FLASK APP — LISTA PARA RENDER.COM
# ==============================================================
app = Flask(__name__)


def run_bot_loop():
    """
    Hilo de fondo (daemon).
    Llama a escanear_mercado() cada 300 segundos.
    Espera 10 s al inicio para que Flask y el webhook arranquen.
    """
    logger.info("🤖 Hilo del bot iniciado — configurando entorno...")
    setup_telegram_webhook()
    time.sleep(10)

    while True:
        try:
            escanear_mercado()
            time.sleep(300)
        except Exception as e:
            logger.error(f"❌ Error en ciclo de escaneo: {e}", exc_info=True)
            time.sleep(60)  # pausa extra ante error


# El inicio del hilo demonio ha sido movido abajo para garantizar dependencias.


@app.route('/')
def index():
    """Ping básico — Render lo usa para verificar que el servicio responde."""
    return "✅ Bot Bitget + Render — en línea.", 200


@app.route('/health', methods=['GET'])
def health_check():
    """
    Estado del bot en JSON.
    Render puede usar este endpoint como Health Check Path.
    """
    try:
        memoria = cargar_memoria()
        status = {
            "status":              "running",
            "timestamp":           datetime.now().isoformat(),
            "operaciones_activas": len(memoria.get("operaciones_activas", [])),
            "simbolos_activos":    memoria.get("operaciones_activas", []),
            "ultima_operacion_ts": memoria.get("ultima_operacion_time", 0),
            "balance_usdt":        obtener_balance_real(),
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error en /health: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    """Recibe updates de Telegram vía webhook (POST JSON)."""
    if request.is_json:
        update = request.get_json()
        logger.info(f"📩 Telegram update: {json.dumps(update)}")
        return jsonify({"status": "ok"}), 200
    return jsonify({"error": "Request must be JSON"}), 400


# ==============================================================
#  9. CONFIGURACIÓN AUTOMÁTICA DEL WEBHOOK DE TELEGRAM
# ==============================================================
def setup_telegram_webhook():
    """
    Al arrancar en Render, registra el webhook de Telegram
    apuntando a {RENDER_EXTERNAL_URL}/webhook.
    """
    token = os.environ.get('TELEGRAM_TOKEN')
    if not token:
        logger.warning("⚠️ TELEGRAM_TOKEN no definido — webhook omitido")
        return

    webhook_url = os.environ.get('WEBHOOK_URL')
    if not webhook_url:
        render_url = os.environ.get('RENDER_EXTERNAL_URL')
        if render_url:
            webhook_url = f"{render_url}/webhook"
        else:
            logger.warning("⚠️ RENDER_EXTERNAL_URL no definida — webhook omitido")
            return

    try:
        logger.info(f"🔗 Registrando webhook Telegram: {webhook_url}")
        # Eliminar webhook anterior antes de registrar el nuevo
        requests.get(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)
        time.sleep(1)
        r = requests.get(
            f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}",
            timeout=10
        )
        if r.status_code == 200:
            logger.info("✅ Webhook de Telegram registrado correctamente")
        else:
            logger.error(f"❌ Error al registrar webhook: {r.status_code} — {r.text}")
    except Exception as e:
        logger.error(f"❌ Excepción al configurar webhook: {e}")


# ==============================================================
#  10. INICIO DEL SISTEMA (THREADING + MAIN)
# ==============================================================

if __name__ == '__main__':
    bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
    bot_thread.start()
    logger.info("🧵 Hilo del bot lanzado en background")
    
    logger.info("🚀 Iniciando bot en Render.com (Modo Desarrollo)...")
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Flask escuchando en 0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
