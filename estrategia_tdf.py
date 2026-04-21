"""
=============================================================
  ESTRATEGIA TDF — Trend Duration Forecast
  Reemplaza el stub escanear_mercado() en render_bitget_core.py

  CÓMO USAR:
    1. Copia este archivo junto a render_bitget_core.py
    2. En render_bitget_core.py, al inicio del archivo agrega:
          from estrategia_tdf import escanear_mercado
    3. Elimina (o comenta) la función escanear_mercado() que
       ya existe en render_bitget_core.py (el stub vacío)

  DEPENDENCIAS:
    pip install numpy ccxt pandas requests flask
=============================================================
"""

import numpy as np
import logging
import time
import sys
from typing import Dict

logger = logging.getLogger(__name__)

# ── Importar desde los módulos de indicadores ───────────────────────────
from tdf_indicator import TrendDurationForecast
from vp_pivots_indicator import VolumeProfilePivots
from macd_sr_indicator import MacdSupportResistance
from adx_di_indicator import calcular_adx_di
from sqz_momentum_indicator import calcular_squeeze_momentum
from supertrend_indicator import Supertrend
from smc_indicator import SmartMoneyConcepts

# bot_web_service es el módulo padre que importa este archivo.
# Se accede vía sys.modules para evitar el circular import.
# Python ya tiene el módulo inicializado cuando escanear_mercado() se ejecuta.
def _core():
    """Retorna el módulo bot_web_service ya inicializado."""
    mod = sys.modules.get('bot_web_service') or sys.modules.get('__main__')
    if mod is None:
        raise RuntimeError("bot_web_service no está en sys.modules")
    return mod


# ==============================================================
#  CONFIGURACIÓN DE LA ESTRATEGIA
# ==============================================================

# Símbolos a escanear (top por volumen)
NUM_MONEDAS_ESCANEAR = 200

# Timeframe de análisis
TIMEFRAME = "15m"

# Cuántas velas descargar (300 mínimo para contextualizar VP y converge MACD EMA)
LIMIT_VELAS = 300

# Parámetros del TDF (iguales al Pine Script original)
TDF_LENGTH      = 50
TDF_SENSITIVITY = 3
TDF_SAMPLES     = 10

# Confianza mínima requerida para abrir operación (0-100)
CONFIANZA_MIN = 60

# Límite de operaciones simultáneas
MAX_OPERACIONES = 5

# Umbral ADX para considerar tendencia activa (4to filtro de confluencia)
ADX_UMBRAL = 20

# Estado de los indicadores por símbolo (se mantiene entre ciclos)
_tdf_instances: Dict[str, TrendDurationForecast] = {}
_vp_instances: Dict[str, VolumeProfilePivots] = {}
_macd_instances: Dict[str, MacdSupportResistance] = {}
_st_instances: Dict[str, Supertrend] = {}
_smc_instances: Dict[str, SmartMoneyConcepts] = {}


# ==============================================================
#  FUNCIÓN PRINCIPAL — reemplaza el stub en render_bitget_core
# ==============================================================

def escanear_mercado():
    """
    Estrategia completa basada en el Trend Duration Forecast.

    Llamada por run_bot_loop() cada 30 segundos.
    Por cada símbolo en SIMBOLOS:
      1. Descarga velas OHLCV confirmadas
      2. Pasa los cierres al TDF para actualizar el estado
      3. Si hay señal nueva (cambio de tendencia) con confianza ≥ CONFIANZA_MIN
         → abre operación con abrir_operacion()
    """
    core = _core()

    memoria = core.cargar_memoria()
    saldo   = core.obtener_balance_real()

    # Sincronizar memoria con el exchange real (Limpiar zombies si tocó TP/SL)
    try:
        posiciones = core.exchange.fetch_positions()
        posiciones_reales = [p['symbol'] for p in posiciones if float(p.get('contracts', 0)) > 0]
        memoria['operaciones_activas'] = list(set(posiciones_reales))
        core.guardar_memoria(memoria)
    except Exception as e:
        logger.error(f"[TDF-BOT] Error sincronizando posiciones: {e}")

    logger.info(f"[TDF-BOT] Ciclo de escaneo | Saldo: {saldo:.2f} USDT | "
                f"Activas: {len(memoria['operaciones_activas'])}/{MAX_OPERACIONES}")

    # No operar si ya se alcanzó el límite máximo
    if len(memoria.get("operaciones_activas", [])) >= MAX_OPERACIONES:
        logger.info(f"[TDF-BOT] Límite de {MAX_OPERACIONES} posiciones alcanzado — esperando cierres")
        return

    # Verificar cooldown global
    if not core.verificar_cooldown(memoria):
        return

    try:
        tickers = core.exchange.fetch_tickers()
        monedas = sorted(
            [{'s': s, 'v': t.get('quoteVolume', 0) or 0} for s, t in tickers.items() if ':USDT' in s],
            key=lambda x: float(x['v']), reverse=True
        )[:NUM_MONEDAS_ESCANEAR]
        simbolos_a_escanear = [m['s'] for m in monedas]
        logger.info(f"[TDF-BOT] 🔍 Escaneando top {NUM_MONEDAS_ESCANEAR} monedas por volumen...")
    except Exception as e:
        logger.error(f"[TDF-BOT] ❌ Error obteniendo símbolos por volumen: {e}")
        return

    for symbol in simbolos_a_escanear:
        # Re-leer memoria en cada ciclo de símbolo para evitar doble apertura
        memoria = core.cargar_memoria()
        if len(memoria.get("operaciones_activas", [])) >= MAX_OPERACIONES:
            logger.info("[TDF-BOT] Límite alcanzado durante el ciclo — deteniendo escaneo")
            break
        try:
            _procesar_simbolo(symbol, memoria, core.exchange, core.abrir_operacion)
        except Exception as e:
            logger.error(f"[TDF-BOT] Error procesando {symbol}: {e}", exc_info=True)
        time.sleep(0.05)   # pausa corta entre símbolos (rate limit manejado por ccxt)

    logger.info(f"[TDF-BOT] ✅ BARRIDO COMPLETO: {len(simbolos_a_escanear)} monedas analizadas. Próximo escaneo en 5 minutos.")


# ==============================================================
#  PROCESAMIENTO POR SÍMBOLO
# ==============================================================

def _procesar_simbolo(symbol, memoria, exchange, abrir_operacion):
    """Descarga velas, evalúa TDF y ejecuta señal si aplica."""

    # 1. Obtener o crear instancias de indicadores para este símbolo
    if symbol not in _tdf_instances:
        _tdf_instances[symbol] = TrendDurationForecast(
            length=TDF_LENGTH,
            trend_sensitivity=TDF_SENSITIVITY,
            samples=TDF_SAMPLES,
        )
        _vp_instances[symbol] = VolumeProfilePivots(
            period=200, 
            bins=50, 
            pivot_length=10, 
            pivot_filter=20
        )
        _macd_instances[symbol] = MacdSupportResistance(
            fast_length=12,
            slow_length=26,
            signal_length=9
        )
        _st_instances[symbol] = Supertrend(period=10, multiplier=3.0)
        _smc_instances[symbol] = SmartMoneyConcepts(pivot_length=10)
        logger.info(f"[TDF-BOT] Instancias TDF, VP, MACD, ST y SMC creadas para {symbol}")

    tdf = _tdf_instances[symbol]
    vp = _vp_instances[symbol]
    macd_ind = _macd_instances[symbol]
    st_ind = _st_instances[symbol]
    smc_ind = _smc_instances[symbol]

    # 2. Descargar velas OHLCV (solo velas cerradas = confirmadas)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT_VELAS)

    if not ohlcv or len(ohlcv) < 100:
        logger.warning(f"[TDF-BOT] {symbol}: insuficientes velas ({len(ohlcv)})")
        return

    # La última vela puede estar abierta — descartar (equivale a barstate.isconfirmed)
    ohlcv_confirmadas = ohlcv[:-1]

    # Extraer columnas individuales
    closes  = np.array([v[4] for v in ohlcv_confirmadas], dtype=float)
    volumes = np.array([v[5] for v in ohlcv_confirmadas], dtype=float)

    precio_actual = float(ohlcv[-1][4])   # cierre de la vela en curso

    # 3. Evaluar señal TDF
    eval_result = tdf.evaluar_entrada(closes, volumes)
    signal      = eval_result["signal"]      # 'BUY' | 'SELL' | None
    confidence  = eval_result["confidence"]
    tendencia   = eval_result["tendencia_str"]
    fuerza      = eval_result["fuerza"]
    detalle     = eval_result["detalle"]

    # Logging del estado (siempre, aunque no haya señal)
    trend_sym  = "↑" if detalle["trend"] == 1 else ("↓" if detalle["trend"] == -1 else "—")
    avg_str    = (f"AvgBull={detalle['avg_bullish']:.1f}" if detalle["avg_bullish"] else "AvgBull=N/A")
    prob_str   = (f"ProbLen={detalle['probable_length']:.1f}" if detalle["probable_length"] else "ProbLen=N/A")
    rem_str    = (f"Rem={detalle['remaining_bars']:.1f}" if detalle['remaining_bars'] is not None else "Rem=N/A")

    logger.info(
        f"[TDF-BOT] {symbol} | {trend_sym} bars={detalle['trend_count']} | "
        f"{avg_str} | {prob_str} | {rem_str} | conf={confidence}"
    )

    # 4. Evaluar Volume Profile + Pivots (Confluencia structural)
    latest_ohlcv = np.array(ohlcv_confirmadas, dtype=float)
    vp_result = vp.update(latest_ohlcv)
    
    # Evaluar MACD S/R (Confluencia de Momentum)
    macd_result = macd_ind.update(latest_ohlcv)

    # Evaluar ADX + DI (4to filtro de confluencia — Fuerza y Dirección de Tendencia)
    adx_result = calcular_adx_di(ohlcv_confirmadas, length=14, umbral=ADX_UMBRAL)
    logger.info(
        f"[ADX] {symbol} | ADX={adx_result['adx']} | DI+={adx_result['di_plus']} "
        f"| DI-={adx_result['di_minus']} | Dominio={adx_result['dominio']} "
        f"| TrendOK={adx_result['trend_ok']}"
    )

    # Evaluar Squeeze Momentum (5to filtro de confluencia — Explosión de volatilidad)
    sqz_result = calcular_squeeze_momentum(ohlcv_confirmadas)
    logger.info(
        f"[SQZMOM] {symbol} | val={sqz_result['val']} | bcolor={sqz_result['bcolor']} "
        f"| sqzOn={sqz_result['sqz_on']} | sqzOff={sqz_result['sqz_off']}"
    )

    # Evaluar Supertrend (6to filtro de confluencia)
    st_result = st_ind.calcular(ohlcv_confirmadas)
    logger.info(
        f"[ST] {symbol} | Bullish={st_result['is_bullish']} | Line={st_result['line']:.4f}"
    )

    # Evaluar SMC (7mo Filtro y Contexto de Order Blocks)
    smc_result = smc_ind.calcular(ohlcv_confirmadas)
    is_bullish_struct = smc_result["is_bullish_structure"]
    struct_str = "ALCISTA" if is_bullish_struct else "BAJISTA"
    logger.info(f"[SMC] {symbol} | Estructura={struct_str} | OBs_Bull={len(smc_result['bullish_obs'])} | OBs_Bear={len(smc_result['bearish_obs'])}")

    # 5. Aplicar Filtro Abierto de Confluencia Múltiple
    if signal is None:
        return

    vpc_delta_pos = vp_result["delta_positive"]
    vpc_poc = vp_result["poc_price"]
    macd_bullish = macd_result["is_bullish"]

    if signal == 'BUY':
        # ── BLOQUE ADX: 4to Filtro de Confluencia ──
        if not adx_result['trend_ok']:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: ADX={adx_result['adx']:.1f} < {ADX_UMBRAL} (sin tendencia).")
            return
        if adx_result['dominio'] != 'BULL':
            logger.info(f"[Confluencia] BUY en {symbol} denegada: DI- domina sobre DI+ (dominio={adx_result['dominio']}).")
            return

        # ── BLOQUE SQZMOM: 5to Filtro de Confluencia ──
        if sqz_result['sqz_on']:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: Squeeze Activo (Mercado en compresión).")
            return
        if not sqz_result['momentum_bullish']:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: SQZMOM Histograma Negativo.")
            return

        # Bloques de Cancelación Duros (VP + MACD)
        if not vpc_delta_pos:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: VP Delta Negativo (Vendedores al mando).")
            return
        if not macd_bullish:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: MACD es Bajista (Bajo Señal).")
            return
            
        # ── BLOQUE SUPERTREND: 6to Filtro de Confluencia ──
        if not st_result["is_bullish"]:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: Supertrend es Bajista.")
            return

        # ── BLOQUE SMC: 7mo Filtro de Confluencia y Localización ──
        if not is_bullish_struct:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: Estructura SMC es Bajista (esperando CHoCH).")
            return
            
        for ob in smc_result["bearish_obs"]:
            distancia_al_ob = (ob['bottom'] - precio_actual) / precio_actual
            if precio_actual <= ob['top'] and distancia_al_ob < 0.008:
                logger.info(f"[Confluencia] BUY en {symbol} denegada: Muro institucional cercano (Bearish OB en {ob['bottom']:.4f}).")
                return

        tendencia += " + 7-X CONFIRMED"

        # Bonus SQZMOM: Squeeze liberando con aceleración
        if sqz_result['sqz_off'] and sqz_result['momentum_accel']:
            confidence += 15
            tendencia += " (SQZ Explosión 🚀)"

        # Bonus ADX: cruce recíente DI+ sobre DI- suma confianza extra
        if adx_result['cruce_long']:
            confidence += 10
            tendencia += " (DI+ Cruce)"
            
        # Bonus SMC: Rebote en Order Block de Demanda
        near_bull_ob = any(ob['bottom'] <= precio_actual <= ob['top'] * 1.005 for ob in smc_result["bullish_obs"])
        if near_bull_ob:
            confidence += 20
            tendencia += " (Rebote OB Demanda)"
        
        # Filtros de Posición y Rebotes para puntuar Confianza Final
        if precio_actual >= vpc_poc:
            confidence += 15
            tendencia += " (Sobre PoC)"
        else:
            near_vp_sup = any(abs(precio_actual - sup)/precio_actual < 0.005 for sup in vp_result["active_supports"])
            if near_vp_sup:
                confidence += 20
                tendencia += " (Rebote PoC-Soporte)"
            else:
                logger.info(f"[Confluencia] BUY en {symbol} denegada: Precio bajo PoC y sin soporte VP cercano.")
                return
                
        # Validación Extra MACD: Rebote en Soporte Dinámico (Opcional, suma mucha confianza)
        near_macd_sup = any(abs(precio_actual - sup)/precio_actual < 0.005 for sup in macd_result["active_supports"])
        if near_macd_sup:
            confidence += 15
            tendencia += " (Rebote MACD)"

    elif signal == 'SELL':
        # ── BLOQUE ADX: 4to Filtro de Confluencia ──
        if not adx_result['trend_ok']:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: ADX={adx_result['adx']:.1f} < {ADX_UMBRAL} (sin tendencia).")
            return
        if adx_result['dominio'] != 'BEAR':
            logger.info(f"[Confluencia] SELL en {symbol} denegada: DI+ domina sobre DI- (dominio={adx_result['dominio']}).")
            return

        # ── BLOQUE SQZMOM: 5to Filtro de Confluencia ──
        if sqz_result['sqz_on']:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: Squeeze Activo (Mercado en compresión).")
            return
        if sqz_result['momentum_bullish']:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: SQZMOM Histograma Positivo.")
            return

        # Bloques de Cancelación Duros (VP + MACD)
        if vpc_delta_pos:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: VP Delta Positivo (Compradores al mando).")
            return
        if macd_bullish:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: MACD es Alcista (Sobre Señal).")
            return
            
        # ── BLOQUE SUPERTREND: 6to Filtro de Confluencia ──
        if st_result["is_bullish"]:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: Supertrend es Alcista.")
            return

        # ── BLOQUE SMC: 7mo Filtro de Confluencia y Localización ──
        if is_bullish_struct:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: Estructura SMC es Alcista (esperando CHoCH).")
            return
            
        for ob in smc_result["bullish_obs"]:
            distancia_al_ob = (precio_actual - ob['top']) / precio_actual
            if precio_actual >= ob['bottom'] and distancia_al_ob < 0.008:
                logger.info(f"[Confluencia] SELL en {symbol} denegada: Muro institucional cercano (Bullish OB en {ob['top']:.4f}).")
                return

        tendencia += " + 7-X CONFIRMED"

        # Bonus SQZMOM: Squeeze liberando con aceleración
        if sqz_result['sqz_off'] and sqz_result['momentum_accel']:
            confidence += 15
            tendencia += " (SQZ Explosión 🚀)"

        # Bonus ADX: cruce reciente DI- sobre DI+ suma confianza extra
        if adx_result['cruce_short']:
            confidence += 10
            tendencia += " (DI- Cruce)"

        # Bonus SMC: Rebote en Order Block de Oferta
        near_bear_ob = any(ob['top'] >= precio_actual >= ob['bottom'] * 0.995 for ob in smc_result["bearish_obs"])
        if near_bear_ob:
            confidence += 20
            tendencia += " (Rechazo OB Oferta)"

        # Filtros de Posición y Rebotes
        if precio_actual <= vpc_poc:
            confidence += 15
            tendencia += " (Bajo PoC)"
        else:
            near_vp_res = any(abs(res - precio_actual)/precio_actual < 0.005 for res in vp_result["active_resistances"])
            if near_vp_res:
                confidence += 20
                tendencia += " (Rechazo PoC-Resist)"
            else:
                logger.info(f"[Confluencia] SELL en {symbol} denegada: Precio sobre PoC y sin resistencia VP cercana.")
                return
                
        # Validación Extra MACD
        near_macd_res = any(abs(res - precio_actual)/precio_actual < 0.005 for res in macd_result["active_resistances"])
        if near_macd_res:
            confidence += 15
            tendencia += " (Rechazo MACD)"

    # Ajustar fuerza final base 7 (luego del bonus de confluencia)
    confidence = min(100, confidence)
    fuerza = round((confidence / 100) * 7)

    # 6. Doble check: confianza mínima post-confluencia
    if confidence < CONFIANZA_MIN:
        logger.info(f"[TDF-BOT] {symbol}: señal {signal} descartada (conf={confidence} < {CONFIANZA_MIN})")
        return

    # 6. El símbolo ya tiene una posición activa
    if symbol in memoria.get("operaciones_activas", []):
        logger.info(f"[TDF-BOT] {symbol}: ya tiene posición activa")
        return

    # 7. Construir DataFrame mínimo para abrir_operacion()
    import pandas as pd
    df = pd.DataFrame(ohlcv_confirmadas[-60:], columns=["ts", "open", "high", "low", "close", "vol"])
    if "st_series" in st_result:
        df["supertrend"] = st_result["st_series"][-60:]

    side = eval_result["side"]   # 'buy' o 'sell'

    logger.info(
        f"[TDF-BOT] 🚀 SEÑAL {signal} en {symbol} | "
        f"Conf={confidence} | Fuerza={fuerza}/7 | {tendencia}"
    )

    # 8. Ejecutar operación
    abrir_operacion(
        symbol    = symbol,
        side      = side,
        entrada   = precio_actual,
        df        = df,
        memoria   = memoria,
        tendencia = tendencia,
        fuerza    = fuerza,
    )
