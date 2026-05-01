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
from wavetrend_indicator import WaveTrendIndicator
from adx_di_indicator import calcular_adx_di
from sqz_momentum_indicator import calcular_squeeze_momentum
from supertrend_indicator import Supertrend
from smc_indicator import SmartMoneyConcepts
from williams_vix_fix import calcular_wvf
from filtro_confluencia_historica import analizar_confluencia_historica
from trendlines_indicator import TrendlinesBreaks
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
TIMEFRAME = "5m"

# Cuántas velas descargar (500 en 5m ≈ ~41 horas de contexto para VP y TDF)
LIMIT_VELAS = 500

# Parámetros del TDF (iguales al Pine Script original)
TDF_LENGTH      = 50
TDF_SENSITIVITY = 3
TDF_SAMPLES     = 10

# Confianza mínima requerida para abrir operación (0-100)
CONFIANZA_MIN = 50

# Límite de operaciones simultáneas
MAX_OPERACIONES = 5

# Umbral ADX para considerar tendencia activa (4to filtro de confluencia)
ADX_UMBRAL = 25

# Umbral 9no Filtro: Movimiento máximo permitido desde el inicio de confluencia
# Subido a 1.2% para dar margen a la confirmación del Filtro 10 (Trendlines)
MOVIMIENTO_MAX_FILTRO_9 = 1.2

# Filtro de Confirmación SuperTrend: Mínimo de velas consecutivas en la misma dirección
# antes de entrar. Evita entradas cuando el ST acaba de cambiar y puede regresar.
ST_CONFIRMACION_BARRAS = 2

# Estado de los indicadores por símbolo (se mantiene entre ciclos)
_tdf_instances: Dict[str, TrendDurationForecast] = {}
_vp_instances: Dict[str, VolumeProfilePivots] = {}
_wt_instances: Dict[str, WaveTrendIndicator] = {}
_st_instances: Dict[str, Supertrend] = {}
_smc_instances: Dict[str, SmartMoneyConcepts] = {}
_tl_instances:  Dict[str, TrendlinesBreaks]   = {}

def _contar_barras_supertrend(ohlcv_confirmadas: list, st_ind) -> int:
    """
    Cuenta cuántas velas consecutivas lleva el SuperTrend en la misma
    dirección que la vela más reciente.
    Mínimo útil: 3 barras. Si retorna < ST_CONFIRMACION_BARRAS, la entrada se rechaza.
    """
    try:
        # Necesitamos el historial de señales del ST para las últimas N velas
        # Calculamos el ST sobre las últimas 20 velas para encontrar el cambio más reciente
        ventana = min(20, len(ohlcv_confirmadas))
        direccion_actual = None
        barras_consecutivas = 0

        for i in range(len(ohlcv_confirmadas) - ventana, len(ohlcv_confirmadas)):
            resultado = st_ind.calcular(ohlcv_confirmadas[:i+1])
            es_bullish = resultado['is_bullish']

            if direccion_actual is None:
                direccion_actual = es_bullish
                barras_consecutivas = 1
            elif es_bullish == direccion_actual:
                barras_consecutivas += 1
            else:
                # Cambio de dirección: reiniciar contador
                direccion_actual = es_bullish
                barras_consecutivas = 1

        return barras_consecutivas
    except Exception:
        # Si falla el cálculo, retornamos el mínimo para no bloquear
        return ST_CONFIRMACION_BARRAS


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
    posiciones_activas_dict = {}
    try:
        posiciones = core.exchange.fetch_positions()
        for p in posiciones:
            if float(p.get('contracts', 0)) > 0:
                posiciones_activas_dict[p['symbol']] = p
                
        memoria['operaciones_activas'] = list(posiciones_activas_dict.keys())
        
        # Limpiar sl_dinamicos zombies
        if 'sl_dinamicos' in memoria:
            zombies = [s for s in memoria['sl_dinamicos'] if s not in memoria['operaciones_activas']]
            for z in zombies:
                del memoria['sl_dinamicos'][z]
                
        core.guardar_memoria(memoria)
    except Exception as e:
        logger.error(f"[TDF-BOT] Error sincronizando posiciones: {e} - ABORTANDO CICLO")
        return  # ABORTO CRÍTICO: Si no sabemos qué hay abierto, no podemos gestionar ni abrir nada.

    logger.info(f"[TDF-BOT] Ciclo de escaneo | Saldo: {saldo:.2f} USDT | "
                f"Activas: {len(memoria.get('operaciones_activas', []))}/{MAX_OPERACIONES}")

    # Verificar cooldown global
    if not core.verificar_cooldown(memoria):
        return

    try:
        tickers = core.exchange.fetch_tickers()
        monedas = sorted(
            [{'s': s, 'v': t.get('quoteVolume', 0) or 0} for s, t in tickers.items() if ':USDT' in s],
            key=lambda x: float(x['v']), reverse=True
        )[:NUM_MONEDAS_ESCANEAR]
        
        # Asegurarnos de que las operaciones activas siempre estén al principio para evaluarlas
        simbolos_a_escanear = list(memoria.get('operaciones_activas', []))
        for m in monedas:
            if m['s'] not in simbolos_a_escanear:
                simbolos_a_escanear.append(m['s'])
                
        logger.info(f"[TDF-BOT] 🔍 Escaneando {len(simbolos_a_escanear)} monedas...")
    except Exception as e:
        logger.error(f"[TDF-BOT] ❌ Error obteniendo símbolos por volumen: {e}")
        return

    for symbol in simbolos_a_escanear:
        # Re-leer memoria en cada ciclo para evitar doble apertura
        memoria = core.cargar_memoria()
        en_posicion = symbol in memoria.get("operaciones_activas", [])
        
        if not en_posicion and len(memoria.get("operaciones_activas", [])) >= MAX_OPERACIONES:
            # Límite alcanzado, pero seguimos iterando para evaluar los que SÍ están activos
            continue
            
        posicion = posiciones_activas_dict.get(symbol)
        
        try:
            _procesar_simbolo(symbol, memoria, core.exchange, core.abrir_operacion, posicion)
        except Exception as e:
            logger.error(f"[TDF-BOT] Error procesando {symbol}: {e}", exc_info=True)
        time.sleep(0.05)   # pausa corta entre símbolos (rate limit manejado por ccxt)

    logger.info(f"[TDF-BOT] ✅ BARRIDO COMPLETO: {len(simbolos_a_escanear)} monedas analizadas. Próximo escaneo en 5 minutos.")


# ==============================================================
#  PROCESAMIENTO POR SÍMBOLO
# ==============================================================

def _procesar_simbolo(symbol, memoria, exchange, abrir_operacion, posicion=None):
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
        _wt_instances[symbol] = WaveTrendIndicator(n1=10, n2=21)
        _st_instances[symbol] = Supertrend(period=10, multiplier=4.0)
        _smc_instances[symbol] = SmartMoneyConcepts(pivot_length=10)
        logger.info(f"[TDF-BOT] Instancias TDF, VP, WaveTrend, ST y SMC creadas para {symbol}")

    tdf = _tdf_instances[symbol]
    vp = _vp_instances[symbol]
    wt_ind = _wt_instances[symbol]
    st_ind = _st_instances[symbol]
    smc_ind = _smc_instances[symbol]

    if symbol not in _tl_instances:
        _tl_instances[symbol] = TrendlinesBreaks(length=10, mult=1.0)
    tl_ind = _tl_instances[symbol]

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
    
    # Evaluar WaveTrend (Confluencia de Momentum)
    import pandas as pd
    wt_df = pd.DataFrame(ohlcv_confirmadas, columns=["ts", "open", "high", "low", "close", "vol"])
    wt_result = wt_ind.calculate(wt_df)

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

    # Evaluar Williams Vix Fix (8vo Filtro — Fondos y Complacencia)
    wvf_result = calcular_wvf(ohlcv_confirmadas)
    if wvf_result:
        logger.info(
            f"[WVF] {symbol} | Val={wvf_result['wvf']:.2f} | BottomSignal={wvf_result['is_bottom_signal']} "
            f"| Complacencia={wvf_result['is_complacency']}"
        )

    # Evaluar Trendlines with Breaks (10mo Filtro - LuxAlgo)
    tl_result = tl_ind.update(ohlcv_confirmadas)
    logger.info(
        f"[TL-LUX] {symbol} | In_Up={tl_result['in_uptrend']} | In_Down={tl_result['in_downtrend']} "
        f"| Break_Up={tl_result['upper_break']} | Break_Down={tl_result['lower_break']}"
    )
    # ── NUEVO: GESTIÓN DE OPERACIONES ACTIVAS (Cierre Jerárquico) ──
    if posicion is not None:
        _evaluar_operacion_activa(
            symbol=symbol,
            posicion=posicion,
            precio_actual=precio_actual,
            adx_result=adx_result,
            sqz_result=sqz_result,
            st_result=st_result,
            wt_result=wt_result,
            vp_result=vp_result,
            smc_result=smc_result,
            wvf_result=wvf_result
        )
        return

    # 5. Aplicar Filtro Abierto de Confluencia Múltiple
    if signal is None:
        return

    side = eval_result["side"]   # 'buy' o 'sell'

    vpc_delta_pos = vp_result["delta_positive"]
    vpc_poc = vp_result["poc_price"]
    wt_bullish = wt_result["is_bullish"]

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

        # Bloques de Cancelación Duros (VP + WaveTrend)
        if not vpc_delta_pos:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: VP Delta Negativo (Vendedores al mando).")
            return
        if not wt_bullish:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: WaveTrend es Bajista (WT1 < WT2).")
            return
            
        # ── BLOQUE SUPERTREND: 6to Filtro de Confluencia ──
        if not st_result["is_bullish"]:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: Supertrend es Bajista.")
            return
            
        # ── FILTRO DE CONFIRMACIÓN ST: Mínimo ST_CONFIRMACION_BARRAS velas alcistas ──
        barras_st = _contar_barras_supertrend(ohlcv_confirmadas, st_ind)
        if barras_st < ST_CONFIRMACION_BARRAS:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: ST recién girado alcista ({barras_st} barra/s < {ST_CONFIRMACION_BARRAS} requeridas). Esperando confirmación.")
            return

        # ── FILTRO 10: TRENDLINES WITH BREAKS (LuxAlgo) ──
        if not tl_result['in_uptrend']:
            logger.info(f"[Confluencia] BUY en {symbol} denegada: Precio bajo línea de resistencia (LuxAlgo).")
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

        # Bonus Filtro 10: Ruptura fresca de trendline (LuxAlgo)
        if tl_result['upper_break']:
            confidence += 25
            tendencia += " (TL Break 🚀)"

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
                
        # Validación Extra WaveTrend: Cruce al alza reciente (Opcional, suma mucha confianza)
        if wt_result['cross_up']:
            confidence += 15
            tendencia += " (WT CrossUp 🟢)"

        # Bonus WVF: Pico de miedo/fondo detectado
        if wvf_result and wvf_result['is_bottom_signal']:
            confidence += 20
            tendencia += " (WVF Bottom 🟢)"

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

        # Bloques de Cancelación Duros (VP + WaveTrend)
        if vpc_delta_pos:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: VP Delta Positivo (Compradores al mando).")
            return
        if wt_bullish:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: WaveTrend es Alcista (WT1 > WT2).")
            return
            
        # ── BLOQUE SUPERTREND: 6to Filtro de Confluencia ──
        if st_result["is_bullish"]:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: Supertrend es Alcista.")
            return
            
        # ── FILTRO DE CONFIRMACIÓN ST: Mínimo ST_CONFIRMACION_BARRAS velas bajistas ──
        barras_st = _contar_barras_supertrend(ohlcv_confirmadas, st_ind)
        if barras_st < ST_CONFIRMACION_BARRAS:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: ST recién girado bajista ({barras_st} barra/s < {ST_CONFIRMACION_BARRAS} requeridas). Esperando confirmación.")
            return

        # ── FILTRO 10: TRENDLINES WITH BREAKS (LuxAlgo) ──
        if not tl_result['in_downtrend']:
            logger.info(f"[Confluencia] SELL en {symbol} denegada: Precio sobre línea de soporte (LuxAlgo).")
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

        # Bonus Filtro 10: Ruptura fresca de trendline (LuxAlgo)
        if tl_result['lower_break']:
            confidence += 25
            tendencia += " (TL Break 📉)"

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
                
        # Validación Extra WaveTrend: Cruce a la baja reciente
        if wt_result['cross_down']:
            confidence += 15
            tendencia += " (WT CrossDown 🔴)"

        # ── BLOQUE WVF: 8vo Filtro de Complacencia para Shorts ──
        if wvf_result and not wvf_result['is_complacency']:
            # Si hay volatilidad bajista alta (miedo), no es buen momento para shortear "tardío"
            if wvf_result['wvf'] > wvf_result['upper_band'] * 0.8:
                logger.info(f"[Confluencia] SELL en {symbol} denegada: WVF alto ({wvf_result['wvf']:.1f}), detectado miedo de fondo.")
                return
            
        if wvf_result and wvf_result['is_complacency']:
            confidence += 15
            tendencia += " (Complacencia WVF 🔵)"

    # ── 9no FILTRO: CONFLUENCIA HISTÓRICA (Entradas Tardías) ──
    # Solo se ejecuta si llegamos hasta aquí (los 8 filtros previos alineados)
    indicadores_para_log = {
        "tdf": eval_result,
        "adx": adx_result,
        "sqz": sqz_result,
        "vp": vp_result,
        "wt": wt_result,
        "st": st_result,
        "smc": smc_result,
        "wvf": wvf_result
    }
    
    pasa_filtro_9, info_filtro_9 = analizar_confluencia_historica(
        symbol=symbol,
        side=side,
        ohlcv_confirmadas=ohlcv_confirmadas,
        indicadores_calculados=indicadores_para_log,
        umbral_movimiento=MOVIMIENTO_MAX_FILTRO_9
    )

    if not pasa_filtro_9:
        logger.info(f"[TDF-BOT] {symbol}: SEÑAL BLOQUEADA por 9no Filtro (Entrada tardía: {info_filtro_9['movimiento_pct']}% > {MOVIMIENTO_MAX_FILTRO_9}%)")
        return

    # Si pasa, añadimos info al log de tendencia
    tendencia += f" | Hist: {info_filtro_9['movimiento_pct']}% ({info_filtro_9['velas_atras']}v)"

    # Ajustar fuerza final base 7 (luego del bonus de confluencia)
    confidence = min(100, confidence)
    fuerza = round((confidence / 100) * 7)

    # 6. Doble check: confianza mínima post-confluencia
    if confidence < CONFIANZA_MIN:
        logger.info(f"[TDF-BOT] {symbol}: señal {signal} descartada (conf={confidence} < {CONFIANZA_MIN})")
        return

    # El chequeo de posición activa ya se realizó en la lógica de Cierre Jerárquico.

    # 7. Construir DataFrame mínimo para abrir_operacion()
    import pandas as pd
    df = pd.DataFrame(ohlcv_confirmadas[-60:], columns=["ts", "open", "high", "low", "close", "vol"])
    if "st_series" in st_result:
        df["supertrend"] = st_result["st_series"][-60:]

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

def _evaluar_operacion_activa(symbol, posicion, precio_actual, adx_result, sqz_result, st_result, wt_result, vp_result, smc_result, wvf_result):
    """
    Evaluación jerárquica de operaciones activas para Cierre Anticipado o Trailing SL.
    TP es 3.2% y SL es 1.6% (Unleveraged price %).
    """
    core = _core()
    entrada = abs(float(posicion.get('entryPrice', 0)))
    
    side_raw = posicion.get('side', '').lower()
    lado = 'buy' if side_raw in ['buy', 'long'] else 'sell'
    
    if lado == 'buy':
        pnl_pct = (precio_actual - entrada) / entrada * 100
    else:
        pnl_pct = (entrada - precio_actual) / entrada * 100
        
    # Verificar SL Dinámico interno si existe
    sl_dinamico = core.obtener_sl_dinamico(symbol)
    if sl_dinamico:
        if lado == 'buy' and precio_actual <= sl_dinamico:
            core.cerrar_operacion_estrategia(symbol, f"TRAILING_SL_TOCADO ({sl_dinamico:.4f})")
            return
        elif lado == 'sell' and precio_actual >= sl_dinamico:
            core.cerrar_operacion_estrategia(symbol, f"TRAILING_SL_TOCADO ({sl_dinamico:.4f})")
            return
            
    filtros_rotos = []
    
    # ── NIVEL 1: FILTROS DUROS (Cierre Estructural) ──
    # Supertrend opuesto es un verdadero Cambio de Carácter (CHoCH)
    if lado == 'buy' and not st_result['is_bullish']:
        filtros_rotos.append("ST_BEARISH")
    if lado == 'sell' and st_result['is_bullish']:
        filtros_rotos.append("ST_BULLISH")
        
    if filtros_rotos:
        logger.warning(f"[MONITOR] {symbol}: FILTROS ROTOS → {filtros_rotos} | PnL: {pnl_pct:.2f}%")
        
        if pnl_pct < 0.2:  # Pérdida o ganancia marginal (< 2% ROE)
            logger.warning(f"[MONITOR] {symbol}: Cerrando inmediato por filtros rotos.")
            core.cerrar_operacion_estrategia(symbol, f"FILTROS_ROTOS: {', '.join(filtros_rotos)}")
            return
        else:
            logger.info(f"[MONITOR] {symbol}: Filtros rotos pero en ganancia ({pnl_pct:.2f}%). Protegiendo con BE.")
            nuevo_sl = entrada * 1.001 if lado == 'buy' else entrada * 0.999
            core.actualizar_sl_dinamico(symbol, nuevo_sl, lado, nivel="BE por Filtros Rotos")
            return
    # En lugar de cerrar el trade, si notamos debilidad, protegemos el capital.
    bonos_perdidos = []
    
    if adx_result['adx'] < 20:
        bonos_perdidos.append("ADX_DEBIL")
        
    if lado == 'buy' and adx_result['dominio'] != 'BULL':
        bonos_perdidos.append("DI_INVERTIDO")
    if lado == 'sell' and adx_result['dominio'] != 'BEAR':
        bonos_perdidos.append("DI_INVERTIDO")
        
    if sqz_result['sqz_on']:
        bonos_perdidos.append("SQZ_COMPRESION")
        
    if lado == 'buy' and not wt_result.get('is_bullish'):
        bonos_perdidos.append("WT_BEARISH")
    if lado == 'sell' and wt_result.get('is_bullish'):
        bonos_perdidos.append("WT_BULLISH")
        
    if lado == 'buy' and not vp_result.get('delta_positive'):
        bonos_perdidos.append("VP_DELTA_NEGATIVO")
    if lado == 'sell' and vp_result.get('delta_positive'):
        bonos_perdidos.append("VP_DELTA_POSITIVO")
        
    # Si perdimos bonos y PnL > 2.0% precio (20% ROE), SL a Breakeven +0.2%
    # Ajustado para el nuevo TP de 3.0%
    if bonos_perdidos and pnl_pct >= 2.0:
        logger.info(f"[MONITOR] {symbol}: Ajustando SL dinámico. Bonos perdidos: {bonos_perdidos}")
        nuevo_sl = entrada * 1.002 if lado == 'buy' else entrada * 0.998
        core.actualizar_sl_dinamico(symbol, nuevo_sl, lado, nivel="Nivel 2: Breakeven")
        
    # ── NIVEL 3: TRAILING AGRESIVO ──
    # Si PnL > 2.3% precio (23% ROE), pegamos el SL a 0.5% del precio actual
    # Se activa antes del TP (3.0%) para asegurar ganancias máximas en el camino
    if pnl_pct >= 2.3:
        distancia = precio_actual * 0.005
        nuevo_sl = precio_actual - distancia if lado == 'buy' else precio_actual + distancia
        core.actualizar_sl_dinamico(symbol, nuevo_sl, lado, nivel="Nivel 3: Trailing Agresivo")
