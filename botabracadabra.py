import os
import json
import time
import logging
import requests
import asyncio
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv

# ==========================================================
# 1. CONFIGURACIÓN DE LOGGING
# ==========================================================
LOG_TO_FILE = os.environ.get('BOT_LOG_TO_FILE', '1') == '1'
LOG_LEVEL = os.environ.get('BOT_LOG_LEVEL', 'INFO')

_handlers = [logging.StreamHandler()]
if LOG_TO_FILE:
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_completo.log")
    _handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_handlers
)
log = logging.getLogger(__name__)

# Memoria temporal
ALERTS_HISTORY = {} 
PEAK_PRICES    = {} 
COOLDOWNS      = {} 
SESSION_ACTIVE_SYMBOLS = set()
DAILY_STATS    = {'tp': 0, 'sl': 0, 'be': 0, 'timeout': 0, 'pnl': 0.0, 'fees': 0.0, 'tp_names': [], 'sl_names': [], 'be_names': [], 'timeout_names': []}
TRADE_ENTRIES  = {}  # entry data for CSV log
TRAIL_COUNTS   = {}
PREMATURE_SL_MONITOR = {}
LAST_KNOWN_INDICATORS = {}
ADVERSE_PRICES = {}
PRICE_PATHS = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_PATHS_DIR = os.path.join(BASE_DIR, 'price_paths')
os.makedirs(PRICE_PATHS_DIR, exist_ok=True)
TRADES_CSV_PATH = os.path.join(BASE_DIR, 'trades.csv')
PREMATURE_SL_CSV_PATH = os.path.join(BASE_DIR, 'premature_sl.csv')
TRADE_ENTRIES_PATH = os.path.join(BASE_DIR, 'trade_entries.json')
SIGNALS_LOG_PATH = os.path.join(BASE_DIR, 'signals_log.csv')

SIGNAL_LOG_HEADERS = [
    'time','symbol','side','price','stc','hma','ema200','poc',
    'sl_proj','tp_proj','volume_ratio','taken','reason_skipped'
]

def _save_trade_entries():
    try:
        data = {}
        for sym, e in TRADE_ENTRIES.items():
            data[sym] = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in e.items()}
        with open(TRADE_ENTRIES_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        log.error(f"Error guardando trade_entries: {ex}")

def _load_trade_entries():
    try:
        if not os.path.exists(TRADE_ENTRIES_PATH): return
        with open(TRADE_ENTRIES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sym, e in data.items():
            for k, v in e.items():
                if k == 'entry_time' and isinstance(v, str):
                    e[k] = datetime.fromisoformat(v)
        TRADE_ENTRIES.update(data)
        log.info(f"📂 Cargadas {len(data)} entradas pendientes de trade_entries.json")
    except Exception as ex:
        log.error(f"Error cargando trade_entries: {ex}")

# ==========================================================
# 2. CREDENCIALES Y CONFIGURACIÓN
# ==========================================================
def crear_config_desde_entorno():
    return {
        'bitget_api_key':    os.environ.get('BITGET_API_KEY',    'bg_0e9c732f2ed08d90c986a7fd9a4cdedd'),
        'bitget_secret_key': os.environ.get('BITGET_SECRET_KEY', '52582b11761d83bce4e4475182b1510617081dd4e56051e787178a2a06a5bd3b'),
        'bitget_passphrase': os.environ.get('BITGET_PASSPHRASE', 'Rasputino977'),
        'telegram_token':    os.environ.get('TELEGRAM_TOKEN',    '8406173543:AAFIuYlFd3jtAF1Q6SNntUGn1PopgkZ7S0k'),
        'telegram_chat_id':  os.environ.get('TELEGRAM_CHAT_ID',  '2108159591'),
    }

config = crear_config_desde_entorno()
API_KEY    = config['bitget_api_key']
SECRET_KEY = config['bitget_secret_key']
PASSPHRASE = config['bitget_passphrase']
TELEGRAM_TOKEN  = config['telegram_token']
TELEGRAM_CHAT_ID = config['telegram_chat_id']

TIMEFRAME          = '2h'  # Optimizado: 2h (barrido 15m-4h)
EMA_MACRO          = 200    # Filtro de tendencia diaria
HMA_SIGNAL         = 25     # Optimizado: 25 (barrido dio +$324k vs +$88k de C3)
STC_FAST           = 25     # Optimizado: 25 (default 23)
STC_SLOW           = 30     # Optimizado: 30 (default 50)
STC_CYCLE          = 15     # Optimizado: 15 (default 10)
STC_UPPER          = 80     # Optimizado: 80 (default 75)
STC_LOWER          = 30     # Optimizado: 30 (default 25)
BE_TRIGGER_PCT     = 0.015  # BE al 1.5%
BE_OFFSET_PCT      = 0.005  # Beneficio asegurado al proteger
TRAILING_DIST_PCT  = 0.012  # Trail 1.2% inicial
SL_LOOKBACK        = 10     # Velas hacia atras para buscar minimo/maximo
MAX_SL_PCT         = 0.02   # SL maximo 2% de distancia
RR_RATIO           = 2.0    # Risk:Reward 1:2
MAX_OPEN_POSITIONS = 5
RISK_PERCENT       = 0.07   # 3% riesgo
LEVERAGE           = 10.0

# ==========================================================
# 3. FUNCIONES AUXILIARES
# ==========================================================
def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_hma(series, length):
    def _wma(s, n):
        weights = np.arange(1, n + 1, dtype=float)
        return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half, sqrtn = length // 2, int(np.sqrt(length))
    raw = 2 * _wma(series, half) - _wma(series, length)
    return _wma(raw, sqrtn)

def calculate_stc(series, fast=23, slow=50, length=10):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    
    macd_min = macd.rolling(window=length).min()
    macd_max = macd.rolling(window=length).max()
    stoch_macd = 100 * (macd - macd_min) / (macd_max - macd_min).replace(0, np.nan)
    stoch_macd = stoch_macd.ffill().fillna(0)
    
    smoothed_stoch = stoch_macd.ewm(alpha=0.5, adjust=False).mean()
    
    stoch_min = smoothed_stoch.rolling(window=length).min()
    stoch_max = smoothed_stoch.rolling(window=length).max()
    stoch_stoch = 100 * (smoothed_stoch - stoch_min) / (stoch_max - stoch_min).replace(0, np.nan)
    stoch_stoch = stoch_stoch.ffill().fillna(0)
    
    return stoch_stoch.ewm(alpha=0.5, adjust=False).mean()

def calculate_poc(df_5m):
    """Calcula el POC exacto sumando el volumen por nivel de precio (24h)."""
    if df_5m.empty: return 0
    # Redondeamos al tick size promedio para agrupar niveles de precio
    tick_size = df_5m['close'].diff().abs().median()
    df_5m['price_level'] = (df_5m['close'] / tick_size).round() * tick_size
    poc = df_5m.groupby('price_level')['volume'].sum().idxmax()
    return poc

def detect_order_blocks(df, lookback=100):
    """Detecta OBs institucionales: Volumen inusual + Imbalance + BOS."""
    obs = {'bullish': [], 'bearish': []}
    avg_vol = df['volume'].rolling(20).mean()
    
    for i in range(len(df) - 5, 20, -1):
        # 1. ¿Hay volumen inusual? (1.5x el promedio)
        if df['volume'].iloc[i] > avg_vol.iloc[i] * 1.5:
            # 2. ¿Hubo un Imbalance (FVG) inmediato?
            # Bullish: Gap entre el High de i-1 y el Low de i+1
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                # 3. ¿Rompió estructura (BOS)? 
                if df['close'].iloc[i+2] > df['high'].iloc[i-10:i].max():
                    obs['bullish'].append(df['low'].iloc[i])
            
            # Bearish: Gap entre el Low de i-1 y el High de i+1
            elif df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                if df['close'].iloc[i+2] < df['low'].iloc[i-10:i].min():
                    obs['bearish'].append(df['high'].iloc[i])
                    
        if len(obs['bullish']) > 2 and len(obs['bearish']) > 2: break
    return obs

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        log.info(f"📤 Telegram: {message[:80].replace(chr(10), ' ')}...")
    except: pass

TRADE_CSV_HEADERS = [
    'entry_time','exit_time','symbol','side','entry_price','exit_price',
    'sl_price','tp_price','sl_pct','tp_pct','quantity','balance_before','balance_after',
    'pnl','fees','net_pnl','status','duration_hours',
    'stc','stc_cross_depth','hma','ema','ema_distance_pct','poc','poc_distance_pct',
    'close_reason',
    'be_triggered','be_price','trail_count','trail_peak_price','trail_final_sl',
    'entry_weekday','entry_hour',
    'stc_exit','hma_exit','ema200_exit','volume_ratio','size_usdt','risk_pct',
    'hours_to_tp','hours_to_sl','max_favorable_pct','max_adverse_pct'
]

def guardar_trade_csv(entry, exit_price, raw_pnl, fees, net, status, close_reason):
    if not entry: return
    now = datetime.now()
    duration = (now - entry['entry_time']).total_seconds() / 3600
    balance_after = entry['balance_before'] + net
    ep, sl, tp = entry['entry_price'], entry['sl_price'], entry['tp_price']
    side = entry['side']
    row = {
        'entry_time': entry['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'exit_time': now.strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': entry['symbol'],
        'side': side,
        'entry_price': ep,
        'exit_price': exit_price,
        'sl_price': sl,
        'tp_price': tp,
        'sl_pct': round(abs(ep - sl) / ep * 100, 2),
        'tp_pct': round(abs(ep - tp) / ep * 100, 2),
        'quantity': entry['quantity'],
        'balance_before': round(entry['balance_before'], 2),
        'balance_after': round(balance_after, 2),
        'pnl': round(raw_pnl, 2),
        'fees': round(fees, 2),
        'net_pnl': round(net, 2),
        'status': status,
        'duration_hours': round(duration, 2),
        'stc': round(entry['stc'], 1),
        'stc_cross_depth': round(entry['stc'] - STC_LOWER, 1) if side == 'long' else round(STC_UPPER - entry['stc'], 1),
        'hma': entry['hma'],
        'ema': entry['ema'],
        'ema_distance_pct': round(abs(ep - entry['ema']) / entry['ema'] * 100, 2),
        'poc': entry['poc'],
        'poc_distance_pct': round(abs(ep - entry['poc']) / entry['poc'] * 100, 2) if entry['poc'] else 0,
        'close_reason': close_reason,
        'be_triggered': 1 if ALERTS_HISTORY.get(f"{entry['symbol']}_be", False) else 0,
        'be_price': round(ALERTS_HISTORY.get(f"{entry['symbol']}_be_price", 0), 4),
        'trail_count': TRAIL_COUNTS.get(entry['symbol'], 0),
        'trail_peak_price': round(PEAK_PRICES.get(entry['symbol'], ep), 4),
        'trail_final_sl': round(ALERTS_HISTORY.get(f"{entry['symbol']}_trail", sl), 4),
        'entry_weekday': entry['entry_time'].weekday(),
        'entry_hour': entry['entry_time'].hour,
        'stc_exit': round(LAST_KNOWN_INDICATORS.get(entry['symbol'], {}).get('stc', 0), 1),
        'hma_exit': round(LAST_KNOWN_INDICATORS.get(entry['symbol'], {}).get('hma', 0), 4),
        'ema200_exit': round(LAST_KNOWN_INDICATORS.get(entry['symbol'], {}).get('ema', 0), 4),
        'volume_ratio': entry.get('volume_ratio', 0),
        'size_usdt': entry.get('size_usdt', 0),
        'risk_pct': entry.get('risk_pct', 0),
        'hours_to_tp': round(duration, 2) if close_reason == 'tp' else '',
        'hours_to_sl': round(duration, 2) if close_reason == 'sl' else '',
        'max_favorable_pct': round(abs(PEAK_PRICES.get(entry['symbol'], ep) - ep) / ep * 100, 2),
        'max_adverse_pct': round(abs(ADVERSE_PRICES.get(entry['symbol'], ep) - ep) / ep * 100, 2)
    }
    csv_path = TRADES_CSV_PATH
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=TRADE_CSV_HEADERS)
            if write_header: w.writeheader()
            w.writerow(row)
    except: pass

PREMATURE_SL_CSV_HEADERS = [
    'entry_time','sl_time','symbol','side','entry_price','sl_price','tp_price',
    'sl_pct','tp_reached','tp_reached_time','hours_to_tp_after_sl',
    'entry_weekday','entry_hour',
    'stc_at_entry','hma_at_entry','ema_at_entry',
    'max_favorable_before_sl','hit_be_before_sl'
]

def guardar_premature_sl_csv(mon, reached, reached_time=None, reached_price=None):
    ep, sl = mon['entry_price'], mon['sl_price']
    row = {
        'entry_time': mon['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'sl_time': mon['sl_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': mon['symbol'],
        'side': mon['side'],
        'entry_price': ep,
        'sl_price': sl,
        'tp_price': mon['tp_price'],
        'sl_pct': round(abs(ep - sl) / ep * 100, 2),
        'tp_reached': 'Yes' if reached else 'No',
        'tp_reached_time': reached_time.strftime('%Y-%m-%d %H:%M:%S') if reached_time else '',
        'hours_to_tp_after_sl': round((reached_time - mon['sl_time']).total_seconds() / 3600, 2) if reached_time else '',
        'entry_weekday': mon['entry_time'].weekday(),
        'entry_hour': mon['entry_time'].hour,
        'stc_at_entry': round(mon.get('stc_at_entry', 0), 1),
        'hma_at_entry': round(mon.get('hma_at_entry', 0), 4),
        'ema_at_entry': round(mon.get('ema_at_entry', 0), 4),
        'max_favorable_before_sl': round(mon.get('max_favorable_before_sl', 0), 4),
        'hit_be_before_sl': 'Yes' if mon.get('hit_be_before_sl') else 'No'
    }
    csv_path = PREMATURE_SL_CSV_PATH
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=PREMATURE_SL_CSV_HEADERS)
            if write_header: w.writeheader()
            w.writerow(row)
    except:
        pass

def guardar_signal_log(symbol, side, price, stc, hma, ema200, poc, sl_proj, tp_proj, volume_ratio, taken=True, reason_skipped=''):
    row = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol, 'side': side, 'price': round(price, 6),
        'stc': round(stc, 1), 'hma': round(hma, 6), 'ema200': round(ema200, 6),
        'poc': round(poc, 6), 'sl_proj': round(sl_proj, 6), 'tp_proj': round(tp_proj, 6),
        'volume_ratio': round(volume_ratio, 2),
        'taken': 'Yes' if taken else 'No',
        'reason_skipped': reason_skipped
    }
    csv_path = SIGNALS_LOG_PATH
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_LOG_HEADERS)
            if write_header: w.writeheader()
            w.writerow(row)
    except:
        pass

def guardar_price_path(symbol, entry_time):
    paths = PRICE_PATHS.pop(symbol, [])
    if not paths: return
    safe_name = symbol.replace('/', '_').replace(':', '_')
    fname = f"{safe_name}_{entry_time.strftime('%Y%m%d_%H%M%S')}.csv"
    fpath = os.path.join(PRICE_PATHS_DIR, fname)
    try:
        with open(fpath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['time', 'price'])
            w.writerows(paths)
    except:
        pass

# ==========================================================
# 4. FETCH ASÍNCRONO DE OHLCV
# ==========================================================
async def _fetch_symbol_async(exch, symbol):
    try:
        ohlcv_main = await exch.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=150)
        ohlcv_5m = await exch.fetch_ohlcv(symbol, timeframe='5m', limit=288)
        return symbol, ohlcv_main, ohlcv_5m
    except:
        return symbol, None, None

async def fetch_all_ohlcv(symbols):
    exch = ccxt_async.bitget({
        'apiKey': API_KEY, 'secret': SECRET_KEY, 'password': PASSPHRASE,
        'enableRateLimit': True, 'options': {'defaultType': 'swap'}
    })
    try:
        results = await asyncio.gather(*[_fetch_symbol_async(exch, s) for s in symbols])
    finally:
        await exch.close()
    return {r[0]: (r[1], r[2]) for r in results}

# ==========================================================
# 5. CONEXIÓN Y GESTIÓN (inicialización lazy)
# ==========================================================
exchange = None

def init_exchange():
    global exchange
    try:
        exchange = ccxt.bitget({'apiKey': API_KEY, 'secret': SECRET_KEY, 'password': PASSPHRASE, 'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        log.info("✅ CONEXIÓN EXITOSA — ESPERANDO NUEVA ESTRATEGIA.")
        return True
    except Exception as e:
        log.critical(f"❌ ERROR: {e}")
        return False

def update_stop_loss(symbol, side, new_sl):
    try:
        new_sl_fmt = exchange.price_to_precision(symbol, new_sl)
        clean_symbol = symbol.split(':')[0].replace('/', '')
        params = {
            'symbol': clean_symbol, 'marginCoin': 'USDT', 'productType': 'USDT-FUTURES',
            'planType': 'pos_loss', 'stopLossTriggerPrice': str(new_sl_fmt), 
            'stopLossTriggerType': 'fill_price', 'holdSide': 'long' if side == 'long' else 'short'
        }
        exchange.private_mix_post_v2_mix_order_place_pos_tpsl(params)
        return True
    except: return False

def manage_escudo_pro(balance=0.0):
    global ALERTS_HISTORY, PEAK_PRICES, COOLDOWNS, DAILY_STATS, SESSION_ACTIVE_SYMBOLS, PREMATURE_SL_MONITOR
    try:
        positions = exchange.fetch_positions()
        active_symbols = [p['symbol'] for p in positions if float(p['contracts']) > 0]
        
        for sym in list(SESSION_ACTIVE_SYMBOLS):
            if sym not in active_symbols:
                COOLDOWNS[sym] = time.time() + 3600
                log.info(f"⏳ {sym} CERRADA. Cooldown 1h activado.")
                try:
                    time.sleep(2)
                    trades = exchange.fetch_my_trades(sym, limit=20)
                    if trades:
                        trade_pnl, trade_fees, last_closing_trade = 0.0, 0.0, None
                        for t in reversed(trades):
                            if float(t['info'].get('profit', 0)) != 0: last_closing_trade = t; break
                        if last_closing_trade:
                            order_id = last_closing_trade.get('order') or last_closing_trade['info'].get('orderId')
                            for t in trades:
                                if (t.get('order') or t['info'].get('orderId')) == order_id:
                                    trade_pnl += float(t['info'].get('profit', 0))
                                    if 'fee' in t and t['fee']: trade_fees += abs(float(t['fee'].get('cost', 0)))
                            net = trade_pnl - trade_fees
                            status = "✅ TP" if net > 0 else "❌ SL"
                            if trade_pnl == 0: status = "⚪ BE"
                            pass # stats now read from trades.csv
                            send_telegram(f"🏁 *{sym} CERRADA*\nPnL: {net:.2f} USDT ({status})\nFees: -{trade_fees:.2f} USDT")
                            entry = TRADE_ENTRIES.pop(sym, None)
                            if entry:
                                exit_px = float(last_closing_trade.get('price', 0))
                                clean_status = status.replace('✅ ', '').replace('❌ ', '').replace('⚪ ', '')
                                reason = 'tp' if trade_pnl > 0 else 'sl' if trade_pnl < 0 else 'be'
                                guardar_trade_csv(entry, exit_px, trade_pnl, trade_fees, net, clean_status, reason)
                                if trade_pnl < 0:
                                    _peak = PEAK_PRICES.get(sym, entry['entry_price'])
                                    PREMATURE_SL_MONITOR[sym] = {
                                        'entry_time': entry['entry_time'],
                                        'sl_time': datetime.now(),
                                        'symbol': sym,
                                        'side': entry['side'],
                                        'entry_price': entry['entry_price'],
                                        'sl_price': entry['sl_price'],
                                        'tp_price': entry['tp_price'],
                                        'stc_at_entry': entry.get('stc', 0),
                                        'hma_at_entry': entry.get('hma', 0),
                                        'ema_at_entry': entry.get('ema', 0),
                                        'hit_be_before_sl': ALERTS_HISTORY.get(f"{sym}_be", False),
                                        'max_favorable_before_sl': _peak
                                    }
                            _save_trade_entries()
                except Exception as e_close:
                    log.error(f"Error procesando cierre de {sym}: {e_close}")
                if sym in PRICE_PATHS:
                    _pp_entry_time = None
                    try: _pp_entry_time = locals().get('entry', {}).get('entry_time')
                    except: pass
                    if _pp_entry_time: guardar_price_path(sym, _pp_entry_time)
                    else: PRICE_PATHS.pop(sym, None)
                PEAK_PRICES.pop(sym, None)
                ADVERSE_PRICES.pop(sym, None)
                LAST_KNOWN_INDICATORS.pop(sym, None)
                if sym in ALERTS_HISTORY: del ALERTS_HISTORY[sym]
                if sym in TRAIL_COUNTS: del TRAIL_COUNTS[sym]
                if sym in SESSION_ACTIVE_SYMBOLS: SESSION_ACTIVE_SYMBOLS.remove(sym)

        for pos in positions:
            symbol, side = pos['symbol'], pos['side']
            if float(pos['contracts']) == 0: continue
            entry, mark = float(pos['entryPrice']), float(pos['markPrice'])
            profit_pct = (mark - entry) / entry if side == 'long' else (entry - mark) / entry

            if symbol not in ADVERSE_PRICES: ADVERSE_PRICES[symbol] = mark
            else: ADVERSE_PRICES[symbol] = min(ADVERSE_PRICES[symbol], mark) if side == 'long' else max(ADVERSE_PRICES[symbol], mark)

            if symbol not in PRICE_PATHS: PRICE_PATHS[symbol] = []
            PRICE_PATHS[symbol].append((datetime.now().isoformat(), mark))

            # 1. Break Even con Beneficio Asegurado
            if profit_pct >= BE_TRIGGER_PCT:
                if symbol not in PEAK_PRICES: PEAK_PRICES[symbol] = mark
                else: PEAK_PRICES[symbol] = max(PEAK_PRICES[symbol], mark) if side == 'long' else min(PEAK_PRICES[symbol], mark)

                if not ALERTS_HISTORY.get(f"{symbol}_be", False):
                    offset = entry * BE_OFFSET_PCT if side == 'long' else -entry * BE_OFFSET_PCT
                    new_sl = entry + offset
                    if update_stop_loss(symbol, side, new_sl):
                        ALERTS_HISTORY[f"{symbol}_be"] = True
                        ALERTS_HISTORY[f"{symbol}_be_price"] = new_sl
                        log.info(f"🛡️ {symbol} protegida en BE+ (Offset {BE_OFFSET_PCT*100:.2f}%)")
                        send_telegram(f"🛡️ *{symbol}* protegida en BE+ (Offset {BE_OFFSET_PCT*100:.2f}%)")

            # 2. Trailing Stop Dinámico y Apretado (Inteligente)
            if profit_pct >= BE_TRIGGER_PCT + 0.005:
                profit_excedente = (profit_pct - BE_TRIGGER_PCT) * 100
                reduccion = profit_excedente * 0.003
                dynamic_trail_pct = max(TRAILING_DIST_PCT - reduccion, 0.003)

                dist = entry * dynamic_trail_pct
                nuevo_sl = PEAK_PRICES[symbol] - dist if side == 'long' else PEAK_PRICES[symbol] + dist
                ultimo = ALERTS_HISTORY.get(f"{symbol}_trail", 0 if side == 'long' else 999999)

                mejora = (nuevo_sl - ultimo) if side == 'long' else (ultimo - nuevo_sl)

                if mejora > (entry * 0.002):
                    if update_stop_loss(symbol, side, nuevo_sl):
                        ALERTS_HISTORY[f"{symbol}_trail"] = nuevo_sl
                        TRAIL_COUNTS[symbol] = TRAIL_COUNTS.get(symbol, 0) + 1
                        log.info(f"🏃 {symbol} Trail Inteligente subió a {nuevo_sl:.4f} (Apretado a {dynamic_trail_pct*100:.2f}%)")
                        send_telegram(f"🏃 *{symbol}* Trailing Dinámico: {nuevo_sl:.4f}\n└ _Distancia apretada a: {dynamic_trail_pct*100:.2f}%_")

            # --- PREMATURE SL MONITOR (seguimiento de SL que luego alcanzaron TP) ---
            for mon_sym in list(PREMATURE_SL_MONITOR.keys()):
                mon = PREMATURE_SL_MONITOR[mon_sym]
                hours_since = (datetime.now() - mon['sl_time']).total_seconds() / 3600
                if hours_since > 24:
                    guardar_premature_sl_csv(mon, False)
                    del PREMATURE_SL_MONITOR[mon_sym]
                    continue
                try:
                    ticker = exchange.fetch_ticker(mon_sym)
                    curr_price = ticker['last']
                    if (mon['side'] == 'long' and curr_price >= mon['tp_price']) or \
                       (mon['side'] == 'short' and curr_price <= mon['tp_price']):
                        guardar_premature_sl_csv(mon, True, datetime.now(), curr_price)
                        log.info(f"📌 {mon_sym}: SL prematuro — alcanzó TP {mon['tp_price']} después del SL")
                        del PREMATURE_SL_MONITOR[mon_sym]
                except:
                    continue

            # --- FILTRO: Cierre automático si >12h en rojo ---
            open_time_ms = float(pos.get('timestamp') or pos['info'].get('openTime', 0) or 0)
            if open_time_ms > 0 and profit_pct < 0:
                horas_abiertas = (time.time() * 1000 - open_time_ms) / 3_600_000
                if horas_abiertas >= 12:
                    try:
                        exchange.close_position(symbol)
                        net_auto = balance * RISK_PERCENT * LEVERAGE * profit_pct
                        pass # stats now read from trades.csv
                        send_telegram(f"⏰ *{symbol} CERRADA automáticamente*\n+12h en rojo ({profit_pct*100:.2f}%) | {horas_abiertas:.1f}h abierta")
                        log.info(f"⏰ {symbol} cerrada por +12h en rojo | {horas_abiertas:.1f}h | PnL: {profit_pct*100:.2f}%")
                        entry = TRADE_ENTRIES.pop(symbol, None)
                        if entry:
                            exit_px = entry['entry_price'] * (1 + profit_pct) if side == 'long' else entry['entry_price'] * (1 - profit_pct)
                            guardar_trade_csv(entry, exit_px, net_auto, 0, net_auto, 'Timeout', 'timeout')
                        _save_trade_entries()
                        if symbol in PRICE_PATHS and entry: guardar_price_path(symbol, entry['entry_time'])
                        if symbol in PEAK_PRICES: del PEAK_PRICES[symbol]
                        if symbol in ALERTS_HISTORY: del ALERTS_HISTORY[symbol]
                        if symbol in TRAIL_COUNTS: del TRAIL_COUNTS[symbol]
                        COOLDOWNS[symbol] = time.time() + 3600
                        if symbol in SESSION_ACTIVE_SYMBOLS: SESSION_ACTIVE_SYMBOLS.remove(symbol)
                    except Exception as e_close:
                        log.error(f"Error cerrando {symbol} por timeout: {e_close}")
    except Exception as e:
        log.error(f"Error en manage_escudo_pro: {e}")

# ==========================================================
# 5. BUCLE PRINCIPAL
# ==========================================================
def main():
    global LAST_KNOWN_INDICATORS, ALERTS_HISTORY, PEAK_PRICES, COOLDOWNS
    global SESSION_ACTIVE_SYMBOLS, DAILY_STATS, TRADE_ENTRIES, TRAIL_COUNTS
    global PREMATURE_SL_MONITOR, ADVERSE_PRICES, PRICE_PATHS, exchange

    if exchange is None:
        if not init_exchange():
            return

    _load_trade_entries()
    for f in os.listdir(PRICE_PATHS_DIR):
        try: os.remove(os.path.join(PRICE_PATHS_DIR, f))
        except: pass
    last_report_day = datetime.now().day - 1

    while True:
        try:
            now = datetime.now()
            if now.hour == 0 and now.day != last_report_day:
                today_str = (now - timedelta(days=1)).strftime('%Y-%m-%d')
                today_trades = []
                try:
                    with open(TRADES_CSV_PATH, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['entry_time'].startswith(today_str):
                                today_trades.append(row)
                except: pass

                total = len(today_trades)
                tps = [r for r in today_trades if r['status'] == 'TP']
                sls = [r for r in today_trades if r['status'] == 'SL']
                bes = [r for r in today_trades if r['status'] == 'BE']
                tos = [r for r in today_trades if r['status'] == 'Timeout']
                pnl_total = sum(float(r['net_pnl']) for r in today_trades)
                fees_total = sum(float(r['fees']) for r in today_trades)
                wr = len(tps) / max(total, 1) * 100

                def fmt_list(nombre, emoji, items, limite=10):
                    lineas = [f"├ {emoji} {nombre}: {len(items)}"]
                    for row in items[:limite]:
                        pnl = float(row['net_pnl'])
                        sym = row['symbol'].split('/')[0]
                        signo = '+' if pnl >= 0 else ''
                        lineas.append(f"│  ├ {sym}: {signo}{pnl:.2f}")
                    if len(items) > limite:
                        lineas.append(f"│  └ ... y {len(items)-limite} más")
                    return '\n'.join(lineas)

                msg = f"📊 *REPORTE DIARIO* ({now.strftime('%d/%m')})\n"
                msg += f"┌ Operaciones: {total}\n"
                msg += fmt_list('TP', '✅', tps) + '\n'
                msg += fmt_list('SL', '❌', sls) + '\n'
                msg += fmt_list('BE', '⚪', bes) + '\n'
                msg += fmt_list('Timeout', '⏰', tos) + '\n'
                msg += f"├ Win Rate: {wr:.0f}%\n"
                msg += f"├ PnL Total: {pnl_total:+.2f} USDT\n"
                msg += f"└ Fees: -{fees_total:.2f} USDT"
                send_telegram(msg)
                last_report_day = now.day

            try:
                balance_data = exchange.fetch_balance()
                balance = float(balance_data['total'].get('USDT', 0))
            except Exception as e:
                log.error(f"Error obteniendo balance: {e}")
                balance = 0.0

            manage_escudo_pro(balance)
            
            positions = exchange.fetch_positions()
            # Actualizamos SESSION_ACTIVE_SYMBOLS con lo que el exchange nos dice
            busy_symbols = {p['symbol'] for p in positions if float(p['contracts']) > 0}
            SESSION_ACTIVE_SYMBOLS.update(busy_symbols)
            
            occupied_str = ", ".join([s.split('/')[0] for s in busy_symbols]) if busy_symbols else "Ninguno"
            log.info(f"🔄 CICLO [{now.strftime('%H:%M:%S')}] | Balance: {balance:.2f} | Ocupados: [{occupied_str}]")
            
            if len(busy_symbols) >= MAX_OPEN_POSITIONS:
                time.sleep(15); continue

            tickers = exchange.fetch_tickers()
            top_30 = [p[0] for p in sorted([(s, float(t.get('quoteVolume', 0))) for s, t in tickers.items() if s.endswith('/USDT:USDT')], key=lambda x: x[1], reverse=True)[:40]]

            ohlcv_data = asyncio.run(fetch_all_ohlcv(top_30))

            for symbol in top_30:
                if symbol in busy_symbols or len(busy_symbols) >= MAX_OPEN_POSITIONS: continue
                if symbol in COOLDOWNS:
                    if time.time() < COOLDOWNS[symbol]: continue
                    else: del COOLDOWNS[symbol]

                try:
                    # Obtenemos TIMEFRAME para indicadores y 5m para el POC preciso
                    ohlcv_main, ohlcv_5m = ohlcv_data.get(symbol, (None, None))
                    if not ohlcv_main or not ohlcv_5m: continue
                    
                    df = pd.DataFrame(ohlcv_main, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # CÁLCULOS PRECISOS
                    df['ema_200'] = calculate_ema(df['close'], EMA_MACRO)
                    df['hma_25']  = calculate_hma(df['close'], HMA_SIGNAL)
                    df['stc']     = calculate_stc(df['close'], STC_FAST, STC_SLOW, STC_CYCLE)
                    poc           = calculate_poc(df_5m)
                    obs           = detect_order_blocks(df)
                    
                    last, prev = df.iloc[-1], df.iloc[-2]
                    price, ema, hma, stc = last['close'], last['ema_200'], last['hma_25'], last['stc']
                    
                    # 1. Filtro Macro: Precio vs EMA 200
                    is_bull_macro = price > ema
                    is_bear_macro = price < ema
                    
                    # 2. Contexto de Volumen: Precio vs POC
                    is_above_poc = price > poc
                    is_below_poc = price < poc

                    # 3. Estructura SMC (Rebote en OB o BOS)
                    # Bullish: Rebote en OB o precio arriba de un OB reciente
                    has_bull_structure = any(price > ob * 0.998 and price < ob * 1.01 for ob in obs['bullish']) or (price > df['high'].iloc[-5:-1].max())
                    # Bearish: Rechazo en OB o precio abajo de un OB reciente
                    has_bear_structure = any(price < ob * 1.002 and price > ob * 0.99 for ob in obs['bearish']) or (price < df['low'].iloc[-5:-1].min())

                    # 4. Confirmación HMA 25 (Pendiente y posición)
                    hma_slope_up = hma > prev['hma_25']
                    hma_slope_down = hma < prev['hma_25']
                    is_above_hma = price > hma
                    is_below_hma = price < hma

                    # 5. Gatillo: Cruce de STC (25 hacia arriba Long, 75 hacia abajo Short)
                    stc_cross_up = stc >= STC_LOWER and prev['stc'] < STC_LOWER
                    stc_cross_down = stc <= STC_UPPER and prev['stc'] > STC_UPPER

                    # --- CONFLUENCIA TOTAL ---
                    buy  = is_bull_macro and is_above_poc and has_bull_structure and hma_slope_up and is_above_hma and stc_cross_up
                    sell = is_bear_macro and is_below_poc and has_bear_structure and hma_slope_down and is_below_hma and stc_cross_down

                    # --- LOG DE INDICADORES POR SÍMBOLO ---
                    fmt_hma = exchange.price_to_precision(symbol, hma)
                    fmt_poc = exchange.price_to_precision(symbol, poc)
                    bull_top = exchange.price_to_precision(symbol, obs['bullish'][0]) if obs['bullish'] else "N/A"
                    bear_top = exchange.price_to_precision(symbol, obs['bearish'][0]) if obs['bearish'] else "N/A"
                    log.info(
                        f"📊 {symbol} | HMA25: {fmt_hma} | POC: {fmt_poc} | STC: {stc:.1f} | "
                        f"OB_Bull: {bull_top} | OB_Bear: {bear_top} | "
                        f"Bull={int(buy)} Bear={int(sell)}"
                    )

                    LAST_KNOWN_INDICATORS[symbol] = {
                        'stc': stc, 'hma': hma, 'ema': ema,
                        'volume_ratio': round(last['volume'] / df['volume'].iloc[-21:-1].mean(), 2)
                    }

                    if (buy or sell):
                        vr = round(last['volume'] / df['volume'].iloc[-21:-1].mean(), 2)
                        _side_log = 'long' if buy else 'short'
                        if buy:
                            _sl_log = max(df['low'].iloc[-SL_LOOKBACK-1:-1].min(), price * (1 - MAX_SL_PCT))
                            _tp_log = price + (price - _sl_log) * RR_RATIO
                        else:
                            _sl_log = min(df['high'].iloc[-SL_LOOKBACK-1:-1].max(), price * (1 + MAX_SL_PCT))
                            _tp_log = price - (_sl_log - price) * RR_RATIO
                        _taken_flag = symbol not in SESSION_ACTIVE_SYMBOLS and len(busy_symbols) < MAX_OPEN_POSITIONS
                        _skip_reason = 'max_open' if not _taken_flag else ''
                        guardar_signal_log(symbol, _side_log, price, stc, hma, ema, poc, _sl_log, _tp_log, vr, _taken_flag, _skip_reason)

                    if (buy or sell) and symbol not in SESSION_ACTIVE_SYMBOLS:
                        # VERIFICACIÓN DE ÚLTIMO SEGUNDO (Anti-recompra)
                        try:
                            check_pos = exchange.fetch_position(symbol)
                            if float(check_pos.get('contracts', 0)) > 0:
                                log.warning(f"⚠️ {symbol} ya tiene posición. Abortando re-entrada.")
                                SESSION_ACTIVE_SYMBOLS.add(symbol)
                                continue
                        except: pass
                        side = 'buy' if buy else 'sell'

                        # --- GESTIÓN DE SL/TP: MIN/MAX ÚLTIMAS 6 VELAS, MÁX 2%, TP 1:2 ---
                        last_lows = df['low'].iloc[-SL_LOOKBACK-1:-1].min() if len(df) > SL_LOOKBACK+1 else price * (1 - MAX_SL_PCT)
                        last_highs = df['high'].iloc[-SL_LOOKBACK-1:-1].max() if len(df) > SL_LOOKBACK+1 else price * (1 + MAX_SL_PCT)
                        if buy:
                            sl_raw = last_lows
                            sl = max(sl_raw, price * (1 - MAX_SL_PCT))
                            tp = price + (price - sl) * RR_RATIO
                        else:
                            sl_raw = last_highs
                            sl = min(sl_raw, price * (1 + MAX_SL_PCT))
                            tp = price - (sl - price) * RR_RATIO
                        
                        # Cálculo de cantidad con redondeo estricto hacia abajo (floor)
                        target_margin = balance * RISK_PERCENT
                        pos_value = target_margin * LEVERAGE
                        raw_qty = pos_value / price
                        
                        # Obtenemos la precisión del lote para este símbolo
                        market = exchange.market(symbol)
                        precision = market['precision']['amount']
                        step = market['limits']['amount']['min'] or (10**-precision)
                        
                        # Forzamos redondeo hacia abajo (floor) al step más cercano
                        qty_precision = (raw_qty // step) * step
                        
                        actual_margin = (qty_precision * price) / LEVERAGE
                        
                        # PROTECCIÓN EXTRA: Si por algún motivo el margen real se pasa, bajamos un lote
                        if actual_margin > target_margin:
                            qty_precision -= step
                            actual_margin = (qty_precision * price) / LEVERAGE

                        log.info(f"⚖️ {symbol} | Objetivo: {target_margin:.2f} | Real: {actual_margin:.2f} (Floor) | SL={sl:.4f} TP={tp:.4f}")

                        params = {
                            'marginCoin': 'USDT', 'marginMode': 'isolated', 'tradeSide': 'open', 
                            'presetStopSurplusPrice': str(exchange.price_to_precision(symbol, tp)), 
                            'presetStopLossPrice': str(exchange.price_to_precision(symbol, sl))
                        }
                        exchange.create_order(symbol, 'market', side, qty_precision, params=params)
                        
                        fmt_price = exchange.price_to_precision(symbol, price)
                        fmt_sl = exchange.price_to_precision(symbol, sl)
                        fmt_tp = exchange.price_to_precision(symbol, tp)

                        msg = f"🏛️ *{symbol} {side.upper()}* ({SL_LOOKBACK} velas / max {MAX_SL_PCT*100:.0f}%)\n"
                        msg += f"Entrada: `{fmt_price}`\n"
                        msg += f"🛑 SL: `{fmt_sl}` (mín {SL_LOOKBACK} velas / max {MAX_SL_PCT*100:.0f}%)\n"
                        msg += f"🎯 TP: `{fmt_tp}` (1:{int(RR_RATIO)})\n"
                        msg += f"\n📊 *Indicadores:*\n"
                        msg += f"  HMA: `{fmt_hma}`\n"
                        msg += f"  POC:  `{fmt_poc}`\n"
                        msg += f"  STC:  `{stc:.1f}`\n"
                        send_telegram(msg)
                        TRADE_ENTRIES[symbol] = {
                            'entry_time': datetime.now(),
                            'symbol': symbol,
                            'side': 'long' if side == 'buy' else 'short',
                            'entry_price': price,
                            'sl_price': sl,
                            'tp_price': tp,
                            'quantity': qty_precision,
                            'balance_before': balance,
                            'stc': stc,
                            'hma': hma,
                            'ema': ema,
                            'poc': poc,
                            'volume_ratio': round(last['volume'] / df['volume'].iloc[-21:-1].mean(), 2),
                            'size_usdt': round(actual_margin, 2),
                            'risk_pct': round(actual_margin / balance * 100, 2)
                        }
                        _save_trade_entries()
                        busy_symbols.add(symbol)
                        SESSION_ACTIVE_SYMBOLS.add(symbol)

                except: continue

            time.sleep(15)
        except Exception as e: log.error(f"Error ciclo: {e}"); time.sleep(15)

if __name__ == '__main__':
    main()
