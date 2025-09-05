import pandas as pd
import requests
import gzip
import io
from datetime import datetime, timedelta
import ccxt 
import time
from pandas_ta.momentum import rsi, cmo, roc
from pandas_ta.trend import adx
from pandas_ta.momentum import stochrsi
import numpy as np

def process_day(date):
    url = f"https://public.bybit.com/spot/BTCUSDT/BTCUSDT_{date.strftime('%Y-%m-%d')}.csv.gz"
    print(f"📥 Downloading: {url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), 'rt') as f:
            df = pd.read_csv(f, header=None, skiprows=1, usecols=[1,2,3,4], 
                                names=['timestamp', 'price', 'volume', 'side'],
                                dtype={'timestamp': np.int64, 'price': np.float32, 'volume': np.float32, 'side': 'category'})
    except Exception as e:
        print(f"❌ Failed {date}: {e}")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp_minute'] = df['timestamp'].dt.floor('min')

    grouped = df.groupby(['timestamp_minute', 'side']).agg(
        count=('volume', 'count'),
        volume=('volume', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()

    pivot = grouped.pivot(index='timestamp_minute', columns='side')
    pivot.columns = ['_'.join([str(c) for c in col]).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    pivot.rename(columns={
        'count_buy': 'buy_count',
        'volume_buy': 'buy_volume',
        'avg_price_buy': 'buy_avg_price',
        'count_sell': 'sell_count',
        'volume_sell': 'sell_volume',
        'avg_price_sell': 'sell_avg_price',
    }, inplace=True)

    for col in ['buy_count', 'buy_volume', 'buy_avg_price',
                'sell_count', 'sell_volume', 'sell_avg_price']:
        if col not in pivot.columns:
            pivot[col] = 0 if 'count' in col or 'volume' in col else float('nan')

    return pivot[['timestamp_minute', 
                    'buy_count', 'buy_volume', 'sell_count', 'sell_volume', 
                    'buy_avg_price', 'sell_avg_price']]



def get_OF_data(start_date, end_date, part):
    import pandas as pd
    import requests
    import gzip
    import io
    from datetime import timedelta
    import numpy as np
    from multiprocessing import Pool, cpu_count

    

    # === Parallelized Execution ===
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    with Pool(cpu_count()) as pool:
        all_data = pool.map(process_day, dates)

    final_df = pd.concat([df for df in all_data if not df.empty])

    final_df['buy_volume'] = final_df['buy_volume'].round(3)
    final_df['sell_volume'] = final_df['sell_volume'].round(3)
    final_df['buy_count'] = final_df['buy_count'].fillna(0).astype(int)
    final_df['sell_count'] = final_df['sell_count'].fillna(0).astype(int)

    final_df.to_csv(f"BTCUSDT_OF_summary_part{part}.csv", index=False)
    print(f"✅ Done: BTCUSDT_OF_summary_part{part}.csv")


#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################

from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import pandas as pd
import time
import ccxt

# --- konfigai ---
SYMBOL = 'BTC/USDT'
TF = '1m'
PAGE_SLEEP = 0.5                 # pauzė tarp puslapių (apsauga nuo rate limit)
MAX_PROCS = min(cpu_count(), 4)  # cap, kad Bybit nepyktų

def _fetch_window(args):
    """Viena diena (ar kitas langas) su paginacija per limit=1000."""
    day_start, day_end = args

    print(f"⛏️ Extracting PRICES {SYMBOL} {TF}"
          f"{day_start:%Y-%m-%d %H:%M} → {day_end:%Y-%m-%d %H:%M}", flush=True)
    ex = ccxt.bybit({'enableRateLimit': True})
    ex.load_markets()

    since = int(day_start.timestamp() * 1000)
    end_ms = int(day_end.timestamp() * 1000)

    rows = []
    while since < end_ms:
        try:
            candles = ex.fetch_ohlcv(SYMBOL, timeframe=TF, since=since, limit=1000)
        except ccxt.RateLimitExceeded:
            time.sleep(PAGE_SLEEP * 2)
            continue
        except Exception:
            time.sleep(PAGE_SLEEP)
            continue

        if not candles:
            break

        # paimam tik iki lango pabaigos
        for c in candles:
            if c[0] >= end_ms:
                break
            rows.append(c)

        # +1min, su apsauga nuo stučio vietoje
        next_since = candles[-1][0] + 60_000
        since = next_since if next_since > since else since + 60_000
        time.sleep(PAGE_SLEEP)

    if not rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])

    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def prices_ccxt_data(start_date, end_date, part):
    """
    Paraleliai surenka Bybit 1m OHLCV tarp start_date (įskaitytinai) ir end_date (neįskaitytinai),
    supjaustydamas į dienos langus. Išsaugo: WITHOUT_BTC_USDT_1m_part{part}.csv
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    if start_date >= end_date:
        raise ValueError("start_date must be < end_date")

    # sudarom dieninius langus [00:00; 00:00)
    days = []
    cur = start_date
    while cur < end_date:
        nxt = min(cur + timedelta(days=1), end_date)
        days.append((cur, nxt))
        cur = nxt
    
    with Pool(processes=MAX_PROCS) as pool:
        dfs = pool.map(_fetch_window, days)

    # concat + filtras + dedupe + sort
    final = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    final = final[final['timestamp'] >= pd.Timestamp(start_date)]
    final = final.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    out = f'WITHOUT_BTC_USDT_1m_part{part}.csv'
    final.to_csv(out, index=False)
    print(f"✅ Done: {out} ({len(final)} rows)")
    return out


#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################



def features_adder(part):

        

    #//// Alternatyva PineScript valuewhen funkcijai ///////////////////////////////////////////////////////////////////////////
    def valuewhen(condition, values):
        result = []
        last = np.nan
        for cond, val in zip(condition, values):
            if cond:
                last = val
            result.append(last)
        return result
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    #//// ATR skaičiavimas /////////////////////////////////////////////////////////////////////////////////////////////////////
    def rma(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()  # RMA = Wilder

    def atr(df, period):
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                abs(df['low'] - df['close'].shift(1))))
        return rma(tr, period) 
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    #//// Algoritmo logika /////////////////////////////////////////////////////////////////////////////////////////////////////
    def Arbiter_of_Fate(df, sensitivity, atr_len=14):

        df = df.copy()
        multiplier = 100 / sensitivity
        hl2 = (df['high'] + df['low']) / 2

        atr_val = atr(df, atr_len)  # 🔹 Skaičiuojamas ATR pagal duotą periodą

        up_series = []
        dn_series = []



        #🔹 Dinamiškas breakout zonų formavimas---------------------------------
        for i in range(len(df)):
            raw_up = hl2.iloc[i] - multiplier * atr_val.iloc[i]
            raw_dn = hl2.iloc[i] + multiplier * atr_val.iloc[i]
            if i == 0:
                up_series.append(raw_up)
                dn_series.append(raw_dn)
            else:
                prev_up = up_series[i - 1]
                prev_dn = dn_series[i - 1]
                up_val = max(raw_up, prev_up) if df['close'].iloc[i - 1] > prev_up else raw_up
                dn_val = min(raw_dn, prev_dn) if df['close'].iloc[i - 1] < prev_dn else raw_dn
                up_series.append(up_val)
                dn_series.append(dn_val)

                
        up1 = pd.Series(up_series)
        dn1 = pd.Series(dn_series)
        #-----------------------------------------------------------------------


        #🔹 Trendo logika (kaip SuperTrend)-------------------------------------
        trend = [1]
        for i in range(1, len(df)):
            prev_trend = trend[i-1]
            close_i = df['close'].iloc[i]
            if prev_trend == -1 and close_i > dn1.iloc[i]:
                trend.append(1)
            elif prev_trend == 1 and close_i < up1.iloc[i]:
                trend.append(-1)
            else:
                trend.append(prev_trend)
        df['trend'] = trend
        #----------------------------------------------------------------------

        trend_series = pd.Series(trend)
        df['cross_up'] = (trend_series == 1) & (trend_series.shift(1) == -1)
        df['cross_dn'] = (trend_series == -1) & (trend_series.shift(1) == 1)

        df['pos'] = np.nan
        df.loc[df['cross_up'], 'pos'] = 1
        df.loc[df['cross_dn'], 'pos'] = -1
        df['pos'] = df['pos'].ffill().fillna(0)

        #🔹 Pratęsti signalą dar 3 žvakėms po kirtimo (viso 4)
        df['cross_up_extended'] = 0
        df['cross_dn_extended'] = 0

        for i in df.index[df['cross_up']]:
            df.loc[i:i+3, 'cross_up_extended'] = 1

        for i in df.index[df['cross_dn']]:
            df.loc[i:i+3, 'cross_dn_extended'] = 1
        

        return df





    def chande_mo(close: pd.Series, length: int = 9) -> pd.Series:
        diff = close.diff()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)

        sum_up = up.rolling(window=length).sum()
        sum_down = down.rolling(window=length).sum()

        cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)
        return cmo

    def stoch_rsi(close: pd.Series, rsi_length=14, stoch_length=14) -> pd.Series:
        rsi_values = rsi(close, length=rsi_length)

        min_rsi = rsi_values.rolling(window=stoch_length).min()
        max_rsi = rsi_values.rolling(window=stoch_length).max()

        stoch_rsi = (rsi_values - min_rsi) / (max_rsi - min_rsi)
        return stoch_rsi * 100


    def order_flow_data_reconstruction(df, windows=(5, 10, 15, 18, 20, 25, 30, 45, 60), eps=1e-9):
        # Užpildom NaN
        for c in ['buy_count','sell_count','buy_volume','sell_volume']:
            df[c] = df[c].fillna(0)

        # Delta Volume & Count
        df['delta_volume'] = df['buy_volume'] - df['sell_volume']
        df['delta_count']  = df['buy_count']  - df['sell_count']

        # Bazė rolling'ams – tik praeitis (no leakage)
        base_vol = df['delta_volume'].shift(1)
        base_cnt = df['delta_count'].shift(1)

        for k in windows:
            # sum / mean
            df[f'delta_vol_sum_{k}']  = base_vol.rolling(k, min_periods=1).sum()
            df[f'delta_cnt_sum_{k}']  = base_cnt.rolling(k, min_periods=1).sum()
            df[f'delta_vol_mean_{k}'] = base_vol.rolling(k, min_periods=max(2, k//3)).mean()
            df[f'delta_cnt_mean_{k}'] = base_cnt.rolling(k, min_periods=max(2, k//3)).mean()

            # std
            std_v = base_vol.rolling(k, min_periods=max(3, k//3)).std()
            std_c = base_cnt.rolling(k, min_periods=max(3, k//3)).std()
            df[f'delta_vol_std_{k}']  = std_v
            df[f'delta_cnt_std_{k}']  = std_c

            # z-score (stabilized su eps)
            df[f'delta_vol_z_{k}'] = (base_vol - df[f'delta_vol_mean_{k}']) / (std_v + eps)
            df[f'delta_cnt_z_{k}'] = (base_cnt - df[f'delta_cnt_mean_{k}']) / (std_c + eps)

            # === Buyers/Sellers volume ratio per k ===
            buyers_vol_sum  = df['buy_volume'].shift(1).rolling(k, min_periods=1).sum()
            sellers_vol_sum = df['sell_volume'].shift(1).rolling(k, min_periods=1).sum()
            df[f'buyers_vs_sellers_vol_ratio_{k}'] = buyers_vol_sum / (sellers_vol_sum + eps)

            # === Buyers/Sellers count ratio per k ===
            buyers_cnt_sum  = df['buy_count'].shift(1).rolling(k, min_periods=1).sum()
            sellers_cnt_sum = df['sell_count'].shift(1).rolling(k, min_periods=1).sum()
            df[f'buyers_vs_sellers_cnt_ratio_{k}'] = buyers_cnt_sum / (sellers_cnt_sum + eps)
        return df




    input_file = f"COMBINED_BTC_USDT_part{part}.csv"  # <- čia įrašyk savo failą
    df = pd.read_csv(input_file)



    df["rsi"] = rsi(df["close"], length=14)
    df["stoch_rsi"] = stoch_rsi(df["close"], rsi_length=14, stoch_length=14)
    df["cmo"] = chande_mo(df["close"], length=9)
    df["roc"] = roc(df["close"], length=9)
    df["adx"] = adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]


    spike_mults = [3, 6, 10]
    df["range"] = (df["high"] - df["low"]).abs()
    df["avg_hist_5"] = df["range"].shift(1).rolling(window=5).mean()
    for m in spike_mults:
        df[f"spike_x{m}"] = (df["range"] > m * df["avg_hist_5"]).astype(int)
    df = df.drop(columns=["avg_hist_5", "range"])
    df = Arbiter_of_Fate(df, sensitivity=9, atr_len=10)
    df.drop(columns=["pos"], inplace=True)
    df['atr14'] = atr(df, 14)

    df["bodysize"] = (df["close"] - df["open"]).abs()


    # --- Order Flow Data Reconstruction ---
    df = order_flow_data_reconstruction(df)

    df = df.iloc[61:]  # <- čia skipini pradžią
    
    
    features = [
    # indikatoriai
    'rsi','stoch_rsi','cmo','roc','adx','atr14','spike_x3','spike_x6','spike_x10',
    'bodysize','cross_up_extended','cross_dn_extended',
    'delta_volume','delta_count',
    'delta_vol_sum_5','delta_cnt_sum_5','delta_vol_mean_5','delta_cnt_mean_5',
    'delta_vol_std_5','delta_cnt_std_5','delta_vol_z_5','delta_cnt_z_5',
    'buyers_vs_sellers_vol_ratio_5','buyers_vs_sellers_cnt_ratio_5',
    'delta_vol_sum_10','delta_cnt_sum_10','delta_vol_mean_10','delta_cnt_mean_10',
    'delta_vol_std_10','delta_cnt_std_10','delta_vol_z_10','delta_cnt_z_10',
    'buyers_vs_sellers_vol_ratio_10','buyers_vs_sellers_cnt_ratio_10',
    'delta_vol_sum_15','delta_cnt_sum_15','delta_vol_mean_15','delta_cnt_mean_15',
    'delta_vol_std_15','delta_cnt_std_15','delta_vol_z_15','delta_cnt_z_15',
    'buyers_vs_sellers_vol_ratio_15','buyers_vs_sellers_cnt_ratio_15',
    'delta_vol_sum_18','delta_cnt_sum_18','delta_vol_mean_18','delta_cnt_mean_18',
    'delta_vol_std_18','delta_cnt_std_18','delta_vol_z_18','delta_cnt_z_18',
    'buyers_vs_sellers_vol_ratio_18','buyers_vs_sellers_cnt_ratio_18',
    'delta_vol_sum_20','delta_cnt_sum_20','delta_vol_mean_20','delta_cnt_mean_20',
    'delta_vol_std_20','delta_cnt_std_20','delta_vol_z_20','delta_cnt_z_20',
    'buyers_vs_sellers_vol_ratio_20','buyers_vs_sellers_cnt_ratio_20',
    'delta_vol_sum_25','delta_cnt_sum_25','delta_vol_mean_25','delta_cnt_mean_25',
    'delta_vol_std_25','delta_cnt_std_25','delta_vol_z_25','delta_cnt_z_25',
    'buyers_vs_sellers_vol_ratio_25','buyers_vs_sellers_cnt_ratio_25',
    'delta_vol_sum_30','delta_cnt_sum_30','delta_vol_mean_30','delta_cnt_mean_30',
    'delta_vol_std_30','delta_cnt_std_30','delta_vol_z_30','delta_cnt_z_30',
    'buyers_vs_sellers_vol_ratio_30','buyers_vs_sellers_cnt_ratio_30',
    'delta_vol_sum_45','delta_cnt_sum_45','delta_vol_mean_45','delta_cnt_mean_45',
    'delta_vol_std_45','delta_cnt_std_45','delta_vol_z_45','delta_cnt_z_45',
    'buyers_vs_sellers_vol_ratio_45','buyers_vs_sellers_cnt_ratio_45',
    'delta_vol_sum_60','delta_cnt_sum_60','delta_vol_mean_60','delta_cnt_mean_60',
    'delta_vol_std_60','delta_cnt_std_60','delta_vol_z_60','delta_cnt_z_60',
    'buyers_vs_sellers_vol_ratio_60','buyers_vs_sellers_cnt_ratio_60',]
    df[features] = df[features].round(2)

    # --- Išsaugojimas ---
    output_file = f'WITH_BTC_USDT_1m_part{part}.csv'
    df.to_csv(output_file, index=False)
    print("Done! File saved:", output_file)






#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################
#################################################################################################################################################################################################

def run_loop():
    start = datetime(2022, 11, 10)
    end = datetime(2025, 8, 8)
    part = 0

    while start < end:
        part += 1
        loop_start = start
        loop_end = min(start + timedelta(days=30*22), end) #22 is max nes kitaip virsys line kieki per csv file

        print(f"\n🔁 Period: {loop_start.date()} to {loop_end.date()} (Part {part})")


        get_OF_data(loop_start, loop_end, part)
        prices_ccxt_data(loop_start, loop_end, part)

        def combine_price_and_OF(part):
            import pandas as pd
            prices = pd.read_csv(f'WITHOUT_BTC_USDT_1m_part{part}.csv', parse_dates=['timestamp'])
            of = pd.read_csv(f'BTCUSDT_OF_summary_part{part}.csv', parse_dates=['timestamp_minute'])
            merged = pd.merge(prices, of, how='left', left_on='timestamp', right_on='timestamp_minute')
            merged.drop(columns=['timestamp_minute'], inplace=True)
            merged.to_csv(f'COMBINED_BTC_USDT_part{part}.csv', index=False)


        combine_price_and_OF(part)
        features_adder(part)

        start = loop_end + timedelta(days=1)
if __name__ == "__main__":
    run_loop() 