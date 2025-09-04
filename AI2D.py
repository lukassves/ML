from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Manager
import threading
import time
import gc
from pympler import asizeof
import psutil
from multiprocessing import freeze_support
import math
from xgboost import XGBClassifier
import shap
from collections import Counter
from collections import defaultdict
from scipy.stats import zscore
from sklearn.utils.class_weight import compute_class_weight
from joblib import load
from datetime import datetime
from dataclasses import dataclass
from xgboost.callback import EarlyStopping as XgbEarlyStopping
import matplotlib.pyplot as plt
from glob import glob
from tqdm import trange
import json
######################## Importu Galas #########################################################################################################################################################################










################# GLOBAL Parameters ################################################################################################################################
fee = 0.00055
initial_balance = 300
leverage = 30
max_used_balance = 28000
n_jobs = 4  
batch_size = 1
entry_price_column = "open"
high_column = "high"
low_column = "low"
chunk_paths = []
encoder = LabelEncoder()

CREATE_ZMOGYSTAI = False   # don't write zmogystai_*.csv
CREATE_MASTERY   = True    # still produce mastery_*.csv

output_folder = "labeliai"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv("WITH_BTC_USDT_1m_part1.csv")
tp_values = [0.1] #for loopo reiksmes / gali but daug
sl_values = [0.1] #for loopo reiksmes / gali but daug
lookahead_values = [20] #for loopo reiksmes / gali but daug

#stage2 param.
OOS_FILE = os.getenv("OOS_FILE", "WITH_BTC_USDT_1m_part2.csv")

# === BALANSAS ===
balance = initial_balance
peak_balance = balance
lowest_balance = balance

# === POZICIJOS TRACKINIMAS ===
in_position = False
position_direction = None
entry_price = None
entry_time = None
entry_confidence = None

# === METRIKOS ===
tp_hits = 0
sl_hits = 0
next_signals_hit = 0
profitable_next_closes = 0
unprofitable_next_closes = 0
labas = False

expected_direction = None  # Pirmam signalui leidžiama bet kokia kryptis
tps_hits=0
sls_hits=0
next_signals_hits=0
trades = 0
wins=0
################# GLOBAL Parameters ################################################################################################################################

















############## ZMOGYSTAI Feature Importance Summary dataframe only (no file) for mastery  ##############################################################
def create_readable_summary(model, feature_names, output_path, model_tp, model_sl, model_lookahead, label_file_path,
                            return_df: bool = False, save: bool = True):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    rows = []

    df_lbl = pd.read_csv(label_file_path)
    unique_labels = sorted(df_lbl["label"].unique())
    for label in unique_labels:
        row = {"tp": model_tp, "sl": model_sl, "lookahead": model_lookahead, "label": label}
        for i, idx in enumerate(sorted_idx[:15], 1):
            row[f"feature_{i}"] = feature_names[idx]
            row[f"importance_{i}"] = round(importances[idx], 5)
        rows.append(row)

    df_out = pd.DataFrame(rows)

    if save and output_path:
        df_out.to_csv(output_path, index=False)
        print(f"📊 Summary išsaugotas: {output_path}")

    if return_df:
        return df_out
############## ZMOGYSTAI Feature Importance Summary dataframe only (no file) for mastery  ##############################################################













###### MASTER FILE #######################################################################################################################
def generate_mastery(zmogystai: pd.DataFrame | None = None,
                     zmogystai_path: str | None = None,
                     chunk_folder: str = "labeliai",
                     output_path: str = "mastery.csv",
                     model_tp: float | None = None,
                     model_sl: float | None = None):

    #1) Load žmogystai either from DF (preferred) or from file path (fallback)
    if zmogystai is not None:
        zmog = zmogystai.copy()
    else:
        if not zmogystai_path or not os.path.exists(zmogystai_path):
            print(f"❌ Nėra žmogystai DF ir failas nerastas: {zmogystai_path}")
            return
        try:
            zmog = pd.read_csv(zmogystai_path)
        except pd.errors.EmptyDataError:
            print(f"⚠️ Sugadintas žmogystai failas: {zmogystai_path}")
            return

    if zmog.empty:
        print("⚠️ Tuščias žmogystai DF.")
        return

    # 2) Load all label chunks
    stats, chunk_data, total_rows_all = [], [], 0
    if not os.path.isdir(chunk_folder):
        print(f"⚠️ Chunk folder nerastas: {chunk_folder}")
        return

    for fname in os.listdir(chunk_folder):
        if fname.endswith(".csv"):
            path = os.path.join(chunk_folder, fname)
            try:
                ch = pd.read_csv(path)
            except Exception:
                continue
            chunk_data.append(ch)
            total_rows_all += len(ch)


    for _, row in zmog.iterrows():
        label = row["label"]
        combo = [
            (row["feature_1"], row["importance_1"]),
            (row["feature_2"], row["importance_2"]),
            (row["feature_3"], row["importance_3"]),
        ]
        tp_hits = 0
        sl_hits = 0
        total = 0
        total_match = 0
        time_to_hit_total = []
        tp = row["tp"]
        sl = row["sl"]

        # extract label type once
        current_type = "tp" if "tp" in label else "sl"

        for chunk in chunk_data:
            for _, r in chunk.iterrows():
                if r["label"] != label:
                    continue  # we only count exact matches now

                match = True

                if match:
                    total_match += 1
                    total += 1
                    if current_type == "tp":
                        tp_hits += 1
                    else:
                        sl_hits += 1

                    if "time_to_hit" in r.index:
                        try:
                            time_to_hit_total.append(float(r["time_to_hit"]))
                        except:
                            pass
    

        if total == 0:
            continue
        win_rate = tp_hits / total if label.startswith("tp_") else sl_hits / total
        rr = tp / sl if sl != 0 else 1
        support = total
        score = win_rate * math.log(support + 1) * rr
        win_rate_pct = round(win_rate * 100, 1)
        density_pct = round((total_match / total_rows_all) * 100, 3)
        if score >= 5:
            reco = "Oh sexy lady, zajebisimo"
        elif score <= 2:
            reco = "Hujnia"
        else: ##tarp 2 ir 5
            reco = "Meh, neidomu, nei ten, nei ten"
        avg_time_to_hit = round(np.mean(time_to_hit_total), 2) if time_to_hit_total else "N/A"
        stats.append({
            "label": label,
            "combo": " + ".join([f"{f}{d}" for f, d in combo]),
            "TP hitai": tp_hits,
            "SL hitai": sl_hits,
            "Total executed": total,
            "Win rate %": win_rate_pct,
            "RR": round(rr, 2),
            "signal density %": density_pct,
            "avg bars to hit": avg_time_to_hit,
            "score": round(score, 2),
            "recommendation": reco
        })

    # === Sujungiam tpslshort / tpsllong ===
    df_stats = pd.DataFrame(stats)

    # Paimam paskutines 4 eilutes (tp/sl long/short)
    # Patikrinam ar visi 4 reikalingi labeliai egzistuoja
    labels = set(df_stats["label"].tolist())
    has_all_4 = (
        any(l.startswith("tp_") and l.endswith("_short") for l in labels) and
        any(l.startswith("sl_") and l.endswith("_short") for l in labels) and
        any(l.startswith("tp_") and l.endswith("_long") for l in labels) and
        any(l.startswith("sl_") and l.endswith("_long") for l in labels)
    )
    tp= model_tp
    sl= model_sl
    # Jei yra visi 4 – tada galima daryti agregavimą
    if has_all_4:
        short = df_stats[df_stats["label"].str.endswith("_short")]
        long = df_stats[df_stats["label"].str.endswith("_long")]

        # SHORT WINRATE
        tp_s = short["TP hitai"].sum()
        sl_s = short["SL hitai"].sum()
        exec_s = short["Total executed"].sum()
        winrate_s = round(100 * tp_s / exec_s, 2) if exec_s > 0 else 0

        # LONG WINRATE
        tp_l = long["TP hitai"].sum()
        sl_l = long["SL hitai"].sum()
        exec_l = long["Total executed"].sum()
        winrate_l = round(100 * tp_l / exec_l, 2) if exec_l > 0 else 0


        df_stats.loc[len(df_stats)] = {
            "label": f"tp{tp}_sl{sl}_short",
            "combo": "COMBINED",
            "TP hitai": tp_s,
            "SL hitai": sl_s,
            "Total executed": exec_s,
            "Win rate %": round(winrate_s, 2),
            "RR": round(short["RR"].mean(), 2),
            "signal density %": round(short["signal density %"].mean(), 3),
            "avg bars to hit": round(np.mean([float(x) for x in short["avg bars to hit"] if x != "N/A"]), 2) if any(short["avg bars to hit"] != "N/A") else "N/A",
            "score": round(short["score"].mean(), 2),
            "recommendation": "AUTO"
        }
        df_stats.loc[len(df_stats)] = {
            "label": f"tp{tp}_sl{sl}_long",
            "combo": "COMBINED",
            "TP hitai": tp_l,
            "SL hitai": sl_l,
            "Total executed": exec_l,
            "Win rate %": round(winrate_l, 2),
            "RR": round(long["RR"].mean(), 2),
            "signal density %": round(long["signal density %"].mean(), 3),
            "avg bars to hit": round(np.mean([float(x) for x in long["avg bars to hit"] if x != "N/A"]), 2) if any(long["avg bars to hit"] != "N/A") else "N/A",
            "score": round(long["score"].mean(), 2),
            "recommendation": "AUTO"
        }

    # Išsaugom
    df_stats.to_csv(output_path, index=False)
    print(f"🏆 Mastery išsaugotas: {output_path}")
###### MASTER FILE #######################################################################################################################












###### BLACK BOX OPINION #################################################################################################################
def black_box_opinion(model, X, y_true, df_original, model_tp, model_sl, threshold=0.6,class_names=None, output_path="black_box_opinion.csv"):
    y_proba = model.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=1)
    y_conf = np.max(y_proba, axis=1)


    if class_names is None:
        # saugiklis – paimti iš modelio
        class_names = getattr(model, "classes_", None)
    if class_names is None:
        class_names = [str(i) for i in range(y_proba.shape[1])]
    class_names = np.array(class_names)


    label_decoded = np.array(class_names)[y_pred]
    label_true_decoded = np.array(class_names)[y_true]
    
    # safe fallbacks if gates absent (old chunks)
    has_gate_long  = "gate_long"  in df_original.columns
    has_gate_short = "gate_short" in df_original.columns

    decisions = []
    for i in range(len(X)):
        pred_lbl = label_decoded[i]
        conf     = y_conf[i]

        gate_ok = True  # default (back-compat)
        if pred_lbl.endswith("_long") and has_gate_long:
            gate_ok = bool(df_original.iloc[i]["gate_long"])
        elif pred_lbl.endswith("_short") and has_gate_short:
            gate_ok = bool(df_original.iloc[i]["gate_short"])

        decisions.append("EXECUTE" if (conf >= threshold and gate_ok) else "IGNORE")

    df_result = pd.DataFrame({
        "true_label": label_true_decoded,
        "predicted_label": label_decoded,
        "confidence": y_conf,
        "decision": decisions
    })

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    n_classes = len(shap_values)
    # Apskaičiuojam dinaminį SHAP threshold
    # Handle both “old” list-of-arrays and new 3D-array SHAP outputs:
    if isinstance(shap_values, list):
        # list of (n_samples × n_features) arrays
        all_shap  = np.vstack(shap_values)                              # (n_classes*n_samples, n_features)
        n_classes = len(shap_values)
    else:
        # single ndarray of shape (n_samples, n_features, n_classes)
        all_shap  = shap_values.reshape(-1, X.shape[1])  # same shape
        n_classes = shap_values.shape[2]

    mean_shap      = np.abs(all_shap).mean() 
    shap_threshold = round(mean_shap * 0.1, 4)
    print(f"🔍 Dynamic SHAP threshold: {shap_threshold}")

    detailed_info = []

    # Debug block: inspect shap_values structure ###############
    print("▶️ shap_values type:", type(shap_values))
    if isinstance(shap_values, list):
        print("  ▶️ Number of classes:", len(shap_values))
        for idx, arr in enumerate(shap_values):
            print(f"    • class {idx} array shape:", arr.shape)
    else:
        print("  ▶️ shap_values array shape:", shap_values.shape)
    print("▶️ X shape:", X.shape)
    print("▶️ y_pred unique values:", np.unique(y_pred))
    # Debug block: inspect shap_values structure ################

    for i in range(len(X)):
        pred = y_pred[i]
        values  = X.iloc[i]         # ← initialize values here
        row_info = []               # ← initialize row_info here

        # extract the right 1D vector of SHAP values
        if isinstance(shap_values, list):
            # list: shap_values[pred][i]
            if 0 <= pred < n_classes and i < shap_values[pred].shape[0]:
                shap_vals = shap_values[pred][i]
            else:
                shap_vals = np.zeros(X.shape[1])
        else:
            # 3D array: shap_values[i, :, pred]
            if i < shap_values.shape[0] and 0 <= pred < shap_values.shape[2]:
                shap_vals = shap_values[i, :, pred]
            else:
                shap_vals = np.zeros(X.shape[1])

        for feat, sv in zip(X.columns, shap_vals):
            if abs(sv) >= shap_threshold:
                row_info.append(f"{feat}:{sv:.3f} (val={values[feat]:.3f})")

        # Min, max, median iš tų naudotų features
        used_feats = [feat for feat, val in zip(X.columns, shap_vals) if abs(val) >= shap_threshold]
        used_vals = [values[feat] for feat in used_feats]
        if used_vals:
            feat_min = round(min(used_vals), 3)
            feat_max = round(max(used_vals), 3)
            feat_med = round(np.median(used_vals), 3)
        else:
            feat_min = feat_max = feat_med = None

        detailed_info.append({
            "important_features_detailed": " | ".join(row_info),
            "min_val": feat_min,
            "max_val": feat_max,
            "median_val": feat_med
        })

    df_details = pd.DataFrame(detailed_info)
    df_result = pd.concat([df_result, df_details], axis=1)
    merged = pd.concat([df_original.reset_index(drop=True), df_result], axis=1)

    # Skaičiuojam TP/SL pagal predicted_label ir true_label
    stats = []
    for label in sorted(set(label_decoded)):
        subset = merged[(merged["predicted_label"] == label) & (merged["true_label"] == label)]
        if subset.empty:
            continue
        total_exec = sum(subset["decision"] == "EXECUTE")
        tp_hits = sum((subset["decision"] == "EXECUTE") & (subset["true_label"].str.startswith("tp")))
        sl_hits = sum((subset["decision"] == "EXECUTE") & (subset["true_label"].str.startswith("sl")))
        winrate = round(100 * tp_hits / total_exec, 2) if total_exec else 0

        
        # viskas likusiam cikle išlieka
        feat_counter = Counter()
        shap_total = {}

        for idx in subset.index:
            pred = y_pred[idx]
            if isinstance(shap_values, list):
                sv = shap_values[pred][idx] if idx < shap_values[pred].shape[0] else np.zeros(X.shape[1])
            else:
                sv = shap_values[idx, :, pred] if idx < shap_values.shape[0] else np.zeros(X.shape[1])

            for feat, val in zip(X.columns, sv):
                if abs(val) >= shap_threshold:
                    feat_counter[feat] += 1
                    shap_total[feat] = shap_total.get(feat, 0) + abs(val)

        # top 5 svarbiausi
        # Initialize SHAP + value containers
        feat_info = defaultdict(lambda: {
            "total_shap": 0.0,
            "count": 0,
            "values": []
        })

        for idx in subset.index:
            pred = y_pred[idx]
            if isinstance(shap_values, list):
                sv = shap_values[pred][idx] if idx < shap_values[pred].shape[0] else np.zeros(X.shape[1])
            else:
                sv = shap_values[idx, :, pred] if idx < shap_values.shape[0] else np.zeros(X.shape[1])

            row_vals = X.iloc[idx]
            for feat, shap_val in zip(X.columns, sv):
                if abs(shap_val) >= shap_threshold:
                    feat_info[feat]["total_shap"] += abs(shap_val)
                    feat_info[feat]["count"] += 1
                    feat_info[feat]["values"].append(row_vals[feat])

        # Construct detailed feature summary
        parts = []
        for feat, info in feat_info.items():
            imp = round(info["total_shap"] / info["count"], 4)
            vals = info["values"]
            stat_min = round(min(vals), 3)
            stat_max = round(max(vals), 3)
            stat_med = round(np.median(vals), 3)
            parts.append(f'{feat}: {{"importance": {imp}, "min": {stat_min}, "max": {stat_max}, "median": {stat_med}}}')

        agg_details = " | ".join(parts)

        stats.append({
            "label": label,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "executed": total_exec,
            "winrate %": winrate,
            "avg_confidence": round(subset["confidence"].mean(), 3),
            "important_features_detailed": agg_details,
        })
    df_stats = pd.DataFrame(stats)  # ← būtina



    # Patikrinam ar visi 4 reikalingi labeliai egzistuoja
    labels = set(df_stats["label"].tolist())
    has_all_4 = (
        any(l.startswith("tp_") and l.endswith("_short") for l in labels) and
        any(l.startswith("sl_") and l.endswith("_short") for l in labels) and
        any(l.startswith("tp_") and l.endswith("_long") for l in labels) and
        any(l.startswith("sl_") and l.endswith("_long") for l in labels)
    )
    if has_all_4:
        short = df_stats[df_stats["label"].str.endswith("_short")]
        long = df_stats[df_stats["label"].str.endswith("_long")]

        # SHORT WINRATE
        tp_s = short["tp_hits"].sum()
        exec_s = short["executed"].sum()
        winrate_s = round(100 * tp_s / exec_s, 2) if exec_s > 0 else 0

        # LONG WINRATE
        tp_l = long["tp_hits"].sum()
        exec_l = long["executed"].sum()
        winrate_l = round(100 * tp_l / exec_l, 2) if exec_l > 0 else 0
        df_stats.loc[len(df_stats)] = {
            "label": f"tp{model_tp}_sl{model_sl}_short",
            "tp_hits": short["tp_hits"].sum(),
            "sl_hits": short["sl_hits"].sum(),
            "executed": short["executed"].sum(),
            "winrate %": round(winrate_s, 2),
            "avg_confidence": round(short["avg_confidence"].mean(), 3),
            "important_features_detailed": "aggregated from short"
        }
        df_stats.loc[len(df_stats)] = {
            "label":  f"tp{model_tp}_sl{model_sl}_long",
            "tp_hits": long["tp_hits"].sum(),
            "sl_hits": long["sl_hits"].sum(),
            "executed": long["executed"].sum(),
            "winrate %": round(winrate_l, 2),
            "avg_confidence": round(long["avg_confidence"].mean(), 3),
            "important_features_detailed": "aggregated from long"
        }
    
    df_stats.to_csv(output_path, index=False)
    print(f"🧠 Black-box opinion saved: {output_path}")


    # 1. aggregate totals from your stats list
    tp_total       = sum(s["tp_hits"]  for s in stats)
    sl_total       = sum(s["sl_hits"]  for s in stats)
    total_executed = sum(s["executed"] for s in stats)

    # 2. compute profit & coefficient
    profit        = tp_total * 100 - sl_total * 100
    current_coeff = total_executed * profit

    # 3. prepare best-run marker file
    best_txt   = "best_blackbox_opinion.txt"
    prev_coeff = float("-inf")

    # read previous if exists
    if os.path.exists(best_txt):
        with open(best_txt, "r") as f:
            content = f.read().strip()
            if content:
                _, prev_coeff = content.split(",")
                prev_coeff = float(prev_coeff)

    # 4. compare and update
    if current_coeff > prev_coeff:
        with open(best_txt, "w") as f:
            f.write(f"{output_path},{current_coeff}")
        print(f"🚀 New best run! {output_path} (coeff={current_coeff})")
    else:
        print(f"No update: current_coeff={current_coeff} ≤ previous best={prev_coeff}")
###### BLACK BOX OPINION #################################################################################################################


















###### RAM ESTIMATION AND SYSTEM USAGE ###################################################################################################
def estimate_gb(obj):
    return asizeof.asizeof(obj) / 1024**3  # GB
def system_used_ram_gb():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1024**3
###### RAM ESTIMATION AND SYSTEM USAGE ###################################################################################################

















###### TRADING STRATEGY ########################################################################################################################
DEBUG_AEA = True

# Kurie flag'ai spausdinami (įtraukiau visus iš compute_aea_signals)
AEA_FLAG_COLS = [
    # kokybės / spike
    "quality", "has_spike",
    # ADX / Stoch / Momentum / RSI
    "adx_ok", "cross_up", "cross_down", "mom_ok_long", "mom_ok_short", "rsi_ok_long", "rsi_ok_short",
    # OF
    "of5_long", "of5_short", "of15_long", "of15_short",
    # režimas
    "_accum", "_distrib", "_accum_p", "_distrib_p", "accumulation_on", "distribution_on",
    # galutiniai
    "_pre_long", "_pre_short", "_pre_both", "long_trigger", "short_trigger",
    # entry
    "entry_long", "entry_short",
]

def print_aea_stats(df_sig: pd.DataFrame, tag: str = "") -> None:
    rows = len(df_sig)
    print(f"\n[AEA STATS {tag}] rows={rows}")
    for c in AEA_FLAG_COLS:
        if c in df_sig.columns:
            cnt = pd.Series(df_sig[c]).fillna(False).astype(bool).sum()
            pct = (cnt / rows * 100.0) if rows else 0.0
            print(f"  {c:18s}: {int(cnt):6d} ({pct:5.1f}%)")
    # Naudinga pamatyti body_cap jei įrašytas
    if "_body_cap" in df_sig.columns:
        try:
            bc = float(pd.to_numeric(df_sig["_body_cap"], errors="coerce").dropna().iloc[0])
            #print(f"  {'body_cap':18s}: {bc:.2f}")
        except Exception:
            pass




def compute_aea_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Quality filtras pagal bodysize hard cap
    bs = pd.to_numeric(out.get("bodysize"), errors="coerce").abs()

    # tiesioginis limitas
    body_cap = 90.0  
    spike10 = pd.to_numeric(out.get("spike_x10", 0), errors="coerce").fillna(0)
    spike6  = pd.to_numeric(out.get("spike_x6", 0),  errors="coerce").fillna(0)
    has_spike = (spike10 > 0) | (spike6 > 0)

    # quality sąlyga
    quality = (~has_spike) & (bs < body_cap)
    
    # stulpai jei kurima nors defe noresi daryti flag() ir daryti kitokia startegy ivystyma negu kituose defuose
    out["_body_cap"] = body_cap
    out["has_spike"] = has_spike.astype(bool)
    out["quality"]   = quality.astype(bool)


    # --- ADX režimas: 8..40 (mean-rev zona) ---
    adx = pd.to_numeric(out["adx"], errors="coerce")
    adx_smooth = adx.ewm(span=5, adjust=False).mean()
    adx_ok = (adx >= 18) & (adx_smooth.diff() > 0)
    adx_ok = adx_ok.fillna(False)

    # --- StochRSI cross'ai (0..100) ---
    st_prev    = out["stoch_rsi"].shift(2)
    cross_up   = (st_prev < 30) & (out["stoch_rsi"] >= 30)   # LONG trigger
    cross_down = (st_prev > 69) & (out["stoch_rsi"] <= 69)   # SHORT trigger




    # Pasirink „griežtumą“
    ROC_L, ROC_S = 0.002, -0.002   # ~0.2% 1m impulsas
    CMO_L, CMO_S = 5.0, -5.0

    roc_s = pd.to_numeric(out["roc"], errors="coerce")
    cmo_s = pd.to_numeric(out["cmo"], errors="coerce")

    roc_e = roc_s.ewm(span=5, adjust=False).mean()
    cmo_m = cmo_s.rolling(5, min_periods=3).mean()

    mom_ok_long  = (roc_e > ROC_L) | (cmo_m >  CMO_L)
    mom_ok_short = (roc_e < ROC_S) | (cmo_m <  CMO_S)


    # --- RSI ribos (ne kraštuose) ---
    rsi_ok_long  = out["rsi"].between(25, 55)
    rsi_ok_short = out["rsi"].between(45, 70)

    













    # --- Order-flow (5): STRICT but lighter ---
    dv5 = pd.to_numeric(out["delta_vol_sum_5"], errors="coerce")
    dc5 = pd.to_numeric(out["delta_cnt_sum_5"], errors="coerce")
    vr5 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_5"], errors="coerce")
    cr5 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_5"], errors="coerce")

    OF5_LONG_Q, OF5_SHORT_Q = 0.85, 0.15
    vr5_hi = float(np.nanquantile(vr5, OF5_LONG_Q)) if vr5.notna().any() else 1.0
    cr5_hi = float(np.nanquantile(cr5, OF5_LONG_Q)) if cr5.notna().any() else 1.0
    vr5_lo = float(np.nanquantile(vr5, OF5_SHORT_Q)) if vr5.notna().any() else 1.0
    cr5_lo = float(np.nanquantile(cr5, OF5_SHORT_Q)) if cr5.notna().any() else 1.0
    # mild guard rails
    vr5_hi = max(vr5_hi, 1.03);  cr5_hi = max(cr5_hi, 1.03)
    vr5_lo = min(vr5_lo, 0.97);  cr5_lo = min(cr5_lo, 0.97)

    of5_long  = (dv5 > 0) & (dc5 > 0) & (vr5 >= vr5_hi) & (cr5 >= cr5_hi)
    of5_short = (dv5 < 0) & (dc5 < 0) & (vr5 <= vr5_lo) & (cr5 <= cr5_lo)











    # --- Order-flow (15): CONFIRMATION (asym: long OR, short AND) ---
    dv15  = pd.to_numeric(out["delta_vol_sum_15"], errors="coerce")
    dc15  = pd.to_numeric(out["delta_cnt_sum_15"], errors="coerce")
    cnt15 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_15"], errors="coerce")
    vol15 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_15"], errors="coerce")

    OF15_LONG_Q, OF15_SHORT_Q = 0.65, 0.35
    cnt15_hi = float(np.nanquantile(cnt15, OF15_LONG_Q)) if cnt15.notna().any() else 1.0
    vol15_hi = float(np.nanquantile(vol15, OF15_LONG_Q)) if vol15.notna().any() else 1.0
    cnt15_lo = float(np.nanquantile(cnt15, OF15_SHORT_Q)) if cnt15.notna().any() else 1.0
    vol15_lo = float(np.nanquantile(vol15, OF15_SHORT_Q)) if vol15.notna().any() else 1.0
    # gentle rails
    cnt15_hi = max(cnt15_hi, 1.00); vol15_hi = max(vol15_hi, 1.01)
    cnt15_lo = min(cnt15_lo, 1.00); vol15_lo = min(vol15_lo, 0.99)


    # asymmetry: long OR, short AND
    of15_long  = (dv15 >= 0) & (dc15 >= 0) & ((cnt15 >= cnt15_hi) | (vol15 >= vol15_hi))
    of15_short = (dv15 <= 0) & (dc15 <= 0) &  (cnt15 <= cnt15_lo)  & (vol15 <= vol15_lo)





        # --- 10 ---
    dv10  = pd.to_numeric(out["delta_vol_sum_10"], errors="coerce")
    dc10  = pd.to_numeric(out["delta_cnt_sum_10"], errors="coerce")
    cnt10 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_10"], errors="coerce")
    vol10 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_10"], errors="coerce")

    OF10_LONG_Q, OF10_SHORT_Q = 0.70, 0.30
    cnt10_hi = float(np.nanquantile(cnt10, OF10_LONG_Q)) if cnt10.notna().any() else 1.0
    vol10_hi = float(np.nanquantile(vol10, OF10_LONG_Q)) if vol10.notna().any() else 1.0
    cnt10_lo = float(np.nanquantile(cnt10, OF10_SHORT_Q)) if cnt10.notna().any() else 1.0
    vol10_lo = float(np.nanquantile(vol10, OF10_SHORT_Q)) if vol10.notna().any() else 1.0
    cnt10_hi = max(cnt10_hi, 1.00); vol10_hi = max(vol10_hi, 1.01)
    cnt10_lo = min(cnt10_lo, 1.00); vol10_lo = min(vol10_lo, 0.99)

    of10_long  = (dv10 >= 0) & (dc10 >= 0) & ((cnt10 >= cnt10_hi) | (vol10 >= vol10_hi))
    of10_short = (dv10 <= 0) & (dc10 <= 0) &  (cnt10 <= cnt10_lo)  & (vol10 <= vol10_lo)











    # --- 15 ---
    dv15  = pd.to_numeric(out["delta_vol_sum_15"], errors="coerce")
    dc15  = pd.to_numeric(out["delta_cnt_sum_15"], errors="coerce")
    cnt15 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_15"], errors="coerce")
    vol15 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_15"], errors="coerce")

    OF15_LONG_Q, OF15_SHORT_Q = 0.70, 0.30
    cnt15_hi = float(np.nanquantile(cnt15, OF15_LONG_Q)) if cnt15.notna().any() else 1.0
    vol15_hi = float(np.nanquantile(vol15, OF15_LONG_Q)) if vol15.notna().any() else 1.0
    cnt15_lo = float(np.nanquantile(cnt15, OF15_SHORT_Q)) if cnt15.notna().any() else 1.0
    vol15_lo = float(np.nanquantile(vol15, OF15_SHORT_Q)) if vol15.notna().any() else 1.0
    cnt15_hi = max(cnt15_hi, 1.00); vol15_hi = max(vol15_hi, 1.01)
    cnt15_lo = min(cnt15_lo, 1.00); vol15_lo = min(vol15_lo, 0.99)

    of15_long  = (dv15 >= 0) & (dc15 >= 0) & ((cnt15 >= cnt15_hi) | (vol15 >= vol15_hi))
    of15_short = (dv15 <= 0) & (dc15 <= 0) &  (cnt15 <= cnt15_lo)  & (vol15 <= vol15_lo)











    # --- 18 ---
    dv18  = pd.to_numeric(out["delta_vol_sum_18"], errors="coerce")
    dc18  = pd.to_numeric(out["delta_cnt_sum_18"], errors="coerce")
    cnt18 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_18"], errors="coerce")
    vol18 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_18"], errors="coerce")

    OF18_LONG_Q, OF18_SHORT_Q = 0.70, 0.30
    cnt18_hi = float(np.nanquantile(cnt18, OF18_LONG_Q)) if cnt18.notna().any() else 1.0
    vol18_hi = float(np.nanquantile(vol18, OF18_LONG_Q)) if vol18.notna().any() else 1.0
    cnt18_lo = float(np.nanquantile(cnt18, OF18_SHORT_Q)) if cnt18.notna().any() else 1.0
    vol18_lo = float(np.nanquantile(vol18, OF18_SHORT_Q)) if vol18.notna().any() else 1.0
    cnt18_hi = max(cnt18_hi, 1.00); vol18_hi = max(vol18_hi, 1.01)
    cnt18_lo = min(cnt18_lo, 1.00); vol18_lo = min(vol18_lo, 0.99)

    of18_long  = (dv18 >= 0) & (dc18 >= 0) & ((cnt18 >= cnt18_hi) | (vol18 >= vol18_hi))
    of18_short = (dv18 <= 0) & (dc18 <= 0) &  (cnt18 <= cnt18_lo)  & (vol18 <= vol18_lo)






    # --- 20 ---
    dv20  = pd.to_numeric(out["delta_vol_sum_20"], errors="coerce")
    dc20  = pd.to_numeric(out["delta_cnt_sum_20"], errors="coerce")
    cnt20 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_20"], errors="coerce")
    vol20 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_20"], errors="coerce")

    OF20_LONG_Q, OF20_SHORT_Q = 0.65, 0.35
    cnt20_hi = float(np.nanquantile(cnt20, OF20_LONG_Q)) if cnt20.notna().any() else 1.0
    vol20_hi = float(np.nanquantile(vol20, OF20_LONG_Q)) if vol20.notna().any() else 1.0
    cnt20_lo = float(np.nanquantile(cnt20, OF20_SHORT_Q)) if cnt20.notna().any() else 1.0
    vol20_lo = float(np.nanquantile(vol20, OF20_SHORT_Q)) if vol20.notna().any() else 1.0
    cnt20_hi = max(cnt20_hi, 1.00); vol20_hi = max(vol20_hi, 1.01)
    cnt20_lo = min(cnt20_lo, 1.00); vol20_lo = min(vol20_lo, 0.99)

    of20_long  = (dv20 >= 0) & (dc20 >= 0) & ((cnt20 >= cnt20_hi) | (vol20 >= vol20_hi))
    of20_short = (dv20 <= 0) & (dc20 <= 0) &  (cnt20 <= cnt20_lo)  & (vol20 <= vol20_lo)









    # --- 25 ---
    dv25  = pd.to_numeric(out["delta_vol_sum_25"], errors="coerce")
    dc25  = pd.to_numeric(out["delta_cnt_sum_25"], errors="coerce")
    cnt25 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_25"], errors="coerce")
    vol25 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_25"], errors="coerce")

    OF25_LONG_Q, OF25_SHORT_Q = 0.65, 0.35
    cnt25_hi = float(np.nanquantile(cnt25, OF25_LONG_Q)) if cnt25.notna().any() else 1.0
    vol25_hi = float(np.nanquantile(vol25, OF25_LONG_Q)) if vol25.notna().any() else 1.0
    cnt25_lo = float(np.nanquantile(cnt25, OF25_SHORT_Q)) if cnt25.notna().any() else 1.0
    vol25_lo = float(np.nanquantile(vol25, OF25_SHORT_Q)) if vol25.notna().any() else 1.0
    cnt25_hi = max(cnt25_hi, 1.00); vol25_hi = max(vol25_hi, 1.01)
    cnt25_lo = min(cnt25_lo, 1.00); vol25_lo = min(vol25_lo, 0.99)

    of25_long  = (dv25 >= 0) & (dc25 >= 0) & ((cnt25 >= cnt25_hi) | (vol25 >= vol25_hi))
    of25_short = (dv25 <= 0) & (dc25 <= 0) &  (cnt25 <= cnt25_lo)  & (vol25 <= vol25_lo)









    # --- 30 ---
    dv30  = pd.to_numeric(out["delta_vol_sum_30"], errors="coerce")
    dc30  = pd.to_numeric(out["delta_cnt_sum_30"], errors="coerce")
    cnt30 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_30"], errors="coerce")
    vol30 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_30"], errors="coerce")

    OF30_LONG_Q, OF30_SHORT_Q = 0.65, 0.35
    cnt30_hi = float(np.nanquantile(cnt30, OF30_LONG_Q)) if cnt30.notna().any() else 1.0
    vol30_hi = float(np.nanquantile(vol30, OF30_LONG_Q)) if vol30.notna().any() else 1.0
    cnt30_lo = float(np.nanquantile(cnt30, OF30_SHORT_Q)) if cnt30.notna().any() else 1.0
    vol30_lo = float(np.nanquantile(vol30, OF30_SHORT_Q)) if vol30.notna().any() else 1.0
    cnt30_hi = max(cnt30_hi, 1.00); vol30_hi = max(vol30_hi, 1.01)
    cnt30_lo = min(cnt30_lo, 1.00); vol30_lo = min(vol30_lo, 0.99)

    of30_long  = (dv30 >= 0) & (dc30 >= 0) & ((cnt30 >= cnt30_hi) | (vol30 >= vol30_hi))
    of30_short = (dv30 <= 0) & (dc30 <= 0) &  (cnt30 <= cnt30_lo)  & (vol30 <= vol30_lo)







    # --- 45 ---
    dv45  = pd.to_numeric(out["delta_vol_sum_45"], errors="coerce")
    dc45  = pd.to_numeric(out["delta_cnt_sum_45"], errors="coerce")
    cnt45 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_45"], errors="coerce")
    vol45 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_45"], errors="coerce")

    OF45_LONG_Q, OF45_SHORT_Q = 0.60, 0.40
    cnt45_hi = float(np.nanquantile(cnt45, OF45_LONG_Q)) if cnt45.notna().any() else 1.0
    vol45_hi = float(np.nanquantile(vol45, OF45_LONG_Q)) if vol45.notna().any() else 1.0
    cnt45_lo = float(np.nanquantile(cnt45, OF45_SHORT_Q)) if cnt45.notna().any() else 1.0
    vol45_lo = float(np.nanquantile(vol45, OF45_SHORT_Q)) if vol45.notna().any() else 1.0
    cnt45_hi = max(cnt45_hi, 1.00); vol45_hi = max(vol45_hi, 1.01)
    cnt45_lo = min(cnt45_lo, 1.00); vol45_lo = min(vol45_lo, 0.99)

    of45_long  = (dv45 >= 0) & (dc45 >= 0) & ((cnt45 >= cnt45_hi) | (vol45 >= vol45_hi))
    of45_short = (dv45 <= 0) & (dc45 <= 0) &  (cnt45 <= cnt45_lo)  & (vol45 <= vol45_lo)











    # --- 60 ---
    dv60  = pd.to_numeric(out["delta_vol_sum_60"], errors="coerce")
    dc60  = pd.to_numeric(out["delta_cnt_sum_60"], errors="coerce")
    cnt60 = pd.to_numeric(out["buyers_vs_sellers_cnt_ratio_60"], errors="coerce")
    vol60 = pd.to_numeric(out["buyers_vs_sellers_vol_ratio_60"], errors="coerce")

    OF60_LONG_Q, OF60_SHORT_Q = 0.60, 0.40
    cnt60_hi = float(np.nanquantile(cnt60, OF60_LONG_Q)) if cnt60.notna().any() else 1.0
    vol60_hi = float(np.nanquantile(vol60, OF60_LONG_Q)) if vol60.notna().any() else 1.0
    cnt60_lo = float(np.nanquantile(cnt60, OF60_SHORT_Q)) if cnt60.notna().any() else 1.0
    vol60_lo = float(np.nanquantile(vol60, OF60_SHORT_Q)) if vol60.notna().any() else 1.0
    cnt60_hi = max(cnt60_hi, 1.00); vol60_hi = max(vol60_hi, 1.01)
    cnt60_lo = min(cnt60_lo, 1.00); vol60_lo = min(vol60_lo, 0.99)

    of60_long  = (dv60 >= 0) & (dc60 >= 0) & ((cnt60 >= cnt60_hi) | (vol60 >= vol60_hi))
    of60_short = (dv60 <= 0) & (dc60 <= 0) &  (cnt60 <= cnt60_lo)  & (vol60 <= vol60_lo)











    # --- Režimai + 2-bar persistence ---
    accum   = (of15_long & of18_long).fillna(False)
    distrib = (of15_short & of18_short).fillna(False)
    accum_p   = accum.rolling(2, min_periods=1).sum() == 2
    distrib_p = distrib.rolling(2, min_periods=1).sum() == 2
    out["accumulation_on"] = (accum_p & ~distrib_p).fillna(False)
    out["distribution_on"] = (distrib_p & ~accum_p).fillna(False)


    EPS = 1e-9
    dv5 = pd.to_numeric(out["delta_vol_sum_5"],  errors="coerce")
    dc5 = pd.to_numeric(out["delta_cnt_sum_5"],  errors="coerce")
    dv15= pd.to_numeric(out["delta_vol_sum_15"], errors="coerce")
    dc15= pd.to_numeric(out["delta_cnt_sum_15"], errors="coerce")

    imp_long  = (dv5 > EPS)  & (dc5 > EPS)
    imp_short = (dv5 < -EPS) & (dc5 < -EPS)
    cont_long  = (dv15 > EPS)  & (dc15 > EPS)
    cont_short = (dv15 < -EPS) & (dc15 < -EPS)

    veto_stoch_short = (out["stoch_rsi"] == 100)
    # --- galutiniai trigeriai ---
    out["long_trigger"]  = (accum).fillna(False) 
    out["short_trigger"] = (distrib).fillna(False)

    # išvengti dviprasmybių
    both = out["long_trigger"] & out["short_trigger"]
    if both.any():
        out.loc[both, ["long_trigger","short_trigger"]] = False

    # suderinamumas su tavo pipeline
    out["entry_long"]  = out["long_trigger"]
    out["entry_short"] = out["short_trigger"]


    # diagnostikai taip pat kad visus isvestu
    out["_body_cap"] = body_cap
    out["has_spike"] = has_spike.astype(bool)
    out["quality"]   = quality.astype(bool)

    out["adx_ok"] = adx_ok.astype(bool)
    out["cross_up"] = cross_up.astype(bool)
    out["cross_down"] = cross_down.astype(bool)

    out["mom_ok_long"]  = mom_ok_long.astype(bool)
    out["mom_ok_short"] = mom_ok_short.astype(bool)
    out["rsi_ok_long"]  = rsi_ok_long.astype(bool)
    out["rsi_ok_short"] = rsi_ok_short.astype(bool)

    out["of5_long"]  = of5_long.astype(bool)
    out["of5_short"] = of5_short.astype(bool)
    out["of15_long"] = of15_long.astype(bool)
    out["of15_short"] = of15_short.astype(bool)

    out["_accum"]    = accum.astype(bool)
    out["_distrib"]  = distrib.astype(bool)
    out["_accum_p"]  = accum_p.astype(bool)
    out["_distrib_p"] = distrib_p.astype(bool)

    out["imp_long"]  = imp_long.astype(bool)
    out["imp_short"] = imp_short.astype(bool)
    out["cont_long"]  = cont_long.astype(bool)
    out["cont_short"] = cont_short.astype(bool)


    if DEBUG_AEA:
        print_aea_stats(out, tag="inline")
    return out
###### TRADING STRATEGY ########################################################################################################################














###### @PLOT SHOW FUNC. ########################################################################################################
def plot_box_steps(df: pd.DataFrame,
                   last: int | str | None = 300,
                   use_prior: bool = True,
                   save_path: str | None = None,
                   max_points: int = 200_000,
                   trades_path: str | None = None,
                   trades_df: pd.DataFrame | None = None):
    """
    Rodo:
      • kainos liniją,
      • (jei yra) accumulation_on / distribution_on šešėlius,
      • strategijos entry_long / entry_short (iš compute_aea_signals),
      • (NAUJA) VISUS simulate_detailed trade'us:
           entry_long, entry_short, take_profit_exit, stop_loss_exit.

    last: "all"/None/-1 -> visa istorija; kitu atveju paskutiniai N barų.
    trades_path: kelias į 'simuliacijos_outputas_*.csv' (jei nepaduosi, paims naujausią).
    trades_df: alternatyviai paduok jau įkeltą DataFrame.
    """
    import os, glob
    # 1) AEA signalai kaip ir anksčiau
    bx_full = compute_aea_signals(df)

    # parenkam langą
    if last in (None, "all", -1):
        bx = bx_full.copy()
    else:
        bx = bx_full.tail(int(last)).copy()

    idx = bx.index
    close = pd.to_numeric(bx["close"], errors="coerce")

    # downsampling tik linijai (markeriai – visi)
    step = max(1, len(bx) // max_points) if len(bx) > max_points else 1
    x_line   = idx[::step]
    close_ds = close.iloc[::step]

    def _flag(name: str):
        return bx[name].fillna(False) if name in bx.columns else pd.Series(False, index=idx)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_line, close_ds, label="Close")

    # režimų šešėliai (jei turi)
    if "accumulation_on" in bx.columns:
        acc = _flag("accumulation_on").to_numpy()
        ax.fill_between(idx, close.min(), close.max(), where=acc, alpha=0.06, label="Accumulation")
    if "distribution_on" in bx.columns:
        dist = _flag("distribution_on").to_numpy()
        ax.fill_between(idx, close.min(), close.max(), where=dist, alpha=0.06, label="Distribution")

    # strategijos įėjimai (iš compute_aea_signals)
    il = bx.index[_flag("entry_long")]
    is_ = bx.index[_flag("entry_short")]
    if len(il): ax.scatter(il,  close.loc[il],  marker="^", s=42, label="entry_long")
    if len(is_): ax.scatter(is_, close.loc[is_], marker="v", s=42, label="entry_short")

    # triggeriai (jei yra)
    if "long_trigger" in bx.columns:
        lt = bx.index[_flag("long_trigger")]
        if len(lt): ax.scatter(lt, close.loc[lt], marker=".", s=16, label="Long trigger")
    if "short_trigger" in bx.columns:
        st = bx.index[_flag("short_trigger")]
        if len(st): ax.scatter(st, close.loc[st], marker=".", s=16, label="Short trigger")

    # 2) (NAUJA) simulate_detailed sandoriai
    sim = None
    if trades_df is not None:
        sim = trades_df.copy()
    else:
        if trades_path is None:
            # pasiimam naujausią 'simuliacijos_outputas_*.csv'
            cands = sorted(glob.glob("simuliacijos_outputas_*.csv"), key=os.path.getmtime)
            trades_path = cands[-1] if cands else None
        if trades_path and os.path.exists(trades_path):
            try:
                sim = pd.read_csv(trades_path)
            except Exception as e:
                print(f"⚠️ Nepavyko nuskaityti {trades_path}: {e}")

    if sim is not None and not sim.empty:
        # Paruošiam timestamp -> bar index map'ą
        ts_map = {}
        if "timestamp" in df.columns:
            df_ts = pd.to_datetime(df["timestamp"], errors="coerce")
            ts_map = {ts: i for i, ts in enumerate(df_ts)}

        # Normalizuojam laukus
        if "timestamp_entry" in sim.columns:
            sim["timestamp_entry"] = pd.to_datetime(sim["timestamp_entry"], errors="coerce")
        if "timestamp_close" in sim.columns:
            sim["timestamp_close"] = pd.to_datetime(sim["timestamp_close"], errors="coerce")

        # entry_bar_idx jei nėra – suskaičiuojam iš timestamp_entry
        if "entry_bar_idx" not in sim.columns or sim["entry_bar_idx"].isna().all():
            sim["entry_bar_idx"] = sim["timestamp_entry"].map(ts_map).astype("Int64")

        # exit bar idx iš timestamp_close (grafikui)
        sim["close_bar_idx"] = sim["timestamp_close"].map(ts_map).astype("Int64") if "timestamp_close" in sim.columns else pd.Series(pd.NA, dtype="Int64", index=sim.index)

        # Filtras į matomą langą
        view_min, view_max = idx.min(), idx.max()
        sim_view = sim.dropna(subset=["entry_bar_idx"]).copy()
        sim_view = sim_view[(sim_view["entry_bar_idx"] >= view_min) & (sim_view["entry_bar_idx"] <= view_max)]

        # Entry markeriai pagal position
        pos_series = sim_view.get("position", pd.Series(index=sim_view.index, dtype=object)).astype(str).str.lower()

        ent_long  = sim_view[pos_series == "long"]
        ent_short = sim_view[pos_series == "short"]

        # helper: jei nėra price – imam close pagal bar idx
        def _vals_price(df_local, price_col, bar_col):
            if price_col in df_local.columns and pd.to_numeric(df_local[price_col], errors="coerce").notna().any():
                return pd.to_numeric(df_local[price_col], errors="coerce").values
            bidx = pd.to_numeric(df_local[bar_col], errors="coerce").astype("Int64")
            return close.reindex(bidx.dropna().astype(int), fill_value=np.nan).values

        if not ent_long.empty:
            ax.scatter(ent_long["entry_bar_idx"].astype(int).values,
                       _vals_price(ent_long, "entry_price", "entry_bar_idx"),
                       marker="^", s=64, label="entry_long (sim)")

        if not ent_short.empty:
            ax.scatter(ent_short["entry_bar_idx"].astype(int).values,
                       _vals_price(ent_short, "entry_price", "entry_bar_idx"),
                       marker="v", s=64, label="entry_short (sim)")

        # Exit markeriai – skirstom pagal reason_for_close
        if "reason_for_close" in sim_view.columns:
            tp = sim_view[sim_view["reason_for_close"] == "take_profit"]
            sl = sim_view[sim_view["reason_for_close"] == "stop_loss"]

            if not tp.empty and tp["close_bar_idx"].notna().any():
                ax.scatter(tp["close_bar_idx"].dropna().astype(int).values,
                           _vals_price(tp, "exit_price", "close_bar_idx"),
                           marker="o", s=46, label="take_profit_exit")

            if not sl.empty and sl["close_bar_idx"].notna().any():
                ax.scatter(sl["close_bar_idx"].dropna().astype(int).values,
                           _vals_price(sl, "exit_price", "close_bar_idx"),
                           marker="x", s=60, label="stop_loss_exit")

    # Titulai/legend
    ax.set_title("AEA + all simulated trades (full)" if last in (None, "all", -1)
                 else f"AEA + all simulated trades (last {last})")
    ax.set_xlabel("Bars"); ax.set_ylabel("Price")

    # sutvarkom dublikatų legendą
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140)
    plt.show()

###### @PLOT SHOW FUNC. ########################################################################################################























##### AI TRAINING ON FILES ##############################################################################################################
def train_on_file(path, tp, sl, lookahead):
    if os.path.getsize(path) < 100:
        print(f"⚠️ Skipped {path} – failas per mažas arba tuščias.")
        return
    df = pd.read_csv(path)
    
    

    # 1) pranešk, tada perfiltruok, ir tiek
    unknown = set(df["label"].unique()) - set(encoder.classes_)
    if unknown:
        print(f"⚠️ {path} – nežinomi labeliai: {unknown}. Praleidžiu šias eilutes.")
        df = df[df["label"].isin(encoder.classes_)]

    if df.empty:
        print(f"⚠️ {path} – neliko eilučių po filtravimo.")
        return

    non_features = {'label','tp','sl','lookahead','time_to_hit','gate_long','gate_short','gate_any','bar_idx'} & set(df.columns)
    X = df.drop(columns=non_features)

    print(f"\n📂 Apdorojamas failas: {path}")
    print(f"🔎 Unikalūs labeliai faile: {df['label'].unique().tolist()}")
    
    labels = df["label"].astype(str).values

    le_local = LabelEncoder()
    y = le_local.fit_transform(labels)          # 0..K-1 ištisiniai
    class_names = list(le_local.classes_)       # vėliau dekodavimui
    # 🥕 Morkų reward sistema
    rewards = []
    for label in df["label"]:
        if label.startswith("tp_"):
            percent = float(label.split("_")[1]) / 100
            reward = int(percent * 100)  # tp_30 → +30
        elif label.startswith("sl_"):
            percent = float(label.split("_")[1]) / 100
            reward = -int(percent * 100)  # sl_15 → –15
        else:
            reward = 0
        rewards.append(reward)
    rewards = np.array(rewards)

    model = XGBClassifier(
          
            n_estimators=2000,
            learning_rate=0.05,
            # struktūra / reguliarizacija
            max_depth=4,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.8,
            gamma=2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            # multi-class
            objective="multi:softprob",
            eval_metric="mlogloss",
            # num_class --- NEBEREIKIA
            # greitis / stabilumas
            tree_method="hist",      # arba "gpu_hist", jei turi GPU
            max_bin=256,
            n_jobs=1,
            random_state=42,
            verbosity=3              # mažiau triukšmo loguose
    )
    print(f"🧪 X shape: {X.shape}, y shape: {y.shape}")
    print(f"🧪 y value counts: {pd.Series(y).value_counts().to_dict()}")
    # Ensure all classes are represented in y, otherwise XGBoost will infer wrong objective
    
        # 1) Skip jei per mažai klasių
    if len(np.unique(y)) < 2:
        print(f"⛔ Skip: per mažai skirtingų klasių – {path}")
        return


    # --- Feedback bank per kombo ---
    fb_exact = f"feedback_bank_tp{int(tp*100)}_sl{int(sl*100)}_look{lookahead}.csv"
    fb_tp_sl = f"feedback_bank_tp{int(tp*100)}_sl{int(sl*100)}.csv"   # Stage2 versija

    if "bar_idx" in df.columns and os.path.exists(fb_exact):
        fb = pd.read_csv(fb_exact)
    elif "bar_idx" in df.columns and os.path.exists(fb_tp_sl):
        fb = pd.read_csv(fb_tp_sl)
    else:
        df["boost"] = 1.0
    if "boost" not in df.columns:
        df["boost"] = 1.0
    df["boost"] = df["boost"].fillna(1.0)


    # 2) Time-based split (paskutiniai 10% -> validacija)
    split = int(len(X) * 0.9)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y[:split], y[split:]
    rewards_tr, rewards_val = rewards[:split], rewards[split:]
    # Patikrinam klasių įvairovę abiejose dalyse
    if np.unique(y_tr).size < 2 or np.unique(y_val).size < 2:
        print(f"⛔ Skip: {path} – per mažai skirtingų klasių train arba validation rinkinyje.")
        return
    # 3) Class weights

    classes_present = np.unique(y_tr)
    cw_vals = compute_class_weight(class_weight='balanced',
                                   classes=classes_present,
                                   y=y_tr)
    cw_map = {c: w for c, w in zip(classes_present, cw_vals)}

    sw_tr = np.array([cw_map[c] for c in y_tr], dtype=float)
    sw_val = np.array([cw_map.get(c, 1.0) for c in y_val], dtype=float)

    # 4) Pasirinktinai įmaišom "morkas" (max ±5% svorio)
    if rewards_tr.size:
        r_tr = np.abs(rewards_tr).astype(float)
        rng_tr = np.ptp(r_tr)                     # vietoj r_tr.ptp()
        if rng_tr == 0:
            r_tr = np.zeros_like(r_tr)
        else:
            r_tr = (r_tr - r_tr.min()) / (rng_tr + 1e-9)
        sw_tr *= (1.0 + 0.05 * r_tr)

    if rewards_val.size:
        r_val = np.abs(rewards_val).astype(float)
        rng_val = np.ptp(r_val)                   # vietoj r_val.ptp()
        if rng_val == 0:
            r_val = np.zeros_like(r_val)
        else:
            r_val = (r_val - r_val.min()) / (rng_val + 1e-9)
        sw_val *= (1.0 + 0.05 * r_val)
    
    # --- Pritaikom feedback boost'ą svoriams ---
    boosts = df["boost"].to_numpy()
    boosts_tr, boosts_val = boosts[:split], boosts[split:]
    sw_tr *= boosts_tr
    sw_val *= boosts_val

    acc_val = 0.0   
    success = False
    try: 
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sw_val],
            verbose=False
                )
        y_pred_val = model.predict(X_val)
        acc_val = accuracy_score(y_val, y_pred_val)
        print(f"✅ Validation accuracy on {path}: {acc_val:.4f}")
        success = True
         # 🥕 Spausdinam bendrą morkų reward sumą
        print(f"🥕 Total morkos (TP+ / SL–): {rewards.sum()}")

    except Exception as e:
        print(f"❌ KLAIDA treniruojant {path}: {e}")
    chunk_number = int(path.split("part")[-1].split(".csv")[0])
    if success==True:
        try:
            black_box_opinion(
                model=model,
                X=X_val,
                y_true=y_val,
                df_original=df.iloc[split:].reset_index(drop=True),
                model_tp=tp,
                model_sl=sl,
                threshold=0.6,  # gali keisti čia
                class_names=class_names,     # <— pridėta
                output_path=f"black_box_opinion_tp{int(tp*100)}_sl{int(sl*100)}_look{lookahead}_part{chunk_number}.csv"
            )
        except Exception as e:
            print(f"❌ KLAIDA black_box_opinion {path}: {e}")
    # ✅ Checkpoint kas 10 chunk arba jei labai aukštas tikslumas
    chunk_number = int(path.split("part")[-1].split(".csv")[0])


    os.makedirs("checkpoints", exist_ok=True)
    key = f"tp{int(tp*100)}_sl{int(sl*100)}"
    best_file = f"checkpoints/best_metric_{key}.txt"
    best = -1.0
    if os.path.exists(best_file):
        with open(best_file, "r") as f:
            best = float((f.read() or "-1").strip())

    
    if acc_val > best + 0.005:
        checkpoint_path = f"checkpoints/model_part{chunk_number}.joblib"
        os.makedirs("checkpoints", exist_ok=True)
        dump(model, checkpoint_path)
        with open(best_file, "w") as f:
            f.write(f"{acc_val:.6f}")

        # Build žmogystai in memory only
        df_zmog = create_readable_summary(
            model,
            feature_names=X.columns.tolist(),
            output_path=None,                 # no file path because we won't save it
            model_tp=tp,
            model_sl=sl,
            model_lookahead=lookahead,
            label_file_path=path,
            return_df=True,
            save=False
        )

        if CREATE_MASTERY:
            mastery_path = f"mastery_tp{int(tp*100)}_sl{int(sl*100)}_look{lookahead}_part{chunk_number}.csv"
            generate_mastery(
                zmogystai=df_zmog,           # pass DF, not a file
                chunk_folder="labeliai",
                model_tp=tp,
                model_sl=sl,
                output_path=mastery_path
            )
        print(f"💾 Checkpoint saved: {checkpoint_path}")


    if success and os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Ištrintas: {path}")
    print(f"Trained on {path} and deleted it.")
    return True
##### AI TRAINING ON FILES ##############################################################################################################



















##### VALID STEPS CALS. ################################################################################################################
def count_valid_idx(df, lookahead):
    spike = ((df["spike_x6"].to_numpy() > 0) | (df["spike_x10"].to_numpy() > 0)).astype(np.uint8)
    ban_anchor = np.convolve(spike, np.ones(11, dtype=np.uint8), mode="full")[:len(spike)] > 0
    tail_cut = np.zeros(len(spike), dtype=bool); tail_cut[-lookahead:] = True
    anchor_mask = ~(ban_anchor | tail_cut)

    df_tmp = compute_aea_signals(df)  
    long_mask  = df_tmp["entry_long"].to_numpy()
    short_mask = df_tmp["entry_short"].to_numpy()
    acc_or_distr = (df_tmp["accumulation_on"] | df_tmp["distribution_on"]).to_numpy()

    signal_mask = long_mask | short_mask | acc_or_distr 
    anchor_mask &= signal_mask
    return np.flatnonzero(anchor_mask).size
##### VALID STEPS CALS. ################################################################################################################
















###### PROGRESS BAR UPDATER ##############################################################################################################
def tqdm_updater(q, total_steps):
    completed = 0
    start_time = time.time()
    total_fmt = f"{total_steps:,} ({total_steps / 1_000_000:.2f}M)"
    print(f"💻 Label Generation | total: {total_fmt} rows")  # ← ← ← ŠITA EILUTĖ
    with tqdm(total=total_steps, desc="💻 Label Generation", dynamic_ncols=True, unit="rows") as pbar:
        while True:
            item = q.get()
            if item == "DONE":
                break
            # item = kiek eilučių sugeneruota
            completed += item
            pbar.update(item)

            if completed % 100000 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = total_steps - completed
                eta_sec = remaining / rate if rate > 0 else 0
                eta_min = int(eta_sec // 60)
                eta_sec = int(eta_sec % 60)
                pbar.set_postfix_str(f"ETA: {eta_min}m {eta_sec}s")
###### PROGRESS BAR UPDATER ##############################################################################################################


















###### GENERATE DATA IN BATCHES AND TRAIN SOMETIMES ##############################################################################################################
def generate_and_train_in_batches(tp, sl, lookahead, df, q):
    global progress_bar
    rows = []
    train_queue = []         # čia kaupiam kelią į kiekvieną 100k chunk'ą (traininsim po generacijos)
    chunk_paths = []  # ← ← ← ŠITAS EILUTĖ BŪTINA
    print("valid_idx =", count_valid_idx(df, lookahead))
    
    entry_features = ['rsi','stoch_rsi','cmo','roc','adx','bodysize','spike_x3','spike_x6','spike_x10',
    'cross_up_extended','cross_dn_extended','delta_volume','delta_count',
    'delta_vol_sum_5','delta_cnt_sum_5','delta_vol_mean_5','delta_cnt_mean_5',
    'buyers_vs_sellers_vol_ratio_5','buyers_vs_sellers_cnt_ratio_5',
    'delta_vol_sum_15','delta_cnt_sum_15','delta_vol_mean_15','delta_cnt_mean_15',
    'buyers_vs_sellers_vol_ratio_15','buyers_vs_sellers_cnt_ratio_15',
    'delta_vol_sum_18','delta_cnt_sum_18','delta_vol_mean_18','delta_cnt_mean_18',
    'buyers_vs_sellers_vol_ratio_18','buyers_vs_sellers_cnt_ratio_18',]

    total_steps = len(df) - lookahead
    chunk_id = 0
    MAX_RAM_GB = 12  # tavo riba
    file_prefix = f"tp{tp}_sl{sl}_look{lookahead}"

    entry_values = {f: df[f].to_numpy() for f in entry_features}
    highs = df[high_column].to_numpy()
    lows  = df[low_column].to_numpy()

    # spike žvakės: x6 arba x10
    spike = ((df["spike_x6"].to_numpy() > 0) | (df["spike_x10"].to_numpy() > 0)).astype(np.uint8)

    # ban inkaro taškams: spike + 10 po jos (1 + 10 = 11)
    kernel = np.ones(11, dtype=np.uint8)
    ban_anchor = np.convolve(spike, kernel, mode="full")[:len(spike)] > 0

    # paskutinės lookahead žvakės negali būti inkaro taškai (nėra pilnos ateities)
    tail_cut = np.zeros(len(spike), dtype=bool)
    tail_cut[-lookahead:] = True

    # kur GALIMA kurti labelius (inkaro indeksai)
    df = compute_aea_signals(df)
    print_aea_stats(df, tag=f"train tp={tp} sl={sl} look={lookahead}")

    long_mask = df["entry_long"].to_numpy()
    short_mask = df["entry_short"].to_numpy()

    signal_mask = long_mask | short_mask
    anchor_mask = ~(ban_anchor | tail_cut) & signal_mask

    # side lock: jei neturim triggerio, remiamės į acc/distribution bias
    side_lock_long  = long_mask  | ((df["accumulation_on"] & ~df["distribution_on"]).to_numpy())
    side_lock_short = short_mask | ((df["distribution_on"] & ~df["accumulation_on"]).to_numpy())

    valid_idx = np.flatnonzero(anchor_mask)
    total_steps = valid_idx.size

    # talpyklos rezultatams
    label = [[] for _ in range(total_steps)]
    time_to_hit = [[] for _ in range(total_steps)]

    batch = 0
    for k, i in enumerate(valid_idx):     # <- einam tik per leidžiamus inkaro indeksus
        batch += 1
        if batch >= 1024:
            q.put(batch)
            batch = 0

        high_price = highs[i]
        low_price = lows[i]
        entry_price = (low_price + high_price) / 2
        balance = initial_balance 
        used_balance = min(balance, max_used_balance)
        position_value = used_balance * leverage

        fee_total = position_value * fee * 2

        # === LONG pozicija ===
        target_net_loss_long = used_balance * sl
        required_gross_loss_long = target_net_loss_long + fee_total
        price_move_sl_long = required_gross_loss_long / position_value
        stop_loss_price_long = entry_price * (1 - price_move_sl_long)

        target_net_pnl_long = used_balance * tp
        required_gross_pnl_long = target_net_pnl_long + fee_total
        price_move_tp_long = required_gross_pnl_long / position_value
        take_profit_price_long = entry_price * (1 + price_move_tp_long)

        # === SHORT pozicija ===
        target_net_loss_short = used_balance * sl
        required_gross_loss_short = target_net_loss_short + fee_total
        price_move_sl_short = required_gross_loss_short / position_value
        stop_loss_price_short = entry_price * (1 + price_move_sl_short)

        target_net_pnl_short = used_balance * tp
        required_gross_pnl_short = target_net_pnl_short + fee_total
        price_move_tp_short = required_gross_pnl_short / position_value
        take_profit_price_short = entry_price * (1 - price_move_tp_short)

        # Hit tracking
        long_tp_hit = None
        long_sl_hit = None
        short_tp_hit = None
        short_sl_hit = None
        
        for j in range(1, lookahead + 1):
            idx = i + j
            if idx >= len(highs):
                break
        
            high = highs[idx]
            low = lows[idx]

            if long_tp_hit is None and high >= take_profit_price_long:
                long_tp_hit = j
            if long_sl_hit is None and low <= stop_loss_price_long:
                long_sl_hit = j
            if short_tp_hit is None and low <= take_profit_price_short:
                short_tp_hit = j
            if short_sl_hit is None and high >= stop_loss_price_short:
                short_sl_hit = j

            if (long_tp_hit is not None or long_sl_hit is not None) and (short_tp_hit is not None or short_sl_hit is not None):
                break


        label[k] = []
        time_to_hit[k] = []

        

        #debugging labels, palik jei kada nors prireiks
        #print(f"[{i}] long_tp_hit: {long_tp_hit}, long_sl_hit: {long_sl_hit}, short_sl_hit: {short_sl_hit}, short_tp_hit: {short_tp_hit}") #debug

        

        #cia parodyta isvedimai labels greiciu ir t.t. kad ateityje galetum is karto debuggint ar pasiziuret ar iszvelgt daugiau sudu, palik
        """[913] long_tp_hit: None, long_sl_hit: 7, short_sl_hit: None, short_tp_hit: 20                                                              
        [859] long_tp_hit: None, long_sl_hit: 75, short_sl_hit: 6, short_tp_hit: 75         
        [1071] long_tp_hit: None, long_sl_hit: 3, short_sl_hit: 1, short_tp_hit: None
        [914] long_tp_hit: None, long_sl_hit: 7, short_sl_hit: None, short_tp_hit: 19       
        [1072] long_tp_hit: None, long_sl_hit: 1, short_sl_hit: None, short_tp_hit: 3
        [860] long_tp_hit: 39, long_sl_hit: None, short_sl_hit: 5, short_tp_hit: None
        [802] long_tp_hit: None, long_sl_hit: 132, short_sl_hit: None, short_tp_hit: 77 
        [915] long_tp_hit: None, long_sl_hit: 6, short_sl_hit: None, short_tp_hit: 18
        [1073] long_tp_hit: None, long_sl_hit: 1, short_sl_hit: None, short_tp_hit: 2
        [861] long_tp_hit: None, long_sl_hit: 18, short_sl_hit: 5, short_tp_hit: None
        [916] long_tp_hit: None, long_sl_hit: 7, short_sl_hit: None, short_tp_hit: 18
        [1074] long_tp_hit: None, long_sl_hit: 1, short_sl_hit: None, short_tp_hit: 13
        [803] long_tp_hit: None, long_sl_hit: 131, short_sl_hit: None, short_tp_hit: 76
        [917] long_tp_hit: None, long_sl_hit: 6, short_sl_hit: None, short_tp_hit: 17
        [862] long_tp_hit: None, long_sl_hit: 72, short_sl_hit: 3, short_tp_hit: 72
        [1075] long_tp_hit: None, long_sl_hit: 1, short_sl_hit: 1, short_tp_hit: None"""



        # === Long pozicija === 

        #ieinam tik jei long_tp_hit ir long_sl_hit yra None arba long_tp_hit greitesnis
        if long_tp_hit is not None and (long_sl_hit is None or long_tp_hit <= long_sl_hit):
            label[k].append(f"tp_{int(tp * 100)}_long")
            time_to_hit[k].append(long_tp_hit)

        # ieinam tik jei long_sl_hit ir long_tp_hit yra None arba long_sl_hit greitesnis
        if long_sl_hit is not None and (long_tp_hit is None or long_sl_hit <= long_tp_hit):
            label[k].append(f"sl_{int(sl * 100)}_long")
            time_to_hit[k].append(long_sl_hit)

        # === Short pozicija ===

        # ieinam tik jei short_tp_hit ir short_sl_hit yra None arba short_tp_hit greitesnis
        if short_tp_hit is not None and (short_sl_hit is None or short_tp_hit <= short_sl_hit):
            label[k].append(f"tp_{int(tp * 100)}_short")
            time_to_hit[k].append(short_tp_hit)

        # ieinam tik jei short_sl_hit ir short_tp_hit yra None arba short_sl_hit greitesnis
        if short_sl_hit is not None and (short_tp_hit is None or short_sl_hit <= short_tp_hit):
            label[k].append(f"sl_{int(sl * 100)}_short")
            time_to_hit[k].append(short_sl_hit)



        #[803] long_tp_hit: None, long_sl_hit: 131, short_sl_hit: None, short_tp_hit: 76
        # Išrenkam pirmą long ir pirmą short hitą

        #logika turi veikti taip: pirmai pagal greiti isrusiuijam in tuples t.y. in long and in short is kiekvienio istraukiam po
        #-> po 1 greiciausia, ir tada akadangi 1 label per 1 candle per prioritizima t.y. (tp/tp) ar (sl/sl) ar (tp/sl) ar (sl/tp) (/tp) (/sl)
        #  ir tsg (/), 
        # ka imti reiktu

        #sita hujova dalis iki debilizmas print continue del greicio tarp short long o ne tp sl prioritizavimo
        long_candidates = [(lbl, t) for lbl, t in zip(label[k], time_to_hit[k]) if "long" in lbl]
        short_candidates = [(lbl, t) for lbl, t in zip(label[k], time_to_hit[k]) if "short" in lbl]
        # nauja:
        if side_lock_long[i] and not side_lock_short[i]:
            short_candidates = []
        elif side_lock_short[i] and not side_lock_long[i]:
            long_candidates = []
        long_hit = min(long_candidates, key=lambda x: x[1]) if long_candidates else None
        short_hit = min(short_candidates, key=lambda x: x[1]) if short_candidates else None


        # Vienas iš jų neegzistuoja
        if long_hit is not None and short_hit is None:
            chosen_label, chosen_time = long_hit
            # continue nereikia
        elif short_hit is not None and long_hit is None:
            chosen_label, chosen_time = short_hit
            # continue nereikia
        elif long_hit is not None and short_hit is not None:
            # Abu egzistuoja – taikom TP prioriteto logiką
            is_tp_long = "tp_" in long_hit[0]
            is_tp_short = "tp_" in short_hit[0]

            if is_tp_long and is_tp_short:
                # abu TP – imti greitesnį
                chosen_label, chosen_time = long_hit if long_hit[1] <= short_hit[1] else short_hit
            elif is_tp_long and not is_tp_short:
                chosen_label, chosen_time = long_hit  # TP > SL
            elif is_tp_short and not is_tp_long:
                chosen_label, chosen_time = short_hit  # TP > SL
            else:
                # abu SL – imti greitesnį
                chosen_label, chosen_time = long_hit if long_hit[1] <= short_hit[1] else short_hit
        else:
            #print("debilizmas") #jis buna, nes failo data galas ir nebera kur kilt nei kur leistis tai pos.==0, nesvarbu kad lookahead nors ir 10k
            #print(f"[{i}] long_candidates: {long_candidates}, short_candidates: {short_candidates}") #for debugs
            #print(f"[{i}] long_tp_hit: {long_tp_hit}, long_sl_hit: {long_sl_hit}, short_sl_hit: {short_sl_hit}, short_tp_hit: {short_tp_hit}") #for debugs
            continue

        row = {
            "label": chosen_label,
            "tp": tp,
            "sl": sl,
            "lookahead": lookahead,
        }

        for feature in entry_features:
            row[f"entry_{feature}"] = entry_values[feature][i]

        row["gate_long"]  = bool(long_mask[i])
        row["gate_short"] = bool(short_mask[i])
        row["gate_any"]   = row["gate_long"] or row["gate_short"]


        row["bar_idx"] = int(i)
        rows.append(row)

        if len(rows) >= 100000:
            df_chunk = pd.DataFrame(rows)
            chunk_name = f"{file_prefix}_part{chunk_id}.csv"
            path = os.path.join(output_folder, chunk_name)
            df_chunk.to_csv(path, index=False)
            rows.clear()            # greitesnė atminties tvarka nei rows = []
            chunk_id += 1
            train_queue.append(path)   # <-- tik pridedam į eilę
            gc.collect()
    # Finalinis flush
    if batch:
        q.put(batch)

    if rows:
        df_chunk = pd.DataFrame(rows)
        chunk_name = f"{file_prefix}_part{chunk_id}.csv"
        path = os.path.join(output_folder, chunk_name)
        df_chunk.to_csv(path, index=False)
        train_queue.append(path)

    # --------- TRAIN PHASE (consumer) ----------
    for path in train_queue:
        print(f"🧠 Mokymas {path} .")
        try:
            ok = train_on_file(path, tp, sl, lookahead)
            if ok:  # sėkmingas mokymas -> iskart Stage2
                r = run_stage2(
                    file_path=OOS_FILE,          # nustatyk viršuje/globaliai
                    tp=tp, sl=sl,
                    initial_balance=300, leverage=30, fee=0.00055
                )
        except Exception as e:
            print(f"❌ train_on_file error {path}: {e}")

    q.put("DONE")  # <- būtinai
    return f"{file_prefix}_done"
###### GENERATE DATA IN BATCHES AND TRAIN SOMETIMES ##############################################################################################################


















#### Atbulinis atsakas ##############################################################################################################
def build_feedback(preds_path="preds_log.csv", trades_path="simuliacijos_outputas.csv",
                out="feedback_bank.csv"):
    if not (os.path.exists(preds_path) and os.path.exists(trades_path)):
        pd.DataFrame({"bar_idx":[], "boost":[]}).to_csv(out, index=False); return

    p = pd.read_csv(preds_path)
    t = pd.read_csv(trades_path)

    m = p.merge(t[["entry_bar_idx","pnl%"]], left_on="bar_idx", right_on="entry_bar_idx", how="left")
    # default neutral
    m["boost"] = 1.0
    # įvykdyti sandoriai → boost pagal PnL
    m.loc[m["pnl%"].notna() & (m["pnl%"] > 0), "boost"] = 1.20   # laimėjimai – švelniai pasvert
    m.loc[m["pnl%"].notna() & (m["pnl%"] <= 0), "boost"] = 0.80  # pralaimėjimai – numazint svorį
    # neįvykdyti (IGNORE) – neutral
    fb = m[["bar_idx","boost"]].drop_duplicates("bar_idx")
    fb.to_csv(out, index=False)
###### Atbulinis atsakas ##############################################################################################################
























##### AI_STAGE2 ################################################################################################################
def run_stage2(file_path: str, tp: float, sl: float,initial_balance: float = 300,leverage: float = 30,fee: float = 0.00055,model_path: str | None = None,threshold: float | None = None) -> dict:
    # === MODELIS ===
    ckpts = sorted(
        glob("checkpoints/model_part*.joblib"),
        key=lambda p: int(p.split("part")[1].split(".")[0])
    )
    model_path = ckpts[-1] if ckpts else "model_part0.joblib"
    model = load(model_path)

    key = f"tp{int(tp*100)}_sl{int(sl*100)}"
    thr_path = f"best_threshold_{key}.json"
    threshold = 0.7
    if os.path.exists(thr_path):
        try:
            threshold = float(json.load(open(thr_path))["best_threshold"])
        except:
            pass
    encoder = load("encoder.joblib")


    

    # Failas
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Stage2 input not found: {file_path}")
    df = pd.read_csv(file_path)


    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    entry_features = ['rsi','stoch_rsi','cmo','roc','adx','bodysize','spike_x3','spike_x6','spike_x10',
    'cross_up_extended','cross_dn_extended','delta_volume','delta_count',
    'delta_vol_sum_5','delta_cnt_sum_5','delta_vol_mean_5','delta_cnt_mean_5',
    'buyers_vs_sellers_vol_ratio_5','buyers_vs_sellers_cnt_ratio_5',
    'delta_vol_sum_15','delta_cnt_sum_15','delta_vol_mean_15','delta_cnt_mean_15',
    'buyers_vs_sellers_vol_ratio_15','buyers_vs_sellers_cnt_ratio_15',
    'delta_vol_sum_18','delta_cnt_sum_18','delta_vol_mean_18','delta_cnt_mean_18',
    'buyers_vs_sellers_vol_ratio_18','buyers_vs_sellers_cnt_ratio_18',]

    # === DUOMENŲ RINKINYS ===
    # 1) Laikas ir rikiavimas
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 2) Patikrinam, kad visi feature'iai yra
    missing = [c for c in entry_features if c not in df.columns]
    if missing:
        raise ValueError(f"Trūksta stulpelių: {missing}")

    # 3) Užtvirtinam skaitinius tipus (OHLC + feature'iai)
    num_cols = ["open", "high", "low", "close", "volume"] + entry_features
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) Inf/NaN -> 0 ir kompaktiškas tipas modelio įėjimui
    df[entry_features] = (
        df[entry_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # (nebūtina, bet gražu) — OHLC į float32
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")


################# Parametru galas #################################################################################################










################# Pradzia stage2 simuliacijos #################################################################################################
    def simulate_detailed(data_rows, balance, expected_direction, winrate, trades, wins, sls_hits, tps_hits, next_signals_hits, peak_balance):
        
        # --- Pre-split į stulpelius (vietoj per-eilutinių tuple unpack) ---
        data = np.asarray(data_rows, dtype=object)
        opens  = data[:, 0].astype(np.float64, copy=False)
        highs  = data[:, 1].astype(np.float64, copy=False)
        lows   = data[:, 2].astype(np.float64, copy=False)
        closes = data[:, 3].astype(np.float64, copy=False)
        signals= data[:, 4]                       # paliekam kaip object ("long"/"short"/None)
        times  = data[:, 5]
        featvecs = data[:, 6]          # ➜ čia bus X_all[i] vektoriai iš #1
        
        entry_feat_snapshot = {}  # kad neužsitemptų į kitą trade
        
        position = None
        entry_price = 0
        exit_price = 0

        tp_long = sl_long = tp_short = sl_short = 0

        timestamp_entry = None
        timestamp_close = None
        price_high_history = []
        price_low_history = []
        peak_balance_list = []
        
        peak_balance_list.append(peak_balance)
        
        # --- Simuliacija ---
        pending_entry = None 
        output_rows=[]

        n = len(data_rows)
        # tqdm su mažesne overhead (dylesnis miniters, mininterval)
        for i in trange(n, desc="💥 Simulating", total=n, dynamic_ncols=True, miniters=20000, mininterval=0.2):            
            
            open_price  = opens[i]
            high_price  = highs[i]
            low_price   = lows[i]
            close_price = closes[i]
            signal      = signals[i]
            timestamp   = times[i]

            pnl = 0
            closed = False
            reason_stop_loss = False
            reason_take_profit = False
            reason_next_signal = False

            pos = position if position else "none"



    ###### TAKE PROFITAS< STOP LOSSAS #############
            if position == "long":
                used_balance = min(balance, 28000)
                position_value = used_balance * leverage 
                fee_total = position_value * fee

                target_net_loss = used_balance * sl
                required_gross_loss = target_net_loss  + fee_total
                price_move_sl = required_gross_loss / position_value
                stop_loss_price = entry_price * (1 - price_move_sl) 

                target_net_pnl = used_balance * tp
                required_gross_pnl = target_net_pnl  + fee_total
                price_move = required_gross_pnl / position_value
                take_profit_price = entry_price * (1 + price_move)
                
                
                price_high_history.append(high_price)
                peak_price = max(price_high_history)

                if low_price <= stop_loss_price and sl!=0: ##stop lossas
                    position_closed = "long"
                    exit_price= stop_loss_price
                    position = None
                    reason_stop_loss = True
                    closed = True
                elif high_price >= take_profit_price and tp!=0: ##take profitas
                    position_closed = "long"
                    exit_price= take_profit_price
                    position = None
                    reason_take_profit = True
                    closed = True

    ###### TAKE PROFITAS< STOP LOSSAS  ##############



    ###### TAKE PROFITAS< STOP LOSSAS ###############
            if position == "short":  #ieis i vidu tik jei 1. Shortas    2. nera pending entry   3. closed=false
                used_balance = min(balance, 28000)
                position_value = used_balance * leverage
                fee_total = position_value * fee 

                target_net_loss = used_balance * sl
                required_gross_loss = target_net_loss + fee_total
                price_move_sl = required_gross_loss / position_value
                stop_loss_price = entry_price * (1 + price_move_sl)

                
                target_net_pnl = used_balance * tp
                required_gross_pnl = target_net_pnl + fee_total
                price_move = required_gross_pnl / position_value
                take_profit_price = entry_price * (1 - price_move)


                #print(f"target_net_loss: {target_net_loss}, required_gross_loss: {required_gross_loss}, fee_total: {fee_total}")
                #print(f"target_net_pnl: {target_net_pnl}, required_gross_pnl: {required_gross_pnl}, fee_total: {fee_total}")
                price_low_history.append(low_price)
                deep_price = min(price_low_history)


                if high_price >= stop_loss_price and sl!=0: ##stop lossas
                    position_closed = "short"
                    exit_price= stop_loss_price
                    position = None
                    reason_stop_loss = True
                    closed = True
                elif low_price <= take_profit_price and tp!=0: ##take profitas
                    position_closed = "short"
                    exit_price= take_profit_price
                    position = None
                    reason_take_profit = True
                    closed = True
    ###### TAKE PROFITAS< STOP LOSSAS  ################

    ######## po LONG ateina SHORT #####################
            if position == "long" and signal=="short" and labas==True: #labas isjungia ta dalyka kad longa gali isjungti short, nebera reason_for_close next_signal
                timestamp_close = timestamp
                position_closed = "long"
                pending_entry = "short"
                exit_price = close_price  #
                reason_next_signal = True #
                closed = True #
                position = None #
    ######## po LONG ateina SHORT ######################



    ######## po SHORT ateina LONG ######################
            if position == "short" and signal=="long" and labas==True: #labas isjungia ta dalyka kad shorta gali isjungti long, nebera reason_for_close next_signal
                timestamp_close = timestamp #
                position_closed = "short" #
                pending_entry = "long" #
                exit_price = close_price #
                reason_next_signal = True #
                closed = True #
                position = None #
    ######## po SHORT ateina LONG #######################



    ######## CLOSED POSITION ############################
            if closed:
                trades+=1
                timestamp_close = timestamp
                price_high_history = []  #isvalom search history : D
                price_low_history = []  #isvalom search history : D
                used_balance = min(balance, 28000)
                position_value = used_balance * leverage
                if position_closed == "long":
                    gross_pnl = (exit_price - entry_price) / entry_price * position_value
                    fee_total = position_value * fee  
                    pnl = gross_pnl - fee_total
                    #print(f"fee_total={fee_total:.6f}  gross_pnl={gross_pnl:.6f}  pnl={pnl:.6f}")
                    pnl_percent = pnl / used_balance * 100 if used_balance > 0 else 0

                elif position_closed  == "short":
                    gross_pnl = (entry_price - exit_price) / entry_price * position_value
                    fee_total = position_value * fee  
                    pnl = gross_pnl - fee_total
                    #print(f"fee_total={fee_total:.6f}  gross_pnl={gross_pnl:.6f}  pnl={pnl:.6f}")
                    pnl_percent = pnl / used_balance * 100

                if pnl > 0:
                    wins+=1

                winrate= wins*100 /trades 
                balance= balance + pnl
                peak_balance_list.append(balance)
                peak_balance= max(peak_balance_list)


                if reason_take_profit == True:
                    reason_for_close = "take_profit"
                    tps_hits += 1
                    if position_closed == "long":
                        tp_long += 1
                    else:  
                        tp_short += 1
                elif reason_stop_loss == True:
                    reason_for_close = "stop_loss"
                    sls_hits += 1
                    if position_closed == "long":
                        sl_long += 1
                    else: 
                        sl_short += 1
                elif reason_next_signal == True:
                    reason_for_close = "next_signal"
                    next_signals_hits += 1



                row_out = {
                    "timestamp_entry": timestamp_entry,
                    "timestamp_close": timestamp_close,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl$": round(pnl, 6),
                    "pnl%": round(pnl_percent, 2),
                    "winrate": winrate, 
                    "reason_for_close": reason_for_close,
                    "balance": round(balance, 4),
                    "peak_balance": round(peak_balance,4),
                    "trades": trades,
                    "tp's hit": tps_hits,
                    "sl's hit": sls_hits,
                    "next_signals' hit": next_signals_hits,
                    "position": position_closed,
                }
                if entry_feat_snapshot:
                    for k, v in entry_feat_snapshot.items():
                        row_out[f"entry_{k}"] = v

                output_rows.append(row_out)
                entry_feat_snapshot = {}  # reset po uždarymo

                
                if balance <=20:
                    print("❌ Balansas per mazas, stabdome prekyba.")
                    kaput =" balansas per mazas, stabdome prekyba"
                    output_rows.append({
                        "pize": kaput,
                    })
                    break
                continue #continue kad neitume i sekancia iteracija jei closed=true,
                         # kadangi jau uzdarem pozicija ir issaugojom duomenis
    ######## CLOSED POSITION ############################


    ####### < PENDING ENTRY > ###########################
            if position is None and pending_entry is not None: #ieis tik jei 1. NERA pozicijos 2. YRA pending entry
                if pending_entry == "long":
                    position = "long"
                elif pending_entry == "short":
                    position = "short"
                entry_price = (low_price + high_price) / 2  
                timestamp_entry = timestamp
                entry_feat_snapshot = { fname: float(val) 
                            for fname, val in zip(entry_features, featvecs[i]) }
                pending_entry = None  
    ####### < PENDING ENTRY > ###########################






    ######## ZERO POSITIONS ITERUOJANT ###########
            if position is None and pending_entry is None and (signal=="long" or signal=="short"): 
                #ieis tik jei 1. NERA pozicijos 2. NERA pending entry 3. closed=false
                #if expected_direction == signal or expected_direction is None:  
                # # jei laukiam tokio paties signalo kaip ir buvo arba pirmas signalas long then short, short then long
                if signal=="long":
                    pending_entry = "long"  #permetame i sekancia candle per PENDING ENTRY
                    expected_direction ="short"  # laukiam short signalo kaip sekancio vietoje long
                    continue
                elif signal=="short":
                    pending_entry = "short"
                    expected_direction ="long"
                    continue
    ######## ZERO POSITIONS ITERUOJANT ###########
        print(
            f"TP long: {tp_long} | SL long: {sl_long} | "
            f"TP short: {tp_short} | SL short: {sl_short}"
        )
        return pd.DataFrame(output_rows)
################# Pabaiga stage2 simuliacijos #################################################################################################


























############### Signalu generavimas for stage2 #######################################################################################
    def generate_signals():
        current_balance = None
        expected_directionG = None  # pradžioje nėra atidarytos pozicijos
        winrateG = 0
        tradesG = 0
        sls_hitsG = 0
        tps_hitsG = 0
        next_signals_hitsG = 0
        peak_balanceG = initial_balance  # pradinis balansas

        # --- ČIA įrašai numpy masyvų paruošimą ---
        opens  = df["open"].to_numpy(dtype=np.float32)
        highs  = df["high"].to_numpy(dtype=np.float32)
        lows   = df["low"].to_numpy(dtype=np.float32)
        closes = df["close"].to_numpy(dtype=np.float32)
        times  = df["timestamp"].to_numpy()
        X_all  = df[entry_features].to_numpy(dtype=np.float32)
        # -----------------------------------------
        # Formuojam data su modelio prediction

        rows = []
        all_rows = []  # kaups visą originalią informaciją
        all_results = []
        pizda = False     # <--- PRIEŠ for i in ...


        aea = compute_aea_signals(df)
        # Saugus būdas pasiimti boolean stulpelį (jei nėra – False su teisingu index)
        def flag(name: str) -> pd.Series:
            return (aea[name].astype(bool).fillna(False)
                    if name in aea.columns
                    else pd.Series(False, index=aea.index, dtype=bool))

        gate_long_sr  = (flag("entry_long"))
        gate_short_sr = (flag("entry_short"))

        # į numpy bool
        gate_long  = gate_long_sr.to_numpy(dtype=bool)
        gate_short = gate_short_sr.to_numpy(dtype=bool)
        pred_rows = []
        key = f"tp{int(tp*100)}_sl{int(sl*100)}"   # Stage2 nenaudoja lookahead → key tik pagal tp/sl
        preds_path = f"preds_log_{key}.csv"


        
        for i in tqdm(range(len(df)), desc="✅ Predicting", dynamic_ncols=True):
            if i < 6:
                signal = None
            else:
                X = X_all[i].reshape(1, -1)
                proba = model.predict_proba(X)[0]
                confidence = np.max(proba)
                label = encoder.inverse_transform([np.argmax(proba)])[0]
                #for cls, p in zip(encoder.classes_, proba):
                    #print(f"{cls}: {p:.4f}")

                #print(label)
                #print(confidence, label)
                #print(type(model.classes_[0]))
                #print(model.classes_)
                # Visi signalai
                #print(f"Signalas {i}: {label} su pasitikėjimu {confidence:.2f}")
                if confidence >= threshold:
                    if label.endswith("_long") and gate_long[i]:
                        signal = "long"
                    elif label.endswith("_short") and gate_short[i]:
                        signal = "short"
                    else:
                        signal = None
                else:
                    signal = None
        
                #print(signal, confidence, label)


                pred_side = "long" if (signal=="long") else ("short" if (signal=="short") else "none")
                pred_rows.append({
                    "bar_idx": i,
                    "confidence": float(confidence),
                    "pred_side": pred_side,
                    "gate_long": bool(gate_long[i]),
                    "gate_short": bool(gate_short[i])
                })

            row = (
            opens[i],
            highs[i],
            lows[i],
            closes[i],
            signal,
            times[i],
            X_all[i]
            )
            rows.append(row)
            #print(row)
            all_rows.append(row)  # <- čia svarbiausia
            #print(row)
            #print(df)

            # 💥 Simuliacija kas 20k
            if i > 0 and i % 20000 == 0:
                chunk = np.array(rows, dtype=object)
                print(f"📊 Simuliuojam chunk #{i/20000} | signalų: {sum(r[4] is not None for r in chunk)}")
                if current_balance is None:
                    result = simulate_detailed(chunk, balance=initial_balance, expected_direction=None, winrate=0, trades=0, wins=0, sls_hits=0, tps_hits=0, next_signals_hits=0, peak_balance=initial_balance)
                    print(f"💰 Starting simulation with balance: {initial_balance}")
                else:
                    result = simulate_detailed(chunk, balance=current_balance, winrate=winrateG, expected_direction=expected_directionG, trades=tradesG, wins=tps_hitsG, sls_hits=sls_hitsG, tps_hits=tps_hitsG, next_signals_hits=next_signals_hitsG, peak_balance=peak_balanceG)
                    print(f"➡️  Next chunk starts with balance: {round(current_balance, 2)}")

                if not result.empty and "balance" in result.columns:
                    current_balance = result["balance"].iloc[-1]
                    winrateG = result["winrate"].iloc[-1]
                    laikinas = result["position"].iloc[-1]
                    if laikinas == "long":
                        expected_directionG = "short"
                    elif laikinas == "short":
                        expected_directionG = "long"
                    else:
                        expected_directionG = None
                    tradesG = result["trades"].iloc[-1]
                    sls_hitsG = result["sl's hit"].iloc[-1]
                    tps_hitsG = result["tp's hit"].iloc[-1]
                    next_signals_hitsG = result["next_signals' hit"].iloc[-1]
                    peak_balanceG = result["peak_balance"].iloc[-1]
                    all_results.append(result)
                else:
                    print(f"⚠️ Empty result at chunk #{i//20000}. No position closed.")
                

                
                
                pizda = False
                if "pize" in result.columns and result["pize"].notna().any():
                    print("🛑 Sustabdom – balansas žemiau 20.")
                    pizda = True
                    break
                rows = []

        # 💥 Final chunk (jei liko mažiau nei 10k)
        if rows and pizda == False:
            chunk = np.array(rows, dtype=object)
            print(f"📊 Simuliuojam final chunk | signalų: {sum(r[4] is not None for r in chunk)}")
            if current_balance is None:
                result = simulate_detailed(chunk, balance=initial_balance, expected_direction=None, winrate=0, trades=0, wins=0, sls_hits=0, tps_hits=0, next_signals_hits=0, peak_balance=initial_balance)
                print(f"💰 Starting simulation with balance: {initial_balance}")
            else:
                result = simulate_detailed(chunk, balance=current_balance, winrate=winrateG, expected_direction=expected_directionG, trades=tradesG, wins=tps_hitsG, sls_hits=sls_hitsG, tps_hits=tps_hitsG, next_signals_hits=next_signals_hitsG, peak_balance=peak_balanceG)
                print(f"➡️  Next chunk starts with balance: {round(current_balance, 2)}")

                if not result.empty and "balance" in result.columns:
                    current_balance = result["balance"].iloc[-1]
                    winrateG = result["winrate"].iloc[-1]
                    laikinas = result["position"].iloc[-1]
                    if laikinas == "long":
                        expected_directionG = "short"
                    elif laikinas == "short":
                        expected_directionG = "long"
                    else:
                        expected_directionG = None
                    tradesG = result["trades"].iloc[-1]
                    sls_hitsG = result["sl's hit"].iloc[-1]
                    tps_hitsG = result["tp's hit"].iloc[-1]
                    next_signals_hitsG = result["next_signals' hit"].iloc[-1]
                    peak_balanceG = result["peak_balance"].iloc[-1]
                    all_results.append(result)
                else:
                    print(f"⚠️ Empty result at chunk #{i//100000}. No position closed.")
                
            if "pize" in result.columns and result["pize"].notna().any():
                print("🛑 Sustabdom – balansas žemiau 20.")
        pd.DataFrame(pred_rows).to_csv(preds_path, index=False)
        return pd.concat(all_results, ignore_index=True)


    df_trades = generate_signals()
    key = f"tp{int(tp*100)}_sl{int(sl*100)}"
    out_trades = f"simuliacijos_outputas_{key}.csv"
    # Map entry timestamp -> bar index (global df is the same one you predicted on)
    ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}

    # Create entry_bar_idx via timestamp_entry
    if not df_trades.empty:
        df_trades["entry_bar_idx"] = df_trades["timestamp_entry"].map(ts_to_idx).astype("Int64")
    else:
        # Make sure required columns exist to avoid KeyError in build_feedback
        df_trades["entry_bar_idx"] = pd.Series(dtype="Int64")
        df_trades["pnl%"] = pd.Series(dtype="float64")

    df_trades.to_csv(out_trades, index=False)

    last_valid = df_trades.dropna(subset=["winrate", "trades"])
    print("Last row winrate:", float(last_valid["winrate"].iloc[-1]) if not last_valid.empty else "N/A")
    print("Last row trades:",  int(last_valid["trades"].iloc[-1])   if not last_valid.empty else "N/A")


    # per-kombo feedback (be lookahead)
    build_feedback(
        preds_path=f"preds_log_{key}.csv",
        trades_path=out_trades,
        out=f"feedback_bank_{key}.csv"
    )
    return {
    "preds":      f"preds_log_{key}.csv",
    "trades":     f"simuliacijos_outputas_{key}.csv",
    "feedback":   f"feedback_bank_{key}.csv",
    "threshold":  threshold,
    "model_path": model_path,
    }
############### Signalu generavimas for stage2 end #######################################################################################
##### AI_STAGE2 ################################################################################################################






















##### MAIN EXECUTION ################################################################################################################
if __name__ == "__main__":
    # Užtikrina multiprocessing veikimą Windows aplinkoje
    freeze_support()





    # Sukuriamos visos galimos klasės modelio treniravimui (tp/sl long ir short variantai)
    ALL_CLASSES = []
    for tp in tp_values:
        tp_label = f"tp_{round(tp * 100)}"
        ALL_CLASSES.append(f"{tp_label}_long")
        ALL_CLASSES.append(f"{tp_label}_short")

    for sl in sl_values:
        sl_label = f"sl_{round(sl * 100)}"
        ALL_CLASSES.append(f"{sl_label}_long")
        ALL_CLASSES.append(f"{sl_label}_short")

    # Užkoduojamos visos klasės į skaitinius labelius
    encoder.fit(ALL_CLASSES)
    dump(encoder, "encoder.joblib")  # ← ← ← ŠITA BŪTINA
    # Visos TP/SL/lookahead kombinacijos (kiekviena bus atskiras darbas)
    combinations = [
        (tp, sl, lookahead)
        for tp in tp_values
        for sl in sl_values
        for lookahead in lookahead_values
    ]

    # Apskaičiuojamas bendras žvakių kiekis visoms kombinacijoms (progress bar tikslui)
    num_combo   = len(tp_values) * len(sl_values)
    total_steps = sum(count_valid_idx(df, l) * num_combo for l in lookahead_values)

    manager = Manager()
    q = manager.Queue()
    t = threading.Thread(target=tqdm_updater, args=(q, total_steps), daemon=True)
    t.start()

    # Paleidžiam paralelinį visų kombinacijų apdorojimą (paleidzia visu darbus per joblib.parallel, kiekvienas darbas iskviecia generate_and_train_in_batches o butent generate_and_train_in_batches fukncija ->
    # -> generuoja labels rows, chunkina, treniruoja modeli, saugo zmogystai ir mastery failus. (kaip run_bot() iceberg.py kode)
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=batch_size)(delayed(generate_and_train_in_batches)(tp, sl, lookahead, df.copy(deep=False), q) for tp, sl, lookahead in combinations)

    # Laukiam, kol progress bar gija baigs darbą
    q.put("DONE")  # ← ← ← VIENINTELIS vietoje visam tqdm uždarymui
    t.join()
    # === Stage2 per combo (sequential, be lookahead) ===
    
    print("Sukurti failai:", results)

    SHOW_BOX_PREVIEW = False
    if SHOW_BOX_PREVIEW:
        df_oos = pd.read_csv(OOS_FILE)           # tas pats failas kaip Stage2
        tr = pd.read_csv("simuliacijos_outputas_tp10_sl10.csv")
        plot_box_steps(df_oos, last="all", trades_df=tr)
##### MAIN EXECUTION ################################################################################################################




