#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight Schedule Optimization


Features:
- NLP + Dual-ML prediction capabilities
- Schedule optimization algorithms
- Runway capacity analysis
- Cascading delay impact modeling
- Peak-time slot recommendations
- Interactive schedule visualization

Save as: app.py
Run: streamlit run app.py -- --features hflights.csv
"""

import argparse, os, re, math, time
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
import itertools

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# sentence-transformers
from sentence_transformers import SentenceTransformer, util


# ------------------------------
# Utilities
# ------------------------------
def parse_time_hhmm(x):
    if pd.isna(x):
        return (np.nan, np.nan)
    try:
        s = str(int(x)).zfill(4)
        hh = int(s[:-2])
        mm = int(s[-2:])
        if 0 <= hh < 24 and 0 <= mm < 60:
            return hh, mm
    except Exception:
        pass
    return (np.nan, np.nan)


def time_to_minutes(hour, minute):
    return hour * 60 + minute


def minutes_to_time(minutes):
    hour = int(minutes // 60) % 24
    minute = int(minutes % 60)
    return hour, minute


# ------------------------------
# Airport Configuration (ADAPTED FOR HFLIGHTS DATA)
# ------------------------------
AIRPORT_CONFIG = {
    "IAH": {  # Houston George Bush Intercontinental
        "runways": 5,
        "capacity_per_hour": 70,  # Combined capacity per runway
        "peak_hours": [7, 8, 9, 16, 17, 18, 19],
        "weather_delays": 0.10,
        "ground_congestion": 0.7
    },
    "DFW": {  # Dallas/Fort Worth
        "runways": 7,
        "capacity_per_hour": 90,
        "peak_hours": [7, 8, 9, 17, 18, 19, 20],
        "weather_delays": 0.12,
        "ground_congestion": 0.8
    },
    "HOU": {  # Houston Hobby
        "runways": 4,
        "capacity_per_hour": 40,
        "peak_hours": [7, 8, 16, 17, 18],
        "weather_delays": 0.09,
        "ground_congestion": 0.5
    }
}

# ------------------------------
# Column Descriptions (ADAPTED FOR HFLIGHTS DATA)
# ------------------------------
COLUMN_DESCRIPTIONS = {
    "ArrDelay": "difference between scheduled and actual arrival time in minutes (negative for early)",
    "DepDelay": "difference between scheduled and actual departure time in minutes",
    "DepTime": "actual departure time in HHMM format",
    "ArrTime": "actual arrival time in HHMM format",
    "UniqueCarrier": "airline carrier code",
    "FlightNum": "unique flight number",
    "TailNum": "aircraft tail number identifier",
    "ActualElapsedTime": "total elapsed time in minutes",
    "AirTime": "time spent airborne in minutes",
    "Origin": "origin airport code",
    "Dest": "destination airport code",
    "Distance": "distance flown (miles)",
    "TaxiIn": "taxi-in time in minutes",
    "TaxiOut": "taxi-out time in minutes",
    "Cancelled": "1 if flight cancelled else 0",
    "CancellationCode": "reason code for cancellation (A=carrier, B=weather, C=NAS, D=security)",
    "Diverted": "1 if flight diverted else 0",
    "Month": "month number 1..12",
    "DayOfWeek": "day of week as integer (1=Mon, 7=Sun)",
    "DayofMonth": "day of the month 1..31"
}

OPERATIONS = {
    "mean": "average / mean",
    "median": "median",
    "max": "maximum",
    "min": "minimum",
    "count": "count / how many",
    "top": "top N / most frequent",
    "distribution": "distribution / histogram",
    "percent": "percentage (for binary columns)",
    "trend": "time trend (by hour or by day)",
    "optimize": "find optimal schedule slots",
    "capacity": "runway capacity analysis",
    "congestion": "congestion analysis"
}


# ------------------------------
# SBERT and TF-IDF caching
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_sbert(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)


@st.cache_data(show_spinner=False)
def precompute_embeddings(_model):
    col_keys = list(COLUMN_DESCRIPTIONS.keys())
    col_texts = [f"{k}: {v}" for k, v in COLUMN_DESCRIPTIONS.items()]
    col_emb = _model.encode(col_texts, convert_to_tensor=True)

    op_keys = list(OPERATIONS.keys())
    op_texts = [f"{k}: {v}" for k, v in OPERATIONS.items()]
    op_emb = _model.encode(op_texts, convert_to_tensor=True)

    return col_keys, col_emb, op_keys, op_emb


# ------------------------------
# Query parsing helpers
# ------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
     "December"], start=1)}
WEEKDAYS = {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7}


def detect_optimization_intent(q: str):
    optimization_keywords = [
        "optimize", "best time", "optimal slot", "schedule", "avoid delay",
        "less congested", "peak hours", "busy time", "runway capacity",
        "recommend", "suggest", "when should", "best departure"
    ]
    return any(keyword in q.lower() for keyword in optimization_keywords)


def detect_month(q: str):
    qq = q.lower()
    for k, v in MONTHS.items():
        if k in qq: return v
    m = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', qq)
    if m:
        short = m.group(1)
        return \
            {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
             'nov': 11,
             'dec': 12}[short]
    return None


def detect_dayofweek(q: str):
    qq = q.lower()
    for k, v in WEEKDAYS.items():
        if k in qq: return v
    if "weekday" in qq: return "weekday"
    if "weekend" in qq: return "weekend"
    return None


def detect_hour_range(q: str):
    qq = q.lower()
    m = re.search(r'(\d{1,2})(?:[:h]?| ?(?:am|pm)?)\s*(?:-|to|and|until)\s*(\d{1,2})(?:[:h]?| ?(?:am|pm)?)', qq)
    if m:
        a = int(m.group(1));
        b = int(m.group(2));
        return min(a, b), max(a, b)
    m2 = re.search(r'\b(at|around|@ )?(\d{1,2})(?:am|pm)?\b', qq)
    if m2:
        return int(m2.group(2)), int(m2.group(2))
    return None


def detect_origin_dest_carrier(q: str, df: pd.DataFrame):
    qq = q.lower()
    origin = None
    dest = None
    carrier = None

    # --- NEW: City to Airport Code Mapping ---
    # This dictionary allows the app to understand city names.
    city_to_code = {
        "houston": "IAH",  # Default Houston to the main international airport
        "dallas": "DFW",
        "hobby": "HOU"     # Specifically for Houston Hobby Airport
    }

    # --- Step 1: Check for City Names First ---
    for city, code in city_to_code.items():
        if city in qq:
            if re.search(r'from\s+' + re.escape(city), qq):
                origin = code
            elif re.search(r'to\s+' + re.escape(city), qq):
                dest = code
            # If "from" or "to" is not specified, assign to origin if it's not taken
            elif origin is None:
                origin = code

    # --- Step 2: Check for Airport Codes (Preserves original functionality) ---
    # This ensures that queries with "IAH", "DFW", etc., still work.
    airport_codes = list(pd.concat([df['Origin'], df['Dest']]).unique())
    for code in airport_codes:
        if code.lower() in qq:
            if re.search(r'from\s+' + re.escape(code.lower()), qq):
                origin = code # Overwrite city match if specific code is used
            elif re.search(r'to\s+' + re.escape(code.lower()), qq):
                dest = code
            elif origin is None:
                origin = code

    # --- Carrier detection (unchanged) ---
    if "UniqueCarrier" in df.columns:
        for c in df["UniqueCarrier"].dropna().astype(str).unique().tolist():
            if c.lower() in qq:
                carrier = c
                break

    return origin, dest, carrier


# ------------------------------
# Schedule Optimization Engine
# ------------------------------
class ScheduleOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        if "DepHour" not in self.df.columns and "DepTime" in self.df.columns:
            dh, dm = zip(*self.df["DepTime"].apply(parse_time_hhmm))
            self.df["DepHour"] = pd.Series(dh, index=self.df.index)
        self._calculate_congestion_metrics()
        self._calculate_delay_patterns()

    def _calculate_congestion_metrics(self):
        congestion_data = []
        airports_to_analyze = list(AIRPORT_CONFIG.keys())

        for airport in airports_to_analyze:
            if "Origin" in self.df.columns and "Dest" in self.df.columns and \
                    (airport in self.df["Origin"].values or airport in self.df["Dest"].values):

                airport_flights = self.df[(self.df["Origin"] == airport) | (self.df["Dest"] == airport)]
                hourly_counts = airport_flights.groupby("DepHour").size()
                config = AIRPORT_CONFIG.get(airport)
                max_capacity = config["runways"] * config["capacity_per_hour"]

                for hour in range(24):
                    count = hourly_counts.get(hour, 0)
                    utilization = count / max_capacity if max_capacity > 0 else 0
                    congestion_data.append({
                        "Airport": airport, "Hour": hour, "FlightCount": count,
                        "Utilization": min(utilization, 1.0),
                        "CongestionLevel": self._get_congestion_level(utilization, config, hour)
                    })

        if congestion_data:
            self.congestion_df = pd.DataFrame(congestion_data)
        else:
            self.congestion_df = pd.DataFrame(
                columns=["Airport", "Hour", "FlightCount", "Utilization", "CongestionLevel"])

    def _get_congestion_level(self, utilization, config, hour):
        base_congestion = utilization
        if hour in config["peak_hours"]: base_congestion *= 1.3
        base_congestion *= (1 + config["weather_delays"])
        base_congestion *= config["ground_congestion"]
        return min(base_congestion, 1.0)

    def _calculate_delay_patterns(self):
        delay_patterns = []
        delay_col = "ArrDelay" if "ArrDelay" in self.df.columns else "DepDelay"

        for airport in AIRPORT_CONFIG.keys():
            airport_data = self.df[self.df["Origin"] == airport]
            if len(airport_data) == 0 or "DepHour" not in airport_data.columns:
                continue

            hourly_delays = airport_data.groupby("DepHour")[delay_col].agg(["mean", "median", "std", "count"]).fillna(0)
            for hour in range(24):
                if hour in hourly_delays.index:
                    stats = hourly_delays.loc[hour]
                    delay_patterns.append({
                        "Airport": airport, "Hour": hour, "AvgDelay": stats["mean"],
                        "MedianDelay": stats["median"], "DelayStd": stats["std"],
                        "FlightCount": stats["count"], "DelayRisk": self._calculate_delay_risk(stats)
                    })
        self.delay_patterns_df = pd.DataFrame(delay_patterns)

    def _calculate_delay_risk(self, stats):
        if stats["count"] == 0: return 0
        avg_delay = max(0, stats["mean"])
        std_delay = stats["std"]
        risk = (avg_delay + std_delay) / 100
        return min(risk, 1.0)

    def find_optimal_slots(self, airport: str, target_hour: int = None, window_hours: int = 3) -> List[Dict]:
        if target_hour is None:
            search_hours = range(24)
        else:
            start_hour = max(0, target_hour - window_hours)
            end_hour = min(24, target_hour + window_hours + 1)
            search_hours = range(start_hour, end_hour)

        recommendations = []
        for hour in search_hours:
            congestion_data = self.congestion_df[
                (self.congestion_df["Airport"] == airport) & (self.congestion_df["Hour"] == hour)]
            delay_data = self.delay_patterns_df[
                (self.delay_patterns_df["Airport"] == airport) & (self.delay_patterns_df["Hour"] == hour)]

            if len(congestion_data) > 0 and len(delay_data) > 0:
                congestion = congestion_data.iloc[0]
                delay_info = delay_data.iloc[0]
                score = (congestion["CongestionLevel"] * 0.4 + delay_info["DelayRisk"] * 0.4 + (
                            congestion["Utilization"] * 0.2))
                recommendations.append({
                    "Hour": hour, "Score": score, "Utilization": congestion["Utilization"],
                    "CongestionLevel": congestion["CongestionLevel"], "DelayRisk": delay_info["DelayRisk"],
                    "AvgDelay": delay_info["AvgDelay"], "FlightCount": congestion["FlightCount"],
                    "Recommendation": self._get_recommendation_text(score, hour)
                })

        recommendations.sort(key=lambda x: x["Score"])
        return recommendations

    def _get_recommendation_text(self, score, hour):
        if score < 0.3:
            return f"Excellent slot at {hour:02d}:00 - Low congestion, minimal delays expected"
        elif score < 0.5:
            return f"Good slot at {hour:02d}:00 - Moderate traffic, acceptable delay risk"
        elif score < 0.7:
            return f"Busy slot at {hour:02d}:00 - High traffic, increased delay risk"
        else:
            return f"Avoid slot at {hour:02d}:00 - Peak congestion, high delay probability"

    def analyze_runway_capacity(self, airport: str) -> Dict:
        airport_data = self.congestion_df[self.congestion_df["Airport"] == airport]
        config = AIRPORT_CONFIG.get(airport)
        if len(airport_data) == 0: return {"max_capacity": 0, "avg_peak_utilization": 0, "avg_offpeak_utilization": 0,
                                           "peak_hours": [], "most_congested_hour": None, "least_congested_hour": None,
                                           "runway_count": 0}

        peak_hours_df = airport_data[airport_data["Hour"].isin(config["peak_hours"])]
        off_peak_hours_df = airport_data[~airport_data["Hour"].isin(config["peak_hours"])]
        return {
            "max_capacity": config["runways"] * config["capacity_per_hour"],
            "avg_peak_utilization": peak_hours_df["Utilization"].mean(),
            "avg_offpeak_utilization": off_peak_hours_df["Utilization"].mean(),
            "peak_hours": config["peak_hours"],
            "most_congested_hour": airport_data.loc[airport_data["CongestionLevel"].idxmax(), "Hour"],
            "least_congested_hour": airport_data.loc[airport_data["CongestionLevel"].idxmin(), "Hour"],
            "runway_count": config["runways"]
        }


# ------------------------------
# Query Mapping (Robust Version)
# ------------------------------
def _determine_intent(query: str, top_op: str, op_score: float) -> str:
    q_lower = query.lower()
    optimization_keywords = ["optimize", "best time", "optimal slot", "recommend", "suggest", "when should",
                             "best departure", "least congested"]
    if any(keyword in q_lower for keyword in optimization_keywords) or (top_op == 'optimize' and op_score > 0.4):
        return "optimize"

    prediction_keywords = ["will", "predict", "probability", "chance", "delayed?", "what are the chances"]
    if any(keyword in q_lower for keyword in prediction_keywords):
        if "delay" in q_lower or "late" in q_lower or "on time" in q_lower:
            return "predict"

    return "statistics"


def map_query(query: str, sbert_model, col_keys, col_emb, op_keys, op_emb, df):
    q_emb = sbert_model.encode(query, convert_to_tensor=True)
    col_scores = util.pytorch_cos_sim(q_emb, col_emb)[0].cpu().tolist()
    col_with_scores = sorted([(col_keys[i], float(col_scores[i])) for i in range(len(col_keys))], key=lambda x: -x[1])
    top_cols = [c for c, s in col_with_scores[:3] if s > 0.28]

    op_scores = util.pytorch_cos_sim(q_emb, op_emb)[0].cpu().tolist()
    op_with_scores = sorted([(op_keys[i], float(op_scores[i])) for i in range(len(op_keys))], key=lambda x: -x[1])
    top_op = op_with_scores[0][0] if op_with_scores and op_with_scores[0][1] > 0.22 else None
    top_op_score = op_with_scores[0][1] if op_with_scores and op_with_scores[0][1] > 0.22 else 0.0

    month = detect_month(query)
    dow = detect_dayofweek(query)
    hour_range = detect_hour_range(query)
    origin, dest, carrier = detect_origin_dest_carrier(query, df)

    intent = _determine_intent(query, top_op, top_op_score)

    groupby = None
    q_lower = query.lower()
    if "by " in q_lower or " per " in q_lower or "each " in q_lower:
        if "carrier" in q_lower:
            groupby = "UniqueCarrier"
        elif "origin" in q_lower:
            groupby = "Origin"
        elif "dest" in q_lower:
            groupby = "Dest"
        else:
            for c in top_cols:
                if c in ["UniqueCarrier", "Origin", "Dest", "TailNum", "Month", "DayOfWeek"]:
                    groupby = c;
                    break

    return {"top_cols": top_cols, "col_expl": col_with_scores[:5], "top_op": top_op, "op_expl": op_with_scores[:4],
            "filters": {"month": month, "dayofweek": dow, "hour_range": hour_range, "origin": origin, "dest": dest,
                        "carrier": carrier},
            "groupby": groupby, "intent": intent}


# ------------------------------
# Execute stats action
# ------------------------------
def execute_stats(mapping, df, raw_query):
    top_cols = mapping["top_cols"]
    op = mapping["top_op"]
    grp = mapping["groupby"]
    filters = mapping["filters"]

    df2 = df.copy()
    if filters["month"]: df2 = df2[df2["Month"] == filters["month"]]
    if filters["dayofweek"]:
        if filters["dayofweek"] == "weekday":
            df2 = df2[df2["DayOfWeek"].isin([1, 2, 3, 4, 5])]
        elif filters["dayofweek"] == "weekend":
            df2 = df2[df2["DayOfWeek"].isin([6, 7])]
        else:
            df2 = df2[df2["DayOfWeek"] == filters["dayofweek"]]
    if filters["origin"]: df2 = df2[df2["Origin"] == filters["origin"]]
    if filters["dest"]: df2 = df2[df2["Dest"] == filters["dest"]]
    if filters["carrier"]: df2 = df2[df2["UniqueCarrier"] == filters["carrier"]]
    if filters["hour_range"]:
        a, b = filters["hour_range"]
        df2 = df2[df2["DepHour"].between(a, b)]

    agg_col = top_cols[0] if top_cols else "ArrDelay"
    if agg_col not in df2.columns: agg_col = "ArrDelay"  # Fallback

    try:
        if op in ["mean", "median", "max", "min", "percent"]:
            if grp:
                res = df2.groupby(grp)[agg_col].mean() * 100 if op == "percent" else getattr(df2.groupby(grp)[agg_col],
                                                                                             op)()
                result = res.reset_index()
            else:
                val = df2[agg_col].mean() * 100 if op == "percent" else getattr(df2[agg_col], op)()
                result = pd.DataFrame([{f"{op}_{agg_col}": val}])
        elif op == "count":
            result = df2.groupby(grp).size().reset_index(name="count") if grp else pd.DataFrame([{"count": len(df2)}])
        elif op == "top":
            result = df2[agg_col].value_counts().head(10).reset_index()
        elif op == "distribution":
            result = {"hist_values": df2[agg_col].dropna().tolist()}
        elif op == "trend":
            result = df2.groupby("DepHour")[agg_col].mean().reindex(range(0, 24)).reset_index()
        else:
            result = df2[
                [agg_col] + [c for c in ["UniqueCarrier", "Origin", "Dest", "DepHour"] if c in df2.columns]].head(50)
    except Exception as e:
        result = pd.DataFrame([{"error": str(e)}])

    return result, "# Generated Code Snippet"


# ------------------------------
# Train ML models
# ------------------------------
@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame, target_col="ArrDelayBinary"):
    df = df.copy()
    if "DepHour" not in df.columns:
        dh, dm = zip(*df["DepTime"].apply(parse_time_hhmm));
        df["DepHour"] = pd.Series(dh, index=df.index)
    if "GroundTime" not in df.columns:
        df["GroundTime"] = df["TaxiIn"].fillna(0) + df["TaxiOut"].fillna(0)

    df["IsPeakHour"] = df["DepHour"].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0)
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in [6, 7] else 0)
    df["IsCongestedAirport"] = df["Origin"].apply(lambda x: 1 if x in ["IAH", "DFW"] else 0)
    df["ArrDelayBinary"] = (df["ArrDelay"].fillna(0) > 15).astype(int)

    feats = ["Month", "DayOfWeek", "DepHour", "Distance", "AirTime", "GroundTime", "IsPeakHour", "IsWeekend",
             "IsCongestedAirport"]
    X = df[feats].fillna(0)
    y = df["ArrDelayBinary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y if y.nunique() > 1 else None)

    rf_pipe = Pipeline([("scaler", StandardScaler()),
                        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10))])
    rf_pipe.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_pipe.predict(X_test))

    if XGBOOST_AVAILABLE:
        model_b = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42,
                                max_depth=5)
        model_b_name = "XGBoost"
    else:
        model_b = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        model_b_name = "GradientBoosting"
    model_b.fit(X_train, y_train)
    model_b_acc = accuracy_score(y_test, model_b.predict(X_test))

    return {"rf_pipe": rf_pipe, "model_b": model_b,
            "meta": {"features": feats, "rf_acc": float(rf_acc), "model_b_acc": float(model_b_acc),
                     "model_b_name": model_b_name}}


# ------------------------------
# Prediction helper
# ------------------------------
def ensemble_predict(rf_pipe, model_b, meta, input_row: pd.DataFrame):
    Xrow = input_row[meta["features"]].fillna(0)
    rf_prob = rf_pipe.predict_proba(Xrow)[:, 1]
    b_prob = model_b.predict_proba(Xrow)[:, 1]
    avg_prob = (rf_prob * 0.5 + b_prob * 0.5)
    return {"rf_prob": float(rf_prob[0]), "b_prob": float(b_prob[0]), "avg_prob": float(avg_prob[0]),
            "confidence": float(abs(avg_prob[0] - 0.5) * 2)}


# ------------------------------
# TF-IDF fallback
# ------------------------------
@st.cache_resource(show_spinner=False)
def build_tfidf_index(df: pd.DataFrame):
    docs = [f"{k}: {v}" for k, v in COLUMN_DESCRIPTIONS.items()]
    vec = TfidfVectorizer().fit(docs)
    doc_mat = vec.transform(docs)
    return vec, doc_mat, list(COLUMN_DESCRIPTIONS.keys())


# ------------------------------
# Visualization Helper Functions
# ------------------------------
def create_congestion_heatmap(optimizer: ScheduleOptimizer, airports: List[str]):
    fig, axes = plt.subplots(1, len(airports), figsize=(5 * len(airports), 5), squeeze=False)
    axes = axes.flatten()
    for i, airport in enumerate(airports):
        airport_data = optimizer.congestion_df[optimizer.congestion_df["Airport"] == airport]
        if airport_data.empty: continue
        heatmap_data = airport_data.pivot_table(index='Airport', columns='Hour', values='CongestionLevel').fillna(0)
        sns.heatmap(heatmap_data, cmap='RdYlBu_r', ax=axes[i], cbar=i == len(airports) - 1)
        axes[i].set_title(f'{airport} Congestion')
        axes[i].set_xlabel('Hour of Day')
        axes[i].set_ylabel('')
    plt.tight_layout()
    return fig


def create_optimization_chart(recommendations: List[Dict], airport: str):
    if not recommendations: return None
    df = pd.DataFrame(recommendations).sort_values("Hour")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    colors = ['#2ca02c' if s < 0.3 else '#ff7f0e' if s < 0.5 else '#d62728' for s in df["Score"]]
    ax1.bar(df["Hour"], df["Score"], color=colors, alpha=0.8, label="Optimization Score (Lower is Better)")
    ax1.set_title(f'{airport} - Hourly Schedule Optimization Scores')
    ax1.set_ylabel('Optimization Score')
    ax1.set_xlabel('Hour of Day')
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


# ------------------------------
# Main Streamlit App UI
# ------------------------------
def main():
    st.set_page_config(page_title="H-Flights Schedule Optimizer", layout="wide")
    st.title("Flight Schedule Optimizer")
    st.markdown("**AI-Powered Schedule Analysis for Houston & DFW Airports**")

    # Sidebar
    with st.sidebar:
        st.header("App Features")
        st.markdown(
            "- NLP Query Understanding\n- Dual-ML Delay Prediction\n- Schedule Slot Optimization\n- Congestion Heatmaps")
        st.header("Supported Airports")
        st.markdown("- **IAH** (Houston)\n- **DFW** (Dallas/Fort Worth)\n- **HOU** (Houston Hobby)")

    # Load data
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--features", default="hflights.csv")
    args, _ = ap.parse_known_args()

    uploaded = st.file_uploader("Upload flight data (CSV)", type=["csv"])
    df = None
    if uploaded:
        df = pd.read_csv(uploaded)
    elif os.path.exists(args.features):
        df = pd.read_csv(args.features)
    else:
        st.error(f"Default file not found: {args.features}. Please upload hflights.csv.")
        st.stop()

    # Feature Engineering
    if "DepHour" not in df.columns and "DepTime" in df.columns:
        dh, dm = zip(*df["DepTime"].apply(parse_time_hhmm))
        df["DepHour"] = pd.Series(dh, index=df.index)

    # UI
    st.metric("Total Flights", f"{len(df):,}")
    with st.expander("📋 Dataset Info", expanded=False):
        st.write("**Available Columns:**", ", ".join(df.columns.tolist()))
        st.dataframe(df.head(3))

    # Initialize engines
    if 'optimizer' not in st.session_state:
        with st.spinner("🔧 Initializing Optimizer..."):
            st.session_state.optimizer = ScheduleOptimizer(df)
    optimizer = st.session_state.optimizer

    with st.spinner("Loading NLP Models..."):
        sbert = load_sbert()
        col_keys, col_emb, op_keys, op_emb = precompute_embeddings(sbert)

    # ML MODEL TRAINING SECTION
    if st.button("Train/Re-train ML Models", help="Train RandomForest + XGBoost models"):
        with st.spinner("🤖 Training Enhanced ML Models..."):
            artifacts = train_models(df)
            st.session_state["artifacts"] = artifacts
            st.success("Models trained successfully!")
    else:
        if "artifacts" not in st.session_state:
            with st.spinner("🤖 Auto-training ML models..."):
                artifacts = train_models(df)
                st.session_state["artifacts"] = artifacts
        else:
            artifacts = st.session_state["artifacts"]

    # MODEL PERFORMANCE DISPLAY
    with st.expander("Model Performance", expanded=False):
        meta = artifacts["meta"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RandomForest Accuracy", f"{meta['rf_acc']:.3f}")
        with col2:
            st.metric(f"{meta['model_b_name']} Accuracy", f"{meta['model_b_acc']:.3f}")
        st.write("**Features used:**", ", ".join(meta["features"]))

    st.markdown("---")
    st.markdown("### 🗣️ Ask a Question (Natural Language)")
    user_q = st.text_input("Query", placeholder="e.g., 'Find optimal departure slots from IAH'")

    # --- UPDATED EXAMPLES SECTION ---
    with st.expander("💡 Example Queries"):
        examples = {
            "Statistics": [
                "What is the average arrival delay from IAH?",
               "What is the average departure delay from IAH on weekends between 4 PM and 7 PM? ",
               " Show the distribution of departure delays"

            ],
            "Predictions": [
                "Predict delay for a flight from IAH to DFW at 8 AM",
                "What is the delay probability for a flight departing Houston on a Sunday?",
                "Will a flight from Hobby be late in the evening?"
            ],
            "Optimization": [
                "When is the best time to depart from IAH?",
                "What are the least congested hours at DFW?",
                "Recommend a schedule to minimize taxi time at Hobby airport"
            ]
        }
        for category, queries in examples.items():
            st.markdown(f"**{category}:**")
            for query in queries:
                if st.button(query, key=query): user_q = query; st.rerun()
    # --- END OF UPDATED SECTION ---

    if user_q:
        mapping = map_query(user_q, sbert, col_keys, col_emb, op_keys, op_emb, df)

        with st.expander("Query Understanding"):
            st.write(f"**Intent:** {mapping['intent'].capitalize()}")
            st.write(f"**Filters:** {mapping['filters']}")

        if mapping["intent"] == "optimize" and optimizer:
            st.markdown("### ⚡ Schedule Optimization Results")
            target_airport = mapping["filters"]["origin"] or "IAH"
            if target_airport not in AIRPORT_CONFIG: target_airport = "IAH"

            recommendations = optimizer.find_optimal_slots(target_airport)

            tab1, tab2, tab3 = st.tabs(["Recommendations", "Analysis Chart", "Congestion Heatmap"])

            with tab1:
                st.markdown(f"#### Top 5 Recommended Departure Slots for **{target_airport}**")
                top_recs = recommendations[:5]

                if not top_recs:
                    st.warning(
                        f"No flight data available in the dataset to generate recommendations for **{target_airport}**.")
                    st.info("Please check if the uploaded CSV contains flights for this airport.")
                else:
                    for i, rec in enumerate(top_recs):
                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
                            with col1:
                                st.markdown(f"### #{i + 1}")
                            with col2:
                                st.markdown(f"**Time Slot:**");
                                st.markdown(f"### {rec['Hour']:02d}:00 - {rec['Hour'] + 1:02d}:00")
                            with col3:
                                st.metric("Score", f"{rec['Score']:.2f}", help="Lower score is better")
                            with col4:
                                st.markdown(f"**Rating:**")
                                if rec["Score"] < 0.3:
                                    st.success("Excellent")
                                elif rec["Score"] < 0.5:
                                    st.info("Good")
                                elif rec["Score"] < 0.7:
                                    st.warning("Busy")
                                else:
                                    st.error("Avoid")
                            st.caption(f"Recommendation: {rec['Recommendation']}")
            with tab2:
                st.markdown(f"#### Optimization Analysis for **{target_airport}**")
                opt_chart = create_optimization_chart(recommendations, target_airport)
                if opt_chart:
                    st.pyplot(opt_chart)
                else:
                    st.info("Analysis chart could not be generated.")
            with tab3:
                st.markdown("#### Airport Congestion Heatmap")
                heatmap_fig = create_congestion_heatmap(optimizer, list(AIRPORT_CONFIG.keys()))
                st.pyplot(heatmap_fig)

        elif mapping["intent"] == "predict":
            st.markdown("### Flight Delay Prediction")
            with st.form("prediction_form"):
                origin = st.selectbox("Origin Airport", list(AIRPORT_CONFIG.keys()), index=0)
                hour = st.slider("Departure Hour", 0, 23, 8)
                dow = st.slider("Day of Week (1=Mon)", 1, 7, 3)
                submitted = st.form_submit_button("Predict Delay")
            if submitted:
                input_features = pd.DataFrame([{"Month": 6, "DayOfWeek": dow, "DepHour": hour,
                                                "Distance": df['Distance'].median(), "AirTime": df['AirTime'].median(),
                                                "GroundTime": df['TaxiIn'].median() + df['TaxiOut'].median(),
                                                "IsPeakHour": 1 if hour in AIRPORT_CONFIG.get(origin, {}).get(
                                                    "peak_hours", []) else 0, "IsWeekend": 1 if dow > 5 else 0,
                                                "IsCongestedAirport": 1 if origin in ["IAH", "DFW"] else 0}])
                result = ensemble_predict(artifacts["rf_pipe"], artifacts["model_b"], artifacts["meta"], input_features)
                st.metric("Predicted Delay Probability", f"{result['avg_prob']:.1%}")
                if result['avg_prob'] < 0.3:
                    st.success("🟢 **LOW RISK**")
                elif result['avg_prob'] < 0.6:
                    st.warning("🟡 **MODERATE RISK**")
                else:
                    st.error("🔴 **HIGH RISK**")

        else:  # Statistics
            st.markdown("### Statistical Analysis Results")
            result, _ = execute_stats(mapping, df, user_q)
            if isinstance(result, pd.DataFrame) and not result.empty:
                st.dataframe(result)
            elif isinstance(result, dict) and "hist_values" in result:
                fig, ax = plt.subplots();
                ax.hist(result["hist_values"], bins=30);
                st.pyplot(fig)
            else:
                st.warning("No results found for this statistical query.")


if __name__ == "__main__":
    main()