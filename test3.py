#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight Schedule Optimization (ADAPTED FOR Flight_Data.csv)

Features:
- NLP + Dual-ML prediction capabilities
- Schedule optimization algorithms
- Runway capacity analysis
- Cascading delay impact modeling
- Peak-time slot recommendations
- Interactive schedule visualization

Save as: app.py
Run: streamlit run app.py
"""

import argparse
import os
import re
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ------------------------------
# NEW: Data Cleaning and Preparation Function for Flight_Data.csv
# ------------------------------
@st.cache_data(show_spinner="Processing and cleaning flight data...")
def clean_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms the raw Flight_Data.csv into a structured format.
    """
    # Forward fill Flight Number and Serial Number
    df['Flight Number'].ffill(inplace=True)
    df['S.No'].ffill(inplace=True)

    # Drop rows where essential data is missing
    df.dropna(subset=['Unnamed: 2', 'From', 'To'], inplace=True)

    # Rename columns for clarity
    df.rename(columns={
        'Unnamed: 2': 'Date',
        'From': 'Origin_City',
        'To': 'Dest_City',
        'Flight time': 'FlightTime'
    }, inplace=True)

    # Extract Airport Codes
    df['Origin'] = df['Origin_City'].str.extract(r'\((\w+)\)').fillna('')
    df['Dest'] = df['Dest_City'].str.extract(r'\((\w+)\)').fillna('')

    # Drop original city columns and other unused columns
    df.drop(columns=[
        'Origin_City', 'Dest_City', 'S.No', 'Unnamed: 10',
        'Unnamed: 12', 'Unnamed: 13'
    ], inplace=True)

    # --- DATETIME CONVERSION ---
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')

    # Function to combine date with time strings and convert to datetime objects
    def to_datetime(series_time):
        return pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + series_time,
            format='%Y-%m-%d %I:%M %p',
            errors='coerce'
        )

    df['STD_dt'] = to_datetime(df['STD'])
    df['ATD_dt'] = to_datetime(df['ATD'])
    df['STA_dt'] = to_datetime(df['STA'])

    # Handle 'Landed HH:MM AM/PM' format in ATA
    ata_time = df['ATA'].str.replace('Landed ', '', regex=False).str.strip()
    df['ATA_dt'] = to_datetime(ata_time)

    # --- FEATURE ENGINEERING ---
    # Calculate Delays in minutes
    df['DepDelay'] = (df['ATD_dt'] - df['STD_dt']).dt.total_seconds() / 60
    df['ArrDelay'] = (df['ATA_dt'] - df['STA_dt']).dt.total_seconds() / 60

    # Fill large negative delays that might be due to date errors (e.g., next day arrival)
    df.loc[df['ArrDelay'] < -1000, 'ArrDelay'] += 1440  # Add 24 hours in minutes
    df.loc[df['DepDelay'] < -1000, 'DepDelay'] += 1440

    # Extract Time-based features
    df['DepHour'] = df['ATD_dt'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Mon, 7=Sun
    df['DayofMonth'] = df['Date'].dt.day

    # Convert FlightTime 'HH:MM' to minutes
    def flight_time_to_minutes(time_str):
        if pd.isna(time_str) or ':' not in str(time_str): return np.nan
        try:
            h, m = map(int, str(time_str).split(':'))
            return h * 60 + m
        except:
            return np.nan

    df['AirTime'] = df['FlightTime'].apply(flight_time_to_minutes)

    # Create a pseudo-distance feature
    df['Distance'] = df['AirTime'] * 8  # Approximate miles assuming avg speed of 8 miles/min

    # Create placeholder columns to match original script structure
    df['Cancelled'] = 0
    df['Diverted'] = 0
    df.rename(columns={'Flight Number': 'UniqueCarrier'}, inplace=True)  # Re-purpose for airline

    # Drop rows with critical null values after processing
    df.dropna(subset=['Date', 'DepHour', 'DepDelay', 'ArrDelay'], inplace=True)

    return df


# ------------------------------
# Utilities (Unchanged)
# ------------------------------
def parse_time_hhmm(x):
    if pd.isna(x):
        return (np.nan, np.nan)
    try:
        s = str(int(x)).zfill(4)
        hh = int(s[:-2]);
        mm = int(s[-2:])
        if 0 <= hh < 24 and 0 <= mm < 60:
            return hh, mm
    except:
        pass
    return (np.nan, np.nan)


# ------------------------------
# Airport Configuration (ADAPTED FOR Flight_Data.csv)
# NOTE: These are ESTIMATES for demonstration purposes.
# ------------------------------
AIRPORT_CONFIG = {
    "BOM": {"runways": 2, "capacity_per_hour": 45, "peak_hours": [8, 9, 10, 17, 18, 19, 20], "weather_delays": 0.15,
            "ground_congestion": 0.85},
    "IXC": {"runways": 1, "capacity_per_hour": 15, "peak_hours": [7, 8, 16, 17], "weather_delays": 0.10,
            "ground_congestion": 0.6},
    "BBI": {"runways": 1, "capacity_per_hour": 20, "peak_hours": [8, 9, 18, 19], "weather_delays": 0.12,
            "ground_congestion": 0.7},
    "DEL": {"runways": 4, "capacity_per_hour": 80, "peak_hours": [7, 8, 9, 18, 19, 20], "weather_delays": 0.18,
            "ground_congestion": 0.9},
    "BLR": {"runways": 2, "capacity_per_hour": 50, "peak_hours": [8, 9, 17, 18, 19], "weather_delays": 0.10,
            "ground_congestion": 0.8},
    "CCU": {"runways": 2, "capacity_per_hour": 35, "peak_hours": [7, 8, 17, 18], "weather_delays": 0.14,
            "ground_congestion": 0.75}
}

# ------------------------------
# Column Descriptions (ADAPTED FOR new DataFrame)
# ------------------------------
COLUMN_DESCRIPTIONS = {
    "ArrDelay": "difference between scheduled and actual arrival time in minutes",
    "DepDelay": "difference between scheduled and actual departure time in minutes",
    "DepHour": "actual departure hour (0-23)",
    "UniqueCarrier": "airline carrier code",
    "Aircraft": "aircraft model and registration",
    "AirTime": "time spent airborne in minutes",
    "Origin": "origin airport code",
    "Dest": "destination airport code",
    "Distance": "estimated distance flown (miles)",
    "Month": "month number 1..12",
    "DayOfWeek": "day of week as integer (1=Mon, 7=Sun)",
}

OPERATIONS = {
    "mean": "average / mean", "median": "median", "max": "maximum", "min": "minimum",
    "count": "count / how many", "top": "top N / most frequent", "distribution": "distribution / histogram",
    "percent": "percentage (for binary columns)", "trend": "time trend (by hour or by day)",
    "optimize": "find optimal schedule slots", "capacity": "runway capacity analysis",
    "congestion": "congestion analysis"
}


# ------------------------------
# SBERT Caching (Unchanged)
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
# Query parsing helpers (ADAPTED)
# ------------------------------
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
     "December"], start=1)}
WEEKDAYS = {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7}


def detect_optimization_intent(q: str):
    optimization_keywords = ["optimize", "best time", "optimal slot", "schedule", "avoid delay", "less congested",
                             "peak hours", "busy time", "runway capacity", "recommend", "suggest", "when should",
                             "best departure"]
    return any(keyword in q.lower() for keyword in optimization_keywords)


def detect_month(q: str):
    qq = q.lower();
    for k, v in MONTHS.items():
        if k in qq: return v
    return None


def detect_dayofweek(q: str):
    qq = q.lower();
    for k, v in WEEKDAYS.items():
        if k in qq: return v
    if "weekday" in qq: return "weekday"
    if "weekend" in qq: return "weekend"
    return None


def detect_hour_range(q: str):
    m = re.search(r'(\d{1,2})(?:[:h]?| ?(?:am|pm)?)\s*(?:-|to|and|until)\s*(\d{1,2})(?:[:h]?| ?(?:am|pm)?)', q.lower())
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return min(a, b), max(a, b)
    return None


def detect_origin_dest_carrier(q: str, df: pd.DataFrame):
    qq = q.lower()
    origin, dest, carrier = None, None, None

    city_to_code = {"mumbai": "BOM", "chandigarh": "IXC", "bhubaneswar": "BBI", "delhi": "DEL", "bangalore": "BLR",
                    "kolkata": "CCU"}

    for city, code in city_to_code.items():
        if city in qq:
            if re.search(r'from\s+' + re.escape(city), qq):
                origin = code
            elif re.search(r'to\s+' + re.escape(city), qq):
                dest = code
            elif origin is None:
                origin = code

    airport_codes = list(pd.concat([df['Origin'], df['Dest']]).unique())
    for code in airport_codes:
        if code.lower() in qq:
            if re.search(r'from\s+' + re.escape(code.lower()), qq):
                origin = code
            elif re.search(r'to\s+' + re.escape(code.lower()), qq):
                dest = code
            elif origin is None:
                origin = code

    if "UniqueCarrier" in df.columns:
        for c in df["UniqueCarrier"].dropna().astype(str).unique().tolist():
            if c.lower() in qq:
                carrier = c;
                break
    return origin, dest, carrier


# ------------------------------
# Schedule Optimization Engine (Unchanged Logic, Adapted Data)
# ------------------------------
class ScheduleOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        self._calculate_congestion_metrics()
        self._calculate_delay_patterns()

    def _calculate_congestion_metrics(self):
        congestion_data = []
        airports_to_analyze = list(AIRPORT_CONFIG.keys())
        for airport in airports_to_analyze:
            airport_flights = self.df[(self.df["Origin"] == airport) | (self.df["Dest"] == airport)]
            if airport_flights.empty: continue

            hourly_counts = airport_flights.groupby("DepHour").size()
            config = AIRPORT_CONFIG.get(airport)
            if not config: continue

            max_capacity = config["runways"] * config["capacity_per_hour"]
            for hour in range(24):
                count = hourly_counts.get(hour, 0)
                utilization = count / max_capacity if max_capacity > 0 else 0
                congestion_data.append({
                    "Airport": airport, "Hour": hour, "FlightCount": count,
                    "Utilization": min(utilization, 1.0),
                    "CongestionLevel": self._get_congestion_level(utilization, config, hour)
                })
        self.congestion_df = pd.DataFrame(congestion_data)

    def _get_congestion_level(self, utilization, config, hour):
        base_congestion = utilization
        if hour in config["peak_hours"]: base_congestion *= 1.3
        base_congestion *= (1 + config["weather_delays"])
        base_congestion *= config["ground_congestion"]
        return min(base_congestion, 1.0)

    def _calculate_delay_patterns(self):
        delay_patterns = []
        delay_col = "ArrDelay"
        for airport in AIRPORT_CONFIG.keys():
            airport_data = self.df[self.df["Origin"] == airport]
            if len(airport_data) == 0: continue
            hourly_delays = airport_data.groupby("DepHour")[delay_col].agg(["mean", "median", "std", "count"]).fillna(0)
            for hour in range(24):
                if hour in hourly_delays.index:
                    stats = hourly_delays.loc[hour]
                    delay_patterns.append({
                        "Airport": airport, "Hour": hour, "AvgDelay": stats["mean"],
                        "DelayRisk": self._calculate_delay_risk(stats)
                    })
        self.delay_patterns_df = pd.DataFrame(delay_patterns)

    def _calculate_delay_risk(self, stats):
        if stats["count"] == 0: return 0
        risk = (max(0, stats["mean"]) + stats["std"]) / 100
        return min(risk, 1.0)

    def find_optimal_slots(self, airport: str):
        recommendations = []
        for hour in range(24):
            congestion_data = self.congestion_df[
                (self.congestion_df["Airport"] == airport) & (self.congestion_df["Hour"] == hour)]
            delay_data = self.delay_patterns_df[
                (self.delay_patterns_df["Airport"] == airport) & (self.delay_patterns_df["Hour"] == hour)]
            if not congestion_data.empty and not delay_data.empty:
                congestion = congestion_data.iloc[0]
                delay_info = delay_data.iloc[0]
                score = (congestion["CongestionLevel"] * 0.5 + delay_info["DelayRisk"] * 0.5)
                recommendations.append({
                    "Hour": hour, "Score": score,
                    "Recommendation": self._get_recommendation_text(score, hour)
                })
        recommendations.sort(key=lambda x: x["Score"])
        return recommendations

    def _get_recommendation_text(self, score, hour):
        if score < 0.3:
            return f"Excellent slot at {hour:02d}:00 - Low congestion, minimal delays"
        elif score < 0.5:
            return f"Good slot at {hour:02d}:00 - Moderate traffic, acceptable delay risk"
        elif score < 0.7:
            return f"Busy slot at {hour:02d}:00 - High traffic, increased delay risk"
        else:
            return f"Avoid slot at {hour:02d}:00 - Peak congestion, high delay probability"


# ------------------------------
# Query Mapping & Execution (Unchanged Logic, Adapted Data)
# ------------------------------
def map_query(query: str, sbert_model, col_keys, col_emb, op_keys, op_emb, df):
    q_emb = sbert_model.encode(query, convert_to_tensor=True)
    col_scores = util.pytorch_cos_sim(q_emb, col_emb)[0].cpu().tolist()
    col_with_scores = sorted([(col_keys[i], float(s)) for i, s in enumerate(col_scores)], key=lambda x: -x[1])
    top_cols = [c for c, s in col_with_scores[:3] if s > 0.28]

    op_scores = util.pytorch_cos_sim(q_emb, op_emb)[0].cpu().tolist()
    op_with_scores = sorted([(op_keys[i], float(s)) for i, s in enumerate(op_scores)], key=lambda x: -x[1])
    top_op = op_with_scores[0][0] if op_with_scores and op_with_scores[0][1] > 0.22 else None
    top_op_score = op_with_scores[0][1] if op_with_scores else 0.0

    intent = "statistics"
    if detect_optimization_intent(query) or (top_op == 'optimize' and top_op_score > 0.4):
        intent = "optimize"
    elif any(k in query.lower() for k in ["predict", "will", "probability", "chance", "delayed?"]):
        intent = "predict"

    filters = {
        "month": detect_month(query), "dayofweek": detect_dayofweek(query),
        "hour_range": detect_hour_range(query),
        "origin": detect_origin_dest_carrier(query, df)[0],
        "dest": detect_origin_dest_carrier(query, df)[1],
        "carrier": detect_origin_dest_carrier(query, df)[2]
    }
    return {"intent": intent, "top_cols": top_cols, "top_op": top_op, "filters": filters}


def execute_stats(mapping, df):
    df2 = df.copy()
    filters = mapping["filters"]
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

    agg_col = mapping["top_cols"][0] if mapping["top_cols"] and mapping["top_cols"][0] in df2.columns else "ArrDelay"
    op = mapping["top_op"]

    if op in ["mean", "median", "max", "min"]:
        result = pd.DataFrame([{f"{op}_{agg_col}": getattr(df2[agg_col], op)()}])
    elif op == "count":
        result = pd.DataFrame([{"count": len(df2)}])
    elif op == "distribution":
        result = {"hist_values": df2[agg_col].dropna().tolist()}
    else:
        result = df2[[agg_col] + [c for c in ["UniqueCarrier", "Origin", "Dest", "DepHour"] if c in df2.columns]].head(
            20)
    return result


# ------------------------------
# Train ML models (ADAPTED features)
# ------------------------------
@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    df = df.copy()

    # Feature Engineering for ML
    df["IsPeakHour"] = df.apply(
        lambda row: 1 if row['Origin'] in AIRPORT_CONFIG and row['DepHour'] in AIRPORT_CONFIG[row['Origin']][
            'peak_hours'] else 0, axis=1)
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in [6, 7] else 0)
    df["ArrDelayBinary"] = (df["ArrDelay"].fillna(0) > 15).astype(int)  # Predict delay > 15 mins

    feats = ["Month", "DayOfWeek", "DepHour", "Distance", "AirTime", "IsPeakHour", "IsWeekend"]
    X = df[feats].fillna(df[feats].median())
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
# Prediction helper (Unchanged)
# ------------------------------
def ensemble_predict(rf_pipe, model_b, meta, input_row: pd.DataFrame):
    Xrow = input_row[meta["features"]].fillna(0)
    rf_prob = rf_pipe.predict_proba(Xrow)[:, 1]
    b_prob = model_b.predict_proba(Xrow)[:, 1]
    avg_prob = (rf_prob * 0.5 + b_prob * 0.5)
    return float(avg_prob[0])


# ------------------------------
# Visualization Helpers (ADAPTED)
# ------------------------------
def create_congestion_heatmap(optimizer: ScheduleOptimizer, airports: list):
    fig, axes = plt.subplots(1, len(airports), figsize=(5 * len(airports), 5), squeeze=False)
    for i, airport in enumerate(airports):
        airport_data = optimizer.congestion_df[optimizer.congestion_df["Airport"] == airport]
        if airport_data.empty: continue
        heatmap_data = airport_data.pivot_table(index='Airport', columns='Hour', values='CongestionLevel')
        sns.heatmap(heatmap_data, cmap='RdYlBu_r', ax=axes.flatten()[i])
        axes.flatten()[i].set_title(f'{airport} Congestion')
    plt.tight_layout()
    return fig


def create_optimization_chart(recommendations: list, airport: str):
    if not recommendations: return None
    df = pd.DataFrame(recommendations).sort_values("Hour")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ca02c' if s < 0.3 else '#ff7f0e' if s < 0.5 else '#d62728' for s in df["Score"]]
    ax.bar(df["Hour"], df["Score"], color=colors)
    ax.set_title(f'{airport} - Hourly Schedule Optimization Scores (Lower is Better)')
    ax.set_xlabel('Hour of Day');
    ax.set_ylabel('Optimization Score')
    return fig


# ------------------------------
# Main Streamlit App UI (ADAPTED)
# ------------------------------
def main():
    st.set_page_config(page_title="Flight Schedule Optimizer", layout="wide")
    st.title("Flight Schedule Optimizer")

    # Load data
    default_file = "Flight_Data.csv"
    uploaded = st.file_uploader("Upload flight data (CSV)", type=["csv"])
    df_raw = None
    if uploaded:
        df_raw = pd.read_csv(uploaded)
    elif os.path.exists(default_file):
        df_raw = pd.read_csv(default_file)
    else:
        st.error(f"Default file not found: {default_file}. Please upload your CSV.")
        st.stop()

    df = clean_flight_data(df_raw)

    # Initialize engines
    optimizer = ScheduleOptimizer(df)
    with st.spinner("Loading NLP Models..."):
        sbert = load_sbert()
        col_keys, col_emb, op_keys, op_emb = precompute_embeddings(sbert)

    if "artifacts" not in st.session_state:
        with st.spinner("🤖 Auto-training ML models..."):
            st.session_state["artifacts"] = train_models(df)
    artifacts = st.session_state["artifacts"]

    # --- UI LAYOUT ---
    st.sidebar.header("App Features")
    st.sidebar.markdown(
        "- NLP Query Understanding\n- Dual-ML Delay Prediction\n- Schedule Slot Optimization\n- Congestion Heatmaps")
    st.sidebar.header("Supported Airports")
    st.sidebar.markdown("\n".join([f"- **{code}**" for code in AIRPORT_CONFIG.keys()]))

    col1, col2 = st.columns(2)
    col1.metric("Total Flights Analyzed", f"{len(df):,}")
    col2.metric("Total Airports", f"{len(AIRPORT_CONFIG)}")

    with st.expander("Model Performance"):
        meta = artifacts["meta"]
        c1, c2 = st.columns(2)
        c1.metric("RandomForest Accuracy", f"{meta['rf_acc']:.3f}")
        c2.metric(f"{meta['model_b_name']} Accuracy", f"{meta['model_b_acc']:.3f}")
        st.caption(f"**Features used:** {', '.join(meta['features'])}")

    st.markdown("---")
    st.markdown("###  Ask a Question in English Language")
    user_q = st.text_input("Query", placeholder="e.g., 'Find optimal departure slots from Mumbai'")

    with st.expander("Example Queries"):
        examples = {
            "Statistics": ["What is the average arrival delay from Mumbai?",
                           "Show departure delay distribution for flights to Delhi"],
            "Predictions": ["Predict delay for a flight from BOM at 8 AM",
                            "What is the delay probability for a flight departing on a Sunday?"],
            "Optimization": ["When is the best time to depart from BOM?",
                             "What are the least congested hours at Delhi?"]
        }
        for category, queries in examples.items():
            st.markdown(f"**{category}:**")
            for query in queries:
                if st.button(query, key=query): user_q = query; st.rerun()

    if user_q:
        mapping = map_query(user_q, sbert, col_keys, col_emb, op_keys, op_emb, df)
        st.info(f"**Interpreted Intent:** {mapping['intent'].capitalize()}")

        if mapping["intent"] == "optimize":
            st.markdown("### ⚡ Schedule Optimization Results")
            target_airport = mapping["filters"]["origin"] or "BOM"
            recommendations = optimizer.find_optimal_slots(target_airport)

            tab1, tab2, tab3 = st.tabs(["Recommendations", "Analysis Chart", "Congestion Heatmap"])
            with tab1:
                st.markdown(f"#### Top 5 Recommended Departure Slots for **{target_airport}**")
                for rec in recommendations[:5]:
                    st.success(
                        f"**{rec['Hour']:02d}:00 - {rec['Hour'] + 1:02d}:00**: {rec['Recommendation']} (Score: {rec['Score']:.2f})")
            with tab2:
                st.pyplot(create_optimization_chart(recommendations, target_airport))
            with tab3:
                st.pyplot(create_congestion_heatmap(optimizer, list(AIRPORT_CONFIG.keys())))

        elif mapping["intent"] == "predict":
            st.markdown("### Flight Delay Prediction")
            with st.form("prediction_form"):
                origin = st.selectbox("Origin Airport", list(AIRPORT_CONFIG.keys()), index=0)
                hour = st.slider("Departure Hour", 0, 23, 8)
                dow = st.slider("Day of Week (1=Mon)", 1, 7, 3)
                submitted = st.form_submit_button("Predict Delay")
            if submitted:
                input_features = pd.DataFrame([{"Month": 7, "DayOfWeek": dow, "DepHour": hour,
                                                "Distance": df['Distance'].median(), "AirTime": df['AirTime'].median(),
                                                "IsPeakHour": 1 if hour in AIRPORT_CONFIG.get(origin, {}).get(
                                                    "peak_hours", []) else 0,
                                                "IsWeekend": 1 if dow > 5 else 0}])
                prob = ensemble_predict(artifacts["rf_pipe"], artifacts["model_b"], artifacts["meta"], input_features)
                st.metric("Predicted Delay Probability", f"{prob:.1%}")
                if prob < 0.3:
                    st.success("🟢 LOW RISK")
                elif prob < 0.6:
                    st.warning("🟡 MODERATE RISK")
                else:
                    st.error("🔴 HIGH RISK")

        else:  # Statistics
            st.markdown("### Statistical Analysis Results")
            result = execute_stats(mapping, df)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif isinstance(result, dict) and "hist_values" in result:
                fig, ax = plt.subplots();
                ax.hist(result["hist_values"], bins=30);
                st.pyplot(fig)
            else:
                st.warning("No results found.")


if __name__ == "__main__":
    main()