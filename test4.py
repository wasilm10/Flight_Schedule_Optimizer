#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Flight Schedule Optimization with Full Requirements Implementation

NEW FEATURES ADDED:
Interactive Schedule Tuning Model with Impact Visualization
Cascading Delay Impact Analysis
Aircraft Rotation Tracking
Enhanced NLP Interface
Advanced Delay Propagation Modeling


"""

import argparse
import os
import re
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, util

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ------------------------------
# ENHANCED Data Cleaning with Aircraft Tracking
# ------------------------------
@st.cache_data(show_spinner="Processing and cleaning flight data with aircraft tracking...")
def clean_flight_data_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced data cleaning with aircraft rotation and cascading analysis support
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

    # Create Aircraft Registration from Aircraft column
    df['Aircraft_Registration'] = df['Aircraft'].str.extract(r'([A-Z]{2}-[A-Z]{3})')
    df['Aircraft_Type'] = df['Aircraft'].str.extract(r'([A-Z0-9]+)')

    # Drop original city columns and other unused columns
    df.drop(columns=[
        'Origin_City', 'Dest_City', 'S.No', 'Unnamed: 10',
        'Unnamed: 12', 'Unnamed: 13'
    ], inplace=True)

    # --- ENHANCED DATETIME CONVERSION ---
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')

    def to_datetime(series_time):
        return pd.to_datetime(
            df['Date'].dt.strftime('%Y-%m-%d') + ' ' + series_time,
            format='%Y-%m-%d %I:%M %p',
            errors='coerce'
        )

    df['STD_dt'] = to_datetime(df['STD'])  # Scheduled Departure
    df['ATD_dt'] = to_datetime(df['ATD'])  # Actual Departure
    df['STA_dt'] = to_datetime(df['STA'])  # Scheduled Arrival

    # Handle 'Landed HH:MM AM/PM' format in ATA
    ata_time = df['ATA'].str.replace('Landed ', '', regex=False).str.strip()
    df['ATA_dt'] = to_datetime(ata_time)  # Actual Arrival

    # --- ENHANCED FEATURE ENGINEERING ---
    # Calculate Delays in minutes (KEY METRICS for analysis)
    df['DepDelay'] = (df['ATD_dt'] - df['STD_dt']).dt.total_seconds() / 60
    df['ArrDelay'] = (df['ATA_dt'] - df['STA_dt']).dt.total_seconds() / 60
    df['TaxiOut'] = (df['ATD_dt'] - df['STD_dt']).dt.total_seconds() / 60  # Gate to runway
    df['TaxiIn'] = (df['ATA_dt'] - df['STA_dt']).dt.total_seconds() / 60  # Runway to gate

    # Fix date rollover issues (flights arriving next day)
    df.loc[df['ArrDelay'] < -1000, 'ArrDelay'] += 1440
    df.loc[df['DepDelay'] < -1000, 'DepDelay'] += 1440

    # Extract Time-based features
    df['DepHour'] = df['ATD_dt'].dt.hour
    df['ArrHour'] = df['ATA_dt'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    df['DayofMonth'] = df['Date'].dt.day

    # Convert FlightTime to minutes
    def flight_time_to_minutes(time_str):
        if pd.isna(time_str) or ':' not in str(time_str):
            return np.nan
        try:
            h, m = map(int, str(time_str).split(':'))
            return h * 60 + m
        except:
            return np.nan

    df['AirTime'] = df['FlightTime'].apply(flight_time_to_minutes)
    df['Distance'] = df['AirTime'] * 8  # Approximate distance

    # NEW: Aircraft Rotation Features for Cascading Analysis
    df = df.sort_values(['Aircraft_Registration', 'ATD_dt'])
    df['Next_Flight_Gap'] = df.groupby('Aircraft_Registration')['STD_dt'].shift(-1) - df['ATA_dt']
    df['Next_Flight_Gap_Minutes'] = df['Next_Flight_Gap'].dt.total_seconds() / 60
    df['Turnaround_Time'] = df['Next_Flight_Gap_Minutes']
    df['Is_Quick_Turnaround'] = (df['Turnaround_Time'] < 90).astype(int)  # < 90 min turnaround

    # Create cascading impact potential score
    df['Cascading_Risk_Score'] = (
            (df['DepDelay'].fillna(0) > 15).astype(int) * 0.4 +  # Departure delay
            df['Is_Quick_Turnaround'] * 0.3 +  # Quick turnaround pressure
            ((df['DepHour'].isin([7, 8, 9, 17, 18, 19, 20])).astype(int)) * 0.3  # Peak hour operation
    )

    # Binary delay classifications for ML
    df['DepDelayBinary'] = (df['DepDelay'].fillna(0) > 15).astype(int)
    df['ArrDelayBinary'] = (df['ArrDelay'].fillna(0) > 15).astype(int)

    # Create placeholder columns for compatibility
    df['Cancelled'] = 0
    df['Diverted'] = 0
    df.rename(columns={'Flight Number': 'UniqueCarrier'}, inplace=True)

    # Final cleanup
    df.dropna(subset=['Date', 'DepHour', 'DepDelay', 'ArrDelay'], inplace=True)

    return df


# ------------------------------
# NEW: Schedule Tuning Model
# ------------------------------
class ScheduleTuner:
    """
    Interactive schedule tuning with delay impact simulation
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df = self.df.sort_values(['Date', 'STD_dt'])
        self._build_aircraft_networks()

    def _build_aircraft_networks(self):
        """Build network graph of aircraft rotations for impact analysis"""
        self.aircraft_networks = {}
        for aircraft in self.df['Aircraft_Registration'].dropna().unique():
            aircraft_flights = self.df[self.df['Aircraft_Registration'] == aircraft].sort_values('STD_dt')
            if len(aircraft_flights) > 1:
                G = nx.DiGraph()
                for i in range(len(aircraft_flights) - 1):
                    current_flight = aircraft_flights.iloc[i]
                    next_flight = aircraft_flights.iloc[i + 1]
                    G.add_edge(
                        f"{current_flight['UniqueCarrier']}_{current_flight.name}",
                        f"{next_flight['UniqueCarrier']}_{next_flight.name}",
                        turnaround_time=current_flight['Turnaround_Time']
                    )
                self.aircraft_networks[aircraft] = G

    def simulate_schedule_change(self, flight_index: int, new_departure_minutes: int):
        """
        Simulate the impact of changing a flight's departure time
        Returns before/after metrics and cascading effects
        """
        if flight_index not in self.df.index:
            return None

        original_flight = self.df.loc[flight_index].copy()

        # Calculate new times
        original_std = original_flight['STD_dt']
        time_shift = timedelta(minutes=new_departure_minutes)
        new_std = original_std + time_shift

        # Simulate impact on current flight
        results = {
            'original_departure': original_std.strftime('%H:%M'),
            'new_departure': new_std.strftime('%H:%M'),
            'time_shift_minutes': new_departure_minutes,
            'original_delay_risk': self._calculate_delay_risk(original_flight),
        }

        # Calculate new delay risk based on congestion at new time
        modified_flight = original_flight.copy()
        modified_flight['DepHour'] = new_std.hour
        modified_flight['STD_dt'] = new_std
        results['new_delay_risk'] = self._calculate_delay_risk(modified_flight)
        results['risk_change'] = results['new_delay_risk'] - results['original_delay_risk']

        # Analyze cascading impact
        cascading_impact = self._analyze_cascading_impact(flight_index, time_shift)
        results.update(cascading_impact)

        return results

    def _calculate_delay_risk(self, flight_data):
        """Calculate delay probability for a flight based on its characteristics"""
        airport = flight_data['Origin']
        hour = flight_data['DepHour'] if 'DepHour' in flight_data else flight_data['STD_dt'].hour

        # Historical delay rate at this airport/hour combination
        similar_flights = self.df[
            (self.df['Origin'] == airport) &
            (self.df['DepHour'] == hour)
            ]

        if len(similar_flights) == 0:
            return 0.3  # Default risk

        delay_rate = (similar_flights['DepDelay'] > 15).mean()
        return delay_rate

    def _analyze_cascading_impact(self, flight_index: int, time_shift: timedelta):
        """Analyze how schedule change affects subsequent flights"""
        flight = self.df.loc[flight_index]
        aircraft_reg = flight['Aircraft_Registration']

        if pd.isna(aircraft_reg) or aircraft_reg not in self.aircraft_networks:
            return {'cascading_flights_affected': 0, 'total_delay_impact': 0}

        # Find subsequent flights on same aircraft
        subsequent_flights = self.df[
            (self.df['Aircraft_Registration'] == aircraft_reg) &
            (self.df['STD_dt'] > flight['STD_dt'])
            ].sort_values('STD_dt')

        affected_flights = 0
        total_impact = 0

        for _, next_flight in subsequent_flights.head(3).iterrows():  # Check next 3 flights
            turnaround_buffer = (next_flight['STD_dt'] - flight['ATA_dt']).total_seconds() / 60

            # If the delay eats into turnaround time
            if time_shift.total_seconds() / 60 > turnaround_buffer:
                affected_flights += 1
                delay_propagation = max(0, time_shift.total_seconds() / 60 - turnaround_buffer)
                total_impact += delay_propagation

        return {
            'cascading_flights_affected': affected_flights,
            'total_delay_impact': total_impact
        }


# ------------------------------
# NEW: Cascading Delay Impact Analyzer
# ------------------------------
class CascadingDelayAnalyzer:
    """
    Analyzes which flights have the biggest cascading impact on schedule delays
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_impact_scores()

    def _calculate_impact_scores(self):
        """Calculate cascading impact potential for each flight"""
        impact_scores = []

        for idx, flight in self.df.iterrows():
            score = self._calculate_single_flight_impact(flight)
            impact_scores.append({
                'flight_index': idx,
                'flight_number': flight['UniqueCarrier'],
                'origin': flight['Origin'],
                'dest': flight['Dest'],
                'departure_time': flight['STD_dt'].strftime('%H:%M') if pd.notna(flight['STD_dt']) else 'N/A',
                'aircraft': flight['Aircraft_Registration'],
                'cascading_impact_score': score,
                'risk_category': self._categorize_risk(score)
            })

        self.impact_df = pd.DataFrame(impact_scores)

    def _calculate_single_flight_impact(self, flight):
        """Calculate impact score for a single flight"""
        score = 0

        # Factor 1: Historical delay probability (40% weight)
        similar_flights = self.df[
            (self.df['Origin'] == flight['Origin']) &
            (self.df['DepHour'] == flight['DepHour'])
            ]
        if len(similar_flights) > 0:
            delay_prob = (similar_flights['DepDelay'] > 15).mean()
            score += delay_prob * 0.4

        # Factor 2: Aircraft utilization intensity (30% weight)
        if pd.notna(flight['Aircraft_Registration']):
            aircraft_flights = self.df[self.df['Aircraft_Registration'] == flight['Aircraft_Registration']]
            daily_flights = len(aircraft_flights) / max(1, aircraft_flights['Date'].nunique())
            utilization_score = min(daily_flights / 8, 1.0)  # Normalize to max 8 flights/day
            score += utilization_score * 0.3

        # Factor 3: Turnaround pressure (20% weight)
        if pd.notna(flight['Turnaround_Time']):
            if flight['Turnaround_Time'] < 60:  # Very tight turnaround
                score += 0.2
            elif flight['Turnaround_Time'] < 90:  # Tight turnaround
                score += 0.15

        # Factor 4: Peak hour operations (10% weight)
        if flight['DepHour'] in [7, 8, 9, 17, 18, 19, 20]:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _categorize_risk(self, score):
        """Categorize cascading risk level"""
        if score >= 0.7:
            return "🔴 HIGH RISK"
        elif score >= 0.5:
            return "🟡 MEDIUM RISK"
        elif score >= 0.3:
            return "🟠 LOW-MEDIUM RISK"
        else:
            return "🟢 LOW RISK"

    def get_highest_impact_flights(self, n=10):
        """Return top N flights with highest cascading impact"""
        return self.impact_df.nlargest(n, 'cascading_impact_score')

    def get_impact_by_airport(self):
        """Get cascading impact analysis by airport"""
        return self.impact_df.groupby('origin').agg({
            'cascading_impact_score': ['mean', 'max', 'count']
        }).round(3)


# ------------------------------
# Enhanced Airport Configuration
# ------------------------------
AIRPORT_CONFIG = {
    "BOM": {
        "name": "Mumbai (Chhatrapati Shivaji)",
        "runways": 2, "capacity_per_hour": 45,
        "peak_hours": [8, 9, 10, 17, 18, 19, 20],
        "weather_delays": 0.15, "ground_congestion": 0.85
    },
    "DEL": {
        "name": "Delhi (Indira Gandhi International)",
        "runways": 4, "capacity_per_hour": 80,
        "peak_hours": [7, 8, 9, 18, 19, 20],
        "weather_delays": 0.18, "ground_congestion": 0.9
    },
    "BLR": {
        "name": "Bangalore (Kempegowda International)",
        "runways": 2, "capacity_per_hour": 50,
        "peak_hours": [8, 9, 17, 18, 19],
        "weather_delays": 0.10, "ground_congestion": 0.8
    },
    "CCU": {
        "name": "Kolkata (Netaji Subhas Chandra Bose)",
        "runways": 2, "capacity_per_hour": 35,
        "peak_hours": [7, 8, 17, 18],
        "weather_delays": 0.14, "ground_congestion": 0.75
    },
    "IXC": {
        "name": "Chandigarh Airport",
        "runways": 1, "capacity_per_hour": 15,
        "peak_hours": [7, 8, 16, 17],
        "weather_delays": 0.10, "ground_congestion": 0.6
    },
    "BBI": {
        "name": "Bhubaneswar (Biju Patnaik)",
        "runways": 1, "capacity_per_hour": 20,
        "peak_hours": [8, 9, 18, 19],
        "weather_delays": 0.12, "ground_congestion": 0.7
    }
}


# ------------------------------
# Enhanced Schedule Optimizer
# ------------------------------
class EnhancedScheduleOptimizer:
    """Enhanced optimizer with cascading analysis and schedule tuning"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.tuner = ScheduleTuner(df)
        self.cascading_analyzer = CascadingDelayAnalyzer(df)
        self._prepare_enhanced_data()

    def _prepare_enhanced_data(self):
        self._calculate_congestion_metrics()
        self._calculate_delay_patterns()
        self._analyze_busiest_slots()

    def _calculate_congestion_metrics(self):
        """Enhanced congestion calculation with real data"""
        congestion_data = []

        for airport in AIRPORT_CONFIG.keys():
            # Get actual flight data for this airport
            airport_flights = self.df[
                (self.df["Origin"] == airport) | (self.df["Dest"] == airport)
                ]

            if airport_flights.empty:
                continue

            # Calculate hourly traffic
            hourly_departures = airport_flights[airport_flights["Origin"] == airport].groupby("DepHour").size()
            hourly_arrivals = airport_flights[airport_flights["Dest"] == airport].groupby("ArrHour").size()

            config = AIRPORT_CONFIG[airport]
            max_capacity = config["capacity_per_hour"]

            for hour in range(24):
                dep_count = hourly_departures.get(hour, 0)
                arr_count = hourly_arrivals.get(hour, 0)
                total_count = dep_count + arr_count

                utilization = total_count / max_capacity if max_capacity > 0 else 0

                congestion_data.append({
                    "Airport": airport,
                    "Hour": hour,
                    "DepartureCount": dep_count,
                    "ArrivalCount": arr_count,
                    "TotalFlights": total_count,
                    "Utilization": min(utilization, 1.0),
                    "CongestionLevel": self._get_enhanced_congestion_level(utilization, config, hour),
                    "IsAvoidableSlot": utilization > 0.8  # Mark as avoidable if >80% capacity
                })

        self.congestion_df = pd.DataFrame(congestion_data)

    def _get_enhanced_congestion_level(self, utilization, config, hour):
        """Enhanced congestion calculation"""
        base_congestion = utilization

        # Peak hour multiplier
        if hour in config["peak_hours"]:
            base_congestion *= 1.4

        # Weather impact
        base_congestion *= (1 + config["weather_delays"])

        # Ground congestion factor
        base_congestion *= config["ground_congestion"]

        return min(base_congestion, 1.0)

    def _calculate_delay_patterns(self):
        """Enhanced delay pattern analysis"""
        delay_patterns = []

        for airport in AIRPORT_CONFIG.keys():
            airport_data = self.df[self.df["Origin"] == airport]
            if len(airport_data) == 0:
                continue

            # Group by hour and calculate comprehensive delay statistics
            hourly_stats = airport_data.groupby("DepHour").agg({
                'DepDelay': ['mean', 'median', 'std', 'count'],
                'ArrDelay': ['mean', 'median', 'std'],
                'Cascading_Risk_Score': 'mean'
            }).fillna(0)

            for hour in range(24):
                if hour in hourly_stats.index:
                    dep_stats = hourly_stats.loc[hour, 'DepDelay']
                    arr_stats = hourly_stats.loc[hour, 'ArrDelay']
                    cascading_risk = hourly_stats.loc[hour, ('Cascading_Risk_Score', 'mean')]

                    delay_patterns.append({
                        "Airport": airport,
                        "Hour": hour,
                        "AvgDepDelay": dep_stats['mean'],
                        "AvgArrDelay": arr_stats['mean'],
                        "DelayVolatility": dep_stats['std'],
                        "FlightCount": dep_stats['count'],
                        "DelayRisk": self._calculate_enhanced_delay_risk(dep_stats, cascading_risk),
                        "CascadingRisk": cascading_risk
                    })

        self.delay_patterns_df = pd.DataFrame(delay_patterns)

    def _calculate_enhanced_delay_risk(self, stats, cascading_risk):
        """Enhanced delay risk calculation including cascading effects"""
        if stats['count'] == 0:
            return 0

        # Base delay risk
        base_risk = (max(0, stats['mean']) + stats['std']) / 100

        # Add cascading risk component
        total_risk = base_risk * 0.7 + cascading_risk * 0.3

        return min(total_risk, 1.0)

    def _analyze_busiest_slots(self):
        """Identify and rank the busiest time slots to avoid"""
        busiest_slots = []

        for airport in AIRPORT_CONFIG.keys():
            airport_congestion = self.congestion_df[self.congestion_df["Airport"] == airport]

            # Rank slots by congestion level
            airport_ranked = airport_congestion.sort_values('CongestionLevel', ascending=False)

            for _, slot in airport_ranked.head(8).iterrows():  # Top 8 busiest slots
                busiest_slots.append({
                    'Airport': airport,
                    'Hour': slot['Hour'],
                    'CongestionLevel': slot['CongestionLevel'],
                    'TotalFlights': slot['TotalFlights'],
                    'Recommendation': self._get_avoidance_recommendation(slot['CongestionLevel']),
                    'AlternativeSlots': self._suggest_alternatives(airport, slot['Hour'])
                })

        self.busiest_slots_df = pd.DataFrame(busiest_slots)

    def _get_avoidance_recommendation(self, congestion_level):
        """Get recommendation text based on congestion level"""
        if congestion_level >= 0.9:
            return "🚫 AVOID - Extreme congestion, high delay risk"
        elif congestion_level >= 0.7:
            return "⚠️  AVOID - Very busy, consider alternatives"
        elif congestion_level >= 0.5:
            return "🟡 CAUTION - Moderately busy, monitor conditions"
        else:
            return "✅ ACCEPTABLE - Low congestion"

    def _suggest_alternatives(self, airport, busy_hour):
        """Suggest alternative time slots"""
        airport_data = self.congestion_df[self.congestion_df["Airport"] == airport]

        # Find hours with low congestion (within 2 hours of busy_hour)
        nearby_hours = [(busy_hour + i) % 24 for i in range(-2, 3) if i != 0]
        alternatives = airport_data[
            airport_data["Hour"].isin(nearby_hours) &
            (airport_data["CongestionLevel"] < 0.5)
            ].sort_values('CongestionLevel')

        if not alternatives.empty:
            alt_hours = alternatives.head(2)["Hour"].tolist()
            return f"Consider {alt_hours[0]:02d}:00" + (f" or {alt_hours[1]:02d}:00" if len(alt_hours) > 1 else "")
        else:
            return "No nearby alternatives available"

    def find_optimal_slots_enhanced(self, airport: str, max_slots: int = 10):
        """Enhanced optimal slot finding with comprehensive scoring"""
        recommendations = []

        airport_congestion = self.congestion_df[self.congestion_df["Airport"] == airport]
        airport_delays = self.delay_patterns_df[self.delay_patterns_df["Airport"] == airport]

        for hour in range(24):
            congestion_data = airport_congestion[airport_congestion["Hour"] == hour]
            delay_data = airport_delays[airport_delays["Hour"] == hour]

            if not congestion_data.empty and not delay_data.empty:
                congestion = congestion_data.iloc[0]
                delay_info = delay_data.iloc[0]

                # Multi-factor scoring
                congestion_score = congestion["CongestionLevel"] * 0.4
                delay_score = delay_info["DelayRisk"] * 0.3
                cascading_score = delay_info["CascadingRisk"] * 0.3

                total_score = congestion_score + delay_score + cascading_score

                recommendations.append({
                    "Hour": hour,
                    "Score": total_score,
                    "CongestionLevel": congestion["CongestionLevel"],
                    "DelayRisk": delay_info["DelayRisk"],
                    "CascadingRisk": delay_info["CascadingRisk"],
                    "TotalFlights": congestion["TotalFlights"],
                    "Recommendation": self._get_enhanced_recommendation_text(total_score, hour),
                    "OptimalityRating": self._get_optimality_rating(total_score)
                })

        recommendations.sort(key=lambda x: x["Score"])
        return recommendations[:max_slots]

    def _get_enhanced_recommendation_text(self, score, hour):
        """Enhanced recommendation text with detailed insights"""
        if score < 0.25:
            return f"⭐ EXCELLENT slot at {hour:02d}:00 - Optimal conditions, minimal delays"
        elif score < 0.4:
            return f"✅ GOOD slot at {hour:02d}:00 - Low congestion, acceptable delay risk"
        elif score < 0.6:
            return f"🟡 MODERATE slot at {hour:02d}:00 - Some congestion, moderate delay risk"
        elif score < 0.75:
            return f"⚠️  BUSY slot at {hour:02d}:00 - High traffic, increased delay risk"
        else:
            return f"🚫 AVOID slot at {hour:02d}:00 - Peak congestion, high delay probability"

    def _get_optimality_rating(self, score):
        """Convert score to star rating"""
        if score < 0.25:
            return "⭐⭐⭐⭐⭐"
        elif score < 0.4:
            return "⭐⭐⭐⭐"
        elif score < 0.6:
            return "⭐⭐⭐"
        elif score < 0.75:
            return "⭐⭐"
        else:
            return "⭐"


# ------------------------------
# Enhanced NLP Query Processing
# ------------------------------
ENHANCED_COLUMN_DESCRIPTIONS = {
    "ArrDelay": "difference between scheduled and actual arrival time in minutes",
    "DepDelay": "difference between scheduled and actual departure time in minutes",
    "DepHour": "actual departure hour (0-23)",
    "ArrHour": "actual arrival hour (0-23)",
    "UniqueCarrier": "airline carrier code",
    "Aircraft_Registration": "aircraft registration number for tracking rotations",
    "AirTime": "time spent airborne in minutes",
    "Origin": "origin airport code",
    "Dest": "destination airport code",
    "Distance": "estimated distance flown (miles)",
    "Month": "month number 1..12",
    "DayOfWeek": "day of week as integer (1=Mon, 7=Sun)",
    "Turnaround_Time": "minutes between arrival and next departure for same aircraft",
    "Cascading_Risk_Score": "probability that this flight will cause delays to other flights"
}

ENHANCED_OPERATIONS = {
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
    "congestion": "congestion analysis",
    "busiest": "busiest time slots to avoid",
    "tune": "tune schedule time and see delay impact",
    "cascade": "analyze cascading delay impacts",
    "simulate": "simulate schedule changes",
    "impact": "flights with biggest cascading impact"
}


# ------------------------------
# Enhanced ML Models with Cascading Features
# ------------------------------
@st.cache_resource(show_spinner=True)
def train_enhanced_models(df: pd.DataFrame):
    """Train enhanced models with cascading delay features"""
    df = df.copy()

    # Enhanced Feature Engineering
    df["IsPeakHour"] = df.apply(
        lambda row: 1 if row['Origin'] in AIRPORT_CONFIG and row['DepHour'] in
                         AIRPORT_CONFIG[row['Origin']]['peak_hours'] else 0, axis=1
    )
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in [6, 7] else 0)
    df["ArrDelayBinary"] = (df["ArrDelay"].fillna(0) > 15).astype(int)

    # NEW: Cascading delay features
    df["HasQuickTurnaround"] = df["Is_Quick_Turnaround"].fillna(0)
    df["TurnaroundPressure"] = (df["Turnaround_Time"].fillna(120) < 90).astype(int)

    # Enhanced feature set
    enhanced_feats = [
        "Month", "DayOfWeek", "DepHour", "Distance", "AirTime",
        "IsPeakHour", "IsWeekend", "HasQuickTurnaround", "TurnaroundPressure",
        "Cascading_Risk_Score"
    ]

    available_feats = [f for f in enhanced_feats if f in df.columns]
    X = df[available_feats].fillna(df[available_feats].median())
    y = df["ArrDelayBinary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Enhanced Random Forest
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1,
            max_depth=12, min_samples_split=5
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_pipe.predict(X_test))

    # Enhanced second model
    if XGBOOST_AVAILABLE:
        model_b = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            n_estimators=150, random_state=42, max_depth=6,
            learning_rate=0.1, subsample=0.8
        )
        model_b_name = "XGBoost"
    else:
        model_b = GradientBoostingClassifier(
            n_estimators=150, random_state=42, max_depth=6,
            learning_rate=0.1, subsample=0.8
        )
        model_b_name = "GradientBoosting"

    model_b.fit(X_train, y_train)
    model_b_acc = accuracy_score(y_test, model_b.predict(X_test))

    return {
        "rf_pipe": rf_pipe,
        "model_b": model_b,
        "meta": {
            "features": available_feats,
            "rf_acc": float(rf_acc),
            "model_b_acc": float(model_b_acc),
            "model_b_name": model_b_name
        }
    }


# ------------------------------
# Enhanced Visualization Functions
# ------------------------------
def create_cascading_impact_chart(analyzer: CascadingDelayAnalyzer):
    """Create interactive chart showing cascading impact analysis"""
    top_flights = analyzer.get_highest_impact_flights(15)

    fig = px.bar(
        top_flights,
        x='cascading_impact_score',
        y='flight_number',
        color='cascading_impact_score',
        color_continuous_scale='Reds',
        title='Top 15 Flights with Highest Cascading Impact',
        labels={'cascading_impact_score': 'Cascading Impact Score', 'flight_number': 'Flight'}
    )
    fig.update_layout(height=600)
    return fig


def create_schedule_tuning_visualization(tuning_results):
    """Create visualization for schedule tuning results"""
    if not tuning_results:
        return None

    # Create before/after comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Delay Risk Comparison', 'Cascading Impact',
            'Time Shift Impact', 'Overall Recommendation'
        ),
        specs=[[{"type": "bar"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "indicator"}]]
    )

    # Delay risk comparison
    fig.add_trace(
        go.Bar(
            x=['Original', 'New Schedule'],
            y=[tuning_results['original_delay_risk'], tuning_results['new_delay_risk']],
            marker_color=['red', 'green'] if tuning_results['risk_change'] < 0 else ['red', 'orange'],
            name='Delay Risk'
        ),
        row=1, col=1
    )

    # Cascading impact indicator
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=tuning_results.get('cascading_flights_affected', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Flights Affected"},
            gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "darkblue"}}
        ),
        row=1, col=2
    )

    fig.update_layout(height=600, title="Schedule Tuning Impact Analysis")
    return fig


def create_busiest_slots_heatmap(optimizer: EnhancedScheduleOptimizer):
    """Create heatmap showing busiest time slots to avoid"""
    pivot_data = optimizer.congestion_df.pivot(
        index='Airport', columns='Hour', values='CongestionLevel'
    )

    fig = px.imshow(
        pivot_data,
        color_continuous_scale='RdYlBu_r',
        title='Airport Congestion Heatmap - Red Areas to Avoid',
        labels={'color': 'Congestion Level'}
    )
    fig.update_layout(height=500)
    return fig


# ------------------------------
# Enhanced Main Application
# ------------------------------
def main():
    st.set_page_config(
        page_title="Enhanced Flight Schedule Optimizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Flight Schedule Optimizer")
    st.markdown("*Complete implementation of all requirements with advanced AI analytics*")

    # Load and process data
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

    # Enhanced data cleaning
    df = clean_flight_data_enhanced(df_raw)

    # Initialize enhanced engines
    optimizer = EnhancedScheduleOptimizer(df)

    # Load NLP models
    with st.spinner("Loading Enhanced NLP Models..."):
        sbert = load_sbert()
        col_keys = list(ENHANCED_COLUMN_DESCRIPTIONS.keys())
        col_emb = sbert.encode([f"{k}: {v}" for k, v in ENHANCED_COLUMN_DESCRIPTIONS.items()], convert_to_tensor=True)
        op_keys = list(ENHANCED_OPERATIONS.keys())
        op_emb = sbert.encode([f"{k}: {v}" for k, v in ENHANCED_OPERATIONS.items()], convert_to_tensor=True)

    # Train enhanced ML models
    if "enhanced_artifacts" not in st.session_state:
        with st.spinner("Training Enhanced ML Models with Cascading Features..."):
            st.session_state["enhanced_artifacts"] = train_enhanced_models(df)

    artifacts = st.session_state["enhanced_artifacts"]

    # --- ENHANCED SIDEBAR ---
    with st.sidebar:


        st.header("Features")
        st.markdown("""
        - **Aircraft Rotation Tracking**
        - **Cascading Delay Analysis** 
        - **Interactive Schedule Tuning**
        - **Advanced ML Predictions**
        - **Real-time Impact Simulation**
        """)

        st.header("Supported Airports")
        for code, config in AIRPORT_CONFIG.items():
            st.markdown(f"**{code}** - {config['name']}")

    # --- ENHANCED METRICS DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Flights", f"{len(df):,}")
    with col2:
        st.metric("Airports", len(AIRPORT_CONFIG))
    with col3:
        avg_cascade_risk = df['Cascading_Risk_Score'].mean()
        st.metric("Avg Cascading Risk", f"{avg_cascade_risk:.2f}")
    with col4:
        high_risk_flights = (df['Cascading_Risk_Score'] > 0.7).sum()
        st.metric("High Risk Flights", high_risk_flights)

    # --- MODEL PERFORMANCE ---
    with st.expander("🤖 Enhanced Model Performance"):
        meta = artifacts["meta"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RandomForest Accuracy", f"{meta['rf_acc']:.3f}")
        with col2:
            st.metric(f"{meta['model_b_name']} Accuracy", f"{meta['model_b_acc']:.3f}")
        st.caption(f"**Enhanced Features:** {', '.join(meta['features'])}")

    st.markdown("---")

    # --- ENHANCED QUERY INTERFACE ---
    st.markdown("### Ask Questions in English Language")

    # Quick action buttons
    st.markdown("**Quick Actions:**")
    quick_actions = st.columns(4)
    with quick_actions[0]:
        if st.button("Find Best Times"):
            user_q = "find optimal departure slots from Mumbai"
    with quick_actions[1]:
        if st.button("Show Busiest Times"):
            user_q = "show busiest time slots to avoid"
    with quick_actions[2]:
        if st.button("Tune Schedule"):
            user_q = "tune schedule time and show impact"
    with quick_actions[3]:
        if st.button("Cascading Impact"):
            user_q = "analyze cascading delay impacts"

    user_q = st.text_input(
        "Query",
        placeholder="e.g., 'Which flights cause the most cascading delays?' or 'Tune departure time for flight and show impact'"
    )

    # --- ENHANCED EXAMPLE QUERIES ---
    with st.expander("Enhanced Example Queries"):
        enhanced_examples = {
            "Optimization": [
                "Find the best departure times from Mumbai to avoid delays",
                "Show optimal landing slots at Delhi airport",
                "When should I schedule flights to minimize cascading impacts?"
            ],
            "Avoidance": [
                "What are the busiest time slots to avoid at BOM?",
                "Show congested hours across all airports",
                "Which time periods have highest delay risks?"
            ],
            "Schedule Tuning": [
                "Tune departure time by +30 minutes and show impact",
                "Simulate moving flight to 2 hours later",
                "What happens if I reschedule this flight?"
            ],
            "Cascading Analysis": [
                "Which flights have the biggest cascading impact?",
                "Show aircraft with highest delay propagation risk",
                "Analyze cascade effects for quick turnaround flights"
            ],
            "Predictions": [
                "Predict delay probability for Sunday morning flight",
                "What's the risk of delays for peak hour departures?",
                "Show delay trends by hour and aircraft type"
            ]
        }

        for category, queries in enhanced_examples.items():
            st.markdown(f"**{category}:**")
            for query in queries:
                if st.button(query, key=f"example_{query}"):
                    user_q = query
                    st.rerun()

    # --- ENHANCED QUERY PROCESSING ---
    if user_q:
        with st.spinner("🔍 Processing your query with enhanced AI..."):
            # Enhanced query mapping
            mapping = map_enhanced_query(user_q, sbert, col_keys, col_emb, op_keys, op_emb, df)

        st.info(f"**Detected Intent:** {mapping['intent'].upper()}")

        # Route to appropriate enhanced handler
        if mapping["intent"] == "optimize":
            handle_optimization_query(mapping, optimizer)
        elif mapping["intent"] == "busiest":
            handle_busiest_slots_query(optimizer)
        elif mapping["intent"] == "tune":
            handle_schedule_tuning_query(df, optimizer)
        elif mapping["intent"] == "cascade":
            handle_cascading_analysis_query(optimizer)
        elif mapping["intent"] == "predict":
            handle_prediction_query(mapping, artifacts, df)
        else:
            handle_statistics_query(mapping, df)


# ------------------------------
# Enhanced Query Handlers
# ------------------------------
def map_enhanced_query(query: str, sbert_model, col_keys, col_emb, op_keys, op_emb, df):
    """Enhanced query mapping with new intents"""
    q_emb = sbert_model.encode(query, convert_to_tensor=True)

    # Calculate similarities
    col_scores = util.pytorch_cos_sim(q_emb, col_emb)[0].cpu().tolist()
    col_with_scores = sorted([(col_keys[i], float(s)) for i, s in enumerate(col_scores)], key=lambda x: -x[1])
    top_cols = [c for c, s in col_with_scores[:3] if s > 0.28]

    op_scores = util.pytorch_cos_sim(q_emb, op_emb)[0].cpu().tolist()
    op_with_scores = sorted([(op_keys[i], float(s)) for i, s in enumerate(op_scores)], key=lambda x: -x[1])
    top_op = op_with_scores[0][0] if op_with_scores and op_with_scores[0][1] > 0.22 else None

    # Enhanced intent detection
    intent = "statistics"  # default
    q_lower = query.lower()

    if any(kw in q_lower for kw in ["optimize", "best time", "optimal", "recommend"]):
        intent = "optimize"
    elif any(kw in q_lower for kw in ["busiest", "avoid", "congested", "busy"]):
        intent = "busiest"
    elif any(kw in q_lower for kw in ["tune", "reschedule", "move", "shift", "simulate"]):
        intent = "tune"
    elif any(kw in q_lower for kw in ["cascade", "cascading", "impact", "propagation", "chain"]):
        intent = "cascade"
    elif any(kw in q_lower for kw in ["predict", "probability", "risk", "chance"]):
        intent = "predict"

    return {
        "intent": intent,
        "top_cols": top_cols,
        "top_op": top_op,
        "query_text": query
    }


def handle_optimization_query(mapping, optimizer):
    """Handle optimization queries"""
    st.markdown("### Optimal Schedule Recommendations")

    # Airport selection
    airports = list(AIRPORT_CONFIG.keys())
    selected_airport = st.selectbox("Select Airport", airports, key="opt_airport")

    # Get recommendations
    recommendations = optimizer.find_optimal_slots_enhanced(selected_airport)

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs([" Recommendations", " Analysis Chart", "Heatmap"])

    with tab1:
        st.markdown(f"####  Top Optimal Slots for **{selected_airport}**")

        for i, rec in enumerate(recommendations[:5]):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{rec['OptimalityRating']} {rec['Hour']:02d}:00-{rec['Hour'] + 1:02d}:00**")
                    st.caption(rec['Recommendation'])
                with col2:
                    st.metric("Score", f"{rec['Score']:.2f}")
                with col3:
                    st.metric("Traffic", int(rec['TotalFlights']))

    with tab2:
        # Interactive chart
        rec_df = pd.DataFrame(recommendations)
        fig = px.bar(
            rec_df, x='Hour', y='Score',
            color='Score', color_continuous_scale='RdYlGn_r',
            title=f'Hourly Optimization Scores for {selected_airport}',
            hover_data=['CongestionLevel', 'DelayRisk', 'CascadingRisk']
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = create_busiest_slots_heatmap(optimizer)
        st.plotly_chart(fig, use_container_width=True)


def handle_busiest_slots_query(optimizer):
    """Handle busiest time slots queries"""
    st.markdown("### Busiest Time Slots to Avoid")

    # Show busiest slots table
    if hasattr(optimizer, 'busiest_slots_df'):
        busiest = optimizer.busiest_slots_df.head(15)

        st.markdown("#### ⚠️ Top 15 Busiest Slots Across All Airports")

        for _, slot in busiest.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{slot['Airport']} - {slot['Hour']:02d}:00**")
                with col2:
                    congestion_color = "🔴" if slot['CongestionLevel'] > 0.8 else "🟡" if slot[
                                                                                            'CongestionLevel'] > 0.5 else "🟢"
                    st.markdown(f"{congestion_color} {slot['CongestionLevel']:.2f}")
                with col3:
                    st.metric("Flights", int(slot['TotalFlights']))
                with col4:
                    st.caption(slot['Recommendation'])
                    if slot['AlternativeSlots'] != "No nearby alternatives available":
                        st.success(f"Alternative: {slot['AlternativeSlots']}")


def handle_schedule_tuning_query(df, optimizer):
    """Handle schedule tuning simulation"""
    st.markdown("### Interactive Schedule Tuning Simulator")

    with st.form("schedule_tuning_form"):
        st.markdown("#### Select Flight to Tune")

        # Flight selection
        sample_flights = df.head(20)[['UniqueCarrier', 'Origin', 'Dest', 'STD']].reset_index()
        flight_options = [
            f"{row['UniqueCarrier']} {row['Origin']}→{row['Dest']} @ {row['STD']}"
            for _, row in sample_flights.iterrows()
        ]

        selected_flight_idx = st.selectbox("Choose Flight", range(len(flight_options)),
                                           format_func=lambda x: flight_options[x])
        flight_index = sample_flights.iloc[selected_flight_idx].name

        # Time adjustment
        time_shift = st.slider("Time Shift (minutes)", -180, 180, 0, 15,
                               help="Positive = later departure, Negative = earlier departure")

        submitted = st.form_submit_button("Simulate Impact")

    if submitted and time_shift != 0:
        with st.spinner("🔄 Simulating schedule change impact..."):
            results = optimizer.tuner.simulate_schedule_change(flight_index, time_shift)

        if results:
            st.success("Simulation Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Departure", results['original_departure'])
                st.metric("New Departure", results['new_departure'])
            with col2:
                st.metric("Risk Change", f"{results['risk_change']:.3f}",
                          delta=f"{results['risk_change']:.3f}")
            with col3:
                st.metric("Affected Flights", results.get('cascading_flights_affected', 0))
                st.metric("Total Impact (min)", f"{results.get('total_delay_impact', 0):.0f}")

            # Visualization
            fig = create_schedule_tuning_visualization(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to simulate - flight not found or insufficient data")


def handle_cascading_analysis_query(optimizer):
    """Handle cascading impact analysis"""
    st.markdown("### Cascading Delay Impact Analysis")

    analyzer = optimizer.cascading_analyzer

    # Top impact flights
    top_impact = analyzer.get_highest_impact_flights(15)

    tab1, tab2, tab3 = st.tabs(["High Impact Flights", "Impact Chart", "By Airport"])

    with tab1:
        st.markdown("####Flights with Highest Cascading Impact")

        for _, flight in top_impact.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{flight['flight_number']}** {flight['origin']}→{flight['dest']}")
                    st.caption(f"Departure: {flight['departure_time']}")
                with col2:
                    st.markdown(flight['risk_category'])
                with col3:
                    st.metric("Impact Score", f"{flight['cascading_impact_score']:.2f}")
                with col4:
                    if flight['aircraft']:
                        st.caption(f"Aircraft: {flight['aircraft']}")

    with tab2:
        fig = create_cascading_impact_chart(analyzer)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        airport_impact = analyzer.get_impact_by_airport()
        st.markdown("#### Cascading Impact by Airport")
        st.dataframe(airport_impact)


def handle_prediction_query(mapping, artifacts, df):
    """Handle prediction queries"""
    st.markdown("### Advanced Delay Prediction")

    with st.form("enhanced_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin = st.selectbox("Origin Airport", list(AIRPORT_CONFIG.keys()))
            hour = st.slider("Departure Hour", 0, 23, 8)
            month = st.slider("Month", 1, 12, 7)
        with col2:
            dow = st.slider("Day of Week (1=Mon)", 1, 7, 3)
            is_quick_turnaround = st.checkbox("Quick Turnaround (<90 min)")

        submitted = st.form_submit_button(" Predict Delay Risk")

    if submitted:
        # Create prediction input
        input_features = pd.DataFrame([{
            "Month": month,
            "DayOfWeek": dow,
            "DepHour": hour,
            "Distance": df['Distance'].median(),
            "AirTime": df['AirTime'].median(),
            "IsPeakHour": 1 if hour in AIRPORT_CONFIG.get(origin, {}).get("peak_hours", []) else 0,
            "IsWeekend": 1 if dow > 5 else 0,
            "HasQuickTurnaround": 1 if is_quick_turnaround else 0,
            "TurnaroundPressure": 1 if is_quick_turnaround else 0,
            "Cascading_Risk_Score": 0.5 if is_quick_turnaround else 0.3
        }])

        # Make prediction
        prob = ensemble_predict_enhanced(artifacts["rf_pipe"], artifacts["model_b"], artifacts["meta"], input_features)

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Delay Probability", f"{prob:.1%}")
        with col2:
            risk_level = "🔴 HIGH" if prob > 0.6 else "🟡 MEDIUM" if prob > 0.3 else "🟢 LOW"
            st.metric("Risk Level", risk_level)
        with col3:
            confidence = min(95, 70 + (prob * 25))  # Higher confidence for extreme predictions
            st.metric("Confidence", f"{confidence:.0f}%")

        # Risk factors breakdown
        with st.expander("🔍 Risk Factors Analysis"):
            factors = []
            if hour in AIRPORT_CONFIG.get(origin, {}).get("peak_hours", []):
                factors.append("Peak hour operation")
            if dow > 5:
                factors.append("Weekend operations")
            if is_quick_turnaround:
                factors.append("⏱Quick turnaround pressure")
            if month in [6, 7, 8]:  # Summer months
                factors.append("High season traffic")

            if factors:
                st.markdown("**Contributing Risk Factors:**")
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.success("No major risk factors identified")


def handle_statistics_query(mapping, df):
    """Handle statistical queries"""
    st.markdown("###Statistical Analysis")

    # Apply filters based on query
    filtered_df = df.copy()

    # Simple filtering based on top columns
    if mapping["top_cols"]:
        target_col = mapping["top_cols"][0]
        if target_col in df.columns:
            st.markdown(f"#### Analysis of: **{target_col}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{df[target_col].mean():.2f}")
                st.metric("Median", f"{df[target_col].median():.2f}")
            with col2:
                st.metric("Max", f"{df[target_col].max():.2f}")
                st.metric("Std Dev", f"{df[target_col].std():.2f}")

            # Distribution chart
            fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Enhanced Helper Functions
# ------------------------------
def ensemble_predict_enhanced(rf_pipe, model_b, meta, input_row: pd.DataFrame):
    """Enhanced ensemble prediction with error handling"""
    try:
        available_features = [f for f in meta["features"] if f in input_row.columns]
        Xrow = input_row[available_features].fillna(0)

        rf_prob = rf_pipe.predict_proba(Xrow)[:, 1] if hasattr(rf_pipe, 'predict_proba') else [0.5]
        b_prob = model_b.predict_proba(Xrow)[:, 1] if hasattr(model_b, 'predict_proba') else [0.5]

        # Weighted ensemble (RF gets higher weight due to typically better performance)
        avg_prob = (rf_prob * 0.6 + b_prob * 0.4)
        return float(avg_prob[0])
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 0.5  # Default probability


def load_sbert(name="all-MiniLM-L6-v2"):
    """Load sentence transformer model with caching"""
    try:
        return SentenceTransformer(name)
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        st.stop()


# ------------------------------
# Application Entry Point
# ------------------------------
if __name__ == "__main__":
    # Set page config first
    st.set_page_config(
        page_title="Enhanced Flight Schedule Optimizer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Run main application
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.markdown("Please check your data file and try again.")


# ------------------------------
# Additional Utility Functions for Enhanced Features
# ------------------------------
def export_analysis_results(optimizer, analyzer):
    """Export analysis results to downloadable formats"""
    results = {
        "optimal_slots": optimizer.find_optimal_slots_enhanced("BOM"),
        "busiest_slots": optimizer.busiest_slots_df.to_dict('records') if hasattr(optimizer,
                                                                                  'busiest_slots_df') else [],
        "high_impact_flights": analyzer.get_highest_impact_flights().to_dict('records'),
        "airport_impact_summary": analyzer.get_impact_by_airport().to_dict()
    }
    return results


def generate_recommendations_report(optimizer):
    """Generate a comprehensive recommendations report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_airports_analyzed": len(AIRPORT_CONFIG),
            "optimization_recommendations": [],
            "avoidance_recommendations": [],
            "high_risk_flights": []
        }
    }

    # Add recommendations for each airport
    for airport in AIRPORT_CONFIG.keys():
        optimal_slots = optimizer.find_optimal_slots_enhanced(airport, max_slots=3)
        report["summary"]["optimization_recommendations"].append({
            "airport": airport,
            "top_slots": [f"{slot['Hour']:02d}:00" for slot in optimal_slots],
            "best_score": optimal_slots[0]["Score"] if optimal_slots else None
        })

    return report


# ------------------------------
# Performance Monitoring
# ------------------------------
def monitor_query_performance():
    """Monitor and log query performance metrics"""
    if "query_stats" not in st.session_state:
        st.session_state.query_stats = {
            "total_queries": 0,
            "optimization_queries": 0,
            "prediction_queries": 0,
            "tuning_queries": 0,
            "cascading_queries": 0
        }

    return st.session_state.query_stats


# ------------------------------
# Data Quality Checks
# ------------------------------
def validate_data_quality(df):
    """Perform data quality validation and report issues"""
    quality_report = {
        "total_records": len(df),
        "missing_data": {},
        "data_anomalies": [],
        "quality_score": 100
    }

    # Check for missing critical data
    critical_columns = ['DepDelay', 'ArrDelay', 'Origin', 'Dest', 'STD_dt', 'ATD_dt']
    for col in critical_columns:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            quality_report["missing_data"][col] = missing_pct
            if missing_pct > 10:
                quality_report["quality_score"] -= 10

    # Check for data anomalies
    if 'DepDelay' in df.columns:
        extreme_delays = (df['DepDelay'] > 300).sum()  # > 5 hours
        if extreme_delays > 0:
            quality_report["data_anomalies"].append(f"{extreme_delays} flights with extreme delays (>5h)")

    return quality_report


# ------------------------------
# Advanced Analytics Extensions
# ------------------------------
def calculate_network_effects(df):
    """Calculate network-wide delay propagation effects"""
    # Build flight network graph
    network = nx.DiGraph()

    # Add flights as nodes
    for idx, flight in df.iterrows():
        network.add_node(idx,
                         flight=flight['UniqueCarrier'],
                         origin=flight['Origin'],
                         dest=flight['Dest'],
                         delay=flight['DepDelay'])

    # Add edges for aircraft rotations
    aircraft_flights = df.groupby('Aircraft_Registration')
    for aircraft, flights in aircraft_flights:
        if len(flights) < 2:
            continue
        flights_sorted = flights.sort_values('STD_dt')
        for i in range(len(flights_sorted) - 1):
            current = flights_sorted.iloc[i].name
            next_flight = flights_sorted.iloc[i + 1].name
            turnaround = flights_sorted.iloc[i]['Turnaround_Time']
            network.add_edge(current, next_flight, turnaround_time=turnaround)

    return network


def predict_seasonal_patterns(df):
    """Predict seasonal delay patterns"""
    monthly_delays = df.groupby('Month')['DepDelay'].agg(['mean', 'std', 'count'])
    seasonal_patterns = {
        "peak_delay_months": monthly_delays['mean'].nlargest(3).index.tolist(),
        "low_delay_months": monthly_delays['mean'].nsmallest(3).index.tolist(),
        "most_volatile_months": monthly_delays['std'].nlargest(3).index.tolist()
    }
    return seasonal_patterns


