import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Custom CSS for modern look (unchanged)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 36px;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #424242;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# List of cities in Punjab (unchanged)
punjab_cities = [
    "Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda", "Hoshiarpur",
    "Mohali", "Pathankot", "Moga", "Firozpur", "Kapurthala", "Sangrur", "Gurdaspur",
    "Rupnagar", "Faridkot", "Muktsar", "Fazilka", "Barnala", "Mansa", "Tarn Taran"
]

# Predefined datasets for each city (100 samples per city, simplified for brevity)
city_datasets = {
    "Amritsar": pd.DataFrame({
        "City": ["Amritsar"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [12.5 + i * 0.247 for i in range(100)],  # 12.5 to 37.2
        "Climate_Score": [0.75 - (i % 50) * 0.015 for i in range(100)],  # 0.0 to 0.75
        "Total_Water_Demand_m3": [15000 + (i % 50) * 300 for i in range(100)],  # 15000 to 29700
        "Aquifer_Recharge_Rate": [1.2 - (i % 50) * 0.02 for i in range(100)],  # 0.2 to 1.2
        "Extraction_Limit_m3": [2000 - (i % 50) * 30 for i in range(100)]  # 500 to 2000
    }),
    "Ludhiana": pd.DataFrame({
        "City": ["Ludhiana"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [15.0 + i * 0.3 for i in range(100)],  # 15.0 to 44.7
        "Climate_Score": [0.8 - (i % 40) * 0.02 for i in range(100)],  # 0.0 to 0.8
        "Total_Water_Demand_m3": [16000 + (i % 60) * 250 for i in range(100)],  # 16000 to 30750
        "Aquifer_Recharge_Rate": [1.0 + (i % 50) * 0.015 for i in range(100)],  # 1.0 to 1.75
        "Extraction_Limit_m3": [1800 + (i % 50) * 20 for i in range(100)]  # 1800 to 2780
    }),
    "Jalandhar": pd.DataFrame({
        "City": ["Jalandhar"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [10.0 + i * 0.2 for i in range(100)],  # 10.0 to 29.8
        "Climate_Score": [0.65 + (i % 60) * 0.006 for i in range(100)],  # 0.65 to 0.95
        "Total_Water_Demand_m3": [14000 + (i % 50) * 200 for i in range(100)],  # 14000 to 23800
        "Aquifer_Recharge_Rate": [0.8 + (i % 50) * 0.01 for i in range(100)],  # 0.8 to 1.3
        "Extraction_Limit_m3": [2100 - (i % 50) * 25 for i in range(100)]  # 850 to 2100
    }),
    "Patiala": pd.DataFrame({
        "City": ["Patiala"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [20.0 + i * 0.15 for i in range(100)],  # 20.0 to 34.85
        "Climate_Score": [0.5 + (i % 50) * 0.01 for i in range(100)],  # 0.5 to 1.0
        "Total_Water_Demand_m3": [17000 + (i % 50) * 260 for i in range(100)],  # 17000 to 29800
        "Aquifer_Recharge_Rate": [0.9 - (i % 50) * 0.01 for i in range(100)],  # 0.4 to 0.9
        "Extraction_Limit_m3": [1900 + (i % 50) * 15 for i in range(100)]  # 1900 to 2635
    }),
    "Bathinda": pd.DataFrame({
        "City": ["Bathinda"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [25.0 + i * 0.1 for i in range(100)],  # 25.0 to 34.9
        "Climate_Score": [0.4 - (i % 50) * 0.008 for i in range(100)],  # 0.0 to 0.4
        "Total_Water_Demand_m3": [18000 + (i % 50) * 240 for i in range(100)],  # 18000 to 29800
        "Aquifer_Recharge_Rate": [0.7 + (i % 50) * 0.012 for i in range(100)],  # 0.7 to 1.3
        "Extraction_Limit_m3": [1700 - (i % 50) * 20 for i in range(100)]  # 700 to 1700
    }),
    "Hoshiarpur": pd.DataFrame({
        "City": ["Hoshiarpur"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [8.0 + i * 0.22 for i in range(100)],  # 8.0 to 29.8
        "Climate_Score": [0.85 + (i % 50) * 0.003 for i in range(100)],  # 0.85 to 1.0
        "Total_Water_Demand_m3": [13000 + (i % 50) * 220 for i in range(100)],  # 13000 to 23800
        "Aquifer_Recharge_Rate": [1.1 - (i % 50) * 0.008 for i in range(100)],  # 0.7 to 1.1
        "Extraction_Limit_m3": [2200 + (i % 50) * 10 for i in range(100)]  # 2200 to 2690
    }),
    "Mohali": pd.DataFrame({
        "City": ["Mohali"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [18.0 + i * 0.18 for i in range(100)],  # 18.0 to 35.82
        "Climate_Score": [0.6 + (i % 50) * 0.008 for i in range(100)],  # 0.6 to 1.0
        "Total_Water_Demand_m3": [16000 + (i % 50) * 280 for i in range(100)],  # 16000 to 29800
        "Aquifer_Recharge_Rate": [0.95 + (i % 50) * 0.01 for i in range(100)],  # 0.95 to 1.45
        "Extraction_Limit_m3": [2000 - (i % 50) * 15 for i in range(100)]  # 1250 to 2000
    }),
    "Pathankot": pd.DataFrame({
        "City": ["Pathankot"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [12.0 + i * 0.2 for i in range(100)],  # 12.0 to 31.8
        "Climate_Score": [0.7 - (i % 50) * 0.01 for i in range(100)],  # 0.2 to 0.7
        "Total_Water_Demand_m3": [14000 + (i % 50) * 200 for i in range(100)],  # 14000 to 23800
        "Aquifer_Recharge_Rate": [1.0 - (i % 50) * 0.012 for i in range(100)],  # 0.4 to 1.0
        "Extraction_Limit_m3": [2100 + (i % 50) * 12 for i in range(100)]  # 2100 to 2688
    }),
    "Moga": pd.DataFrame({
        "City": ["Moga"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [22.0 + i * 0.12 for i in range(100)],  # 22.0 to 33.88
        "Climate_Score": [0.45 + (i % 50) * 0.01 for i in range(100)],  # 0.45 to 0.95
        "Total_Water_Demand_m3": [17000 + (i % 50) * 260 for i in range(100)],  # 17000 to 29800
        "Aquifer_Recharge_Rate": [0.8 + (i % 50) * 0.008 for i in range(100)],  # 0.8 to 1.2
        "Extraction_Limit_m3": [1800 - (i % 50) * 18 for i in range(100)]  # 900 to 1800
    }),
    "Firozpur": pd.DataFrame({
        "City": ["Firozpur"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [30.0 + i * 0.1 for i in range(100)],  # 30.0 to 39.9
        "Climate_Score": [0.3 - (i % 50) * 0.006 for i in range(100)],  # 0.0 to 0.3
        "Total_Water_Demand_m3": [19000 + (i % 50) * 220 for i in range(100)],  # 19000 to 29800
        "Aquifer_Recharge_Rate": [0.6 + (i % 50) * 0.01 for i in range(100)],  # 0.6 to 1.1
        "Extraction_Limit_m3": [1600 + (i % 50) * 14 for i in range(100)]  # 1600 to 2290
    }),
    "Kapurthala": pd.DataFrame({
        "City": ["Kapurthala"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [14.0 + i * 0.16 for i in range(100)],  # 14.0 to 29.84
        "Climate_Score": [0.8 + (i % 50) * 0.004 for i in range(100)],  # 0.8 to 1.0
        "Total_Water_Demand_m3": [15000 + (i % 50) * 240 for i in range(100)],  # 15000 to 26800
        "Aquifer_Recharge_Rate": [1.05 - (i % 50) * 0.01 for i in range(100)],  # 0.55 to 1.05
        "Extraction_Limit_m3": [2000 + (i % 50) * 10 for i in range(100)]  # 2000 to 2490
    }),
    "Sangrur": pd.DataFrame({
        "City": ["Sangrur"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [26.0 + i * 0.14 for i in range(100)],  # 26.0 to 39.86
        "Climate_Score": [0.5 - (i % 50) * 0.01 for i in range(100)],  # 0.0 to 0.5
        "Total_Water_Demand_m3": [18000 + (i % 50) * 250 for i in range(100)],  # 18000 to 30250
        "Aquifer_Recharge_Rate": [0.85 + (i % 50) * 0.01 for i in range(100)],  # 0.85 to 1.35
        "Extraction_Limit_m3": [1700 - (i % 50) * 16 for i in range(100)]  # 900 to 1700
    }),
    "Gurdaspur": pd.DataFrame({
        "City": ["Gurdaspur"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [10.0 + i * 0.18 for i in range(100)],  # 10.0 to 27.82
        "Climate_Score": [0.75 + (i % 50) * 0.005 for i in range(100)],  # 0.75 to 1.0
        "Total_Water_Demand_m3": [14000 + (i % 50) * 220 for i in range(100)],  # 14000 to 24800
        "Aquifer_Recharge_Rate": [1.0 - (i % 50) * 0.008 for i in range(100)],  # 0.6 to 1.0
        "Extraction_Limit_m3": [2200 + (i % 50) * 8 for i in range(100)]  # 2200 to 2592
    }),
    "Rupnagar": pd.DataFrame({
        "City": ["Rupnagar"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [16.0 + i * 0.15 for i in range(100)],  # 16.0 to 30.85
        "Climate_Score": [0.65 - (i % 50) * 0.01 for i in range(100)],  # 0.15 to 0.65
        "Total_Water_Demand_m3": [16000 + (i % 50) * 260 for i in range(100)],  # 16000 to 28800
        "Aquifer_Recharge_Rate": [0.9 + (i % 50) * 0.012 for i in range(100)],  # 0.9 to 1.5
        "Extraction_Limit_m3": [1900 - (i % 50) * 14 for i in range(100)]  # 1200 to 1900
    }),
    "Faridkot": pd.DataFrame({
        "City": ["Faridkot"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [28.0 + i * 0.12 for i in range(100)],  # 28.0 to 39.88
        "Climate_Score": [0.4 + (i % 50) * 0.008 for i in range(100)],  # 0.4 to 0.8
        "Total_Water_Demand_m3": [19000 + (i % 50) * 240 for i in range(100)],  # 19000 to 30800
        "Aquifer_Recharge_Rate": [0.7 - (i % 50) * 0.01 for i in range(100)],  # 0.2 to 0.7
        "Extraction_Limit_m3": [1600 + (i % 50) * 16 for i in range(100)]  # 1600 to 2384
    }),
    "Muktsar": pd.DataFrame({
        "City": ["Muktsar"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [32.0 + i * 0.1 for i in range(100)],  # 32.0 to 41.9
        "Climate_Score": [0.3 - (i % 50) * 0.006 for i in range(100)],  # 0.0 to 0.3
        "Total_Water_Demand_m3": [20000 + (i % 50) * 220 for i in range(100)],  # 20000 to 30800
        "Aquifer_Recharge_Rate": [0.65 + (i % 50) * 0.01 for i in range(100)],  # 0.65 to 1.15
        "Extraction_Limit_m3": [1500 - (i % 50) * 12 for i in range(100)]  # 900 to 1500
    }),
    "Fazilka": pd.DataFrame({
        "City": ["Fazilka"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [35.0 + i * 0.08 for i in range(100)],  # 35.0 to 42.92
        "Climate_Score": [0.25 + (i % 50) * 0.005 for i in range(100)],  # 0.25 to 0.5
        "Total_Water_Demand_m3": [21000 + (i % 50) * 200 for i in range(100)],  # 21000 to 30800
        "Aquifer_Recharge_Rate": [0.6 - (i % 50) * 0.008 for i in range(100)],  # 0.2 to 0.6
        "Extraction_Limit_m3": [1400 + (i % 50) * 14 for i in range(100)]  # 1400 to 2086
    }),
    "Barnala": pd.DataFrame({
        "City": ["Barnala"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [20.0 + i * 0.14 for i in range(100)],  # 20.0 to 33.86
        "Climate_Score": [0.55 + (i % 50) * 0.009 for i in range(100)],  # 0.55 to 1.0
        "Total_Water_Demand_m3": [17000 + (i % 50) * 250 for i in range(100)],  # 17000 to 29250
        "Aquifer_Recharge_Rate": [0.85 + (i % 50) * 0.01 for i in range(100)],  # 0.85 to 1.35
        "Extraction_Limit_m3": [1800 - (i % 50) * 10 for i in range(100)]  # 1300 to 1800
    }),
    "Mansa": pd.DataFrame({
        "City": ["Mansa"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [30.0 + i * 0.1 for i in range(100)],  # 30.0 to 39.9
        "Climate_Score": [0.35 - (i % 50) * 0.007 for i in range(100)],  # 0.0 to 0.35
        "Total_Water_Demand_m3": [19000 + (i % 50) * 230 for i in range(100)],  # 19000 to 30300
        "Aquifer_Recharge_Rate": [0.75 + (i % 50) * 0.01 for i in range(100)],  # 0.75 to 1.25
        "Extraction_Limit_m3": [1600 + (i % 50) * 12 for i in range(100)]  # 1600 to 2188
    }),
    "Tarn Taran": pd.DataFrame({
        "City": ["Tarn Taran"] * 100,
        "Timestamp": [datetime.now() - timedelta(days=x) for x in range(100)],
        "Groundwater_Level_m": [15.0 + i * 0.17 for i in range(100)],  # 15.0 to 31.83
        "Climate_Score": [0.7 + (i % 50) * 0.006 for i in range(100)],  # 0.7 to 1.0
        "Total_Water_Demand_m3": [15000 + (i % 50) * 240 for i in range(100)],  # 15000 to 26800
        "Aquifer_Recharge_Rate": [1.0 - (i % 50) * 0.01 for i in range(100)],  # 0.5 to 1.0
        "Extraction_Limit_m3": [2000 + (i % 50) * 10 for i in range(100)]  # 2000 to 2490
    })
}

# Train Random Forest model on combined dataset
combined_dataset = pd.concat([city_datasets[city] for city in punjab_cities])
X = combined_dataset[["Groundwater_Level_m", "Climate_Score", "Total_Water_Demand_m3", "Aquifer_Recharge_Rate"]]
y = combined_dataset["Extraction_Limit_m3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.session_state["rf_model"] = rf_model
st.session_state["model_mse"] = mse

# Function to get dataset
def get_groundwater_dataset(city):
    """Return the predefined dataset for the given city."""
    return city_datasets[city]

# Placeholder functions (updated to use predefined datasets)
def fetch_weather_data(city, date):
    """Fetch weather data for a city and date using dataset."""
    dataset = get_groundwater_dataset(city)
    latest_climate_score = dataset.iloc[-1]["Climate_Score"]
    condition = "Wet" if latest_climate_score > 0.5 else "Dry"
    return {
        "temperature": random.randint(15, 35),  # Daily variability
        "condition": condition,
        "humidity": random.randint(30, 90),
        "precipitation": random.uniform(0, 20) if latest_climate_score > 0.5 else random.uniform(0, 5)
    }

def fetch_groundwater_data(city, period):
    """Fetch groundwater data from predefined dataset for a city and time period."""
    dataset = get_groundwater_dataset(city)
    if period == "1month":
        days = 30
    elif period == "2months":
        days = 60
    else:  # 6months
        days = 180
    # Filter dataset for the specified period
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_data = dataset[dataset["Timestamp"] >= cutoff_date][["Timestamp", "Groundwater_Level_m"]]
    filtered_data = filtered_data.rename(columns={"Timestamp": "Date"})
    return filtered_data

def fetch_water_demand(city):
    """Fetch water demand data from predefined dataset for a city."""
    dataset = get_groundwater_dataset(city)
    latest_demand = dataset.iloc[-1]["Total_Water_Demand_m3"]
    # Split into sectors (fixed proportions)
    industries = latest_demand * 0.3
    agriculture = latest_demand * 0.5
    household = latest_demand * 0.2
    return {
        "industries": industries,
        "agriculture": agriculture,
        "household": household
    }

def fetch_water_sources(city):
    """Fetch water supply sources for a city."""
    dataset = get_groundwater_dataset(city)
    gw_level = dataset.iloc[-1]["Groundwater_Level_m"]
    # Higher groundwater level -> lower groundwater source percentage
    gw_percentage = 50 - (gw_level / 50) * 20  # Ranges from 30% to 50%
    sources = ["Groundwater", "Surface Water", "Canals", "Rainwater Harvesting"]
    percentages = [gw_percentage, 30, 15, 100 - (gw_percentage + 30 + 15)]
    return pd.DataFrame({"Source": sources, "Percentage": percentages})

def ai_extraction_recommendation(city):
    """AI model to predict sustainable water extraction using Random Forest."""
    dataset = get_groundwater_dataset(city)
    latest_data = dataset.iloc[-1]
    
    # Parameters from dataset
    gw_level = latest_data["Groundwater_Level_m"]
    climate_score = latest_data["Climate_Score"]
    demand = latest_data["Total_Water_Demand_m3"]
    recharge_rate = latest_data["Aquifer_Recharge_Rate"]
    
    # Prepare input for Random Forest
    input_data = np.array([[gw_level, climate_score, demand, recharge_rate]])
    recommendation = st.session_state["rf_model"].predict(input_data)[0]
    
    # Ensure recommendation is within bounds
    recommendation = max(500, min(2500, recommendation))
    
    # Factors for display
    factors = {
        "Groundwater Level": f"Current level: {gw_level:.2f} meters",
        "Climate Forecast": f"Score: {climate_score:.2f} ({'Wet' if climate_score > 0.5 else 'Dry'})",
        "Water Demand": f"Total demand: {demand:.2f} cubic meters/day",
        "Geological Factors": f"Aquifer recharge rate: {recharge_rate:.2f} meters/year",
        "Model MSE": f"Mean Squared Error: {st.session_state['model_mse']:.2f}"
    }
    
    return recommendation, factors

# Streamlit app (unchanged)
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Feature", [
        "Home",
        "Weather Conditions",
        "Groundwater Levels",
        "Water Demand",
        "Water Supply Sources",
        "AI Extraction Recommendation"
    ])

    if page == "Home":
        st.markdown("<div class='header'>AI-Driven Groundwater Monitoring System</div>", unsafe_allow_html=True)
        st.write("Welcome to the Smart Groundwater Monitoring and Sustainable Extraction System. Select your location below to explore features.")

        # State and city selection
        state = st.selectbox("State", ["Punjab"])  # Fixed to Punjab for now
        city = st.selectbox("City", punjab_cities)

        # Store city in session state
        st.session_state["selected_city"] = city

        st.markdown("<div class='subheader'>Explore Features</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Weather Conditions"):
                st.session_state["page"] = "Weather Conditions"
        with col2:
            if st.button("Groundwater Levels"):
                st.session_state["page"] = "Groundwater Levels"
        with col3:
            if st.button("Water Demand"):
                st.session_state["page"] = "Water Demand"
        
        col4, col5, _ = st.columns(3)
        with col4:
            if st.button("Water Supply Sources"):
                st.session_state["page"] = "Water Supply Sources"
        with col5:
            if st.button("AI Extraction Recommendation"):
                st.session_state["page"] = "AI Extraction Recommendation"

    elif page == "Weather Conditions":
        st.markdown("<div class='header'>Weather Conditions</div>", unsafe_allow_html=True)
        city = st.session_state.get("selected_city", punjab_cities[0])
        st.write(f"Showing weather data for {city}")

        # Calendar for date selection
        selected_date = st.date_input("Select Date", datetime.now())

        # Fetch and display weather data
        weather = fetch_weather_data(city, selected_date)
        st.markdown(f"**Date**: {selected_date}")
        st.markdown(f"**Temperature**: {weather['temperature']} °C")
        st.markdown(f"**Condition**: {weather['condition']}")
        st.markdown(f"**Humidity**: {weather['humidity']}%")
        st.markdown(f"**Precipitation**: {weather['precipitation']} mm")

    elif page == "Groundwater Levels":
        st.markdown("<div class='header'>Groundwater Levels</div>", unsafe_allow_html=True)
        city = st.session_state.get("selected_city", punjab_cities[0])
        st.write(f"Groundwater levels for {city}")

        # Period selection
        period = st.selectbox("Select Time Period", ["1 Month", "2 Months", "6 Months"])
        period_map = {"1 Month": "1month", "2 Months": "2months", "6 Months": "6months"}
        data = fetch_groundwater_data(city, period_map[period])

        # Plot groundwater levels
        fig = px.line(data, x="Date", y="Groundwater_Level_m", title=f"Groundwater Levels - Past {period}")
        st.plotly_chart(fig)

        # Additional info
        dataset = get_groundwater_dataset(city)
        latest_level = dataset.iloc[-1]["Groundwater_Level_m"]
        trend = "Stable" if abs(latest_level - dataset.iloc[-10]["Groundwater_Level_m"]) < 5 else "Declining" if latest_level > dataset.iloc[-10]["Groundwater_Level_m"] else "Rising"
        st.markdown("**Additional Insights**:")
        st.write(f"- Trend: {trend}")
        st.write(f"- Aquifer Health: Current level at {latest_level:.2f} meters.")

    elif page == "Water Demand":
        st.markdown("<div class='header'>Water Demand</div>", unsafe_allow_html=True)
        city = st.session_state.get("selected_city", punjab_cities[0])
        st.write(f"Water demand for {city}")

        # Fetch demand data
        demand = fetch_water_demand(city)

        # Display demand
        st.markdown("**Demand Breakdown**:")
        st.write(f"- Industries: {demand['industries']:.2f} cubic meters/day")
        st.write(f"- Agriculture: {demand['agriculture']:.2f} cubic meters/day")
        st.write(f"- Household: {demand['household']:.2f} cubic meters/day")

        # Bar chart
        demand_df = pd.DataFrame({
            "Sector": ["Industries", "Agriculture", "Household"],
            "Demand": [demand["industries"], demand["agriculture"], demand["household"]]
        })
        fig = px.bar(demand_df, x="Sector", y="Demand", title="Water Demand by Sector")
        st.plotly_chart(fig)

    elif page == "Water Supply Sources":
        st.markdown("<div class='header'>Water Supply Sources</div>", unsafe_allow_html=True)
        city = st.session_state.get("selected_city", punjab_cities[0])
        st.write(f"Water supply sources for {city}")

        # Fetch sources data
        sources = fetch_water_sources(city)

        # Pie chart
        fig = px.pie(sources, names="Source", values="Percentage", title="Water Supply Distribution")
        st.plotly_chart(fig)

    elif page == "AI Extraction Recommendation":
        st.markdown("<div class='header'>AI Extraction Recommendation</div>", unsafe_allow_html=True)
        city = st.session_state.get("selected_city", punjab_cities[0])
        st.write(f"Optimal water extraction for {city}")

        # Fetch AI recommendation
        recommendation, factors = ai_extraction_recommendation(city)

        # Display recommendation
        st.markdown(f"**Recommended Extraction**: {recommendation:.2f} cubic meters/day")
        st.markdown("**Contributing Factors**:")
        for factor, value in factors.items():
            st.write(f"- {factor}: {value}")

        # Plot historical extraction recommendations
        dataset = get_groundwater_dataset(city)
        fig = px.line(dataset.tail(30), x="Timestamp", y="Extraction_Limit_m3", title="Historical Extraction Recommendations (Last 30 Days)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
