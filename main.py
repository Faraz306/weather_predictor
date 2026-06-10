import datetime
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
country_df = pd.read_csv("cities.csv")
st.title("Yamaan Faraz YF Weather predictor")

# 1. Country Selection Dropdown
country = st.selectbox("Choose Country", country_df["country"].unique())

# 2. City Selection Dropdown (Dynamically changes based on chosen Country)
filtered_cities = country_df[country_df["country"] == country]
city = st.selectbox("Choose City", filtered_cities["city_name"].unique())

days = st.slider("How many days to predict?", 1, 700)
predict = st.button("Predict")

if predict:
    with st.spinner(f"Training Yamaan's AI Model specifically for {city}..."):
        weather_df = pd.read_parquet("daily_weather.parquet")

        # 3. Find the EXACT station_id for the selected city
        target_station = filtered_cities.loc[
            filtered_cities["city_name"] == city, "station_id"
        ].values[0]

        # 4. Filter historical data strictly for THAT specific city station
        asli_df = weather_df[weather_df["station_id"] == target_station].copy()

        # Clean your local data timeline
        asli_df["date"] = pd.to_datetime(asli_df["date"])
        asli_df = asli_df.dropna(
            subset=["date", "avg_temp_c", "precipitation_mm"]
        )

        # --- Train Time-Based Regressors ---
        X = pd.DataFrame()
        X["month"] = asli_df["date"].dt.month
        X["day_of_year"] = asli_df["date"].dt.dayofyear

        model_temp = RandomForestRegressor(n_estimators=50, random_state=42)
        model_temp.fit(X, asli_df["avg_temp_c"])

        model_rain = RandomForestRegressor(n_estimators=50, random_state=42)
        model_rain.fit(X, asli_df["precipitation_mm"])

    # --- Generate Predictions for that specific city ---
    with st.spinner(f"Calculating local trends for {city}..."):
        today = datetime.date.today()
        future_dates = [today + datetime.timedelta(days=i) for i in range(days)]

        future_df = pd.DataFrame({"date": future_dates})
        future_df["date"] = pd.to_datetime(future_df["date"])

        future_X = pd.DataFrame()
        future_X["month"] = future_df["date"].dt.month
        future_X["day_of_year"] = future_df["date"].dt.dayofyear

        # Predict the exact numerical metrics using your parallel models
        # Predict the exact numerical metrics using your parallel models
        predicted_temps = model_temp.predict(future_X)
        predicted_rains = model_rain.predict(future_X)

        # 1. Build the single 'Weather' column using combined rules
        # 1. Build the single 'Weather' column using realistic thresholds
        weather_conditions = []
        for temp, rain in zip(predicted_temps, predicted_rains):
            if rain > 5.0:
                weather_conditions.append("⛈️ Stormy")
            # INCREASE THIS: Only call it Rainy if there is real rainfall (above 2.5 mm)
            elif rain > 2.5:
                weather_conditions.append("🌧️ Rainy")
            # CHECK TEMPERATURE FIRST: If it's scorching hot and barely drizzling, it's Scorching!
            elif temp >= 35.0:
                weather_conditions.append("🔥 Scorching")
            elif temp >= 28.0:
                weather_conditions.append("☀️ Sunny")
            elif temp < 15.0:
                weather_conditions.append("❄️ Cold")
            else:
                weather_conditions.append("🌤️ Pleasant")
        # 2. Assemble exactly the 4 columns you want
        final_table = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Weather": weather_conditions,
            "Temp (°C)": [round(t, 1) for t in predicted_temps],
            "mm": [round(r, 2) for r in predicted_rains]
        })

        # --- Display Results ---
        st.subheader(f"Weather Forecast Timeline for {city}, {country}")
        st.dataframe(final_table, use_container_width=True, hide_index=True)
