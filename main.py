import datetime
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Cache the lightweight cities file
country_df = pd.read_csv("cities.csv")
st.title("Yamaan Faraz YF Weather predictor")

country = st.selectbox("Choose Country", country_df["country"].unique())
filtered_cities = country_df[country_df["country"] == country]
city = st.selectbox("Choose City", filtered_cities["city_name"].unique())
days = st.slider("How many days to predict?", 1, 700)
predict = st.button("Predict")

if predict:
    with st.spinner(f"Training Yamaan's AI Model specifically for {city}..."):
        # OPTIMIZATION 1: Use filters directly while reading parquet to save RAM
        target_station = filtered_cities.loc[filtered_cities["city_name"] == city, "station_id"].values[0]

        # This reads ONLY the rows for that specific station instead of the entire global file
        asli_df = pd.read_parquet(
            "daily_weather.parquet",
            filters=[("station_id", "==", target_station)]
        )

        asli_df["date"] = pd.to_datetime(asli_df["date"])
        asli_df = asli_df.dropna(subset=["date", "avg_temp_c", "precipitation_mm"])

        X = pd.DataFrame()
        X["month"] = asli_df["date"].dt.month
        X["day_of_year"] = asli_df["date"].dt.dayofyear

        # OPTIMIZATION 2: Lower n_estimators to 15 to prevent memory spikes
        model_temp = RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=1)
        model_temp.fit(X, asli_df["avg_temp_c"])

        model_rain = RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=1)
        model_rain.fit(X, asli_df["precipitation_mm"])

    with st.spinner(f"Calculating local trends for {city}..."):
        today = datetime.date.today()
        future_dates = [today + datetime.timedelta(days=i) for i in range(days)]

        future_df = pd.DataFrame({"date": future_dates})
        future_df["date"] = pd.to_datetime(future_df["date"])

        future_X = pd.DataFrame()
        future_X["month"] = future_df["date"].dt.month
        future_X["day_of_year"] = future_df["date"].dt.dayofyear

        predicted_temps = model_temp.predict(future_X)
        predicted_rains = model_rain.predict(future_X)

        weather_conditions = []
        for temp, rain in zip(predicted_temps, predicted_rains):
            if rain > 5.0:
                weather_conditions.append("⛈️ Stormy")
            elif rain > 2.5:
                weather_conditions.append("🌧️ Rainy")
            elif temp >= 35.0:
                weather_conditions.append("🔥 Scorching")
            elif temp >= 28.0:
                weather_conditions.append("☀️ Sunny")
            elif temp < 15.0:
                weather_conditions.append("❄️ Cold")
            else:
                weather_conditions.append("🌤️ Pleasant")

        final_table = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Weather": weather_conditions,
            "Temp (°C)": [round(t, 1) for t in predicted_temps],
            "mm": [round(r, 2) for r in predicted_rains]
        })

        st.subheader(f"Weather Forecast Timeline for {city}, {country}")
        st.dataframe(final_table, use_container_width=True, hide_index=True)
