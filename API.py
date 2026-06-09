import datetime
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

# Initialize FastAPI app
app = FastAPI(title="Yamaan Faraz YF Weather Predictor API")

# Load data at startup to keep endpoints fast
CITIES_PATH = "cities.csv"
WEATHER_PATH = "daily_weather.parquet"

if not os.path.exists(CITIES_PATH) or not os.path.exists(WEATHER_PATH):
    raise FileNotFoundError("Missing data files! Ensure 'cities.csv' and 'daily_weather.parquet' are in the directory.")

country_df = pd.read_csv(CITIES_PATH)
weather_df = pd.read_parquet(WEATHER_PATH)


# Define request schema
class PredictionRequest(BaseModel):
    country: str
    city_name: str
    days: int = 7  # Default to 7 days if not provided


# Define response schema
class ForecastItem(BaseModel):
    date: str
    weather: str
    temp_c: float
    rain_mm: float


# --- FIX FOR {"detail":"Not Found"} ERROR ---
@app.get("/")
def read_root():
    """Root endpoint to verify the API is running."""
    return {
        "status": "Online",
        "message": "Welcome to the Yamaan Faraz YF Weather Predictor API!",
        "docs_url": "http://127.0.0"
    }


@app.post("/predict", response_model=list[ForecastItem])
def predict_weather(request: PredictionRequest):
    # 1. Validate inputs and filter country
    filtered_country = country_df[country_df["country"].str.lower() == request.country.lower()]
    if filtered_country.empty:
        raise HTTPException(status_code=404, detail=f"Country '{request.country}' not found.")

    # 2. Filter city and find the EXACT station_id
    city_match = filtered_country[filtered_country["city_name"].str.lower() == request.city_name.lower()]
    if city_match.empty:
        raise HTTPException(status_code=404, detail=f"City '{request.city_name}' not found in {request.country}.")

    target_station = city_match["station_id"].values[0]

    # 3. Validate slider limits equivalent (1 to 700 days)
    if not (1 <= request.days <= 700):
        raise HTTPException(status_code=400, detail="Days must be between 1 and 700.")

    # 4. Filter historical data strictly for THAT specific city station
    asli_df = weather_df[weather_df["station_id"] == target_station].copy()
    if asli_df.empty:
        raise HTTPException(status_code=404, detail=f"No historical weather data found for station {target_station}.")

    # Clean local data timeline
    asli_df["date"] = pd.to_datetime(asli_df["date"])
    asli_df = asli_df.dropna(subset=["date", "avg_temp_c", "precipitation_mm"])

    if len(asli_df) < 5:  # Sanity check for training data size
        raise HTTPException(status_code=500, detail="Insufficient data to train the AI model for this city.")

    # --- Train Time-Based Regressors ---
    try:
        X = pd.DataFrame()
        X["month"] = asli_df["date"].dt.month
        X["day_of_year"] = asli_df["date"].dt.dayofyear

        model_temp = RandomForestRegressor(n_estimators=50, random_state=42)
        model_temp.fit(X, asli_df["avg_temp_c"])

        model_rain = RandomForestRegressor(n_estimators=50, random_state=42)
        model_rain.fit(X, asli_df["precipitation_mm"])

        # --- Generate Predictions for that specific city ---
        today = datetime.date.today()
        future_dates = [today + datetime.timedelta(days=i) for i in range(request.days)]

        future_df = pd.DataFrame({"date": future_dates})
        future_df["date"] = pd.to_datetime(future_df["date"])

        future_X = pd.DataFrame()
        future_X["month"] = future_df["date"].dt.month
        future_X["day_of_year"] = future_df["date"].dt.dayofyear

        predicted_temps = model_temp.predict(future_X)
        predicted_rains = model_rain.predict(future_X)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {str(e)}")

    # --- Build Weather Conditions ---
    response_data = []
    for i, (temp, rain) in enumerate(zip(predicted_temps, predicted_rains)):
        if rain > 5.0:
            weather_condition = "⛈️ Stormy"
        elif rain > 2.5:
            weather_condition = "🌧️ Rainy"
        elif temp >= 35.0:
            weather_condition = "🔥 Scorching"
        elif temp >= 28.0:
            weather_condition = "☀️ Sunny"
        elif temp < 15.0:
            weather_condition = "❄️ Cold"
        else:
            weather_condition = "🌤️ Pleasant"

        response_data.append(
            ForecastItem(
                date=future_dates[i].strftime("%Y-%m-%d"),
                weather=weather_condition,
                temp_c=round(temp, 1),
                rain_mm=round(rain, 2)
            )
        )

    return response_data
