from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import losses, metrics
import pickle
from datetime import datetime, timedelta
import requests
import logging
from dotenv import load_dotenv
import os
import traceback
from utils.weather_utils import (
    process_weather_data, 
    inverse_transform_rainfall,
    format_monthly_forecast,
    get_crop_recommendations,
    RAINFALL_FEATURE_INDEX,
    validate_prediction,
    get_seasonal_features
)
from dateutil.relativedelta import relativedelta
from auth import auth_bp  # Import the blueprint

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Register the auth blueprint
app.register_blueprint(auth_bp)

# Define custom objects for model loading
custom_objects = {
    'MeanSquaredError': metrics.MeanSquaredError,
    'MeanAbsoluteError': metrics.MeanAbsoluteError,
    'mean_squared_error': losses.MeanSquaredError(),
    'mean_absolute_error': losses.MeanAbsoluteError(),
    'mse': losses.MeanSquaredError(),
    'mae': losses.MeanAbsoluteError()
}

# OpenWeather API Config
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
if not OPENWEATHER_API_KEY:
    raise ValueError("OpenWeather API key not configured")
BASE_WEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"
GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"

def get_coordinates(location):
    """Get coordinates for a given location name using OpenWeather Geocoding API"""
    try:
        # Add common Indian cities for states/regions
        state_to_city = {
            'meghalaya': 'Shillong',
            'kerala': 'Thiruvananthapuram',
            'tamil nadu': 'Chennai',
            'karnataka': 'Bangalore',
            'theni': 'Theni',  # Added Theni explicitly
            # Add more state-city mappings as needed
        }
        
        # If the location is a state/city, use its mapping
        search_location = location.lower()
        if search_location in state_to_city:
            search_location = f"{state_to_city[search_location]},IN"
        elif not ',' in location:
            # If no comma, assume it's an Indian location
            search_location = f"{location},IN"
            
        params = {
            'q': search_location,
            'limit': 1,
            'appid': OPENWEATHER_API_KEY
        }
        
        logger.info(f"Attempting to get coordinates for location: {search_location}")
        response = requests.get(GEOCODING_URL, params=params, verify=True)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            # Try without country code if first attempt fails
            params['q'] = location
            response = requests.get(GEOCODING_URL, params=params, verify=True)
            response.raise_for_status()
            data = response.json()
            
        if not data:
            raise ValueError(f"Location '{location}' not found")
            
        logger.info(f"Successfully found coordinates for {location}")
        return {
            'latitude': data[0]['lat'],
            'longitude': data[0]['lon'],
            'name': data[0]['name'],
            'country': data[0].get('country', 'IN')
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Geocoding API error: {str(e)}")
        raise ValueError(f"Failed to get location coordinates: {str(e)}")

try:
    # Load model and scaler
    model_path = os.path.join('models', 'rainfall_model_aggregated.h5')
    scaler_path = os.path.join('models', 'scaler_aggregated.pkl')
    
    model = load_model(model_path, custom_objects=custom_objects)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    raise RuntimeError("Could not load prediction models. Service unavailable.")

@app.route('/predict', methods=['POST'])
def predict_rainfall():
    try:
        data = request.get_json()

        location = data.get('location')
        months = data.get('months', 3)

        if not location:
            return jsonify({'error': 'Location is required'}), 400

        if months not in [3, 5]:
            return jsonify({'error': 'Invalid months parameter'}), 400

        # STEP 1: Coordinates
        location_data = get_coordinates(location)

        # STEP 2: Weather data
        weather_params = {
            'lat': location_data['latitude'],
            'lon': location_data['longitude'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'cnt': 40
        }

        weather_raw = requests.get(BASE_WEATHER_URL, params=weather_params).json()

        # STEP 3: Preprocess once
        processed_data, actual_weather = process_weather_data(weather_raw, scaler)

        # Reuse this array and avoid creating new ones
        current_sequence = processed_data.copy()

        predictions = []
        start_date = datetime.now().replace(day=1)

        for month in range(months):

            # MODEL PREDICTION — FAST because model is preloaded
            pred_scaled = model.predict(current_sequence, verbose=0)[0][0]

            # Unscale rainfall
            rainfall_mean = scaler.mean_[RAINFALL_FEATURE_INDEX]
            rainfall_scale = scaler.scale_[RAINFALL_FEATURE_INDEX]
            rainfall_pred = float(pred_scaled * rainfall_scale + rainfall_mean)

            # Cleanup negative / impossible values
            rainfall_pred = validate_prediction(
                rainfall_pred,
                (start_date.month + month - 1) % 12 + 1,
                location_data,
                actual_weather['current_conditions']
            )

            predictions.append({
                "month": (start_date + relativedelta(months=month)).strftime("%B %Y"),
                "date": (start_date + relativedelta(months=month)).strftime("%Y-%m-%d"),
                "rainfall_mm": round(rainfall_pred, 2),
                "confidence": "high" if month == 0 else "medium" if month == 1 else "low"
            })

            # SHIFT the sequence (avoid heavy recomputation)
            current_sequence[:, :-1, :] = current_sequence[:, 1:, :]
            current_sequence[:, -1, :] = current_sequence[:, -2, :]

        # Visualization
        visualization_data = format_monthly_forecast(predictions, start_date, actual_weather)

        # Crop AI
        crop_recommendations = get_crop_recommendations(
            rainfall_data=visualization_data['rainfall'],
            temperature_data=visualization_data['max_temp']
        )

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "visualization_data": visualization_data,
            "crop_recommendations": crop_recommendations,
            "location": location_data,
            "forecast_period": f"{months} months"
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'service': 'Rainfall Prediction API'
    })

@app.route('/test-api', methods=['GET'])
def test_api():
    """Test route to verify OpenWeather API connectivity"""
    try:
        # 1. Test Geocoding API
        test_location = "Theni,IN"
        geo_params = {
            'q': test_location,
            'limit': 1,
            'appid': OPENWEATHER_API_KEY
        }
        
        logger.info("Testing Geocoding API...")
        geo_response = requests.get(GEOCODING_URL, params=geo_params, verify=True)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return jsonify({
                'status': 'error',
                'message': 'Geocoding API returned no data',
                'api_key_status': 'API key might be invalid'
            }), 400
            
        # 2. Test Weather API
        weather_params = {
            'lat': geo_data[0]['lat'],
            'lon': geo_data[0]['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
            'cnt': 1  # Just get one data point for testing
        }
        
        logger.info("Testing Weather Forecast API...")
        weather_response = requests.get(BASE_WEATHER_URL, params=weather_params, verify=True)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        if 'list' not in weather_data or not weather_data['list']:
            return jsonify({
                'status': 'error',
                'message': 'Weather API returned no data',
                'api_key_status': 'API key might be invalid'
            }), 400
            
        # If we get here, both APIs are working
        return jsonify({
            'status': 'success',
            'message': 'OpenWeather APIs are working correctly',
            'api_key_status': 'valid',
            'geocoding_response': {
                'location': test_location,
                'coordinates': {
                    'lat': geo_data[0]['lat'],
                    'lon': geo_data[0]['lon']
                }
            },
            'weather_sample': {
                'temperature': weather_data['list'][0]['main']['temp'],
                'humidity': weather_data['list'][0]['main']['humidity'],
                'description': weather_data['list'][0]['weather'][0]['description']
            }
        })
        
    except requests.exceptions.SSLError as e:
        return jsonify({
            'status': 'error',
            'message': 'SSL Certificate verification failed. Try updating your SSL certificates.',
            'error': str(e)
        }), 500
        
    except requests.exceptions.ConnectionError as e:
        return jsonify({
            'status': 'error',
            'message': 'Failed to connect to OpenWeather API. Check your internet connection.',
            'error': str(e)
        }), 500
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': 'API request failed',
            'error': str(e)
        }), 500
    
@app.route("/")
def home():
    return "Rainfall Prediction API is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 