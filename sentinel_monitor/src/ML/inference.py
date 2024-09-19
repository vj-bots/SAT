import joblib
from src.ML.data_preprocessing import preprocess_vi_image, preprocess_sensor_data
from src.ML.ml_models import predict_crop_health, predict_irrigation, predict_pest_presence, predict_yield

def load_models():
    crop_health_model = joblib.load('crop_health_model.joblib')
    irrigation_model = joblib.load('irrigation_model.joblib')
    pest_detection_model = joblib.load('pest_detection_model.joblib')
    yield_prediction_model = joblib.load('yield_prediction_model.joblib')
    return crop_health_model, irrigation_model, pest_detection_model, yield_prediction_model

def predict(image_path, sensor_data):
    crop_health_model, irrigation_model, pest_detection_model, yield_prediction_model = load_models()
    
    image_features = preprocess_vi_image(image_path)
    sensor_features = preprocess_sensor_data(sensor_data)
    
    health_prediction = predict_crop_health(crop_health_model, image_features)
    irrigation_prediction = predict_irrigation(irrigation_model, sensor_features)
    pest_prediction = predict_pest_presence(pest_detection_model, image_features)
    yield_prediction = predict_yield(yield_prediction_model, image_features)
    
    return health_prediction, irrigation_prediction, pest_prediction, yield_prediction