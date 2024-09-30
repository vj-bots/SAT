import time
import joblib
import logging
import signal
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from contextlib import contextmanager
from .visualization import plot_confusion_matrix, plot_feature_importance, plot_random_forest_learning_curve

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Tempo limite atingido")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def evaluate_classification_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    logger.info(f"Acurácia: {accuracy:.4f}")
    logger.info(f"Precisão: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def evaluate_regression_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    return mae, mse, r2

def optimize_hyperparameters(X, y, model_type):
    if model_type == 'classifier':
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [5, 10, 15, None],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2']
        }
        scoring = 'accuracy'
    elif model_type == 'regressor':
        model = RandomForestRegressor(random_state=42)
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [5, 10, 15, None],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2']
        }
        scoring = 'neg_mean_squared_error'
    else:
        raise ValueError("Invalid model_type. Choose 'classifier' or 'regressor'.")

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=10, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
    random_search.fit(X, y)

    logger.info(f"Melhores hiperparâmetros: {random_search.best_params_}")
    return random_search.best_estimator_

def train_crop_health_model(X, y):
    return train_generic_model(X, y, 'classifier', n_estimators=100, random_state=42)

def predict_crop_health(model, features):
    return model.predict(features)[0]

def train_irrigation_model(X, y):
    return train_generic_model(X, y, 'regressor', n_estimators=100, random_state=42)

def predict_irrigation(model, features):
    return model.predict(features)

def train_pest_detection_model(X, y):
    return train_generic_model(X, y, 'classifier', n_estimators=100, random_state=42)

def predict_pest_presence(model, features):
    return model.predict(features)[0]

def train_yield_prediction_model(X, y):
    return train_generic_model(X, y, 'regressor', n_estimators=100, random_state=42)

def predict_yield(model, features):
    return model.predict(features)[0]

def train_generic_model(X, y, model_type, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'classifier':
        model = RandomForestClassifier(**kwargs)
        scoring = 'accuracy'
    elif model_type == 'regressor':
        model = RandomForestRegressor(**kwargs)
        scoring = 'neg_mean_squared_error'
    else:
        raise ValueError("Invalid model_type. Choose 'classifier' or 'regressor'.")

    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if model_type == 'classifier':
        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_test, y_pred_test)
    else:
        train_score = mean_squared_error(y_train, y_pred_train)
        test_score = mean_squared_error(y_test, y_pred_test)
    
    logger.info(f"Train score: {train_score}")
    logger.info(f"Test score: {test_score}")
    
    return model