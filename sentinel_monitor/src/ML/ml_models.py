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

def train_crop_health_model(X, y, model_type='classifier'):
    if os.environ.get('TESTING') == 'True':
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        model = optimize_hyperparameters(X, y, model_type)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    logger.info("Resultados no conjunto de treinamento:")
    evaluate_classification_model(y_train, y_pred_train)
    
    logger.info("Resultados no conjunto de teste:")
    evaluate_classification_model(y_test, y_pred_test)
    
    joblib.dump(model, 'crop_health_model.joblib')
    
    plot_confusion_matrix(y_test, y_pred_test, classes=['Saudável', 'Estressada', 'Doente'])
    plot_feature_importance(model, [f'Feature {i}' for i in range(X.shape[1])], 'Importância das Features - Saúde da Cultura')
    plot_random_forest_learning_curve(model, X, y, 'Curva de Aprendizado - Saúde da Cultura')
    
    return model

def predict_crop_health(model, features):
    return model.predict(features)[0]

def train_irrigation_model(X, y, model_type='regressor'):
    model = optimize_hyperparameters(X, y, model_type)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    logger.info(f"MSE médio do modelo de irrigação (validação cruzada): {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
    logger.info(f"R² médio do modelo de irrigação (validação cruzada): {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    logger.info("Resultados no conjunto de treinamento:")
    evaluate_regression_model(y_train, y_pred_train)
    
    logger.info("Resultados no conjunto de teste:")
    evaluate_regression_model(y_test, y_pred_test)
    
    joblib.dump(model, 'irrigation_model.joblib')
    
    plot_feature_importance(model, X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])], 'Importância das Features - Irrigação')
    plot_random_forest_learning_curve(model, X, y, 'Curva de Aprendizado - Irrigação', is_classifier=False)
    
    return model

def predict_irrigation(model, features):
    return model.predict(features)

def train_pest_detection_model(X, y, model_type='classifier'):
    X_2d = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=50, cv=5, scoring='f1', n_jobs=-1, random_state=42, verbose=1)
    
    start_time = time.time()
    max_time = 600  # 10 minutos
    
    try:
        random_search.fit(X_train_resampled, y_train_resampled)
    except Exception as e:
        logger.warning(f"Erro durante a busca aleatória: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_time:
            logger.warning("A busca aleatória atingiu o limite de tempo. Usando os melhores parâmetros encontrados até agora.")
    
    logger.info(f"Melhores hiperparâmetros para detecção de pragas: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    logger.info("Resultados no conjunto de treinamento:")
    evaluate_classification_model(y_train, y_pred_train)
    
    logger.info("Resultados no conjunto de teste:")
    evaluate_classification_model(y_test, y_pred_test)
    
    joblib.dump(model, 'pest_detection_model.joblib')
    
    plot_confusion_matrix(y_test, y_pred_test, classes=['Sem Pragas', 'Com Pragas'])
    plot_feature_importance(model, [f'Feature {i}' for i in range(X_2d.shape[1])], 'Importância das Features - Detecção de Pragas')
    plot_random_forest_learning_curve(model, X_2d, y, 'Curva de Aprendizado - Detecção de Pragas', is_classifier=True)
    
    return model

def predict_pest_presence(model, features):
    return model.predict(features)[0]

def train_yield_prediction_model(X, y, model_type='regressor'):
    X_2d = X.reshape(X.shape[0], -1)
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_2d)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [3, 5, 10, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4)
    }
    
    model = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                       n_jobs=-1, random_state=42, verbose=1)
    
    random_search.fit(X_train, y_train)
    
    logger.info(f"Melhores hiperparâmetros para previsão de colheita: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    logger.info(f"MSE médio do modelo de previsão de colheita (validação cruzada): {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
    
    r2_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
    logger.info(f"R² médio do modelo de previsão de colheita (validação cruzada): {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    logger.info("Resultados no conjunto de treinamento:")
    evaluate_regression_model(y_train, y_pred_train)
    
    logger.info("Resultados no conjunto de teste:")
    evaluate_regression_model(y_test, y_pred_test)
    
    joblib.dump((model, poly_features), 'yield_prediction_model.joblib')
    
    plot_feature_importance(model, [f'Feature {i}' for i in range(X_poly.shape[1])], 'Importância das Features - Previsão de Colheita')
    plot_random_forest_learning_curve(model, X_poly, y, 'Curva de Aprendizado - Previsão de Colheita', is_classifier=False)
    
    return model, poly_features

def predict_yield(model, features):
    return model.predict(features)[0]