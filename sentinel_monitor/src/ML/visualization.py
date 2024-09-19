import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_ndvi_time_series(dates, ndvi_values):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, ndvi_values)
    plt.title('Série Temporal de NDVI')
    plt.xlabel('Data')
    plt.ylabel('NDVI')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ndvi_time_series.png')
    plt.close()

def plot_yield_prediction(actual_yield, predicted_yield):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_yield, predicted_yield)
    plt.plot([min(actual_yield), max(actual_yield)], [min(actual_yield), max(actual_yield)], 'r--')
    plt.title('Previsão de Rendimento vs. Rendimento Real')
    plt.xlabel('Rendimento Real')
    plt.ylabel('Rendimento Previsto')
    plt.tight_layout()
    plt.savefig('yield_prediction.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Curva de Perda')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Curva de Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def plot_random_forest_learning_curve(estimator, X, y, title, cv=5, n_jobs=-1, is_classifier=True):
    from sklearn.model_selection import learning_curve
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy" if is_classifier else "neg_mean_squared_error"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Tamanho do conjunto de treinamento")
    plt.ylabel("Acurácia" if is_classifier else "Erro Quadrático Médio Negativo")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treino")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validação cruzada")
    plt.legend(loc="best")
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()
    