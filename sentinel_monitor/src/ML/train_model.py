import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from .data_preprocessing import preprocess_vi_image, preprocess_sensor_data, load_data, validate_data

def create_model(input_shape_images, input_shape_sensors):
    image_input = tf.keras.Input(shape=input_shape_images)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    sensor_input = tf.keras.Input(shape=(input_shape_sensors,))
    y = tf.keras.layers.Dense(16, activation='relu')(sensor_input)

    combined = tf.keras.layers.concatenate([x, y])
    combined = tf.keras.layers.Dense(32, activation='relu')(combined)

    irrigation_output = tf.keras.layers.Dense(1, activation='sigmoid', name='irrigation')(combined)
    invasion_output = tf.keras.layers.Dense(1, activation='sigmoid', name='invasion')(combined)
    health_output = tf.keras.layers.Dense(3, activation='softmax', name='health')(combined)
    yield_output = tf.keras.layers.Dense(1, name='yield')(combined)

    model = tf.keras.Model(inputs=[image_input, sensor_input], outputs=[irrigation_output, invasion_output, health_output, yield_output])

    model.compile(optimizer='adam',
                  loss={'irrigation': 'binary_crossentropy', 
                        'invasion': 'binary_crossentropy', 
                        'health': 'sparse_categorical_crossentropy',
                        'yield': 'mse'},
                  metrics={'irrigation': 'accuracy', 
                           'invasion': 'accuracy', 
                           'health': 'accuracy',
                           'yield': 'mse'})

    return model

def train_model(test_mode=False):
    (X_combined, X_sensors), (y_irrigation, y_invasion, y_health, y_yield) = load_data()

    # Validação de dados
    validate_data(X_combined)
    validate_data(X_sensors)
    validate_data(y_irrigation)
    validate_data(y_invasion)
    validate_data(y_health)
    validate_data(y_yield)

    print("Forma original de X_combined:", X_combined.shape)
    print("Forma original de X_sensors:", X_sensors.shape)

    # Pré-processamento dos dados
    X_processed = np.array([preprocess_vi_image(img) for img in X_combined])
    X_sensors_processed = preprocess_sensor_data(X_sensors)

    print("Forma de X_processed após pré-processamento:", X_processed.shape)
    print("Forma de X_sensors_processed após pré-processamento:", X_sensors_processed.shape)

    # Certifique-se de que todas as entradas têm o mesmo número de amostras
    n_samples = min(X_processed.shape[0], X_sensors_processed.shape[0], len(y_irrigation))
    X_processed = X_processed[:n_samples]
    X_sensors_processed = X_sensors_processed[:n_samples]
    y = np.column_stack((y_irrigation[:n_samples], y_invasion[:n_samples], y_health[:n_samples], y_yield[:n_samples]))
    
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, X_sensors_train, X_sensors_test, y_train, y_test = train_test_split(
        X_processed, X_sensors_processed, y, test_size=0.2, random_state=42
    )
    
    # Criar e treinar o modelo
    model = create_model(input_shape_images=X_train.shape[1:], input_shape_sensors=X_sensors_train.shape[1])
    
    # Treinar o modelo
    epochs = 5 if test_mode else 50
    batch_size = 16 if test_mode else 32
    
    history = model.fit(
        [X_train, X_sensors_train],
        {
            'irrigation': y_train[:, 0],
            'invasion': y_train[:, 1],
            'health': y_train[:, 2],
            'yield': y_train[:, 3]
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Modelo treinado e salvo com sucesso.")