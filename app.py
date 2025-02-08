from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import os

app = Flask(__name__)
CORS(app)

# Cargar datos
file_path = './datos.csv'
data = pd.read_csv(file_path, sep=';')

# Revisar la distribución de los datos
print(data['output'].describe())

# Preparar datos
encoder = OneHotEncoder(sparse_output=False)
dominant_foot_encoded = encoder.fit_transform(data[['dominantFoot']])
dominant_foot_df = pd.DataFrame(dominant_foot_encoded, columns=encoder.get_feature_names_out(['dominantFoot']))

# Incluir 'achievements', 'injuryHistory', y 'trainingHoursPerWeek' en las features
features = pd.concat([data.drop(columns=['output', 'dominantFoot']), dominant_foot_df], axis=1)

X = features.values
y = data['output'].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Cargar o entrenar modelo
if os.path.exists('modelo_entrenado.h5'):
    model = tf.keras.models.load_model('modelo_entrenado.h5')
else:
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)
    model.save('modelo_entrenado.h5')

# Predicción
def predecir_puntaje(nueva_jugadora):
    try:
        dominant_foot = encoder.transform([[nueva_jugadora['dominantFoot']]])
        features = [
            nueva_jugadora['position'],
            nueva_jugadora['height'],
            nueva_jugadora['weight'],
            nueva_jugadora['experience'],
            nueva_jugadora['videoUploaded'],
            nueva_jugadora['ambidextrous'],
            nueva_jugadora['versatility'],
            nueva_jugadora['achievements'],
            nueva_jugadora['injuryHistory'],
            nueva_jugadora['trainingHoursPerWeek']
        ]
        features.extend(dominant_foot.flatten())
        features_scaled = scaler.transform([features])
        puntaje = float(model.predict(features_scaled)[0][0] * 100)
        return max(0, min(100, puntaje))
    except Exception as e:
        return {"error": f"Error al procesar la entrada: {str(e)}"}

# Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "El cuerpo de la solicitud debe ser JSON"}), 400
    try:
        data = request.get_json()
        puntaje = predecir_puntaje(data)
        return jsonify({"puntaje": puntaje})
    except Exception as e:
        return jsonify({"error": f"Error al predecir el puntaje: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
