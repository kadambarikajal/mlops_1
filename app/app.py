# Flask application code
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
MODEL_PATH = 'model.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Model is ready!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        print("Prediction based on provided features")
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
