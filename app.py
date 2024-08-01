import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


# Load the trained model (assuming it's saved as 'model.pkl')
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
