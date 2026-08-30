from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # All origins allowed for CORS

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diabetes_xgboost_pipeline.pkl')

model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model loading failed: {e}")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is live', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded'}), 500

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data received'}), 400

        input_df = pd.DataFrame([data])

        # Step 1: One-Hot Encoding for text values
        categorical_cols = input_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            input_df = pd.get_dummies(input_df)

        # Step 2: Feature alignment check for both raw models & pipelines
        expected_features = None
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        elif hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected_features = step.feature_names_in_
                    break

        if expected_features is not None:
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_features]

        # Step 3: Probability calculation
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0][1])
        else:
            prob = float(model.predict(input_df)[0])

        is_diabetic = bool(prob >= 0.1268)

        return jsonify({
            'risk_score': round(prob * 100, 1),
            'is_high_risk': is_diabetic
        })

    except Exception as e:
        return jsonify({'error': f"Model Execution Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)