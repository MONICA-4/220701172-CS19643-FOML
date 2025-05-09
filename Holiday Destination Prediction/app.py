from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
import numpy as np

app = Flask(__name__)

def feature_engineering(X):
    X['Beaches_Mountains'] = X['Beaches'] * X['Mountains']
    X['Historical_Romantic'] = X['Historical'] * X['Romantic Place']
    X['Family_Kids'] = X['Family Place'] * X['Kids Place']
    X['Days_Squared'] = X['No. of Days'] ** 2
    X['Location_Complexity'] = X['Within India'] + X['Outside India']
    return X

def load_model():
    if not all(os.path.exists(f) for f in ['model.pkl', 'label_encoders.pkl', 'target_encoder.pkl', 'scaler.pkl', 'final_features.pkl']):
        raise FileNotFoundError("Model files not found. Please run train_model.py first.")
    
    model = joblib.load('model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    final_features = joblib.load('final_features.pkl')
    
    return model, label_encoders, target_encoder, scaler, final_features

# Load model at startup
print("Loading trained model...")
model, label_encoders, target_encoder, scaler, final_features = load_model()
print("Model loaded successfully!")

@app.route('/', methods=['GET', 'POST'])
def index():
    columns = ['Peace Loving', 'Beaches', 'Mountains', 'Historical', 'Within India', 
              'Outside India', 'No. of Days', 'Romantic Place', 'Kids Place', 'Family Place']
    user_inputs = {}
    prediction_result = None

    if request.method == 'POST':
        for col in columns:
            value = request.form.get(col)
            if value is not None:
                user_inputs[col] = int(value) if value.isdigit() else value

        prediction_result = predict(user_inputs)

    return render_template('index.html', columns=columns, prediction=prediction_result)

def predict(input_data):
    try:
        print(f"Input data: {input_data}")
        
        # Fill missing values with 0
        default_columns = ['Peace Loving', 'Beaches', 'Mountains', 'Historical', 'Within India', 
                           'Outside India', 'No. of Days', 'Romantic Place', 'Kids Place', 'Family Place']
        for col in default_columns:
            input_data.setdefault(col, 0)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        print(f"Input DataFrame before FE: {input_df}")

        # Apply feature engineering
        input_df = feature_engineering(input_df)

        # Label encoding
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

        # Ensure all final features are present
        for feature in final_features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Reorder to match training order
        input_df = input_df[final_features]

        print("Final input to scaler:", input_df)

        # Scaling
        input_df_scaled = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(input_df_scaled, columns=final_features)

        # Prediction
        prediction = model.predict(input_df_scaled)[0]
        predicted_place = target_encoder.inverse_transform([prediction])[0]

        # Confidence
        probabilities = model.predict_proba(input_df_scaled)[0]
        confidence = max(probabilities) * 100

        return f"{predicted_place} (Confidence: {confidence:.2f}%)"
    
    except Exception as e:
        import traceback
        return f"Error making prediction: {str(e)}\n{traceback.format_exc()}"

if __name__ == '__main__':
    app.run(debug=True)
