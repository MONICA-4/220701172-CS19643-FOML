import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

def load_data():
    print("Loading dataset...")
    data_path = "Holiday_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found. Please ensure 'Holiday_data.csv' is in the correct directory.")
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
    return data

def feature_engineering(X):
    X['Beaches_Mountains'] = X['Beaches'] * X['Mountains']
    X['Historical_Romantic'] = X['Historical'] * X['Romantic Place']
    X['Family_Kids'] = X['Family Place'] * X['Kids Place']
    X['Days_Squared'] = X['No. of Days'] ** 2
    X['Location_Complexity'] = X['Within India'] + X['Outside India']
    return X

def preprocess_data(data):
    target_column = 'Places'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    label_encoders = {}

    X = feature_engineering(X)
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    scaler = StandardScaler()
    X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(X.select_dtypes(include=['int64', 'float64']))
    
    return X, y, label_encoders, target_encoder, scaler

def train_model():
    print("\nStarting model training process...")
    data = load_data()
    X, y, label_encoders, target_encoder, scaler = preprocess_data(data)
    
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    param_grids = {
        'random_forest': {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
        'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        'logistic_regression': {'C': [0.1, 1], 'penalty': ['l2']}
    }

    best_model = None
    best_score = 0
    best_model_name = None
    best_params = None
    
    print("\nTraining and evaluating models...")
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = model_name
            best_params = grid_search.best_params_
            print(f"New best model found: {model_name}")

    print("\nTraining completed!")
    print(f"Best model: {best_model_name}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}\n")
    
    print("\n🔹 Evaluating the Best Model on Full Dataset:")
    y_pred = best_model.predict(X)
    
    print("\n🔹 Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\n🔹 Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print("\nSaving the model and preprocessing objects...")
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'final_features.pkl')
    print("✅ Model saved successfully!")
    
    return best_model, best_score

if __name__ == '__main__':
    print("Starting model training...")
    model, score = train_model()
    print(f"\n✅ Training completed with score: {score:.4f}")
