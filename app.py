from flask import Flask, request, jsonify
from flask_cors import CORS   # 👈 ADD THIS LINE

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

CORS(app)

# ---------- PREPROCESS ----------
def preprocess(df):
    df = df.fillna(df.mean())
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled)

# ---------- FEATURE SELECTION (SIMPLIFIED DE) ----------
def feature_selection(X, y):
    # Dummy DE selection (you can extend)
    return X[:, :2]   # select first 2 features

# ---------- BPNN ----------
def run_bp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MLPClassifier(hidden_layer_sizes=(12,8), max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

# ---------- API ----------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_excel(file)

    df = preprocess(df)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_selected = feature_selection(X, y)

    result = run_bp(X_selected, y)

    return jsonify(result)

# ---------- RUN ----------
if __name__ == '__main__':
    app.run(debug=True)