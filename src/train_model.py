import os
import joblib
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------- MODEL REGISTRY ---------------------------

def get_model(model_type):
    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        return SVC(probability=True)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=100)
    elif model_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=5000, random_state=42)
    elif model_type == "ensemble":
        return VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svm', SVC(probability=True))
        ], voting='soft')
    elif model_type == "stacked":
        base_estimators = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svm', SVC(probability=True))
        ]
        final_estimator = LogisticRegression()
        return StackingClassifier(estimators=base_estimators, final_estimator=final_estimator, passthrough=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --------------------------- TRAINING FUNCTION ---------------------------

def train_and_save_model(df, label_col="label", model_type="logistic", output_dir="models", test_size=0.2, use_scaler=True):
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=[label_col, "filename"], errors="ignore")
    y = df[label_col]

    X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
        X, y, df["filename"], test_size=test_size, stratify=y, random_state=42
    )

    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(output_dir, f"{model_type}_scaler.pkl"))
    else:
        scaler = None

    model = get_model(model_type)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(output_dir, f"{model_type}.pkl"))

    split_info = pd.DataFrame({
        "filename": list(fn_train) + list(fn_test),
        "split": ["train"] * len(fn_train) + ["test"] * len(fn_test)
    })
    split_info.to_csv(os.path.join(output_dir, "split_info.csv"), index=False)

    # Save feature column names
    feature_cols = X.columns.tolist()
    with open(os.path.join(output_dir, "features_used.json"), "w") as f:
        json.dump(feature_cols, f)

    return model, scaler, X_test, y_test
