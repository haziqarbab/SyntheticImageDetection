import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json

def evaluate_model(df, model_type, model_dir="models/", result_dir="results/", visualize_misclassifications=True):
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_type}.pkl")
    scaler_path = os.path.join(model_dir, f"{model_type}_scaler.pkl")
    split_path = os.path.join(model_dir, "split_info.csv")

    if not (os.path.exists(model_path) and os.path.exists(split_path)):
        raise FileNotFoundError("Model or split_info not found.")

    model = joblib.load(model_path)
    split_info = pd.read_csv(split_path)

    # Try to load scaler if available
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    test_files = split_info[split_info["split"] == "test"]["filename"]
    df_test = df[df["filename"].isin(test_files)].copy()

    # Load feature column names
    features_path = os.path.join(model_dir, "features_used.json")
    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    X_test = df_test[feature_cols]

    if scaler:
        X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_scaled = X_test.copy()

    y_test = df_test["label"]
    y_pred = model.predict(X_scaled)

    # Get predicted probabilities for ROC AUC
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            positive_class_index = list(model.classes_).index("Fake")
            y_prob = model.predict_proba(X_scaled)[:, positive_class_index]
        except ValueError:
            print("[WARNING] 'Fake' not found in model classes, skipping AUC computation.")

    # Save confusion matrix
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(result_dir, "classification_report.csv"))

    # Save ROC AUC and ROC curve
    # Save ROC AUC
    auc_val = None
    if y_prob is not None:
        try:
            # Determine index of the "Fake" class in model.classes_
            positive_class = "Fake"
            pos_index = list(model.classes_).index(positive_class)

            y_prob = model.predict_proba(X_scaled)[:, pos_index]
            binary_labels = (y_test == positive_class).astype(int)  # 1 for Fake, 0 for Real

            auc_val = roc_auc_score(binary_labels, y_prob)
            with open(os.path.join(result_dir, "roc_auc.txt"), "w") as f:
                f.write(f"{auc_val:.4f}")

            # ROC Curve
            fpr, tpr, _ = roc_curve(binary_labels, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "roc_curve.png"))
            plt.close()
        except Exception as e:
            print(f"[WARNING] Could not compute AUC/ROC curve: {e}")

    # Save misclassifications
    if visualize_misclassifications:
        misclassified = df_test[y_test != y_pred].copy()
        misclassified.loc[:, "predicted"] = y_pred[y_test != y_pred]
        misclassified.to_csv(os.path.join(result_dir, "misclassified.csv"), index=False)

    return y_pred, y_test
