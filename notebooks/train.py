
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, "saved")

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

print("\nLoading dataset...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

TARGET_COLUMN = "Result"

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

# Ensure train and test have the same feature columns
X_test = X_test[X_train.columns]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 3. Define five classification models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": Pipeline([
        ("classifier", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}


# --------------------------------------------------
# 4. Train models and evaluate performance
# --------------------------------------------------

results = []

print("\nStarting model training...\n")

for model_name, model in models.items():

    print("=" * 60)
    print("Training:", model_name)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate prediction probabilities for AUC
    positive_class_index = list(model.classes_).index(1)
    y_probability = model.predict_proba(X_test)[:, positive_class_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        (y_test == 1).astype(int),
        y_probability
    )

    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Save model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    model_path = os.path.join(SAVED_MODEL_DIR, model_filename)

    joblib.dump(model, model_path)

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })

    print("Accuracy :", round(accuracy, 4))
    print("AUC      :", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC      :", round(mcc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    print("Saved model:", model_filename)


# --------------------------------------------------
# 5. Display model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nMODEL PERFORMANCE COMPARISON")
print("=" * 90)

print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. Save evaluation results
# --------------------------------------------------

results_path = os.path.join(MODEL_DIR, "results.json")

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)


# --------------------------------------------------
# 7. Save feature information
# --------------------------------------------------

feature_info = {
    "feature_columns": X_train.columns.tolist(),
    "target_column": TARGET_COLUMN,
    "target_mapping": {
        "-1": "Legitimate",
        "1": "Phishing"
    }
}

feature_path = os.path.join(MODEL_DIR, "feature_info.json")

with open(feature_path, "w") as file:
    json.dump(feature_info, file, indent=4)


print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("Five models trained and saved.")
print("Models location:", SAVED_MODEL_DIR)
print("Results saved:", results_path)
print("=" * 60)
