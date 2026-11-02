
import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, "saved")

FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.json")


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Phishing Website Detection")
st.write(
    "Upload website feature data and select a machine learning "
    "model to identify phishing websites."
)


# --------------------------------------------------
# 3. Load feature information
# --------------------------------------------------

if not os.path.exists(FEATURE_INFO_PATH):
    st.error("Feature information file not found.")
    st.stop()

with open(FEATURE_INFO_PATH, "r") as file:
    feature_info = json.load(file)

FEATURE_COLUMNS = feature_info["feature_columns"]
TARGET_COLUMN = feature_info["target_column"]


# --------------------------------------------------
# 4. Model selection
# --------------------------------------------------

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

st.subheader("1. Select Machine Learning Model")

selected_model = st.selectbox(
    "Choose a classification algorithm",
    list(model_files.keys())
)


# --------------------------------------------------
# 5. Upload test dataset
# --------------------------------------------------

st.subheader("2. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(uploaded_df.head())

    st.write("Dataset Shape:", uploaded_df.shape)

    # Check required feature columns
    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in uploaded_df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required feature columns: {missing_columns}"
        )
        st.stop()

    # Prepare features in the original training order
    X = uploaded_df[FEATURE_COLUMNS]

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        st.error("All input features must be numeric.")
        st.stop()

    if X.isnull().any().any():
        st.error("Uploaded data contains missing feature values.")
        st.stop()

    # Load selected model
    model_path = os.path.join(
        SAVED_MODEL_DIR,
        model_files[selected_model]
    )

    if not os.path.exists(model_path):
        st.error(f"Saved model not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(X)

    st.subheader("3. Prediction Results")

    if TARGET_COLUMN in uploaded_df.columns:

        y_true = uploaded_df[TARGET_COLUMN]

        if not y_true.isin([-1, 1]).all():
            st.error("Target column must contain only -1 and 1.")
            st.stop()

        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)

        precision = precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        )

        recall = recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        )

        f1 = f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        )

        mcc = matthews_corrcoef(y_true, predictions)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X)[:, positive_index]

        auc = roc_auc_score(
            (y_true == 1).astype(int),
            probabilities
        )

        # Display metrics
        st.subheader("4. Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.2%}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.2%}")
        col5.metric("F1 Score", f"{f1:.2%}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.subheader("5. Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            predictions,
            labels=[-1, 1]
        )

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Legitimate", "Phishing"]
        )

        display.plot(ax=ax, cmap="Blues", values_format="d")

        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.subheader("6. Classification Report")

        report = classification_report(
            y_true,
            predictions,
            labels=[-1, 1],
            target_names=["Legitimate", "Phishing"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(pd.DataFrame(report).transpose())

    else:

        st.info(
            "No Result column found. Evaluation metrics cannot be "
            "calculated, but predictions are available."
        )

    # Display predictions
    result_df = uploaded_df.copy()

    result_df["Predicted_Result"] = predictions

    result_df["Prediction"] = result_df["Predicted_Result"].map({
        1: "Phishing",
        -1: "Legitimate"
    })

    st.subheader("7. Website Classification")

    st.dataframe(result_df)

    # Download predictions
    csv_data = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv_data,
        file_name="phishing_predictions.csv",
        mime="text/csv"
    )
