# Phishing Website Detection Using Machine Learning

## 1. Project Overview

This project focuses on detecting phishing websites using machine learning classification algorithms. It analyzes website features and classifies websites into two categories:

- **Phishing (1):** Websites identified as phishing.
- **Legitimate (-1):** Genuine websites.

Five machine learning classification algorithms are implemented and evaluated to compare their performance.

## 2. Dataset Description

**Dataset:** Phishing Websites

The dataset contains:
- Total instances: 11,055
- Input features: 30
- Target variable: `Result`
- Phishing instances: 6,157
- Legitimate instances: 4,898

The dataset contains numerical features representing different characteristics of websites.

## 3. Machine Learning Algorithms

The following five classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

- Loaded the dataset from the ARFF file.
- Converted the dataset into CSV format.
- Checked missing values and data types.
- Used a group-aware stratified train-test split to prevent identical feature combinations from appearing in both training and testing datasets.
- Applied StandardScaler for Logistic Regression, KNN and SVM.
- Used the original numerical features for Decision Tree and Gaussian Naive Bayes.

## 5. Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 92.31% | 0.9782 | 93.17% | 93.02% | 93.10% | 0.8443 |
| Decision Tree | 93.13% | 0.9396 | 95.23% | 92.29% | 93.73% | 0.8618 |
| KNN | 91.32% | 0.9648 | 92.76% | 91.56% | 92.16% | 0.8245 |
| Gaussian Naive Bayes | 60.94% | 0.9645 | 98.94% | 30.19% | 46.27% | 0.3939 |
| SVM | 94.30% | 0.9845 | 94.10% | 95.78% | 94.93% | 0.8845 |

### Observations

- SVM achieved the highest accuracy of 94.30%.
- SVM also achieved the highest AUC, F1 Score, Recall and MCC.
- Decision Tree achieved the highest precision among the models with strong overall performance.
- Gaussian Naive Bayes achieved high precision but low recall, indicating that it missed many phishing websites.
- Overall, SVM performed best in this experiment.

## 6. Project Structure

```text
Phishing_Website_Detection/
│
├── data/
│   ├── phishing.arff
│   ├── phishing.csv
│   ├── train_data.csv
│   └── test_data.csv
│
├── model/
│   ├── saved/
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree.joblib
│   │   ├── knn.joblib
│   │   ├── naive_bayes.joblib
│   │   └── svm.joblib
│   │
│   ├── results.json
│   └── feature_info.json
│
├── notebooks/
│   └── train.py
│
├── screenshots/
│
├── app.py
├── requirements.txt
└── README.md
```

## 7. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/silaparasettirisith/Phishing_Website_Detection.git
cd Phishing_Website_Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Model Training

Run the training script from the project root:

```bash
python notebooks/train.py
```

The script trains all five classification models, evaluates their performance and saves the trained models inside the `model/saved/` directory.

## 9. Run the Streamlit Application

Start the application:

```bash
streamlit run app.py
```

The application provides the following functionality:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Generate phishing or legitimate predictions.
- Display evaluation metrics when actual labels are available.
- Display a confusion matrix and classification report.
- Download prediction results as a CSV file.

The application loads saved models and does not retrain them during prediction.

## 10. Links

- **GitHub Repository:** https://github.com/silaparasettirisith/Phishing_Website_Detection
- **Live Streamlit Application:** https://phishingwebsitedetection-mqjdpnhhgvsud2oxvw2xgm.streamlit.app/
- **BITS Virtual Lab Screenshot:** See the `screenshots/` directory.

## 11. Conclusion

This project demonstrates the application of machine learning classification algorithms for phishing website detection. Five classification models were implemented and evaluated using multiple performance metrics.

Based on the experimental results, Support Vector Machine achieved the best overall performance with 94.30% accuracy and an AUC of 0.9845.
