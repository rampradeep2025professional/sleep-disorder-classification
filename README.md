# Sleep Disorder Classification Using Machine Learning

A machine-learning web app that classifies sleep-health records into **No Sleep Disorder**, **Insomnia** or **Sleep Apnea**, built with scikit-learn and Streamlit.

> ⚠️ **Disclaimer:** This application is developed for educational and machine-learning demonstration purposes only. It is not a medical diagnostic tool. Please consult a qualified healthcare professional for medical concerns.

## 1. Project Overview
Users enter sleep, lifestyle and health details. A saved scikit-learn model predicts the sleep-disorder category and shows estimated class probabilities.

## 2. Problem Statement
Sleep disorders are often noticed late. This project explores whether simple lifestyle and health attributes can be used to classify sleep-disorder categories with machine learning.

## 3. Objectives
- Clean and preprocess the Sleep Health and Lifestyle dataset
- Explore the data with visualizations
- Train and compare several classifiers and choose the best one fairly
- Deploy the saved model in an easy-to-use Streamlit app

## 4. Dataset
`data/dataset.csv` - Sleep Health and Lifestyle dataset (Kaggle; 374 rows, 13 columns).
- Target: `Sleep Disorder` (None 219, Sleep Apnea 78, Insomnia 77)
- `Person ID` is an identifier and is **not** used as a feature.
- 242 rows were identical to another row apart from `Person ID`; after removing them, **132 records** remain (None 73, Sleep Apnea 30, Insomnia 29).
- In the CSV the text `None` is a real class label; it is read with `keep_default_na=False` so pandas does not treat it as missing.

## 5. Technologies Used
Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit, Git/GitHub, Streamlit Community Cloud.

## 6. Machine Learning Algorithms
Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Gradient Boosting (extra model).

## 7. Data Preprocessing
Implemented once in `preprocessing.py` and reused by training and the app:
1. Drop `Person ID`; remove duplicate records (before splitting, so copies cannot leak into the test set)
2. Merge BMI labels "Normal Weight" and "Normal"
3. Split `Blood Pressure` (e.g. `126/83`) into `Systolic BP` and `Diastolic BP`
4. Standard-scale numerical columns, one-hot encode `Gender`, `Occupation`, `BMI Category` (`ColumnTransformer` + `Pipeline`)
5. Stratified 80/20 train-test split (105 train / 27 test)

## 8. Exploratory Data Analysis
Run `python eda.py`. 12 plots are saved in `images/` (class distribution, feature distributions, features vs. disorder, correlation heatmap).

## 9. Model Evaluation
Test set = 27 records; "CV F1" = 5-fold stratified cross-validation macro-F1 on the training set. Precision, recall and F1 are macro-averaged. Full table: `results/model_comparison.csv`; per-class reports: `results/classification_reports.txt`; confusion matrices: `images/`.

| Model | Accuracy | Precision | Recall | F1 | CV F1 (mean ± std) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7037 | 0.6833 | 0.6556 | 0.6571 | 0.7156 ± 0.0633 |
| Decision Tree | 0.5556 | 0.5074 | 0.4667 | 0.4554 | 0.4519 ± 0.0257 |
| Random Forest | 0.6296 | 0.5595 | 0.5444 | 0.5452 | 0.5465 ± 0.0723 |
| K-Nearest Neighbors | 0.6667 | 0.6508 | 0.6333 | 0.6094 | 0.6345 ± 0.0981 |
| Gradient Boosting | 0.6296 | 0.5500 | 0.5444 | 0.5459 | 0.5209 ± 0.1178 |

## 10. Best Model
**Logistic Regression.** It was chosen by the highest cross-validated macro-F1 on the *training* data (the test set was not used for selection). Macro-F1 gives each class equal weight, which matters because "None" has more samples. It was also best on the test set. With only 27 test records, differences are noisy - treat the numbers as rough estimates.

## 11. Streamlit Application
`app.py` loads `model/model.pkl` and `model/preprocessor.pkl` (it never retrains), validates inputs, shows the prediction with class probabilities, model metrics, a sidebar (About, Model Information, Disclaimer) and friendly error messages.

## 12. Project Structure
```
sleep-disorder-classification/
├── app.py                # Streamlit app
├── train_model.py        # training + comparison + saving
├── preprocessing.py      # shared cleaning/preprocessing pipeline
├── predict_utils.py      # loading, validation, prediction helpers
├── eda.py                # EDA plots
├── requirements.txt
├── README.md
├── .gitignore
├── data/dataset.csv
├── model/                # model.pkl, preprocessor.pkl, metadata.json
├── results/              # metrics tables and reports
├── notebooks/sleep_disorder_analysis.ipynb
├── docs/                 # report, slides and viva content
└── images/               # plots and screenshots
```

## 13. Installation
```bash
git clone <repository-url>
cd sleep-disorder-classification
pip install -r requirements.txt
streamlit run app.py
```
To retrain: `python train_model.py` (then commit the updated `model/` files).

## 14. Deployment (Streamlit Community Cloud)
1. Push the project to GitHub (the `model/` folder must be included).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repository, branch `main`, main file `app.py`.
4. In **Advanced settings** choose Python 3.11 or newer (required by the pinned scikit-learn).
5. Click **Deploy**.

## 15. Limitations
- Very small dataset (132 unique records, 27 test records), so metrics are noisy
- Occupation is strongly linked to the label in this data (e.g. most Nurses are Sleep Apnea) - the model may learn dataset quirks rather than real medical patterns
- Inputs are limited to the ranges in the training data; no clinical validation

## 16. Future Enhancements
Larger and more diverse data, hyperparameter tuning, explainability (feature importance/SHAP), more clinically relevant features.

## 17. Disclaimer
This application is developed for educational and machine-learning demonstration purposes only. It is not a medical diagnostic tool. Please consult a qualified healthcare professional for medical concerns.
