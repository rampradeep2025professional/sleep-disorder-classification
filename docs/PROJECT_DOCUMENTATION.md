# Sleep Disorder Classification - Report, Slides and Viva Content
(All numbers come from the actual run of `train_model.py` on `data/dataset.csv`.)

---
# PART A - MINI PROJECT REPORT CONTENT

## 1. Title
Sleep Disorder Classification Using Machine Learning

## 2. Abstract
Sleep disorders affect health and daily performance, but they are often noticed late. This project builds a machine learning system that classifies a person into No Sleep Disorder, Insomnia or Sleep Apnea using sleep, lifestyle and health attributes. The Sleep Health and Lifestyle dataset (374 records, 13 columns) was cleaned, duplicate records were removed (132 unique records remained) and the data was preprocessed with a scikit-learn pipeline. Five classifiers were compared: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors and Gradient Boosting. Logistic Regression was selected using cross-validated macro F1-score and reached 70.37% accuracy and 0.657 macro F1-score on a small test set of 27 records. The model is deployed as a Streamlit web application. The system is for education only and is not a medical diagnostic tool.

## 3. Introduction
Machine learning lets computers learn patterns from data. Sleep health depends on factors such as sleep duration, stress, physical activity, body weight and heart health. This project applies supervised classification to these factors to predict a sleep-disorder category, and presents the result in a simple web interface.

## 4. Problem Statement
To build a machine learning model that predicts the sleep-disorder category (None, Insomnia, Sleep Apnea) from health and lifestyle information, and to provide it as an easy-to-use web application.

## 5. Existing System
Sleep disorders are normally identified through clinical consultation and sleep studies, which need specialists and equipment and are not always easily available. Simple self-check tools usually give only general advice and do not learn from data.

## 6. Proposed System
A data-driven classifier trained on a sleep health dataset. The user enters basic details in a web form and the model returns a predicted category with estimated probabilities. It is a learning demonstration and does not replace a doctor.

## 7. Objectives
1. Preprocess and analyze the sleep health dataset.
2. Train and compare multiple classification algorithms.
3. Select the best model using suitable metrics.
4. Build and deploy a Streamlit application.
5. Present results honestly, including limitations.

## 8. Scope
Educational classification of three categories using 11 input attributes. It covers data analysis, modelling and web deployment. It does not cover medical diagnosis, treatment or clinical validation.

## 9. Dataset Description
- Source file: Sleep Health and Lifestyle dataset (Kaggle CSV), 374 rows x 13 columns, no missing cells in the feature columns.
- Features: Gender, Age, Occupation, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps. `Person ID` is an identifier and was not used.
- Target: Sleep Disorder - None 219, Sleep Apnea 78, Insomnia 77.
- 242 records were duplicates apart from Person ID; after removal 132 remain (None 73, Sleep Apnea 30, Insomnia 29).
- Note: the text "None" is a real label, not a missing value.
- (Add the exact Kaggle dataset page and author in your references.)

## 10. Data Preprocessing
1. Loaded the CSV with `keep_default_na=False` so "None" stays a label.
2. Dropped Person ID.
3. Removed duplicate records before splitting, so identical rows cannot appear in both train and test sets.
4. Merged BMI labels "Normal Weight" and "Normal".
5. Split Blood Pressure (e.g. 126/83) into Systolic and Diastolic numbers.
6. Scaled numerical columns (StandardScaler); one-hot encoded Gender, Occupation, BMI Category.
7. Stratified 80/20 split: 105 training and 27 test records.
8. Everything is in one scikit-learn Pipeline/ColumnTransformer, so training and prediction use identical steps.

## 11. Exploratory Data Analysis
Twelve plots are saved in `images/`: class distribution, distributions of age, sleep duration, sleep quality, stress, physical activity and BMI, box/count plots of sleep duration, stress, sleep quality and BMI against sleep disorder, and a correlation heatmap. Open these images and describe what you observe in your own words (for example, how the three classes differ in stress level and sleep quality) - look at the plots before writing conclusions.

## 12. Algorithms Used
- **Logistic Regression** - linear model that estimates class probabilities.
- **Decision Tree** - asks a series of yes/no questions on features.
- **Random Forest** - many decision trees voting together.
- **K-Nearest Neighbors** - predicts using the 5 most similar records.
- **Gradient Boosting** - trees built one after another, each correcting earlier errors.

## 13. System Architecture
User Input (Streamlit form) → Validation → Saved Preprocessing Pipeline → Saved Model → Prediction + Probabilities → Displayed Category

## 14. Methodology
Data collection → cleaning and duplicate removal → preprocessing pipeline → train/test split → train 5 models → 5-fold cross-validation on training data → test-set evaluation (accuracy, precision, recall, F1, confusion matrix) → select best model by cross-validated macro F1 → save with Joblib → build Streamlit app → deploy from GitHub.

## 15. Implementation
Python with Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib and Streamlit. Files: `preprocessing.py` (shared pipeline), `train_model.py` (training), `eda.py` (plots), `predict_utils.py` (validation and prediction), `app.py` (web app). The app loads `model/model.pkl` and `model/preprocessor.pkl` and never retrains.

## 16. Results
Test set of 27 records; precision, recall and F1 are macro-averaged.

| Model | Accuracy | Precision | Recall | F1 | CV F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7037 | 0.6833 | 0.6556 | 0.6571 | 0.7156 |
| Decision Tree | 0.5556 | 0.5074 | 0.4667 | 0.4554 | 0.4519 |
| Random Forest | 0.6296 | 0.5595 | 0.5444 | 0.5452 | 0.5465 |
| K-Nearest Neighbors | 0.6667 | 0.6508 | 0.6333 | 0.6094 | 0.6345 |
| Gradient Boosting | 0.6296 | 0.5500 | 0.5444 | 0.5459 | 0.5209 |

Logistic Regression was selected because it had the highest cross-validated macro F1 and also the best test scores. Its per-class F1 on the test set: Insomnia 0.57, None 0.80, Sleep Apnea 0.60. Because the test set has only 27 records, these results are approximate.

## 17. Advantages
- Simple, fast and easy to use; saved model loads instantly.
- Same preprocessing in training and prediction.
- Duplicate removal and cross-validation give more honest results.
- Shows probabilities instead of only a label.

## 18. Limitations
- Very small dataset (132 unique records); noisy metrics.
- Occupation is strongly tied to the label in this data, so the model may learn dataset-specific patterns.
- Inputs only valid within the training data ranges.
- Not clinically validated; not a diagnostic tool.

## 19. Future Enhancement
Larger, more diverse and clinically sourced data; hyperparameter tuning; feature-importance or SHAP explanations; more medical features; model monitoring.

## 20. Conclusion
The project delivers a complete, reproducible ML pipeline and web application for sleep-disorder classification. Logistic Regression performed best among the tested models, reaching about 70% accuracy on a small test set. Results are limited by the small dataset, so the system should be used only for learning and demonstration.

## 21. References
1. Sleep Health and Lifestyle Dataset - Kaggle (add the exact page URL and author you downloaded it from).
2. Pedregosa et al., "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research, 2011.
3. Streamlit documentation - https://docs.streamlit.io
4. Pandas documentation - https://pandas.pydata.org/docs
5. Matplotlib and Seaborn documentation.

---
# PART B - PRESENTATION (12 slides)

1. **Title** - Sleep Disorder Classification Using ML; your name, guide, college, year.
2. **Introduction** - Sleep affects health; ML can find patterns in lifestyle data; web app for demo.
3. **Problem Statement** - Predict None / Insomnia / Sleep Apnea from health and lifestyle data.
4. **Objectives** - Preprocess data; compare models; choose best fairly; deploy Streamlit app.
5. **Dataset** - 374 rows, 13 columns; 132 unique after duplicate removal; classes None 73, Apnea 30, Insomnia 29.
6. **Data Preprocessing** - Drop ID; remove duplicates; merge BMI labels; split blood pressure; scale + one-hot; 80/20 stratified split.
7. **ML Algorithms** - Logistic Regression, Decision Tree, Random Forest, KNN, Gradient Boosting.
8. **System Architecture** - Input → Preprocessing → Model → Prediction → Category (diagram).
9. **Model Evaluation** - Metrics table; confusion matrix; selection by CV macro F1.
10. **Streamlit Application** - Screenshot of form, sidebar, prediction and probabilities.
11. **Results** - Logistic Regression: 70.37% accuracy, 0.657 macro F1 (27 test records); honest limits.
12. **Conclusion & Future Scope** - Working end-to-end system; needs more data, explainability, clinical validation.

---
# PART C - VIVA QUESTIONS AND ANSWERS

1. **What is Machine Learning?** Teaching computers to learn patterns from data instead of writing fixed rules.
2. **What is classification?** Predicting which category (class) an input belongs to, e.g. Insomnia or not.
3. **What is sleep disorder classification?** Predicting a person's sleep-disorder category from sleep, lifestyle and health data.
4. **Why this project?** Sleep health matters, and it is a good practical example of the full ML workflow plus deployment.
5. **What dataset was used?** The Sleep Health and Lifestyle dataset: 374 rows, 13 columns; 132 unique records after removing duplicates.
6. **What is preprocessing?** Cleaning and converting raw data (removing duplicates, scaling, encoding) so models can use it.
7. **Why encoding?** Models need numbers; one-hot encoding turns categories like Occupation into 0/1 columns.
8. **Why train-test split?** To test the model on data it has not seen, giving an honest estimate of performance.
9. **What is Logistic Regression?** A model that computes class probabilities from a weighted sum of features; despite the name it is used for classification.
10. **What is Random Forest?** Many decision trees trained on random parts of the data whose votes are combined.
11. **What is KNN?** Classifies a record by the majority class of its K nearest records (K = 5 here).
12. **What is overfitting?** When a model memorizes training data and performs poorly on new data. A Decision Tree overfits easily.
13. **What is accuracy?** Correct predictions divided by total predictions.
14. **What is precision?** Of the records predicted as a class, how many truly belong to it.
15. **What is recall?** Of the records that truly belong to a class, how many the model found.
16. **What is F1-score?** The harmonic mean of precision and recall; macro F1 averages it over classes equally.
17. **What is a confusion matrix?** A table of actual versus predicted classes showing which classes get confused.
18. **Why did you select the final model?** Logistic Regression had the highest cross-validated macro F1 on training data (0.716) and also the best test results; macro F1 is fairer than accuracy because classes are imbalanced.
19. **Why did you remove duplicates?** 242 rows were identical except Person ID; keeping them would put copies in both train and test sets and inflate scores.
20. **Why not use Person ID?** It is just an identifier with no real meaning; using it would be misleading.
21. **Why Streamlit?** It turns Python scripts into web apps quickly, with no HTML/JavaScript, and deploys free on Streamlit Community Cloud.
22. **How is the model deployed?** Code and saved model files go to GitHub; Streamlit Community Cloud runs `app.py` from the repository.
23. **Why save the model with Joblib?** So the app loads the trained model instantly instead of retraining each time.
24. **What are the limitations?** Small dataset, noisy metrics, occupation bias in the data, not clinically validated, not a diagnostic tool.
25. **What is cross-validation?** Splitting training data into folds and rotating which fold is used for validation, to get a more stable estimate.
