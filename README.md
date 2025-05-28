# MachineLearning
# Predictive Modeling of Diabetes Using Machine Learning

## Overview
This project presents a comprehensive machine learning framework for the **early prediction of diabetes** using clinical and physiological attributes. The solution leverages multiple supervised learning algorithms to identify individuals at risk, supporting timely intervention and informed medical decisions.

The model was developed using the **Pima Indians Diabetes Dataset**, a widely used benchmark for diabetes classification problems. Key stages include data preprocessing, model training, hyperparameter tuning, performance evaluation, and comparison with existing approaches.

---

## Dataset Description
The dataset comprises health indicators from female patients of Pima Indian heritage, including:

| Feature                 | Description                                  |
|------------------------|----------------------------------------------|
| Pregnancies            | Number of times pregnant                     |
| Glucose                | Plasma glucose concentration                 |
| BloodPressure          | Diastolic blood pressure (mm Hg)             |
| SkinThickness          | Triceps skin fold thickness (mm)             |
| Insulin                | 2-Hour serum insulin (mu U/ml)               |
| BMI                    | Body mass index (weight in kg/(height²))     |
| DiabetesPedigreeFunction | Genetic predisposition indicator           |
| Age                    | Age (years)                                  |
| Outcome                | Diabetes presence (1) or absence (0)         |

---

##  Methodology

### Data Preprocessing
- Missing values imputed using median strategy
- Feature scaling and normalization
- Outlier detection using **Local Outlier Factor (LOF)**
- Data partitioned via **K-Fold Cross Validation**

### Exploratory Data Analysis
- Distribution analysis and correlation mapping
- Feature importance analysis
- Visualizations of key predictors (Glucose, BMI, Age)

###  Algorithms Implemented
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **LightGBM**
- **XGBoost (Highest performing model)**

---

## Model Evaluation

Evaluation metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Sensitivity (True Positive Rate)**

| Algorithm           | Accuracy (%) | Sensitivity (%) |
|---------------------|--------------|------------------|
| Logistic Regression | 82.46        | 68.23            |
| SVM                 | 79.22        | 59.99            |
| Naive Bayes         | 79.22        | 64.44            |
| Random Forest       | 81.81        | 68.88            |
| **XGBoost**         | **90.00**    | —                |

> The **XGBoost** model achieved the highest accuracy, making it the most robust model for diabetes classification in this study.

---

##  Key Insights
- **BMI**, **Glucose**, and **Diabetes Pedigree Function** emerged as the most predictive features.
- **XGBoost** provided superior performance due to its boosting and regularization capabilities.
- Ensemble methods like Random Forest and Gradient Boosting demonstrated high generalization with reduced variance.

---

##  Future Enhancements
- Introduce **feature engineering** (e.g., interaction terms, binning)
- Apply **SMOTE/ADASYN** for class imbalance
- Incorporate **explainable AI** tools like SHAP/LIME
- Deploy using **FastAPI** or **Flask** as a clinical decision support tool
- Leverage **automated hyperparameter tuning** (Bayesian Optimization)


