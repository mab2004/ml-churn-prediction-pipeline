# 📊 End-to-End ML Pipeline: Telco Customer Churn Prediction

## Overview

This repository contains a complete, production-ready machine learning pipeline developed as part of an AI/ML Engineering Internship. The objective was to build a reusable solution for predicting customer churn using standard ML engineering practices.

The solution uses the **Telco Customer Churn Dataset** to train and optimize a **Random Forest Classifier** within a single, unified scikit-learn pipeline.


## 🎯 Objective

The primary goal of this task was to construct a robust, production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn Dataset. This project demonstrates core ML Engineering skills, including pipeline construction, comprehensive data preprocessing, hyperparameter optimization, and model serialization for reusability.

The final deliverable is a single, exported model object capable of handling raw input data, preprocessing it, and generating an accurate prediction, adhering to **production-readiness standards**.


## ⚙️ Technologies and Tools

* **Python 3.x**
* **scikit-learn:** For the Pipeline, ColumnTransformer, and Model (Random Forest).
* **GridSearchCV:** For hyperparameter tuning.
* **joblib:** For serializing and exporting the final model.
* **Pandas/NumPy:** For data handling.

## 🚀 Key Features and Deliverables

* **End-to-End ML Pipeline:** Handles all steps from raw data imputation and scaling/encoding to final prediction.
* **Hyperparameter Tuning:** Optimized the Random Forest model using **5-fold Cross-Validation** via `GridSearchCV`.
* **Production-Ready Export:** The entire pipeline object (including the preprocessor and the best model) is exported using `joblib` for seamless deployment.

## 🧩 Methodology / Approach

### 1. Data Preparation and Transformation

The initial dataset was loaded, cleaned, and split into training/testing sets. A **`ColumnTransformer`** was implemented within a scikit-learn pipeline to ensure appropriate preprocessing for different feature types:

* **Numerical Features:** Missing values were imputed (median) and features were scaled using **`StandardScaler`**.
* **Categorical Features:** Missing values were imputed (most frequent) and features were encoded using **`OneHotEncoder`**.

### 2. Model Selection and Optimization

A **Random Forest Classifier** was selected for the prediction task. The full pipeline (preprocessor + model) was subjected to hyperparameter tuning using **`GridSearchCV`** with 5-fold cross-validation (`cv=5`). This systematic search was conducted to find the optimal combination of `n_estimators`, `max_depth`, and `min_samples_leaf`.

### 3. Production Export

The best-performing model—which includes the trained preprocessing steps—was serialized and saved as a single object (`churn_prediction_pipeline.joblib`) using the **`joblib`** library. This ensures the entire workflow is reusable and deployable.


## 📁 Repository Structure

The core logic is contained in a single Python script (or notebook cells) which performs the following steps sequentially:

| File/Section | Description |
| :--- | :--- |
| **`Telco-Customer-Churn.csv`** | The raw dataset used for training. |
| **`churn_prediction_pipeline.joblib`** | The final exported ML object. |
| **Code Cells (or `ml-churn-prediction-pipeline.ipynb`)** | Implementation of data loading, preprocessing (`ColumnTransformer`), pipeline construction, and `GridSearchCV` tuning. |

## ✨ Key Results & Observations

### Optimized Model Performance

The Random Forest Pipeline achieved the following performance metrics on the held-out test set:

| Metric | Score |
| :--- | :--- |
| **Best CV Score** (Training Accuracy) | ~80.2% |
| **Final Test Set Accuracy** (Validation) | **~80.9%** |

### Observation on Reusability:

The successful export and reload of the `joblib` file confirmed that the complete preprocessing logic and the final model are bundled together. This verified the **reusability** of the pipeline, which is essential for a production environment.

### Best Model Hyperparameters:

The optimal configuration found by `GridSearchCV` was:
```python
{'classifier__max_depth': 10, 
 'classifier__min_samples_leaf': 4, 
 'classifier__n_estimators': 100}
```

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details.

