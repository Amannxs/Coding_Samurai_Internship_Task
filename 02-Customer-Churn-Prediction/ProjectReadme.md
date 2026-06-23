# Telecom Customer Churn Prediction Dashboard

This repository contains the end-to-end machine learning pipeline and local Streamlit deployment code for predicting customer churn risk. The core engine is built using LightGBM and achieves highly robust classification bounds on a dense dataset containing over 440,000 subscriber records.

---

## 🛠️ Dev Log: What We Tackled & Solved

During the implementation phases inside our Kaggle environment, we ran into a couple of workflow errors and design choices that we successfully resolved:

### 1. The Missing Pipeline Object (`NameError`)
* **The Bug:** While testing early cells, we threw a `NameError: name 'pipeline' is not defined` when trying to call `.fit()` inside the validation blocks. 
* **The Fix:** We realized the structural `Pipeline` combining our `ColumnTransformer` preprocessor and the `LGBMClassifier` wasn't explicitly initialized in the preceding workspace cells. We cleanly integrated Scikit-Learn's `Pipeline` wrapper to pack both scaling and model constraints together before triggering structural data loops.

### 2. Manual Encoding and Feature Scaling Mismatch
* **The Problem:** The initial raw telecom dataset included mixed datatypes such as string categoricals for `Gender`, `Contract Length`, and `Subscription Type`. Standard numeric estimators require direct arithmetic representations.
* **The Fix:** We built explicit manual dictionaries (`contract_mapping`, `sub_mapping`, `gender_mapping`) to systematically transfigure columns into meaningful numeric steps (e.g., mapping Monthly contracts to 1, Annual to 12) rather than random categorical hashes. This kept feature evaluation mathematically sensible for the tree-based splits.

### 3. Training Splitting Strategy
* **The Implementation:** We initially split the training array down using a stratified 80/20 train-test split (`X_train_split`, `y_train_split`) to compute validation baselines. The pipeline hit a near-perfect metric array (1.00 Precision/Recall on test frames). Once verified, we refitted the entire pipeline object against the **whole training dataset (`X_train`, `y_train`) containing exactly 440,832 data rows** to ensure maximum pattern generalizability before final deployment.

### 4. Model Export & File Link Dowload Setup
* **The Workflow:** We wrapped the fully trained state using `joblib.dump()` into a standalone file `final_customer_churn_model.pkl`. To pull it down to our local VS Code desktop environment seamlessly without fighting the Kaggle storage sidebar, we used `from IPython.display import FileLink` to generate a direct download route right inside the cell outputs.

---

## 📊 Pipeline Data Architecture & Features

The baseline architecture handles the following 10 structural customer metrics mapped inside a single automated data transformer layout:

* **Age:** Numerical age of the customer (Highly correlated).
* **Gender:** Binary feature mapped as `{'Male': 0, 'Female': 1}`.
* **Tenure:** Number of continuous subscription months.
* **Usage Frequency:** Number of times the customer interacted with the service per month.
* **Support Calls:** Number of technical assist requests logged.
* **Payment Delay:** Historical payment grace delays counted in days.
* **Subscription Type:** Categorical tiers mapped hierarchically as `{'Premium': 1, 'Standard': 2, 'Basic': 3}`.
* **Contract Length:** Temporal commitments mapped as `{'Monthly': 1, 'Quarterly': 6, 'Annual': 12}`.
* **Total Spend:** Float representation of total historical transactional values.
* **Last Interaction:** Days elapsed since the subscriber last utilized the core platform ecosystem.

---

## 🚀 Step-by-Step Local Deployment Guide

1. Clone or extract this project setup into a clean folder on your local computer using VS Code.
2. Download your generated model checkpoint file (`final_customer_churn_model.pkl`) from your Kaggle workspace output directory and drop it into the exact same folder where `churn_app.py` sits.
3. Open up your VS Code terminal window and spin up the dependency stack:
   ```bash
   pip install -r requirements.txt