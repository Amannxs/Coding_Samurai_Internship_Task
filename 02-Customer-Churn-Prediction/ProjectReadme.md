Elecom Customer Churn Prediction Dashboard
A complete machine learning pipeline with a local Streamlit dashboard for predicting customer churn risk. The model uses LightGBM and was trained on over 440,000 subscriber records.

What I Ran Into While Building This
Working through the development in Kaggle, I hit a few snags that needed sorting out:

The Pipeline Error
Early on, I kept getting a NameError: name 'pipeline' is not defined when trying to run .fit() in my validation code. Turned out I hadn't actually initialized the Scikit-Learn Pipeline that combined my preprocessor and the LGBMClassifier. Once I wrapped everything together properly, it started working.

Handling Mixed Data Types
The raw telecom data had a mix of text and numbers - things like Gender, Contract Length, and Subscription Type were all strings. Instead of letting the model guess how to handle them, I created simple mapping dictionaries (contract_mapping, sub_mapping, gender_mapping) to convert these into meaningful numbers. For example, Monthly contracts became 1, Annual became 12 - this made more sense for the tree-based model than random category codes.

Training and Validation
I started with a stratified 80/20 split to get initial performance metrics. The model performed really well (hit 1.00 Precision/Recall on the test set). After confirming it worked, I retrained on the full dataset (all 440,832 rows) to squeeze out every bit of pattern recognition before deployment.

Getting the Model Out of Kaggle
Instead of fighting with Kaggle's file system, I used joblib.dump() to save the trained model as final_customer_churn_model.pkl, then used from IPython.display import FileLink to generate a direct download link right in the notebook cell. Made it super easy to grab the file for local use.

What's Inside the Pipeline
The model looks at 10 customer features:

Age: Customer's age (strong predictor)

Gender: Mapped as Male=0, Female=1

Tenure: Months they've been subscribed

Usage Frequency: Monthly service interactions

Support Calls: Number of tech support requests

Payment Delay: Days late on payments

Subscription Type: Premium=1, Standard=2, Basic=3

Contract Length: Monthly=1, Quarterly=6, Annual=12

Total Spend: Historical spending total

Last Interaction: Days since last platform activity

Running It Locally
Download the model file (final_customer_churn_model.pkl) from your Kaggle workspace and place it in the same folder as churn_app.py

Open the project in VS Code

In the terminal, run:

bash
pip install -r requirements.txt
Launch the Streamlit app:

bash
streamlit run churn_app.py
