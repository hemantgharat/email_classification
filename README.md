# 📧 Email Classification Using Logistic Regression,  TF-IDF & Flask

This project implements an **email classification system** using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Logistic Regression**. Additionally, a **Flask API** is provided to predict whether an email is **spam or ham**.

## **Project Structure**


📂 email-classification 
   - **📄email_classification.ipynb**: This Jupyter notebook contains the code to train and evaluate the email classification model.
   - **📄email_classifier_model.pkl**: The trained Logistic Regression model saved for later use in predictions.
   - **📄tfidf_vectorizer.pkl**: The saved TF-IDF vectorizer used to preprocess email content.
   - **📄email_dataset.csv**: A sample dataset containing emails and their corresponding labels.
   - **📄app.py**: A Flask API that serves the model and provides an endpoint for email classification.
   - **📄README.md**: Project documentation.