# Symptom-Disease-Medication Predictor

This project is a web application that takes a list of symptoms as input, predicts a potential disease based on a training dataset, and suggests a corresponding medication. The application consists of a Flask-based Python backend for disease prediction and medication mapping, and an HTML/JavaScript frontend for user interaction.

## Features

* **Symptom Input:** Users can add multiple symptoms via a simple web interface.
* **Disease Prediction:** A machine learning model (Decision Tree Classifier) predicts the likely disease based on the combination of entered symptoms.
* **Medication Suggestion:** Provides a suggested medication associated with the predicted disease.
* **RESTful API:** The backend exposes a `/predict_disease` API endpoint for symptom submission and result retrieval.
