import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer # This import is not used, can be removed if not needed
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- ADD THIS LINE
import json # Import json module

app = Flask(__name__)
CORS(app) # <--- ADD THIS LINE

# --- Load and Prepare Data ---
try:
    # Assuming training.csv is in the same directory as app.py
    df = pd.read_csv('training.csv')

    # The last two columns are 'prognosis' (disease) and 'medicine'
    diseases = df['prognosis']
    medications = df['medicine']

    # All columns except 'prognosis' and 'medicine' are symptoms
    symptom_columns = df.columns[:-2].tolist()
    X = df[symptom_columns] # Features (symptoms encoded as 0s and 1s)
    y = diseases # Target (disease)

    # Train a Decision Tree Classifier model
    # Using the entire dataset for training for simplicity, as it's a mapping task
    # For a real application, you'd use train_test_split for robust evaluation
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Create a mapping from disease to medication
    disease_med_map = df.set_index('prognosis')['medicine'].to_dict()

    print("Model trained and data loaded successfully!")

except FileNotFoundError:
    print("Error: training.csv not found. Make sure it's in the same directory as app.py")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or model training: {e}")
    exit()

# --- API Endpoint ---
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    user_symptoms = data.get('symptoms', [])

    if not user_symptoms:
        return jsonify({'error': 'No symptoms provided.'}), 400

    # Initialize a symptom vector with all zeros
    # This vector must have the same order and length as symptom_columns used in training
    input_symptom_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)

    # Set 1 for symptoms present in user_symptoms
    for symptom in user_symptoms:
        # Normalize symptom name by replacing spaces with underscores and converting to lowercase
        normalized_symptom = symptom.strip().replace(' ', '_').lower()

        # Check if the normalized symptom is in our model's symptom columns
        if normalized_symptom in input_symptom_vector.columns: # Corrected from previous error
            input_symptom_vector[normalized_symptom] = 1
        else:
            print(f"Warning: Symptom '{symptom}' (normalized: '{normalized_symptom}') not found in training data. Ignoring.")


    # Make prediction
    predicted_disease = model.predict(input_symptom_vector)[0]
    predicted_medication = disease_med_map.get(predicted_disease, "No specific medication found for this disease.")

    return jsonify({
        'disease': predicted_disease,
        'medication': predicted_medication
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Run on port 5000