# ============================================================
# AI-Based Hospital Readmission Prediction System
# Deployment Application using Flask
# ============================================================

# Step 1: Import required libraries
# Flask is used to create the web application.
# joblib is used to load the trained machine learning model.
# pandas is used to create the input DataFrame.
# os is used to create reliable file paths.

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os


# ============================================================
# Step 2: Create Flask Application
# ============================================================

# Create the Flask application.
# The templates folder will automatically be detected
# because it is inside the same folder as app.py.

app = Flask(__name__)


# ============================================================
# Step 3: Define Project Paths
# ============================================================

# Get the main project folder.
# app.py is inside:
# AI-Hospital-Readmission-Prediction/07_Deployment/
# Therefore, moving one level up gives the main project folder.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Define the location of the saved final machine learning model.

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "final_readmission_model.pkl"
)


# Define the location of the saved model input column names.

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_input_columns.pkl"
)


# Define the location of the training data.
# We use this data to create default values for fields
# that are not directly entered through the web form.

TRAIN_DATA_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_X_train.pkl"
)


# ============================================================
# Step 4: Load the Trained Model
# ============================================================

# Load the final model selected during model evaluation.

model = joblib.load(MODEL_PATH)


# Load the exact input column names expected by the model.

model_input_columns = joblib.load(COLUMNS_PATH)


# Load the training data.
# This helps us generate sensible default values
# for fields that are not shown in the simple interface.

X_train = joblib.load(TRAIN_DATA_PATH)


# ============================================================
# Step 5: Create Default Values
# ============================================================

# Create an empty dictionary to store default values.

default_values = {}


# Go through every feature used by the model.

for column in X_train.columns:

    # If the column contains numerical values,
    # use the median value from the training data.

    if pd.api.types.is_numeric_dtype(X_train[column]):

        default_values[column] = X_train[column].median()

    # Otherwise, use the most frequently occurring category.

    else:

        mode_value = X_train[column].mode(dropna=True)

        # If a mode exists, use the first mode value.

        if not mode_value.empty:

            default_values[column] = mode_value.iloc[0]

        # If no mode exists, use an empty value.

        else:

            default_values[column] = ""


# ============================================================
# Step 6: Home Page Route
# ============================================================

# This route displays the prediction form.

@app.route("/", methods=["GET", "POST"])
def home():

    # Store prediction result.
    prediction = None

    # Store prediction probability.
    probability = None

    # Store error message if something goes wrong.
    error = None


    # ========================================================
    # Step 7: Handle Prediction Request
    # ========================================================

    # Check whether the user submitted the form.

    if request.method == "POST":

        try:

            # Start with default values for all model features.
            # This ensures that the model receives every
            # feature it was trained on.

            input_data = default_values.copy()


            # =================================================
            # Step 8: Read User Input
            # =================================================

            # Read age category entered by the user.

            input_data["age"] = request.form["age"]


            # Read gender entered by the user.

            input_data["gender"] = request.form["gender"]


            # Read length of hospital stay.

            input_data["time_in_hospital"] = float(
                request.form["time_in_hospital"]
            )


            # Read number of laboratory procedures.

            input_data["num_lab_procedures"] = float(
                request.form["num_lab_procedures"]
            )


            # Read number of procedures.

            input_data["num_procedures"] = float(
                request.form["num_procedures"]
            )


            # Read number of medications.

            input_data["num_medications"] = float(
                request.form["num_medications"]
            )


            # Read previous outpatient visits.

            input_data["number_outpatient"] = float(
                request.form["number_outpatient"]
            )


            # Read previous emergency visits.

            input_data["number_emergency"] = float(
                request.form["number_emergency"]
            )


            # Read previous inpatient visits.

            input_data["number_inpatient"] = float(
                request.form["number_inpatient"]
            )


            # Read number of diagnoses.

            input_data["number_diagnoses"] = float(
                request.form["number_diagnoses"]
            )


            # Read diabetes medication status.

            input_data["diabetesMed"] = request.form["diabetesMed"]


            # =================================================
            # Step 9: Recalculate Engineered Features
            # =================================================

            # Calculate total number of previous healthcare visits.

            input_data["total_prior_visits"] = (
                input_data["number_outpatient"]
                + input_data["number_emergency"]
                + input_data["number_inpatient"]
            )


            # Calculate total number of procedures.

            input_data["total_procedures"] = (
                input_data["num_lab_procedures"]
                + input_data["num_procedures"]
            )


            # Calculate medications per diagnosis.

            if input_data["number_diagnoses"] > 0:

                input_data["medications_per_diagnosis"] = (
                    input_data["num_medications"]
                    / input_data["number_diagnoses"]
                )

            else:

                input_data["medications_per_diagnosis"] = 0


            # =================================================
            # Step 10: Create Model Input DataFrame
            # =================================================

            # Convert the dictionary into a one-row DataFrame.

            input_df = pd.DataFrame([input_data])


            # Arrange columns in exactly the same order
            # expected by the trained model.

            input_df = input_df[model_input_columns]


            # =================================================
            # Step 11: Generate Prediction
            # =================================================

            # Generate the predicted class.
            # 1 = likely readmitted within 30 days
            # 0 = not predicted as readmitted within 30 days

            prediction_value = model.predict(input_df)[0]


            # Convert prediction into a user-friendly message.

            if prediction_value == 1:

                prediction = "High Risk of Readmission Within 30 Days"

            else:

                prediction = "Low Risk of Readmission Within 30 Days"


            # =================================================
            # Step 12: Calculate Prediction Probability
            # =================================================

            # Get probability of the positive class if supported
            # by the selected model.

            if hasattr(model, "predict_proba"):

                probability_value = model.predict_proba(input_df)[0][1]

                probability = round(
                    probability_value * 100,
                    2
                )


        # =====================================================
        # Step 13: Handle Errors
        # =====================================================

        except Exception as e:

            # Store the error so that it can be displayed
            # on the web page instead of stopping the application.

            error = str(e)


    # =========================================================
    # Step 14: Display Web Page
    # =========================================================

    # Send prediction, probability and error information
    # to the HTML interface.

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


# ============================================================
# Step 15: Run Flask Application
# ============================================================

# Start the Flask development server.

if __name__ == "__main__":

    app.run(debug=True)