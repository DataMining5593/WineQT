# Site for datamining course
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

import mining




wine_data_all = None



# Initialisation
def setupServer():
    print("Start setup")
    # Load global variables
    global wine_data_all

    # Load dataset
    wine_data_all = pd.read_csv('mysite/winequality-all.csv', sep=";")

    # Train all models (can take some times)
    mining.modelsTraining()

    print("Ended setup")



setupServer()
# Create server
app = Flask(__name__)


# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Page 1 : show all record
@app.route('/view-data')
def view_data():
    return render_template('view_data.html', tables=[wine_data_all.to_html(classes='data')], titles=wine_data_all.columns.values)

# Page 2 : visualisation
@app.route('/visualize')
def visualise():
    return render_template('DataMining_Visualise.html')

# Page 3 : classify
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == "POST":
        # Get form data
        try:
            color = 1 if request.form["color"] == "red" else 0
            fixed_acidity = float(request.form["fixed_acidity"])
            volatile_acidity = float(request.form["volatile_acidity"])
            citric_acid = float(request.form["citric_acid"])
            residual_sugar = float(request.form["residual_sugar"])
            chlorides = float(request.form["chlorides"])
            free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density = float(request.form["density"])
            pH = float(request.form["pH"])
            sulphates = float(request.form["sulphates"])
            alcohol = float(request.form["alcohol"])
            
            # Combine inputs into a feature array
            features = np.array([[
                color, fixed_acidity, volatile_acidity, citric_acid,
                residual_sugar, chlorides, free_sulfur_dioxide,
                total_sulfur_dioxide, density, pH, sulphates, alcohol
            ]])
            
            # Predict quality using the model
            # predicted_quality = model.predict(features)[0]
            predictions = mining.getModelsResult(features)
            return render_template("result.html", grubb=predictions[0], ridge=predictions[1])
            
        
        except Exception as e:
            return f"Error in input: {e}", 400
    
    return render_template("classify.html")


