# Site for datamining course
from flask import Flask, render_template, request, redirect, url_for
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load dataset
wine_data_red = pd.read_csv('mysite/winequality-red.csv', sep=";")
wine_data_white = pd.read_csv('mysite/winequality-white.csv', sep=";")
wine_data_all = pd.read_csv('mysite/winequality-all.csv', sep=";")

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
    return render_template('classify.html')

