from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib, os

app = Flask(__name__)

def train_model():
    # Dataset Student Performance (UCI) - données réelles
    df = pd.read_csv("student-mat.csv", sep=';')
    df = pd.read_csv(url, sep=';')
    
    features = ['G1','G2','studytime','failures','absences',
                'Medu','Fedu','goout','health','Dalc',
                'Walc','traveltime','famrel','freetime','age']
    
    X = df[features]
    y = df['G3']
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    return model

# Charge ou réentraîne le modèle
try:
    model = joblib.load("model.pkl")
    # Test de compatibilité
    model.predict(np.zeros((1, 15)))
except Exception:
    model = train_model()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            G1 = float(request.form["G1"])
            G2 = float(request.form["G2"])
            studytime = float(request.form["studytime"])
            failures = float(request.form["failures"])
            absences = float(request.form["absences"])
            Medu = float(request.form["Medu"])
            Fedu = float(request.form["Fedu"])
            goout = float(request.form["goout"])
            health = float(request.form["health"])
            Dalc = float(request.form["Dalc"])
            Walc = float(request.form["Walc"])
            traveltime = float(request.form["traveltime"])
            famrel = float(request.form["famrel"])
            freetime = float(request.form["freetime"])
            age = float(request.form["age"])

            input_data = np.array([[G1, G2, studytime, failures, absences,
                                    Medu, Fedu, goout, health, Dalc,
                                    Walc, traveltime, famrel, freetime, age]])
            prediction = round(model.predict(input_data)[0], 2)
        except Exception as e:
            error = str(e)
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
