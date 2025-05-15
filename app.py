from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")  # Assure-toi que le modèle est dans le même dossier

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Récupération des 15 variables du formulaire
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
            age = float(request.form["age"])  # 15e variable

            # Création du tableau de données pour la prédiction
            input_data = np.array([[G1, G2, studytime, failures, absences,
                                    Medu, Fedu, goout, health, Dalc,
                                    Walc, traveltime, famrel, freetime, age]])

            # Prédiction avec le modèle
            prediction = round(model.predict(input_data)[0], 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    print("Lancement du serveur Flask...")
    app.run(debug=True)
