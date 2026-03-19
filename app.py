import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load trained pipeline model
try:
    model = joblib.load("heart_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Feature names (same order as dataset)
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        # Convert input to dataframe
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                    restecg, thalach, exang, oldpeak,
                                    slope, ca, thal]], columns=columns)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        probability = model.predict_proba(input_data)

        risk_prob = round(probability[0][0] * 100, 2)
        safe_prob = round(probability[0][1] * 100, 2)

        # Label interpretation
        if prediction == 0:
            result = f"⚠️ Heart Disease Detected ({risk_prob}% risk)"
        else:
            result = f"✅ No Heart Disease ({safe_prob}% safe)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", prediction_text="❌ Error processing prediction. Please check inputs.")




if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)