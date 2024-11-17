from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the trained model from JSON
MODEL_PATH = "xgb_model3.json"
xgb_model = xgb.Booster()
xgb_model.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        state = request.form.get('state')
        tavg = float(request.form.get('tavg'))
        tmin = float(request.form.get('tmin'))
        tmax = float(request.form.get('tmax'))
        prcp = float(request.form.get('prcp'))
        snow = float(request.form.get('snow'))
        wdir = float(request.form.get('wdir'))
        wspd = float(request.form.get('wspd'))
        pres = float(request.form.get('pres'))

        # Prepare input data as a DataFrame
        input_df = pd.DataFrame([{
            "state": state,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "prcp": prcp,
            "snow": snow,
            "wdir": wdir,
            "wspd": wspd,
            "pres": pres
        }])

        # Convert the 'state' column to categorical
        input_df['state'] = pd.Categorical(input_df['state'])

        # Prepare the feature vector
        input_dmatrix = xgb.DMatrix(input_df, enable_categorical=True)

        # Predict
        predictions = xgb_model.predict(input_dmatrix)
        predicted_label = np.argmax(predictions, axis=1)[0]

        # Map predictions to severity categories
        categories = ['Low', 'Medium', 'High']
        predicted_category = categories[predicted_label]

        return jsonify({"prediction": predicted_category})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
