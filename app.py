from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load the trained model from JSON
MODEL_PATH = "xgb_model.json"
xgb_model = xgb.Booster()
xgb_model.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        tavg = float(request.form.get('tavg'))
        prcp = float(request.form.get('prcp'))
        
        # Prepare input data as a DataFrame
        input_df = pd.DataFrame([{"tavg": tavg, "prcp": prcp}])
        input_dmatrix = xgb.DMatrix(input_df)
        
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
