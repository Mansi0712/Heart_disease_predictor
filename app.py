from flask import Flask, render_template, request
import pandas as pd
import joblib

app=Flask(__name__)

#open the files
model = joblib.load('knn_heart_model.pkl')
scaler=joblib.load('heart_scaler.pkl')
model_columns=joblib.load('heart_columns.pkl')



@app.route('/')
def home():
    # This renders the index.html file you just created
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from the HTML form
    form_data = request.form.to_dict()
    
    # 2. Convert to DataFrame
    df = pd.DataFrame([form_data])
    
    # 3. Ensure all columns match heart_columns.pkl (fill missing with 0)
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # 4. Scale and Predict
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)
    
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
    
    # 5. Send the result back to the HTML page
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)