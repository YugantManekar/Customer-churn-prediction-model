from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model components
with open("model/knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



with open("model/le.pkl", "rb") as f:  
    le = pickle.load(f)






with open("model/columns.pkl", "rb") as f:
    col_data = pickle.load(f)
    cat_col = col_data['cat_col']
    num_col = col_data['num_col']

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get form data
        user_input = [[
            int(request.form['age']),
            request.form['gender'],
            int(request.form['tenure']),
            int(request.form['usage_frequency']),
            int(request.form['support_calls']),
            int(request.form['payment_delay']),
            request.form['subscription_type'],
            request.form['contract_length'],
            float(request.form['total_spend']),
            int(request.form['last_interaction'])
        ]]

        user_df = pd.DataFrame(user_input, columns=[
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
            'Payment Delay', 'Subscription Type', 'Contract Length',
            'Total Spend', 'Last Interaction'
        ])

        # Encode categorical features using saved label encoders
        for col in cat_col:
          user_df[col] = le[col].transform(user_df[col].astype(str).str.strip())


        # Scale numerical features
        user_scaled = scaler.transform(user_df[num_col])

        # Combine for final input
        user_final = np.hstack((user_scaled, user_df[cat_col].values))

        # Predict churn
        result = knn.predict(user_final)
        prediction = "Yes" if result[0] == 1 else "No"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
