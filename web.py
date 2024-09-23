from flask import Flask,render_template,request
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import gzip

# Initialize the Flask app
app = Flask(__name__)

# Load the trained RandomForest model, encoders, and scaler
# Load the compressed model
with gzip.open('credit_rf_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

# Assuming you saved the encoder and scaler after training, load them:
# Load the compressed scaler
with gzip.open('scaler.pkl.gz', 'rb') as f:
    scaler = pickle.load(f)

# Load the compressed encoder for 'Credit_Mix'
with gzip.open('encoder_credit_mix.pkl.gz', 'rb') as f:
    encoder_credit_mix = pickle.load(f)

# Load the compressed encoder for 'Payment_of_Min_Amount'
with gzip.open('encoder_payment_of_min_amount.pkl.gz', 'rb') as f:
    encoder_payment_of_min_amount = pickle.load(f)

# Load the compressed encoder for 'Payment_Behaviour'
with gzip.open('encoder_payment_behaviour.pkl.gz', 'rb') as f:
    encoder_payment_behaviour = pickle.load(f)

    
# Route to render the HTML form
@app.route('/')
def index():
    print("Index function called!")
    return render_template('index.html')

# Define the route for predictions
#@app.route('/predict', methods=['POST'])
#def predict():
    # Get JSON data from the request
 #   data = request.get_json()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form  # For form data submission
    # Extract and calculate features from the input
    Amount_invested_monthly = float(data['Amount_invested_monthly'])
    Monthly_Inhand_Salary = float(data['Monthly_Inhand_Salary'])
    Saving_ratio = Amount_invested_monthly / Monthly_Inhand_Salary if Monthly_Inhand_Salary > 0 else 0
    
    # Extract and calculate features from the input
   #Amount_invested_monthly = data['Amount_invested_monthly']
    #Monthly_Inhand_Salary = data['Monthly_Inhand_Salary']
    #if Monthly_Inhand_Salary > 0:
     #   Saving_ratio = Amount_invested_monthly / Monthly_Inhand_Salary
    #else:
        #Saving_ratio = 0  # Avoid division by zero

    # Prepare raw feature inputs
    features = pd.DataFrame([[
        int(data['Month']),
        int(data['Age']),
        float (data['Annual_Income']),
        int(data['Num_Bank_Accounts']),
        int(data['Num_Credit_Card']),
        float( data['Interest_Rate']),
        int(data['Num_of_Loan']),        
        int(data['Delay_from_due_date']),
        int(data['Num_of_Delayed_Payment']),
        float( data['Changed_Credit_Limit']),
        int(data['Num_Credit_Inquiries']),
        data['Credit_Mix'],            # Needs encoding
        float(data['Outstanding_Debt']),
        float(data['Credit_Utilization_Ratio']),
        int(data['Credit_History_Age']),
        data['Payment_of_Min_Amount'],    # Needs encoding
        float(data['Total_EMI_per_month']),
        data['Payment_Behaviour'],     # Needs encoding
        float(data['Monthly_Balance']),
        Saving_ratio,
        
    ]], columns=[
        'Month', 'Age', 'Annual_Income','Num_Bank_Accounts', 'Num_Credit_Card',
       'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
       'Credit_Utilization_Ratio', 'Credit_History_Age',
       'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Payment_Behaviour',
       'Monthly_Balance', 'Saving_ratio'
    ])

    # 1. Encode the categorical features
    features['Credit_Mix'] = encoder_credit_mix.transform(features[['Credit_Mix']])
    features['Payment_of_Min_Amount'] = encoder_payment_of_min_amount.transform(features[['Payment_of_Min_Amount']])
    features['Payment_Behaviour'] = encoder_payment_behaviour.transform(features[['Payment_Behaviour']])

    print("Features DataFrame:\n", features)

   # features=np.array([features])
    features_scale = scaler.transform(features)

    # Make a prediction using the processed features
    prediction = model.predict( features_scale)
    
    # Return the result as a JSON response
   # return jsonify({'Credit_Score_Prediction': prediction[0]})

     # Render the result in an HTML template
    return render_template('result.html', prediction=prediction[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True,port=8000,use_reloader=False)
