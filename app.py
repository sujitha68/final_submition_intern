import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import gzip
import pickle

# Load the dataset
df0 = pd.read_csv('D:\Internship ICT\Credit Score\credit.csv')
df1 = df0.copy()

# Drop irrelevant columns
df1.drop(['ID', 'Customer_ID', 'SSN', 'Occupation', 'Type_of_Loan', 'Name'], axis=1, inplace=True)

# Create new feature [Savings_Ratio = Amount_invested_monthly / Monthly_Inhand_Salary]
df1['Saving_ratio'] = df1['Amount_invested_monthly'] / df1['Monthly_Inhand_Salary']

# Drop Columns Monthly_Inhand_Salary and Amount_invested_monthly as they are highly correlated
df1.drop(['Monthly_Inhand_Salary', 'Amount_invested_monthly'], axis=1, inplace=True)

# Encode the categorical variables
Credit_Mix_oe = OrdinalEncoder()
Payment_of_Min_Amount_oe = OrdinalEncoder()
Payment_Behaviour_oe = OrdinalEncoder()

# Ordinal Encoding for 'Credit_Mix'
df1['Credit_Mix'] = Credit_Mix_oe.fit_transform(df1[['Credit_Mix']])

# Ordinal Encoding for 'Payment_of_Min_Amount'
df1['Payment_of_Min_Amount'] = Payment_of_Min_Amount_oe.fit_transform(df1[['Payment_of_Min_Amount']])

# Ordinal Encoding for 'Payment_Behaviour'
df1['Payment_Behaviour'] = Payment_Behaviour_oe.fit_transform(df1[['Payment_Behaviour']])

# Log transformation to avoid issues with zero values
columns_to_transform = ['Annual_Income', 'Delay_from_due_date', 'Outstanding_Debt', 'Total_EMI_per_month', 'Monthly_Balance', 'Saving_ratio']
df1[columns_to_transform] = df1[columns_to_transform].apply(np.log1p)

# Initialize RobustScaler
scaler = RobustScaler()

# Separate the features and the target variable
X1 = df1.drop(columns=['Credit_Score'])
y1 = df1['Credit_Score']

# Apply SMOTE for oversampling
smote = SMOTE()
X, y = smote.fit_resample(X1, y1)

# Step 1: Split the data into 70% training and 30% temporary sets
X_train1, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 2: Split the temporary set into 50% validation and 50% test sets
X_val1, X_test1, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train1)

# Use the same transformation on the test and validation data
X_test = scaler.transform(X_test1)
X_val = scaler.transform(X_val1)

# Best parameters from RandomSearchCV
best_params = {
    'bootstrap': False,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 200
}

# Initialize the RandomForestClassifier with the best parameters
best_rf_model = RandomForestClassifier(
    bootstrap=best_params['bootstrap'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# Fit the RandomForest model on the entire training dataset
best_rf_model.fit(X_train, y_train)

# Compress and Save the RobustScaler using gzip
with gzip.open('scaler.pkl.gz', 'wb') as f:
    pickle.dump(scaler, f)

# Compress and Save the OrdinalEncoders using gzip
with gzip.open('encoder_credit_mix.pkl.gz', 'wb') as f:
    pickle.dump(Credit_Mix_oe, f)

with gzip.open('encoder_payment_of_min_amount.pkl.gz', 'wb') as f:
    pickle.dump(Payment_of_Min_Amount_oe, f)

with gzip.open('encoder_payment_behaviour.pkl.gz', 'wb') as f:
    pickle.dump(Payment_Behaviour_oe, f)

# Compress and Save the RandomForest model using gzip
with gzip.open('credit_rf_model.pkl.gz', 'wb') as f:
    pickle.dump(best_rf_model, f)

