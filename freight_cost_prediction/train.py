import joblib
import os
# Import functions from your custom scripts
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

# 1. Configuration
# Make sure this points to the exact path of your database!
DB_PATH = r'C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\data\inventory.db'
MODELS_DIR = r'C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\models'

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. Data Pipeline
print("Loading data...")
df = load_vendor_invoice_data(DB_PATH)

print("Preparing features...")
X, y = prepare_features(df)

print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Model Training
print("Training models...")
lr_model = train_linear_regression(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# 4. Evaluation
lr_score = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
dt_score = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
rf_score = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# 5. Save the Best Model
# Based on the video, Linear Regression performs the best for Freight Cost
best_model_path = os.path.join(MODELS_DIR, 'predict_freight_model.pkl')
# Save the Best Model (Linear Regression)
joblib.dump(lr_model, r'C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\models\predict_freight_model.pkl')
print("Freight model saved successfully!")