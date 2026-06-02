import joblib
import os

# Import functions from your custom scripts
from data_preprocessing import (
    load_invoice_data,
    apply_labels_and_features,
    split_and_scale_data,
)
from model_evaluation import train_tuned_random_forest, evaluate_classifier

# 1. Configuration (Make sure these paths match your machine!)
DB_PATH = (
    r"C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\data\inventory.db"
)
MODELS_DIR = r"C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\models"

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# 2. Data Pipeline
print("Loading invoice data...")
df = load_invoice_data(DB_PATH)

print("Applying labels and selecting features...")
X, y = apply_labels_and_features(df)

print("Splitting and scaling data...")
X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y, SCALER_PATH)
print(f"Scaler saved to: {SCALER_PATH}")

# 3. Model Training (Tuning)
print("Training model with Hyperparameter Tuning...")
best_rf_model = train_tuned_random_forest(X_train_scaled, y_train)

# 4. Evaluation
evaluate_classifier(best_rf_model, X_test_scaled, y_test)

# 5. Save the Best Model
model_save_path = r"C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\models\predict_flag_invoice.pkl"
joblib.dump(best_rf_model, model_save_path)
print(f"\nBest model (Random Forest) saved successfully!")
