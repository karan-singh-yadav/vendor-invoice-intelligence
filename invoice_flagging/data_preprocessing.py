import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_invoice_data(db_path):
    """Connects to the database and runs the complex join query."""
    conn = sqlite3.connect(db_path)

    # Using the corrected column names (PONumber, PODate, etc.)
    query = """
    SELECT 
        p.PONumber as po_number,
        COUNT(DISTINCT p.Brand) as total_brands,
        SUM(p.Quantity) as total_item_quantity,
        SUM(p.Dollars) as total_item_dollars,
        AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber;
    """
    purchase_df = pd.read_sql_query(query, conn)

    vi_query = """
    SELECT 
        PONumber as po_number, 
        Quantity as invoice_quantity, 
        Dollars as invoice_dollars, 
        Freight as freight_cost 
    FROM vendor_invoice;
    """
    vi_df = pd.read_sql_query(vi_query, conn)

    # Merge and drop NAs
    final_df = pd.merge(purchase_df, vi_df, on="po_number", how="left").dropna()
    conn.close()

    return final_df


def apply_labels_and_features(df):
    """Creates the target variable and drops non-significant features."""
    # Label 1 (Flagged) if quantities don't match OR delay > 10 days
    df["flag_invoice"] = np.where(
        (df["total_item_quantity"] != df["invoice_quantity"])
        | (df["avg_receiving_delay"] > 10),
        1,
        0,
    )

    # The creator dropped 'total_brands' and 'avg_receiving_delay' after T-Testing
    X = df[
        [
            "total_item_quantity",
            "total_item_dollars",
            "invoice_quantity",
            "invoice_dollars",
        ]
    ]
    y = df["flag_invoice"]

    return X, y


def split_and_scale_data(X, y, scaler_save_path):
    """Splits data and applies Standard Scaling, saving the scaler for the web app."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and fit the scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(
        X_test
    )  # Transform test data using the fitted scaler

    # Save the scaler so app.py can use it later!
    joblib.dump(scaler, scaler_save_path)

    return X_train_scaled, X_test_scaled, y_train, y_test
