import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path):
    """Connects to the database, runs SQL queries, and merges the dataframes."""
    # Use raw string (r'') to prevent Windows path errors
    conn = sqlite3.connect(db_path)

    # Aggregated Purchases Query (with our corrected column names)
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
    purchase_agg_df = pd.read_sql_query(query, conn)

    # Vendor Invoice Query
    vi_query = """
    SELECT 
        PONumber as po_number, 
        Quantity as invoice_quantity, 
        Dollars as invoice_dollars, 
        Freight as freight_cost 
    FROM vendor_invoice;
    """
    vendor_invoice_df = pd.read_sql_query(vi_query, conn)

    # Merge and drop NAs
    final_df = pd.merge(purchase_agg_df, vendor_invoice_df, on="po_number", how="left")
    final_df = final_df.dropna()

    conn.close()
    return final_df


def prepare_features(df):
    """Selects the features (X) and target (y) for the model."""
    X = df[["invoice_quantity", "invoice_dollars"]]
    y = df["freight_cost"]
    return X, y


def split_data(X, y):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)
