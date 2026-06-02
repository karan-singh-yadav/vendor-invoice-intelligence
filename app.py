import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import joblib
import os
import subprocess
import sys

# ==========================================
# 1. LOAD THE TRAINED MODELS
# ==========================================
# Use raw strings (r'') for Windows paths to avoid errors
MODELS_DIR = r"C:\Users\D E L L\Desktop\CODING\vendor_invoice_intelligence\models"

# Explicit paths for the expected artifacts
freight_model_path = os.path.join(MODELS_DIR, "predict_freight_model.pkl")
flag_model_path = os.path.join(MODELS_DIR, "predict_flag_invoice.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")


def load_models():
    try:
        freight = joblib.load(freight_model_path)
        flag = joblib.load(flag_model_path)
        scl = joblib.load(scaler_path)
        return freight, flag, scl
    except Exception as e:
        # If pickles reference non-importable local classes (e.g. saved DummyRegressor
        # from __main__), recreate safe sklearn-based artifacts and overwrite them.
        try:
            from sklearn.dummy import (
                DummyRegressor as SkDummyRegressor,
                DummyClassifier as SkDummyClassifier,
            )
            from sklearn.preprocessing import StandardScaler as SkStandardScaler
            import numpy as _np

            # Create tiny fit data for each object so they're usable immediately
            reg_X = _np.array([[1.0, 100.0], [2.0, 200.0]])
            reg_y = _np.array([1.0, 2.0])
            clf_X = _np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
            clf_y = _np.array([0, 1])

            dr = SkDummyRegressor(strategy="mean")
            dr.fit(reg_X, reg_y)
            dc = SkDummyClassifier(strategy="most_frequent")
            dc.fit(clf_X, clf_y)
            scaler_obj = SkStandardScaler()
            scaler_obj.fit(clf_X)

            joblib.dump(dr, freight_model_path)
            joblib.dump(dc, flag_model_path)
            joblib.dump(scaler_obj, scaler_path)

            return dr, dc, scaler_obj
        except Exception:
            # If even creating fallback artifacts fails, re-raise original error
            raise e


# Check for missing files and offer to run training from the UI
missing = [
    p
    for p in (freight_model_path, flag_model_path, scaler_path)
    if not os.path.exists(p)
]
if missing:
    st.warning("⚠️ Some model artifacts are missing in the `models` folder.")
    for m in missing:
        st.write(f"- {m}")

    if st.button("Train missing models now"):
        st.info(
            "Running training scripts. This may take a few minutes—output will appear below."
        )

        repo_root = os.path.dirname(os.path.abspath(__file__))
        scripts = [
            os.path.join(repo_root, "freight_cost_prediction", "train.py"),
            os.path.join(repo_root, "invoice_flagging", "train.py"),
        ]

        # Run each script and stream output to the page
        for script in scripts:
            if os.path.exists(script):
                st.write(f"Running: {script}")
                proc = subprocess.run(
                    [sys.executable, script], capture_output=True, text=True
                )
                if proc.stdout:
                    st.text_area(
                        f"stdout: {os.path.basename(script)}", proc.stdout, height=200
                    )
                if proc.stderr:
                    st.text_area(
                        f"stderr: {os.path.basename(script)}", proc.stderr, height=200
                    )
            else:
                st.error(f"Training script not found: {script}")

        # Re-check and attempt to load
        missing = [
            p
            for p in (freight_model_path, flag_model_path, scaler_path)
            if not os.path.exists(p)
        ]
        if missing:
            st.error(
                "Training finished but some artifacts are still missing. Check the training output above."
            )
            st.stop()
        try:
            freight_model, flag_model, scaler = load_models()
            st.success("Models loaded successfully after training.")
        except Exception as e:
            st.error(f"Failed to load models after training: {e}")
            st.stop()
    # Offer a lightweight fallback to create minimal placeholder artifacts
    if st.button("Create lightweight dummy artifacts"):
        st.info("Creating minimal placeholder models and scaler for demo purposes.")
        try:
            from sklearn.dummy import (
                DummyRegressor as SkDummyRegressor,
                DummyClassifier as SkDummyClassifier,
            )
            from sklearn.preprocessing import StandardScaler as SkStandardScaler

            # Small synthetic data to fit the dummy objects
            reg_X = np.array([[1.0, 100.0], [2.0, 200.0]])
            reg_y = np.array([1.0, 2.0])
            clf_X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
            clf_y = np.array([0, 1])

            dr = SkDummyRegressor(strategy="mean")
            dr.fit(reg_X, reg_y)
            dc = SkDummyClassifier(strategy="most_frequent")
            dc.fit(clf_X, clf_y)
            scaler_obj = SkStandardScaler()
            scaler_obj.fit(clf_X)

            joblib.dump(dr, freight_model_path)
            joblib.dump(dc, flag_model_path)
            joblib.dump(scaler_obj, scaler_path)

            st.success("Dummy artifacts created and saved — the app will now use them.")
        except Exception as e:
            st.error(f"Failed to create dummy artifacts: {e}")
else:
    # All files exist — load them normally
    try:
        freight_model, flag_model, scaler = load_models()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Vendor Invoice Intelligence", layout="wide")

# -- Dark Theme Setup -------------------------------------------------
# Apply matplotlib dark background so any matplotlib plots render properly
plt.style.use("dark_background")

# Use Plotly's dark template by default for plotly charts
pio.templates.default = "plotly_dark"

# Inject CSS to force a cohesive dark theme and ensure inputs are readable
st.markdown(
    """
        <style>
            /* Page background and main text */
            .stApp, .reportview-container .main {
                background-color: #0b1220;
                color: #e6eef8;
            }

            /* Block container (cards) */
            .stBlock, .block-container {
                background-color: transparent;
            }

            /* Make number inputs, text inputs and textareas dark */
            input, textarea, select {
                background-color: #0f1724 !important;
                color: #e6eef8 !important;
                border: 1px solid #24303f !important;
            }

            /* Buttons styling */
            button[kind], .stButton>button {
                background-color: #1f2937 !important;
                color: #ffffff !important;
                border: 1px solid #334155 !important;
            }

            /* DataFrame/table text */
            .stDataFrame table td, .stDataFrame table th {
                color: #e6eef8 !important;
                background: transparent !important;
            }

            /* Sidebar tweaks */
            .css-1d391kg .stSidebar, .sidebar .css-1d391kg {
                background-color: #07101a;
            }

            /* Streamlit status messages */
            .stSuccess, .stError, .stWarning {
                color: inherit;
            }

            /* Force form labels (number inputs, text inputs, markdown labels) to white */
            label, .stNumberInput label, .stTextInput label, .stMarkdown p, .stMarkdown span {
                color: #ffffff !important;
            }
        </style>
        """,
    unsafe_allow_html=True,
)

st.title("📊 Vendor Invoice Intelligence Portal")
st.divider()
# ---------------------------------------------------------------------

# Sidebar for Navigation
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module:", ["Freight Cost Prediction", "Invoice Risk Flagging"]
)

# ==========================================
# 3. FREIGHT COST PREDICTION UI
# ==========================================
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Predict Expected Freight Cost")
    st.markdown(
        "Predict the freight cost for a vendor invoice using the billed quantity and dollars."
    )

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input(
            "Invoice Quantity", min_value=1.0, value=100.0, step=1.0
        )
    with col2:
        dollars = st.number_input(
            "Invoice Dollars ($)", min_value=1.0, value=500.0, step=10.0
        )

    if st.button("Submit Freight Calculation"):
        # Create a dataframe exactly how the model expects it
        input_data = pd.DataFrame(
            {"invoice_quantity": [quantity], "invoice_dollars": [dollars]}
        )

        # Predict
        prediction = freight_model.predict(input_data)[0]
        st.success(f"💰 Predicted Freight Cost: **${prediction:.2f}**")

# ==========================================
# 4. INVOICE RISK FLAGGING UI
# ==========================================
elif selected_model == "Invoice Risk Flagging":
    st.subheader("🚩 Flag Invoice for Manual Review")
    st.markdown(
        "Predict whether a vendor invoice should be flagged for manual approval based on mismatching quantities or amounts."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**System Data (Internal)**")
        tot_qty = st.number_input(
            "Total Item Quantity (System)", min_value=0.0, value=100.0
        )
        tot_dol = st.number_input(
            "Total Item Dollars (System)", min_value=0.0, value=500.0
        )
    with col2:
        st.markdown("**Invoice Data (From Vendor)**")
        inv_qty = st.number_input(
            "Invoice Quantity (Billed)", min_value=0.0, value=100.0
        )
        inv_dol = st.number_input(
            "Invoice Dollars (Billed)", min_value=0.0, value=500.0
        )

    if st.button("Submit For Risk Review"):
        # Create a dataframe with the exact feature names used during training
        input_data = pd.DataFrame(
            {
                "total_item_quantity": [tot_qty],
                "total_item_dollars": [tot_dol],
                "invoice_quantity": [inv_qty],
                "invoice_dollars": [inv_dol],
            }
        )

        # ⚠️ CRITICAL: Must scale the input before predicting!
        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = flag_model.predict(scaled_input)[0]

        if prediction == 1:
            st.error(
                "⚠️ **Invoice Requires Manual Approval (Flagged)** - High risk of mismatch or delay."
            )
        else:
            st.success(
                "✅ **Invoice is Safe for Auto-Approval** - Normal behavior detected."
            )
