"""
Inference Example - how FastAPI / Streamlit should use the saved artifacts.
=============================================================================
Run this AFTER run_training.py has produced the artifacts/ folder.

This mirrors exactly what a FastAPI endpoint or a Streamlit form handler
would do: take raw user input (same shape as original CSV columns, with
ORDERDATE as a plain date string), and return a predicted DEALSIZE label.

No manual date-feature engineering is needed here - it lives inside the
saved pipeline itself (see dealsize_pipeline.add_date_features).
"""

import json

import joblib
import pandas as pd

from dealsize_pipeline import predict_new_data

# --------------------------------------------------------------------------
# Load artifacts once at app startup (NOT on every request - this is the
# correct pattern for FastAPI: load in a global/startup event, reuse across
# requests).
# --------------------------------------------------------------------------
model = joblib.load("artifacts/dealsize_pipeline.pkl")

with open("artifacts/inference_config.json") as f:
    inference_config = json.load(f)

REQUIRED_COLUMNS = inference_config["required_input_columns"]
LARGE_THRESHOLD = inference_config["large_threshold"]


def predict_single_order(order: dict) -> dict:
    """
    Example function a FastAPI POST endpoint would call.

    order = {
        "QUANTITYORDERED": 30,
        "PRICEEACH": 95.70,
        "ORDERLINENUMBER": 2,
        "DAYS_SINCE_LASTORDER": 828,
        "MSRP": 95,
        "ORDERDATE": "24/02/2018",
        "PRODUCTLINE": "Motorcycles",
        "COUNTRY": "France",
    }
    """
    missing = set(REQUIRED_COLUMNS) - set(order.keys())
    if missing:
        raise ValueError(f"Missing fields in request: {missing}")

    raw_df = pd.DataFrame([order])
    result = predict_new_data(model, raw_df, large_threshold=LARGE_THRESHOLD)

    return {
        "predicted_dealsize": result["PREDICTED_DEALSIZE"].iloc[0],
        "confidence": round(float(result["CONFIDENCE"].iloc[0]), 4),
    }


if __name__ == "__main__":
    sample_order = {
        "QUANTITYORDERED": 30,
        "PRICEEACH": 95.70,
        "ORDERLINENUMBER": 2,
        "DAYS_SINCE_LASTORDER": 828,
        "MSRP": 95,
        "ORDERDATE": "24/02/2018",
        "PRODUCTLINE": "Motorcycles",
        "COUNTRY": "France",
    }
    prediction = predict_single_order(sample_order)
    print("Sample prediction:", prediction)

    # ---- FastAPI equivalent (for reference, not executed here) ----
    #
    # from fastapi import FastAPI
    # app = FastAPI()
    #
    # @app.post("/predict")
    # def predict(order: dict):
    #     return predict_single_order(order)
