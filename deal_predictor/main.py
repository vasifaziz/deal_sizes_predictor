from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from dealsize_pipeline import load_pipeline, predict_new_data


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deal Size Predictor API",
    description="FastAPI backend for the Deal Size Predictor machine learning model.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
# Allows the separate HTML/CSS/JS frontend to call this API.
# For a small student project, allowing all origins is acceptable.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Load the trained pipeline ONCE when the API starts.
#
# load_pipeline() is used instead of joblib.load() directly because the
# training pipeline contains the custom add_date_features transformer.
# It also loads inference_config.json.
# ---------------------------------------------------------------------------

model, config = load_pipeline()


# ---------------------------------------------------------------------------
# Pydantic request model
#
# These names intentionally match the exact input columns used by the
# trained pipeline and the Streamlit application.
# ---------------------------------------------------------------------------

class DealFeatures(BaseModel):
    QUANTITYORDERED: int = Field(ge=1, le=97)
    PRICEEACH: float = Field(ge=20, le=300)
    DAYS_SINCE_LASTORDER: int = Field(ge=42, le=3562)
    MSRP: float = Field(ge=33, le=214)
    ORDERLINENUMBER: int = Field(ge=1, le=18)

    PRODUCTLINE: Literal[
        "Motorcycles",
        "Classic Cars",
        "Trucks and Buses",
        "Vintage Cars",
        "Planes",
        "Ships",
        "Trains",
    ]

    COUNTRY: Literal[
        "USA",
        "France",
        "Norway",
        "Australia",
        "Finland",
        "Austria",
        "UK",
        "Spain",
        "Sweden",
        "Singapore",
        "Canada",
        "Japan",
        "Italy",
        "Denmark",
        "Belgium",
        "Philippines",
        "Germany",
        "Switzerland",
        "Ireland",
    ]

    ORDERDATE: str

    @field_validator("ORDERDATE")
    @classmethod
    def validate_order_date(cls, value: str) -> str:
        """
        The trained pipeline expects DD/MM/YYYY.

        We also accept YYYY-MM-DD because that is the normal value produced
        by an HTML <input type="date"> element, then normalize it to the
        format required by the model.
        """
        for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                parsed_date = datetime.strptime(value, date_format)
                return parsed_date.strftime("%d/%m/%Y")
            except ValueError:
                continue

        raise ValueError(
            "ORDERDATE must be a valid date in DD/MM/YYYY or YYYY-MM-DD format."
        )


# ---------------------------------------------------------------------------
# Root / health endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Deal Size Predictor API is running",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(features: DealFeatures):
    """
    Receive validated JSON from the frontend, send it through the saved
    scikit-learn pipeline, and return the predicted deal size + confidence.
    """

    try:
        # Convert the Pydantic object into the same raw-column shape used
        # during model training.
        input_data = pd.DataFrame(
            [
                {
                    "QUANTITYORDERED": features.QUANTITYORDERED,
                    "PRICEEACH": features.PRICEEACH,
                    "DAYS_SINCE_LASTORDER": features.DAYS_SINCE_LASTORDER,
                    "MSRP": features.MSRP,
                    "ORDERLINENUMBER": features.ORDERLINENUMBER,
                    "PRODUCTLINE": features.PRODUCTLINE,
                    "COUNTRY": features.COUNTRY,
                    "ORDERDATE": features.ORDERDATE,
                }
            ]
        )

        # Use the same prediction helper as the Streamlit application.
        # This preserves:
        # - required feature order
        # - date preprocessing inside the pipeline
        # - Large-class threshold
        # - label mapping
        # - confidence calculation
        result = predict_new_data(
            model,
            input_data,
            large_threshold=config["large_threshold"],
        ).iloc[0]

        prediction = str(result["PREDICTED_DEALSIZE"])
        confidence = float(result["CONFIDENCE"])

        return {
            "predicted_dealsize": prediction,
            "confidence": round(confidence, 4),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Run locally:
#
# uvicorn main:app --reload
#
# Then open:
# http://127.0.0.1:8000/docs
#
# Render start command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
# ---------------------------------------------------------------------------
