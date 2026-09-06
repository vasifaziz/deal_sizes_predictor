import streamlit as st
import pandas as pd
from datetime import date

from dealsize_pipeline import load_pipeline, predict_new_data


@st.cache_resource
def load_model():
    """Load the trained model pipeline and its inference config"""
    return load_pipeline()

model, config = load_model()


st.title('Deal Size Predictor')

st.write("'Predict Deal Size using Machine Learning! Classify Dealsize as multiclass from low to high value.'")


QUANTITYORDERED = st.slider("Quantity Ordered", min_value = 1, max_value = 97, value = 35)

PRICEEACH = st.number_input("Unit Price", min_value = 20, max_value = 300, value = 95)

DAYS_SINCE_LASTORDER = st.slider("Days Since Last Order", min_value = 42, max_value = 3562, value = 3184)

MSRP = st.slider("Manufacturing Selling Retail Price", min_value = 33, max_value = 214, value = 99)

ORDERLINENUMBER = st.slider("Order Line Number", min_value = 1, max_value = 18, value = 6)

PRODUCTLINE = st.selectbox("AutoMotive Type", ['Motorcycles', 'Classic Cars', 'Trucks and Buses', 'Vintage Cars', 'Planes', 'Ships', 'Trains'])

COUNTRY = st.selectbox("Country", ['USA', 'France', 'Norway', 'Australia', 'Finland', 'Austria', 'UK', 'Spain', 'Sweden', 'Singapore', 'Canada', 'Japan', 'Italy', 'Denmark', 'Belgium', 'Philippines', 'Germany', 'Switzerland', 'Ireland'])

ORDERDATE = st.date_input(
    "Order Date",
    value=date.today()
)

ORDERDATE = ORDERDATE.strftime(config["date_format"])

input_data = pd.DataFrame({
	"QUANTITYORDERED": [QUANTITYORDERED],
	"PRICEEACH": [PRICEEACH],
	"DAYS_SINCE_LASTORDER": [DAYS_SINCE_LASTORDER],
	"MSRP": [MSRP],
	"ORDERLINENUMBER": [ORDERLINENUMBER], 
	"PRODUCTLINE": [PRODUCTLINE],
	"COUNTRY": [COUNTRY],
	"ORDERDATE": [ORDERDATE]
	})

st.subheader("Input Sent to the Pipeline")

st.dataframe(input_data)


if st.button("Predict"):
	# predict_new_data reorders the columns to match training, applies the
	# configured Large-class threshold, and maps the index back to a label.
	result = predict_new_data(
		model, input_data, large_threshold=config["large_threshold"]
	).iloc[0]

	prediction = result["PREDICTED_DEALSIZE"]

	confidence = result["CONFIDENCE"]

	st.subheader("Prediction Result")

	if prediction == "Small":
		st.success("The Deal Size is Small")

	elif prediction == "Medium":
		st.warning("The Deal Size is Medium")

	else:
		st.error("The Deal Size is Large")

	st.write(
		f"**Probability of Deal Size is {prediction}: **"
		f"{confidence:.2%}"
		)
