import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load model
model = joblib.load("models/model_small.pkl")

# page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# title
st.title("🏠 House Price Prediction App")
st.info("This model predicts house prices based on key features like quality, area, and garage capacity.")
st.markdown("### Predict house prices using Machine Learning")

st.markdown("---")

# layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    garage_cars = st.number_input("Garage Capacity", 0, 5, 1)

with col2:
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    total_bsmt = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

st.markdown("---")

# prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'TotalBsmtSF': [total_bsmt]
    })

    prediction = model.predict(input_df)
    prediction = np.expm1(prediction)

    st.success(f"💰 Estimated House Price: ₹ {int(prediction[0]):,}")

    with st.expander("ℹ️ Feature Info"):
     st.write("""
    - Overall Quality: Overall material and finish quality
    - Living Area: Total above ground living area (sq ft)
    - Garage Cars: Size of garage in car capacity
    - Basement Area: Total basement area (sq ft)
    """)
     