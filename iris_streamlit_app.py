import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessors
model = load_model("iris_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🌸 Iris Flower Species Prediction")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

if st.button("Predict"):
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])
    st.success(f"The predicted species is: 🌼 **{predicted_class[0]}**")
