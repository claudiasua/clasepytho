#app para predicción de problema cardiaco
NO CORRER CON EL BOTON DE RuntimeError

...
se ejecuta primero cargan las bibliotecas requeridas con
python -r install requirements.txt (tenga creado ese archivo)

se ejecuta la aplicacion con
steamlit run app.py dede la linea del terminal o de su powershell

...


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained SVC model and the scaler
try:
    svc_model_loaded = joblib.load('svc_model.pkl')
    # Assuming the scaler used for normalization in the notebook is saved as 'scaler.pkl'
    # You might need to save the scaler in your original notebook before deploying
    # joblib.dump(scaler, 'scaler.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'svc_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

# Streamlit UI
st.title("Predicción de Problemas Cardíacos")

st.write("""
Esta aplicación predice la probabilidad de que un paciente sufra problemas cardíacos
basado en su edad y nivel de colesterol, utilizando un modelo de clasificación SVC.
""")

# Input fields for user data
st.header("Introduce los datos del paciente:")

edad = st.number_input("Edad del paciente:", min_value=1, max_value=120, value=50)
colesterol = st.number_input("Nivel de Colesterol:", min_value=50, max_value=600, value=200)

# Create a DataFrame for the new patient data
new_patient_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Scale the new patient data using the loaded scaler
try:
    new_patient_scaled = scaler.transform(new_patient_data)
    new_patient_scaled_df = pd.DataFrame(new_patient_scaled, columns=['edad', 'colesterol'])
except Exception as e:
    st.error(f"Error al escalar los datos de entrada: {e}")
    st.stop()

# Make prediction when button is clicked
if st.button("Predecir"):
    try:
        prediction = svc_model_loaded.predict(new_patient_scaled_df)

        # Interpret the prediction (assuming 0 for 'No problema' and 1 for 'Sí problema')
        if prediction[0] == 1:
            result = 'Sí sufrirá problemas cardiacos'
            st.error(f"Predicción: **{result}**")
        else:
            result = 'No sufrirá problemas cardiacos'
            st.success(f"Predicción: **{result}**")

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")

