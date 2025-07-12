#app para predicción de problema cardiaco

# prompt: creamos un promt detallando lo que deseamos que la aplicacion haga. ejemplo: Eres experto en steamlit y requiero realizar un deployment de la aplicación para determinar si un paciente sufrira o no del corazon. el modelo fue entrenado usando sklearn con svc y los datos de entrada fueron escalados usando mixmax scaler modelo:svc model.jb escalador: scale.jb los modelos fueron guardados usando joblib. coloque un titulo asi: modelo IA para predicción de problemas cardiacos. haga un resumen de como funciona el modelo para los usuarios. coloque en la parte de abajo elaborado por:Alfredo Diaz un emoji de copy right unab 2025. En el lado izquierdo en sidebar donde con slider el usuario escoja lo siguiente: Edad:20 años a 80 años, con incremento de 1 año. Colesterol:Use los valores de parametros de niveles de colesterol, por ejemplo 230, 240 etc. Por defecto: los valores seleccionados sean: Edad 20, Colesterol:200. Estos datos deben pasar por el scaler. los resultados son: 0 no sufrira del corazón, ponerlo en fondo verde y letras negras y un emoji feliz y debajo aparece una imagen llamada No sufre.jpg 1:Sufrira del corazón con fondo rojo y letras negras y un emoji triste y abajo una imagen llamda Si sufre.jpg. Antes del título poner una imagen tipo banner llamada cabezote.jpg

import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the models and scaler
# Assumes 'svc_model.jb' and 'scaler.jb' are in the same directory as the app.py file
try:
    svc_model = joblib.load('svc_model.jb')
    scaler = joblib.load('scaler.jb')
except FileNotFoundError:
    st.error("Error loading models. Please make sure 'svc_model.jb' and 'scaler.jb' are in the same directory.")
    st.stop()


# Load images
try:
    cabezote_img = Image.open('cabezote.jpg')
    no_sufre_img = Image.open('NoSufre.jpg')
    si_sufre_img = Image.open('Sisufre.jpg')
except FileNotFoundError:
     st.warning("Image files not found. Please make sure 'cabezote.jpg', 'No sufre.jpg', and 'Si sufre.jpg' are in the same directory.")
     cabezote_img = None
     no_sufre_img = None
     si_sufre_img = None


# App title
if cabezote_img:
    st.image(cabezote_img, use_container_width =True)

st.title("Modelo IA para predicción de problemas cardiacos")

# Summary of the model
st.write("""
Este modelo de inteligencia artificial utiliza un clasificador de Máquinas de Vectores de Soporte (SVC)
para predecir la probabilidad de que un paciente sufra problemas cardiacos
basado en su edad y nivel de colesterol. Los datos de entrada son escalados
para mejorar el rendimiento del modelo.
""")

# Sidebar for user input
st.sidebar.header('Parámetros del Paciente')

edad = st.sidebar.slider('Edad', min_value=20, max_value=80, value=20, step=1)
colesterol = st.sidebar.slider('Colesterol', min_value=100, max_value=400, value=200, step=10)

# Prepare data for prediction
input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Scale the input data
scaled_input_data = scaler.transform(input_data)
scaled_input_data_df = pd.DataFrame(scaled_input_data, columns=['edad', 'colesterol'])

# Make prediction
prediction = svc_model.predict(scaled_input_data_df)

# Display results
st.subheader("Resultado de la Predicción:")

if prediction[0] == 0:
    st.markdown("<h3 style='color: black; background-color: lightgreen;'>No sufrirá del corazón 😊</h3>", unsafe_allow_html=True)
    if no_sufre_img:
        st.image(no_sufre_img, use_container_width =True)
else:
    st.markdown("<h3 style='color: black; background-color: red;'>Sufrirá del corazón 😟</h3>", unsafe_allow_html=True)
    if si_sufre_img:
        st.image(si_sufre_img, use_container_width =True)

st.markdown("---")
st.write("Elaborado por: Alfredo Diaz © UNAB 2025")
