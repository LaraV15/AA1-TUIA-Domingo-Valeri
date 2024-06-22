import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd

st.title('Predicción de lluvia en Australia')
st.subheader('El objetivo de esta app es predecir si llueve mañana y la cantidad de lluvia en función de las variables que se encuentra específicadas debajo.')

# Definimos el rango de fechas permitido
# No ponemos una fecha máxima aunque hay que tener en cuenta que una fecha muy alejada
# de la ultima fecha del data set podria no realizar buenas predicciones
min_date = datetime(2007, 11, 1)

Date = st.date_input('Fecha', value=min_date, min_value=min_date)
st.write('<span style="font-size: 14px;">No se pueden utilizar fechas anteriores al 11-01-2008. Para fechas muy alejadas del 2017-06-24, las predicciones pueden no ser buenas.</span>', unsafe_allow_html=True)

# En el caso de slider para máx y mín tomamos como referencia los del data set y aplicamos un rango mayor, 
# En el caso que configuramos un valor inicial se corresponde con el promedio de esa variable.
# En algunos casos, con st.write() aclaramos algo si es necesario.

MinTemp = st.slider('MinTemp', 10.0, 40.0, 10.93)
MaxTemp = st.slider('MaxTemp', 0.0, 50.0, 21.9)
Rainfall = st.slider('Rainfall', 0.0, 200.0, 2.0)
Evaporation = st.slider('Evaporation', 0.0, 100.0, 5.0)
Sunshine = st.slider('Sunshine', 0.0, 20.0, 7.42)
WindGustDir = st.selectbox('WindgustDir', ['', 'W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
if WindGustDir == '':
     WindGustDir = None  
WindGustSpeed = st.slider('WindGustSpeed', 5.0, 130.0, 40.68)
WindDir9am = st.selectbox('WindDir9am', ['','W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
if WindDir9am  == '':
     WindDir9am  = None  
WindDir3pm = st.selectbox('WindDir3pm', ['','W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
if WindDir3pm == '':
     WindDir3pm = None  
WindSpeed9am = st.slider('WindSpeed9am', 0.0, 75.0, 13.92)
WindSpeed3pm = st.slider('WindSpeed3pm', 0.0, 85.0, 18.83)
Humidity9am = st.slider('Humidity9am', 0.0, 100.0, 69.22)
Humidity3pm = st.slider('Humidity3pm', 0.0, 100.0, 49.89)
Pressure9am = st.slider('Pressure9am', 970.0, 1045.0, 1018.27)
Pressure3pm = st.slider('Pressure3pm', 970.0, 1040.0, 1016.17)
Cloud9am = st.slider('Cloud9am', -5.0, 10.0, 4.78)
Cloud3pm = st.slider('Cloud3pm', 0.0, 10.0, 4.77)
Temp9am = st.slider('Temp9am', -5.0, 40.0, 15.24)
Temp3pm = st.slider('Temp3pm', 2.0, 50.0, 2.0)
Location = st.selectbox('Location',['','Canberra', 'Sydney', 'Brisbane', 'Perth', 'Melbourne', 'Adelaide', 'Darwin',
             'Hobart', 'Albany', 'Albury', 'Townsville', 'Penrith', 'AliceSprings',
             'MountGambier', 'MountGinini', 'Wollongong', 'Launceston', 'Cairns',
             'Tuggeranong', 'Ballarat', 'Bendigo', 'Newcastle', 'GoldCoast', 'Nuriootpa',
             'NorfolkIsland', 'SalmonGums', 'CoffsHarbour', 'BadgerysCreek', 'WaggaWagga',
             'NorahHead', 'Watsonia', 'Woomera', 'Portland', 'Walpole', 'Richmond',
             'Mildura', 'MelbourneAirport', 'PerthAirport', 'Sale', 'SydneyAirport',
             'Williamtown', 'PearceRAAF', 'Dartmoor', 'Cobar', 'Moree', 'Witchcliffe',
             'Nhil', 'Katherine', 'Uluru'])
if Location == '':
     Location = None  
RainToday = st.selectbox('Raintoday', ['','Yes', 'No'])
if RainToday == '':
       st.write('Debe seleccionar una opción para RainToday')


# Create DataFrame
data = {
    'Date': [Date],  # You can modify this to include actual date if needed
    'Location': [Location],
    'MinTemp': [MinTemp],
    'MaxTemp': [MaxTemp],
    'Rainfall': [Rainfall],
    'Evaporation': [Evaporation],
    'Sunshine': [Sunshine],
    'WindGustDir': [WindGustDir],
    'WindGustSpeed': [WindGustSpeed],
    'WindDir9am': [WindDir9am],
    'WindDir3pm': [WindDir3pm],
    'WindSpeed9am': [WindSpeed9am],
    'WindSpeed3pm': [WindSpeed3pm],
    'Humidity9am': [Humidity9am],
    'Humidity3pm': [Humidity3pm],
    'Pressure9am': [Pressure9am],
    'Pressure3pm': [Pressure3pm],
    'Cloud9am': [Cloud9am],
    'Cloud3pm': [Cloud3pm],
    'Temp9am': [Temp9am],
    'Temp3pm': [Temp3pm],
    'RainToday': [RainToday]
}

to_predict = pd.DataFrame(data)


st.write('')
if st.button('Predecir'):
     if RainToday != '':
          st.write('Espere mientras procesamos la información')
          
          # dataset
          from pipeline import regression_pipeline
          from pipeline import classification_pipeline
          X = regression_pipeline(to_predict)
          Xc = classification_pipeline(to_predict)
          
          # models load
          model1 = joblib.load('regression.joblib')
          nn = joblib.load('classification.joblib')

          # predictions
          y = model1.predict(X)
          yc = nn.predict_ones_and_zeros(Xc)
          
          if yc[0][0]==0:
               st.write('Creemos que mañana no lloverá.')
          else:
               st.write('Creemos que mañana lloverá.')
          
          st.write(f'En caso de lluvia, estimamos que caerán {y[0].round(2)} milímetros.')
          

     else:
        st.write('Complete RainToday para continuar')