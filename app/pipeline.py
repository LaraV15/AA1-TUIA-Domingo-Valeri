import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE

def common_pipeline(to_predict):
	# Date conversion
	to_predict['Date'] = pd.to_datetime(to_predict['Date'])
	
	# Dummies
	palabrasObject = ["Location",  "RainToday"]
	to_predict = pd.get_dummies(to_predict, columns = palabrasObject, drop_first=True, dtype=int)

	# Wind code
	values = ['NW', 'ENE', 'SSE', 'SE', 'E', 'S', 'N', 'WNW', 'ESE', 'NE', 'NNE', 'NNW', 'SW', 'W', 'WSW', 'SSW']
	codes = [315, 67.5, 157.5, 135, 90, 180, 0, 292.5, 112.5, 45, 22.5, 337.5, 225, 270, 247.5, 202.5]
	wind_coded = pd.DataFrame({'value': values, 'code': codes})
	for columna in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
		to_predict[columna] = to_predict[columna].replace(dict(zip(wind_coded['value'], wind_coded['code'])))
	

	# Cycle date
	to_predict['dia'] = to_predict['Date'].dt.dayofyear
	to_predict['dia'] = np.sin(2 * np.pi * to_predict['dia'] / 365)
	to_predict['año'] = to_predict['Date'].dt.year

	return to_predict

def regression_pipeline(to_predict):
	to_predict = common_pipeline(to_predict)
	
	# Saco columnas de clasificacion
	todas_las_columnas = list(to_predict.columns)
	columnas_reg = []
	for i in todas_las_columnas:
		if "Location" not in i and "_Yes" not in i and "Date" not in i:
			columnas_reg.append(i)
	to_predict = to_predict[columnas_reg]

	# Features
	features = ['Humidity3pm', 'Sunshine', 'Rainfall', 'Cloud3pm', 'WindGustSpeed', 'Pressure3pm', 'Pressure9am', 'Cloud9am', 'Humidity9am', 'Temp3pm']
	to_predict = to_predict[features]

	

	# 
	return to_predict

def classification_pipeline(to_predict):
	to_predict = common_pipeline(to_predict)
	
	# Saco columnas de regresión lineal
	todas_las_columnas = list(to_predict.columns)
	columnas_clas = []
	for i in todas_las_columnas:
		if "Location" not in i and "RainfallTomorrow" not in i and "Date" not in i:
			columnas_clas.append(i)
	to_predict = to_predict[columnas_clas]

	# Scale
	scaler = StandardScaler()
	to_predict = scaler.fit_transform(to_predict)
	
	return to_predict