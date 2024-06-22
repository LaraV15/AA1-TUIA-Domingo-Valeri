from pipeline import regression_pipeline
from pipeline import classification_pipeline
import pandas as pd

# Create DataFrame
data = {
    'Date': ['2024-06-01'],  # You can modify this to include actual date if needed
    'Location': ['Camberra'],
    'MinTemp': [0],
    'MaxTemp': [10],
    'Rainfall': [0],
    'Evaporation': [5],
    'Sunshine': [1],
    'WindGustDir': ['W'],
    'WindGustSpeed': [1],
    'WindDir9am': ['W'],
    'WindDir3pm': ['W'],
    'WindSpeed9am': [0],
    'WindSpeed3pm': [1],
    'Humidity9am': [0],
    'Humidity3pm': [0],
    'Pressure9am': [0],
    'Pressure3pm': [0],
    'Cloud9am': [1],
    'Cloud3pm': [1],
    'Temp9am': [32],
    'Temp3pm': [33],
    'RainToday': ['No']
}

to_predict = pd.DataFrame(data)

X = regression_pipeline(to_predict)
Xc = classification_pipeline(to_predict)

print(Xc.info())