import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split

class data():
    def __init__(self, dataset=None):
        self.dataset = dataset
    
    def load(self):
        file_path= 'weatherAUS.csv'
        self.dataset = pd.read_csv(file_path, sep=',',engine='python')
        self.dataset.drop(self.dataset.columns[0], axis=1, inplace=True)
    
    def convert_date(self):
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
    
    def order_by_date(self):
        self.dataset = self.dataset.sort_values(by='Date', ignore_index=True)
    
    def filter_cities(self):
        ciudades_deseadas = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport']
        self.dataset = self.dataset[self.dataset['Location'].isin(ciudades_deseadas)]

    def merge_airports(self):
        df = self.dataset.copy()
        df = df.sort_values(by='Date', ignore_index=True)
        df_Sydney = df[(df["Location"]=="Sydney")]
        df_aSydney = df[(df["Location"]=="SydneyAirport")]
        df_Melbourne = df[(df["Location"]=="Melbourne")]
        df_AMelbourne = df[(df["Location"]=="MelbourneAirport")]

        # Hacemos un join outer sobre "Date" para mantener todos los registros
        df_s = df_Sydney.merge(df_aSydney, on='Date', how='outer', suffixes=('', '_ASydney'))
        df_m = df_Melbourne.merge(df_AMelbourne, on='Date', how='outer', suffixes=('', '_AMelbourne'))

        # Eliminamos la columna Location_Aeropuerto que no necesitamos, y llenamos las vacías con lo que corresponda para cada ciudad
        # (en caso de que el registro haya existido en el aeropuerto, y no en la ciudad)
        df_s.drop("Location_ASydney", axis=1, inplace=True)
        df_m.drop("Location_AMelbourne", axis=1, inplace=True)
        df_s["Location"] = "Sydney"
        df_m["Location"] = "Melbourne"

        # Generamos una lista de columnas numéricas y categóricas que tiene sentido ser completadas con los valores de los aeropuertos.
        columnas_numericas = ['MinTemp', 'MaxTemp','Evaporation', 'Sunshine', 'Humidity9am', 'Humidity3pm','Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm','Temp9am', 'Temp3pm', "Rainfall", 'RainfallTomorrow', "WindGustSpeed","WindSpeed9am","WindSpeed3pm"]
        columnas_categoricas = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

        # Completamos nulos para esas columnas, y a su vez las eliminamos
        for columna in columnas_numericas+columnas_categoricas:
            columna_as = columna+"_ASydney"
            columna_am = columna+"_AMelbourne"
            df_s[columna] = df_s[columna].fillna(df_s[columna_as])
            df_m[columna] = df_m[columna].fillna(df_m[columna_am])
            df_s.drop(columna_as, axis=1, inplace=True)
            df_m.drop(columna_am, axis=1, inplace=True)

        # Generemos el nuevo dataframe
        df_without_sydney_melbourne = df[(df["Location"]!="Sydney")&(df["Location"]!="SydneyAirport")&(df["Location"]!="Melbourne")&(df["Location"]!="MelbourneAirport")]
        df = pd.concat([df_without_sydney_melbourne, df_s, df_m], ignore_index=True)
        self.dataset = df

    def get_dummies(self):
        palabrasObject = ["Location",  "RainToday", "RainTomorrow"]
        self.dataset = pd.get_dummies(self.dataset, columns = palabrasObject, drop_first=True, dtype=int)

    def code_wind(self):
        values = ['NW', 'ENE', 'SSE', 'SE', 'E', 'S', 'N', 'WNW', 'ESE', 'NE', 'NNE', 'NNW', 'SW', 'W', 'WSW', 'SSW']
        codes = [315, 67.5, 157.5, 135, 90, 180, 0, 292.5, 112.5, 45, 22.5, 337.5, 225, 270, 247.5, 202.5]
        wind_coded = pd.DataFrame({'value': values, 'code': codes})

        # Reemplazar los valores originales por los codificados
        for columna in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            self.dataset[columna] = self.dataset[columna].replace(dict(zip(wind_coded['value'], wind_coded['code'])))

    def add_date_as_cycle(self):
        # Extraemos el día del año (1 a 365)
        self.dataset['dia'] = self.dataset['Date'].dt.dayofyear

        # Calculamos la característica trigonométrica (solo el seno)
        self.dataset['dia'] = np.sin(2 * np.pi * self.dataset['dia'] / 365)

        # Agregamos el año como una característica adicional
        self.dataset['año'] = self.dataset['Date'].dt.year

    def complete_nulls_1(self):
        df = self.dataset.copy()
        df_canberra = df[df["Location_Canberra"]==1]
        df_cobar = df[df["Location_Cobar"]==1]
        df_dartmoor = df[df["Location_Dartmoor"]==1]
        df_melbourne = df[df["Location_Melbourne"]==1]
        df_mountgambier = df[df["Location_MountGambier"]==1]
        df_sydney = df[df["Location_Sydney"]==1]
        df_adelaide = df[(df["Location_Canberra"]==0)&(df["Location_Cobar"]==0)&(df["Location_Dartmoor"]==0)&(df["Location_Melbourne"]==0)&(df["Location_MountGambier"]==0)&(df["Location_Sydney"]==0)]

        df_cities = [df_adelaide,df_canberra, df_cobar, df_dartmoor, df_melbourne, df_mountgambier, df_sydney]

        for city in df_cities:
            for variable in list(df.columns):
                # Rellenamos los valores nulos con el valor de los tres días anteriores (de este año o el anterior)
                # si existen (pues el df está ordenado por fecha)
                city[variable] = city[variable].fillna(city[variable].shift(-1))
                city[variable] = city[variable].fillna(city[variable].shift(-2))
                city[variable] = city[variable].fillna(city[variable].shift(-3))

                city[variable] = city[variable].fillna(city[variable].shift(-365))
                city[variable] = city[variable].fillna(city[variable].shift(-365-1))
                city[variable] = city[variable].fillna(city[variable].shift(-365-2))
                city[variable] = city[variable].fillna(city[variable].shift(-365-3))
        df = pd.concat(df_cities)
        self.dataset = df

    def complete_nulls_2(self):
        df_training = self.dataset_training.copy()

        t_df_canberra = df_training[df_training["Location_Canberra"]==1]
        t_df_cobar = df_training[df_training["Location_Cobar"]==1]
        t_df_dartmoor = df_training[df_training["Location_Dartmoor"]==1]
        t_df_melbourne = df_training[df_training["Location_Melbourne"]==1]
        t_df_mountgambier = df_training[df_training["Location_MountGambier"]==1]
        t_df_sydney = df_training[df_training["Location_Sydney"]==1]
        t_df_adelaide = df_training[(df_training["Location_Canberra"]==0)&(df_training["Location_Cobar"]==0)&(df_training["Location_Dartmoor"]==0)&(df_training["Location_Melbourne"]==0)&(df_training["Location_MountGambier"]==0)&(df_training["Location_Sydney"]==0)]
        t_df_cities = [t_df_adelaide,t_df_canberra,t_df_cobar,t_df_dartmoor,t_df_melbourne,t_df_mountgambier,t_df_sydney]


        #lara, puede ser que faltaba la linea del def
        df = self.dataset.copy()
        df_canberra = df[df["Location_Canberra"]==1]
        df_cobar = df[df["Location_Cobar"]==1]
        df_dartmoor = df[df["Location_Dartmoor"]==1]
        df_melbourne = df[df["Location_Melbourne"]==1]
        df_mountgambier = df[df["Location_MountGambier"]==1]
        df_sydney = df[df["Location_Sydney"]==1]
        df_adelaide = df[(df["Location_Canberra"]==0)&(df["Location_Cobar"]==0)&(df["Location_Dartmoor"]==0)&(df["Location_Melbourne"]==0)&(df["Location_MountGambier"]==0)&(df["Location_Sydney"]==0)]
        df_cities = [df_adelaide,df_canberra, df_cobar, df_dartmoor, df_melbourne, df_mountgambier, df_sydney]

        columns = ["WindGustDir", "WindGustSpeed", "WindDir9am", "Evaporation", "Humidity9am", "Humidity3pm", "Sunshine", "Cloud9am", "Cloud3pm"]

        for i in range(len(df_cities)):
            for variable in columns:
                mediana = t_df_cities[i][variable].median()
                promedio_total = df_training[variable].mean()
                df_cities[i][variable].fillna(mediana, inplace=True)
                df_cities[i][variable].fillna(promedio_total, inplace=True)

        df = pd.concat(df_cities)
        self.dataset = df

    def drop_targets(self):
        todas_las_columnas = list(self.dataset.columns)
        columnas_reg = []
        columnas_clas = []
        for i in todas_las_columnas:
            if "Location" not in i and "_Yes" not in i and "Date" not in i:
                columnas_reg.append(i)
            if "Location" not in i and "RainfallTomorrow" not in i and "Date" not in i:
                columnas_clas.append(i)
        df_regression = self.dataset[columnas_reg]
        df_clasification = self.dataset[columnas_clas]
        self.dataset_regression = df_regression
        self.dataset_clasification = df_clasification
    
    def filter_features_reg(self):
        features = ['Humidity3pm', 'Sunshine', 'Rainfall', 'Cloud3pm', 'WindGustSpeed', 'Pressure3pm', 'Pressure9am', 'Cloud9am', 'Humidity9am', 'Temp3pm']
        self.X = self.dataset_regression[features]
        self.y = self.dataset_regression['RainfallTomorrow']

    def divide(self):
        n = self.dataset.shape[0]
        m = int((n/100)*80)
        self.cut = m
        self.dataset_training = self.dataset.head(m)
        self.dataset_test = self.dataset.tail(n-m)
    
    def train_test_regression(self):
        self.X_train = self.X.iloc[:self.cut]
        self.X_test = self.X.iloc[self.cut:]
        self.y_train = self.y.iloc[:self.cut]
        self.y_test = self.y.iloc[self.cut:]
    
    def train_test_classification(self):
        self.Xc = self.dataset_clasification.drop(columns="RainTomorrow_Yes")
        self.yc = self.dataset_clasification['RainTomorrow_Yes']
        self.Xc_train = self.Xc.iloc[:self.cut]
        self.Xc_test = self.Xc.iloc[self.cut:]
        self.yc_train = self.yc.iloc[:self.cut]
        self.yc_test = self.yc.iloc[self.cut:]

    def classification_scale(self):
        scaler = StandardScaler()
        self.Xc_train = scaler.fit_transform(self.Xc_train)
        self.Xc_test = scaler.transform(self.Xc_test)
        import joblib
        joblib.dump(scaler, 'nn_scaler.joblib')

    def classification_smote(self):
        smote = SMOTE(random_state=42)
        Xc_train_resampled, yc_train_resampled = smote.fit_resample(self.Xc_train, self.yc_train)
        self.Xc_train = Xc_train_resampled
        self.yc_train = yc_train_resampled

    def train_validation_classification(self):
        Xc_train, Xc_valid, yc_train, yc_valid = train_test_split(self.Xc_train, self.yc_train, test_size=0.2, random_state=42)
        self.Xc_train = Xc_train
        self.Xc_valid = Xc_valid
        self.yc_train = yc_train
        self.yc_valid = yc_valid