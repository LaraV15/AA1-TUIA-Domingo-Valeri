import numpy as np
import pandas as pd

from dataset import data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline

class model():
	def __init__(self):
		self.model = make_pipeline(StandardScaler(),
                        ElasticNetCV(
                              l1_ratio        = [0, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                              alphas          = np.logspace(-10, 10, 200),
                              cv              = 10))
	
	def fit(self, df : data):
		_ = self.model.fit(df.X_train, df.y_train)
	
	def predict(self, rows):
		if type(rows)==type(pd.DataFrame()) and rows.shape[1]==10:
			return self.model.predict(rows)
		else:
			print("Error en los datos ingresados. Verifique que el objeto sea un Dataframe y que tenga 10 columnas sin valores nulos.")
			return None