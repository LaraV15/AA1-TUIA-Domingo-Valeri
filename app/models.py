##################
# DATASET
##################
from dataset import data

def python_models():
	# Iniciamos el dataset
	df = data()

	# Cargamos el dataset
	df.load()

	# Convertimos la columna "Date" y ordenamos por fecha
	df.convert_date()
	df.order_by_date()

	# Filtramos las ciudades de interés
	df.filter_cities()

	# Mergeamos la data de los aeropuertos para completar faltantes
	df.merge_airports()

	# Hacemos que las ciudades y las target de clasificacion sean dummies
	df.get_dummies()

	# Codificamos las columnas de viento
	df.code_wind()

	# Agregamos columnas que indiquen 'lo cíclico' de las fechas
	df.add_date_as_cycle()

	# Completamos nulos
	df.complete_nulls_1()

	# Dividimos en training y test
	df.divide()

	# Completamos nulos usando variables estadísticas
	df.complete_nulls_2()

	# Actualizamos los argumentos dataset_training y dataset_test
	df.divide()

	# Borramos las columnas _Yes del atributo dataset_regression y las Rainfall de dataset_clasification
	df.drop_targets()

	# Nos quedamos con las 10 columnas con más correlación para regresión lineal
	df.filter_features_reg()

	# Generamos las variables X e y para correr nuestro modelo
	df.train_test_regression()


	##################
	# ELASTICNET
	##################
	from elasticnet import model as reg_model

	# Construimos un modelo de elastic net que incluye un standard scaler
	modelo1 = reg_model()

	# Entrenamos el modelo enviando nuestro objeto data
	modelo1.fit(df)


	"""
	# Obtenemos predicciones (UAT only)
	y_pred = modelo.predict(df.X_test)
	print(df.y_test)
	print(y_pred)
	print(df.y_test.shape[0])
	print(len(y_pred))
	"""




	##################
	# NEURAL NETWORK FOR CLASSIFICATION
	##################
	from neuralnetwork import NeuralNetwork as clas_model

	# Dividimos el dataset para clasificación
	df.train_test_classification()

	# Escalamos los datos de clasificación
	df.classification_scale()

	# Aplicamos oversampling con SMOTE al dataset de clasificación
	df.classification_smote()

	# Generamos los datasets de validación
	df.train_validation_classification()

	# Generamos nuestro modelo con los hiperparámetros optimizados
	nn = clas_model(
		input_shape=df.Xc_train.shape[1], 
		num_classes=1, 
		learning_rate=0.00518, 
		num_layers=5, 
		units=[64,128,64,16,112], 
		activations=['sigmoid', 'relu', 'tanh', 'sigmoid', 'relu']
	)

	# Construimos el modelo
	nn.build_model()

	# Lo entrenamos con los datasets generados
	nn.train(df.Xc_train, df.yc_train, df.Xc_valid, df.yc_valid, epochs=10, batch_size=48)

	"""
	# Obtenemos predicciones (UAT only)
	yc_pred = nn.predict_ones_and_zeros(df.Xc_test)
	print(df.yc_test)
	print(yc_pred)
	#print(df.yc_test.shape[0])
	#print(len(yc_pred))
	"""


	##################
	# MODELS DUMP
	##################
	import joblib
	joblib.dump(modelo1, 'regression.joblib')
	joblib.dump(nn, 'classification.joblib')

python_models()