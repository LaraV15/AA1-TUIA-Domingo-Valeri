from dataset import data

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

print(df.dataset.info())