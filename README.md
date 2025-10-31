# surface_recognition
Para comenzar, el repositorio contiene dos scripts principales: uno de entrenamiento y otro de ejecución. El primero, llamado train_surfaces.py, se ejecuta y se encarga de procesar un conjunto de imágenes clasificadas por tipo de superficie para calcula(o en un computador conectado a su IP) y usa esos centroides para clasificar en tiempo real lo que el robot ve.

Antes de entrenar, se prepara un pequeño conjunto de imágenes organizadas por carpetas dentro de una carpeta principal llamada dataset. 

El entrenamiento se ejecuta simplemente corriendo el archivo train_surfaces.py desde la terminal. Este script recorre las carpetas, extrae características de color y textura de cada imagen, y calcula el promedio (centroide) de cada clase. Como resultado, genera un archivo llamado centroides_superficies.json que contiene toda la información del modelo: los nombres de las clases, los vectores promedio y los parámetros usados.

Una vez  ese archivo JSON, cuando ejecutas el programa, este se conecta a la cámara de Pepper, obtiene los frames, calcula las mismas características de color y textura que en el entrenamiento, y compara con los centroides del modelo. Según la clase más cercana, el robot pronuncia un mensaje y ajusta su velocidad de movimiento.

La clasificación se basa en dos partes. Por un lado, el histograma HSV describe la distribución de colores de la superficie, y por otro, el LBP mide la variación de textura. Estas dos características se concatenan y normalizan para formar un vector que se compara con los centroides del modelo entrenado. Si el color y la textura del frame actual se parecen más al vector promedio de “pasto”, el robot lo clasificará así. Todo este cálculo es rápido y no necesita GPU.
