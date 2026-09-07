import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Cargar el archivo de landmarks
datos = pd.read_csv("landmarks.csv")

# Separar etiquetas y características
X = datos.drop("letra", axis=1)
y = datos["letra"]

# Dividir datos para entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrenando modelo...")

# Crear modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Entrenar
modelo.fit(X_entrenamiento, y_entrenamiento)

# Evaluar
predicciones = modelo.predict(X_prueba)

precision = accuracy_score(y_prueba, predicciones)

print("\n==============================")
print("ENTRENAMIENTO TERMINADO")
print("==============================")
print("Muestras para entrenamiento:", len(X_entrenamiento))
print("Muestras para prueba:", len(X_prueba))
print("Precisión:", precision)

# Guardar modelo
joblib.dump(modelo, "modelo_letras.pkl")

print("Modelo guardado como: modelo_letras.pkl")