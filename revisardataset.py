import pandas as pd

# Cargar el archivo de landmarks
df = pd.read_csv("landmarks.csv")

print("\n==============================")
print("      REVISION DEL DATASET")
print("==============================")

# Total de muestras
print("\nTotal de muestras:", len(df))

# Cantidad de muestras por letra
conteo = df["letra"].value_counts().sort_index()

print("\nMuestras por letra:")
print(conteo)

# Valores mínimo y máximo
print("\n==============================")
print("ESTADISTICAS")
print("==============================")

print("Mínimo de muestras:", conteo.min())
print("Máximo de muestras:", conteo.max())

# Letras con pocas muestras
print("\nLetras con menos de 350 muestras:")

pocas = conteo[conteo < 350]

if len(pocas) == 0:
    print("Ninguna")
else:
    print(pocas)

print("\n==============================")
print("REVISION TERMINADA")
print("==============================")