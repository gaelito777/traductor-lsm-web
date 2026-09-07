import cv2
import mediapipe as mp
import os
import csv

# Carpeta principal del dataset
carpeta_dataset = "dataset"

# Archivo donde guardaremos los landmarks
archivo_salida = "landmarks.csv"

# MediaPipe Hands
mp_manos = mp.solutions.hands

manos = mp_manos.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.8
)

# CREAR ENCABEZADOS DEL CSV
encabezados = ["letra"]

for i in range(21):
    encabezados.append(f"x{i}")
    encabezados.append(f"y{i}")
    encabezados.append(f"z{i}")

# PROCESAR DATASET

total_imagenes = 0
total_procesadas = 0
total_sin_mano = 0

with open(archivo_salida, "w", newline="", encoding="utf-8") as archivo:

    escritor = csv.writer(archivo)

    escritor.writerow(encabezados)

    # Recorrer las carpetas A, B, C...
    for letra in sorted(os.listdir(carpeta_dataset)):

        carpeta_letra = os.path.join(carpeta_dataset, letra)

        # Verificar que sea una carpeta
        if not os.path.isdir(carpeta_letra):
            continue

        print(f"\nProcesando letra: {letra}")

        # Recorrer fotografías
        for nombre_imagen in sorted(os.listdir(carpeta_letra)):

            ruta_imagen = os.path.join(
                carpeta_letra,
                nombre_imagen
            )

            # Solo procesar imágenes
            if not nombre_imagen.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            total_imagenes += 1

            # Leer imagen
            imagen = cv2.imread(ruta_imagen)

            if imagen is None:
                print("No se pudo leer:", ruta_imagen)
                continue

            # Convertir BGR → RGB
            imagen_rgb = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2RGB
            )

            # Detectar mano
            resultado = manos.process(imagen_rgb)

            # Verificar si encontró una mano
            if not resultado.multi_hand_landmarks:

                total_sin_mano += 1

                print(
                    "Sin mano:",
                    nombre_imagen
                )

                continue

            # Obtener la primera mano
            mano = resultado.multi_hand_landmarks[0]

            fila = [letra]

            # Obtener los 21 landmarks
            for punto in mano.landmark:

                fila.append(punto.x)
                fila.append(punto.y)
                fila.append(punto.z)

            # Guardar fila
            escritor.writerow(fila)

            total_procesadas += 1

        print(f"Letra {letra} terminada.")

# RESULTADOS

manos.close()

print("\n==============================")
print("PROCESO TERMINADO")
print("==============================")
print("Imágenes encontradas:", total_imagenes)
print("Imágenes procesadas:", total_procesadas)
print("Imágenes sin mano:", total_sin_mano)
print("Archivo creado:", archivo_salida)