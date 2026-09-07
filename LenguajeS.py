import cv2
import mediapipe as mp
import time
import joblib

# ==========================================
# CARGAR MODELO ENTRENADO
# ==========================================

modelo = joblib.load("modelo_letras.pkl")

print("Modelo cargado correctamente.")

# ACTIVAR CÁMARA

cap = cv2.VideoCapture(0)

mp_manos = mp.solutions.hands

manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.8
)

mp_dibujo = mp.solutions.drawing_utils

# ==========================================
# VARIABLES PARA TEXTO
# ==========================================

texto_final = ""
ultima_letra = ""
tiempo_letra = 0

# ==========================================
# BUCLE PRINCIPAL
# ==========================================

while True:

    success, img = cap.read()

    if not success:
        break

    # Voltear imagen
    img = cv2.flip(img, 1)

    # ======================================
    # ENCABEZADO
    # ======================================

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], 60),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        img,
        "Traductor LSM",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ======================================
    # CONVERTIR BGR -> RGB
    # ======================================

    imRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Tiempo de espera
    time.sleep(0.02)

    # ======================================
    # DETECTAR MANO
    # ======================================

    resultado = manos.process(imRGB)

    # ======================================
    # REGION DE INTERES (ROI)
    # ======================================

    x1, y1 = 150, 50
    x2, y2 = 500, 400

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "Coloque la mano dentro del recuadro",
        (x1, y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ======================================
    # VARIABLE DE LETRA
    # ======================================

    texto = ""

    # ======================================
    # SI SE DETECTA UNA MANO
    # ======================================

    if resultado.multi_hand_landmarks:

        # Solo utilizamos la primera mano
        mano = resultado.multi_hand_landmarks[0]

        # Dibujar landmarks
        mp_dibujo.draw_landmarks(
            img,
            mano,
            mp_manos.HAND_CONNECTIONS
        )

        punto = []

        alto, ancho, _ = img.shape

        # ==================================
        # OBTENER LOS 21 LANDMARKS
        # ==================================

        for lm in mano.landmark:

            x = int(lm.x * ancho)
            y = int(lm.y * alto)

            punto.append((x, y))

        # ==================================
        # VALIDAR QUE EXISTAN 21 PUNTOS
        # ==================================

        if len(punto) == 21:

            # Posición de la muñeca
            x_muneca = punto[0][0]
            y_muneca = punto[0][1]

            # ==================================
            # VALIDAR ROI
            # ==================================

            if not (
                x1 <= x_muneca <= x2
                and
                y1 <= y_muneca <= y2
            ):

                cv2.putText(
                    img,
                    "MANO FUERA DE ROI",
                    (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                continue

            # ==================================
            # CREAR CARACTERISTICAS
            # ==================================

            caracteristicas = []

            for lm in mano.landmark:

                caracteristicas.append(lm.x)
                caracteristicas.append(lm.y)
                caracteristicas.append(lm.z)

            # ==================================
            # RECONOCER LETRA CON EL MODELO
            # ==================================

            prediccion = modelo.predict(
                [caracteristicas]
            )

            texto = prediccion[0]

            # ==================================
            # MOSTRAR LETRA ACTUAL
            # ==================================

            cv2.putText(
                img,
                "LETRA: " + texto,
                (20, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                (0, 255, 255),
                3
            )

            # ==================================
            # GUARDAR LETRA DESPUES DE 1 SEGUNDO
            # ==================================

            if texto != "":

                if texto != ultima_letra:

                    tiempo_letra = time.time()
                    ultima_letra = texto

                if time.time() - tiempo_letra > 1:

                    texto_final += texto

                    ultima_letra = ""

    # ======================================
    # BARRA INFERIOR
    # ======================================

    cv2.rectangle(
        img,
        (0, img.shape[0] - 70),
        (img.shape[1], img.shape[0]),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        img,
        "Texto generado:",
        (20, img.shape[0] - 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        texto_final,
        (230, img.shape[0] - 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # ======================================
    # MOSTRAR VENTANA
    # ======================================

    cv2.imshow(
        "Traductor LSM",
        img
    )

    # ======================================
    # TECLAS
    # ======================================

    tecla = cv2.waitKey(1) & 0xFF

    # ESC = salir
    if tecla == 27:
        break

    # R = borrar texto
    if tecla == ord("r"):
        texto_final = ""
        ultima_letra = ""

# ==========================================
# CERRAR
# ==========================================

cap.release()
cv2.destroyAllWindows()