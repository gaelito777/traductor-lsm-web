import cv2
import mediapipe as mp
import time
import os

# Activar cámara
cap = cv2.VideoCapture(0)

# Comando manos
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.8
)

mp_dibujo = mp.solutions.drawing_utils

# Variables para texto
texto_final = ""
ultima_letra = ""
tiempo_letra = 0

while True:

    success, img = cap.read()

    if not success:
        break

    # Voltear imagen
    img = cv2.flip(img, 1)
    cv2.rectangle(
        img,
        (0,0),
        (img.shape[1],60),
        (35,35,35),
        -1
        )
    cv2.putText(
        img,
        "Traductor LSM",
        (20,40),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255,255,255),
        2
        )

    # Convertir a RGB
    imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tiempo 2 ms
    time.sleep(0.02)

    # Procesar imagen
    resultado = manos.process(imRGB)
    x1, y1 = 150, 50
    x2, y2 = 500, 400
    
    #cuadro
    color = (255,255,255)
    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),   
        color,
        2
        )
    cv2.putText(
        img,
        "Coloque la mano dentro del recuadro",
        (x1, y1+15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    texto = ""

    if resultado.multi_hand_landmarks:

        for idx, mano in enumerate(resultado.multi_hand_landmarks):

            mp_dibujo.draw_landmarks(
                img,
                mano,
                mp_manos.HAND_CONNECTIONS
            )

            punto = []

            alto, ancho, _ = img.shape

            # Obtener landmarks
            for id, lm in enumerate(mano.landmark):

                x = int(lm.x * ancho)
                y = int(lm.y * alto)

                punto.append((x, y))

            if len(punto) == 21:
                
                 # POSICION DE LA MUÑECA
                 x_muneca = punto[0][0]
                 y_muneca = punto[0][1]
                 
                 # VALIDAR ROI
                 if not (
                     x1 <= x_muneca <= x2 and
                     y1 <= y_muneca <= y2
                     ):
                     cv2.putText( img,
                                 "MANO FUERA DE ROI",
                                 (10, 260),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 1,(0, 0, 255), 2
                                 )
                     continue
            dedos = []
            # Detectar tipo de mano
            tipo_mano = "Right"
            if (
                    resultado.multi_handedness and
                    len(resultado.multi_handedness) > idx
                ):
                    tipo_mano = (
                        resultado.multi_handedness[idx]
                        .classification[0]
                        .label
                    )
                    
            # DETECTAR PALMA O DORSO

            x5 = punto[5][0]
            x17 = punto[17][0]

            if tipo_mano == "Right":

                    if x5 < x17:
                        lado = "PALMA"
                    else:
                        lado = "DORSO"

            else:

                    if x5 > x17:
                        lado = "PALMA"
                    else:
                        lado = "DORSO"
                        
                # DETECCION DEL PULGAR
            if tipo_mano == "Right":

                    if punto[4][0] < punto[3][0]:
                        dedos.append(1)
                    else:
                        dedos.append(0)

            else:

                    if punto[4][0] > punto[3][0]:
                        dedos.append(1)
                    else:
                        dedos.append(0)

            # OTROS DEDOS
            for i in [8, 12, 16, 20]:
                    diferencia = punto[i - 2][1] - punto[i][1]
                    if diferencia > 20:
                        dedos.append(1)
                    else:
                        dedos.append(0)

            # RECONOCIMIENTO DE LETRAS
            # A
            if dedos == [1,0,0,0,0]:
                    texto = "A"
                # B
            elif dedos == [0,1,1,1,1]:
                    texto = "B"
                # D
            elif dedos == [0,1,0,0,0]:
                    texto = "D"
                # E
            elif dedos == [1,0,0,0,1]:
                    texto = "E"
                # F
            elif dedos == [0,0,1,1,1]:
                    texto = "F"
                # I
            elif dedos == [0,0,0,0,1]:
                    texto = "I"
                # S
            elif dedos == [0,0,0,0,0]:
                    texto = "S"
                # U
            elif dedos == [0,1,1,0,0]:
                    texto = "U"
                # L
            elif (
                    dedos[0] == 1 and
                    dedos[1] == 1 and
                    sum(dedos) == 2
                ):
                    texto = "L"

            # GUARDAR LETRAS
            if texto != "":
                    if texto != ultima_letra:
                        tiempo_letra = time.time()
                        ultima_letra = texto

                    if time.time() - tiempo_letra > 1:
                        texto_final += texto
                        ultima_letra = ""

                # Mostrar letra actual
            if texto != "":
             cv2.putText(
                 img,
                 "LETRA: " + texto,
                 (20,120),
                 cv2.FONT_HERSHEY_DUPLEX,
                 1.8,
                 (0,255,255),
                 3
            )
        # Barra inferior de texto
    cv2.rectangle(
        img,
        (0, img.shape[0]-70),
        (img.shape[1], img.shape[0]),
        (35,35,35),
        -1
    )

    cv2.putText(
        img,
        "Texto generado:",
        (20, img.shape[0]-40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        img,
        texto_final,
        (230, img.shape[0]-40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.imshow("Traductor LSM", img)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27: 
        break
    if tecla == ord("r"):
        texto_final = ""

cap.release()
cv2.destroyAllWindows()