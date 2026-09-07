import cv2
import os

# Carpeta donde se guardarán las imágenes
carpeta = "dataset/M"
#carpeta = os.path.join("dataset", "A")

if not os.path.exists(carpeta):
    os.makedirs(carpeta)

cap = cv2.VideoCapture(0)

contador = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    cv2.rectangle(frame,(150,50),(500,400),(0,255,0),2)

    cv2.putText(
        frame,
        f"Fotos: {contador}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Captura Dataset",frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("c"):

        roi = frame[50:400,150:500]

        nombre = os.path.join(carpeta,f"{contador}.jpg")

        guardado = cv2.imwrite(nombre,roi)
        #print("ruta:", nombre)
        #print("se guardo?", guardado)
        if guardado:
            contador += 1

    elif tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()