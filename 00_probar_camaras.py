import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cámara {i}: no disponible")
        continue

    ok, frame = cap.read()
    if ok:
        alto, ancho = frame.shape[:2]
        print(f"Cámara {i}: funciona ({ancho}x{alto})")
        cv2.imshow(f"Camara {i} - apretá una tecla para pasar a la siguiente", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Cámara {i}: abre pero no da imagen")
    cap.release()