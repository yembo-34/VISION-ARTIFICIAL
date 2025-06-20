import cv2
import numpy as np

# 1. Cargar imagen
img = cv2.imread('hotwheels.jpg')

if img is None:
    print("Error: No se pudo cargar la imagen. Verifica el path.")
    exit()

# 2. Mostrar ventana para seleccionar ROI con el mouse
roi = cv2.selectROI("Selecciona el ROI", img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Selecciona el ROI")  # Cerrar ventana tras seleccionar

# roi = (x, y, w, h)
x, y, w, h = map(int, roi)

# 3. Crear máscara (negra afuera del ROI, blanca dentro)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[y:y+h, x:x+w] = 255

# 4. Aplicar máscara
img_roi = cv2.bitwise_and(img, img, mask=mask)

# 5. Convertir a escala de grises para detección de esquinas
gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

# 6. Detectar esquinas con Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)

# 7. Dibujar esquinas detectadas
if corners is not None:
    corners = corners.astype(int)
    for i in corners:
        x_c, y_c = i.ravel()
        cv2.circle(img_roi, (x_c, y_c), 5, (0, 255, 0), -1)

# 8. Mostrar resultado final
cv2.imshow('ROI con esquinas', img_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
