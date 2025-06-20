import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('F_mas.png', cv2.IMREAD_GRAYSCALE)

# Verificar que se haya cargado correctamente
if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica el nombre y la ruta.")

# --- Filtros de detección de bordes ---

# 1. Laplaciano (detecta bordes en todas direcciones)
laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
laplaciano = cv2.convertScaleAbs(laplaciano)

# 2. Sobel en X (bordes verticales)
sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

# 3. Sobel en Y (bordes horizontales)
sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)

# 4. Canny (detector de bordes más completo)
canny = cv2.Canny(imagen, 100, 200)

# --- Mostrar resultados ---
titulos = ['Original', 'Laplaciano', 'Sobel X', 'Sobel Y', 'Canny']
imagenes = [imagen, laplaciano, sobelx, sobely, canny]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(titulos[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

