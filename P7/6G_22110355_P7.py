import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
imagen = cv2.imread('F_mas.png', cv2.IMREAD_GRAYSCALE)  # Cambia el nombre si es F-.png

# Verifica que la imagen se haya cargado
if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica el nombre y la ruta.")

# Crear un elemento estructurante (puedes ajustar el tamaño)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ------- OPERACIONES MORFOLÓGICAS --------
# Apertura (elimina ruido blanco pequeño)
apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

# Cierre (rellena huecos negros pequeños)
cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)

# Tophat = Original - Apertura (resalta detalles claros)
tophat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)

# Blackhat = Cierre - Original (resalta detalles oscuros)
blackhat = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)

# -------- MOSTRAR RESULTADOS --------
titulos = ['Original', 'Apertura', 'Cierre', 'Tophat', 'Blackhat']
imagenes = [imagen, apertura, cierre, tophat, blackhat]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imagenes[i], cmap='gray')
    plt.title(titulos[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

