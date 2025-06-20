import cv2
import numpy as np

# === 1. Cargar imagen original en escala de grises ===
imagen = cv2.imread('hotwheels.png', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    raise ValueError("No se pudo cargar la imagen. Verifica el nombre y la ruta.")

# === 2. Crear un template desde un ROI manualmente ===
# Coordenadas de recorte: (x, y, ancho, alto) — AJUSTA ESTAS SI ES NECESARIO
x, y, w, h = 70, 70, 50, 50  # Puedes ajustar para tu imagen
template = imagen[y:y+h, x:x+w]

# Guardar template para verificarlo si quieres
cv2.imwrite('template_generado.png', template)

# === 3. Aplicar template matching ===
resultado = cv2.matchTemplate(imagen, template, cv2.TM_CCOEFF_NORMED)

# === 4. Encontrar coincidencias con similitud ≥ 0.85 ===
umbral = 0.85
loc = np.where(resultado >= umbral)

# === 5. Dibujar coincidencias en la imagen original (convertida a color) ===
imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
conteo = 0
for pt in zip(*loc[::-1]):
    cv2.rectangle(imagen_color, pt, (pt[0] + w, pt[1] + h), (255, 2, 45), 0)
    conteo += 1

print(f"Se encontraron {conteo} coincidencias con umbral ≥ {umbral}")

# === 6. Mostrar resultados ===
cv2.imshow('Detecciones de Template Matching', imagen_color)
cv2.imshow('Template utilizado', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

