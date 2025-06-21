from deepface import DeepFace
import cv2

# Inicializar cámara
cap = cv2.VideoCapture(0)
frame_count = 0
last_emotion = "Detectando..."
skip_frames = 10  # Analizar cada 10 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % skip_frames == 0:
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Si DeepFace retorna lista (más de un rostro), tomamos el primero
            if isinstance(results, list):
                results = results[0]

            # Obtener emoción principal
            last_emotion = results.get("dominant_emotion", "Desconocida")

            # Obtener y dibujar región si existe
            region = results.get("region", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        except Exception as e:
            print("Error:", e)

    # Mostrar emoción en pantalla
    cv2.putText(frame, f'Emocion: {last_emotion}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Analisis de Emociones", frame)

    # Salir con tecla Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
