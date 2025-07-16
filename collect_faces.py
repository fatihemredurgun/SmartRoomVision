import cv2
import os
from deepface import DeepFace
# RTSP URL (config.py'den alınabilir)
RTSP_URL = ""  # kendi adresinle değiştir
cap = cv2.VideoCapture(RTSP_URL)

SAVE_DIR = "data/faces/fatih/"
os.makedirs(SAVE_DIR, exist_ok=True)
img_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

# 🔍 Haarcascade yüz algılama (çok hızlı)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[i] 's' tuşu ile yüzü kaydedebilirsin. Çıkmak için 'q'.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Görüntü alınamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = frame[y:y+h, x:x+w]

            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_count += 1
                path = os.path.join(SAVE_DIR, f"img_{img_count:02d}.jpg")
                cv2.imwrite(path, face_img)
                print(f"[+] {path} kaydedildi.")
    else:
        cv2.putText(frame, "❗ Yüz bulunamadi", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Yüz Toplama - 's' ile kaydet, 'q' ile çık", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Toplam {img_count} yüz görüntüsü kaydedildi.")