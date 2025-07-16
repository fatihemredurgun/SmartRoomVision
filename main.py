import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# -------------------- AYARLAR --------------------
RTSP_URL = ""  # 0 = webcam. 1 veya 2 harici kamera olabilir.
MODEL_NAME = "Facenet"
DETECTOR = "opencv"
DB_PATH = "data/faces"
THRESHOLD = 0.4  # cosine mesafe
DETECTION_INTERVAL = 3

# -------------------- KAMERA --------------------
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

# -------------------- MODEL --------------------
print("🔧 DeepFace modeli yükleniyor...")
model = DeepFace.build_model(MODEL_NAME)
print("✅ Model yüklendi.")

# -------------------- VERİTABANINI VEKÖRLE --------------------
print("📦 Yüz vektörleri hazırlanıyor...")
face_db = []

for person in os.listdir(DB_PATH):
    person_dir = os.path.join(DB_PATH, person)
    if not os.path.isdir(person_dir):
        continue
    for img in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img)
        try:
            embedding_obj = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False
            )
            if isinstance(embedding_obj, list):
                embedding = embedding_obj[0]["embedding"]
                face_db.append({"name": person, "embedding": embedding})
        except Exception as e:
            print(f"[!] {img_path} için hata: {e}")

print(f"✅ {len(face_db)} yüz vektörü yüklendi.")

# -------------------- TANIMA --------------------
def recognize_face_fast(frame):
    frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:
        if frame is None or frame.shape[0] == 0:
            print("❌ Boş veya hatalı frame")
            return "Frame hatası"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Yüz var mı kontrol et
        faces = DeepFace.extract_faces(
            img_path=rgb_frame,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

        if len(faces) != 1 or faces[0]['facial_area']['w'] < 60:
            return "Yuz Yok"

        # 2. Embedding çıkar (hata fırlatırsa try yakalayacak)
        result = DeepFace.represent(
            img_path=rgb_frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )

        embedding = result[0]["embedding"]

        # 3. Gürültü filtreleme (norm kontrolü)
        if np.linalg.norm(embedding) < 10:
            return "Yüz Bozuk"

        # 4. En yakın eşleşmeyi bul
        min_dist = float("inf")
        identity = "Taninmadi"

        for person in face_db:
            dist = cosine(embedding, person["embedding"])
            if dist < min_dist:
                min_dist = dist
                identity = person["name"]

        if min_dist > THRESHOLD:
            return "Taninmadi"
        return identity

    except Exception as e:
        print("[!] Tanima hatasi:", e)
        return "Yuz Algilanamadi"




# -------------------- ANA DÖNGÜ --------------------
last_detection_time = 0
last_detected_name = "..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Görüntü alinamadi.")
        break

    current_time = time.time()
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        name = recognize_face_fast(frame)
        last_detected_name = name
        last_detection_time = current_time

    # Etiketle ve göster
    label = f"Kisi: {last_detected_name}"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("SmartRoomVision - Hızlı Tanıma", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("test_frame.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()
