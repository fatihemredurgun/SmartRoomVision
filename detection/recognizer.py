# detection/recognizer.py
import os
import cv2
import numpy as np
from deepface import DeepFace

# Yüz veritabanı dizini
FACE_DB_DIR = "data/faces"

def recognize_face(frame, model):
    try:
        # RGB dönüşümü (DeepFace için zorunlu)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # DeepFace ile yüz tanıma
        results = DeepFace.find(
            img_path=img_rgb,
            db_path=FACE_DB_DIR,
            model_name="Facenet",
            model=model,
            detector_backend="opencv",
            enforce_detection=False,
            distance_metric='cosine',
            silent=True
        )

        # Eğer tanınan bir kişi varsa
        if isinstance(results, list) and len(results) > 0 and not results[0].empty:
            identity_path = results[0].iloc[0]["identity"]
            person_name = os.path.basename(os.path.dirname(identity_path))
            return person_name
        else:
            return "Bilinmiyor"

    except Exception as e:
        print(f"[!] Tanıma hatası: {e}")
        return "Hata"
