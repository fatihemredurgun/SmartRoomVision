import cv2
from deepface import DeepFace

img_path = "data/faces/fatih/img_01.jpg"

# Yüz önizlemesi yap
img = cv2.imread(img_path)
cv2.imshow("Yuz", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Yüz algılama denemesi
try:
    obj = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        detector_backend="mtcnn",  # daha güçlü!
        enforce_detection=True
    )
    print("✅ Yüz başarıyla tespit edildi.")
except Exception as e:
    print(f"❌ Yüz tespit edilemedi: {e}")
