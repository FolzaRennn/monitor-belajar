import cv2
from ultralytics import YOLO
import pygame
import time

# --- PENGATURAN ---
FILE_SUARA = "fahhh.mp3" # Ganti dengan nama file suaramu!
DETIK_JEDA = 5           # Jeda suara biar gak spam

# Inisialisasi Suara
pygame.mixer.init()
try:
    suara = pygame.mixer.Sound(FILE_SUARA)
except:
    print(f"ERROR: File {FILE_SUARA} gak ketemu di folder ini!")

# Load Model AI (Otomatis download pas pertama kali jalan)
model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)

# TURUNKAN RESOLUSI BIAR GAK LAG (Penting!)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_played = 0

print("CCTV Pemantau Belajar Aktif! Tekan 'q' untuk berhenti.")

while True:
    success, img = cap.read()
    if not success: break

    # AI Deteksi Objek
    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # Ambil koordinat kotak (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Ubah ke integer

            # Cek apakah itu HP
            nama_benda = model.names[int(box.cls[0])]
            
            if nama_benda == "cell phone":
                # 1. GAMBAR KOTAK MERAH (Thickness: 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # 2. KASIH TULISAN "TERCIDUK!" di atas kotak
                cv2.putText(img, "TERCIDUK!", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # 3. BUNYIKAN SUARA
                if time.time() - last_played > DETIK_JEDA:
                    try:
                        suara.play()
                        last_played = time.time()
                    except: pass

    cv2.imshow("Monitor Belajar - ANTI HP", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
