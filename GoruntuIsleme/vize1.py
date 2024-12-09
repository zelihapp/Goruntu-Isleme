import cv2
import numpy as np

# Video yolu
video_path = ("C:\\Users\\zelih\\OneDrive\\Belgeler\\OPENCV\\GoruntuIslemeOdev\\train.mp4")

# Video yakalama
cap = cv2.VideoCapture(video_path)

# Arkaplan çıkarma algoritması
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=True)

# Sayaçlar
vehicle_count = 0
pedestrian_count = 0

# Daha önce sayılan nesnelerin merkezi noktalarını tutmak için liste
counted_pedestrians = []
counted_vehicles = []

# Çizgi pozisyonları
pedestrian_line_position = 200
vehicle_line_position = 400
tolerance = 20

# Optical Flow parametreleri
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# İlk frame ve grey tanımlama
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)

# Yaya ve araç sınıflandırmak için boyut ve oran eşikleri
def classify_object(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    # Boyut ve oranlara göre sınıflandırma
    if area > 5000 and aspect_ratio > 1.2:
        return "Arac"
    elif area > 500:
        return "Yaya"
    return None

# İki nokta arasındaki mesafeyi hesapla
def distance_between_points(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Çerçeve boyutunu al
    height, width = frame.shape[:2]

    # Çizgi pozisyonlarını çiz
    cv2.line(frame, (0, pedestrian_line_position), (width, pedestrian_line_position), (0, 255, 0), 2)  # Yaya çizgisi
    cv2.line(frame, (0, vehicle_line_position), (width, vehicle_line_position), (255, 0, 0), 2)        # Araç çizgisi

    # Gri tonlama ve bulanıklaştırma
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Arkaplan çıkarma
    fg_mask = background_subtractor.apply(blurred)

    # Gölgeleri temizleme
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # Gürültü azaltma ve kontur birleştirme
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Kontur bulma
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hareketli nesnelerin hızlarını hesaplamak için verileri tutmak
    moving_objects = []

    # Tespit edilen nesnelerin işlenmesi
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Küçük gürültüyü yok say
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        classification = classify_object(contour)

        if classification == "Yaya":
            # Yayanın çizgiyi geçip geçmediğini kontrol et
            if abs(center_y - pedestrian_line_position) < tolerance:
                is_counted = False
                for cx, cy in counted_pedestrians:
                    if distance_between_points((center_x, center_y), (cx, cy)) < 50:
                        is_counted = True
                        break

                if not is_counted:
                    counted_pedestrians.append((center_x, center_y))
                    pedestrian_count += 1

        elif classification == "Arac":
            # Araç için çizgi kontrolü
            if abs(center_y - vehicle_line_position) < tolerance:
                is_counted = False
                for cx, cy in counted_vehicles:
                    if distance_between_points((center_x, center_y), (cx, cy)) < 50:
                        is_counted = True
                        break

                if not is_counted:
                    counted_vehicles.append((center_x, center_y))
                    vehicle_count += 1

        # Hareket eden nesneleri listeye ekle
        moving_objects.append((center_x, center_y, classification))

        # Çizim ve etiketleme
        color = (0, 255, 0) if classification == "Yaya" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, classification, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Optical Flow Hesaplama
    next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(moving_objects) > 0:
        p0 = np.array([[x, y] for x, y, _ in moving_objects], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

        # Hareket eden nesnelerin hızlarını çiz
        for i, (new, old) in enumerate(zip(p1, p0)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Hız hesaplama
            speed = np.sqrt((a - c) ** 2 + (b - d) ** 2) * 30 / cap.get(cv2.CAP_PROP_FPS)  # Piksel başına cm/s

            # Hızı ekrana yazdır (sadece hareket eden nesneler için)
            cv2.putText(frame, f"Speed: {speed:.2f} cm/s", (int(a), int(b) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Sayaçları ekranda gösterme
    cv2.putText(frame, f"Yaya: {pedestrian_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Tasit: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Görüntüyü gösterme
    cv2.imshow("Yaya ve Tasit Tespiti", frame)

    # Mevcut frame'i önceki frame olarak güncelle
    prev_gray = next_gray.copy()

    # Çıkış için 'q' tuşu
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
