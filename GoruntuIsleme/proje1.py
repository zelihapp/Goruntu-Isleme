import cv2
import numpy as np

# Video yolu
video_path = ("C:\\Users\\zelih\\OneDrive\\Belgeler\\OPENCV\\GoruntuIslemeOdev\\train.mp4")

# Video yakalama
cap = cv2.VideoCapture(video_path)

# Arkaplan çıkarma algoritması
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=True)

# Pedestrian crossing coordinates (adjusted to match video resolution)
pedestrian_areas = [
    np.array([(2, 308), (430, 54), (530, 96), (130, 380)], dtype=np.int32),   # Area 1
    np.array([(602, 73), (684, 24), (950, 125), (900, 220)], dtype=np.int32),  # Area 2
    np.array([(650, 560), (870, 232), (1020, 304), (900, 560)], dtype=np.int32),  # Area 3
    np.array([(0, 550), (120, 458), (273, 577), (42, 577)], dtype=np.int32)    # Area 4


    
]

# Create stop zones for vehicles (offset areas in front of pedestrian crossings)
stop_zones = [
    np.array([(2, 278), (430, 24), (530, 66), (130, 350)], dtype=np.int32),   # Area 1 Stop Zone
    np.array([(602, 43), (684, -6), (950, 95), (900, 190)], dtype=np.int32),  # Area 2 Stop Zone
    np.array([(650, 530), (870, 202), (1020, 274), (900, 530)], dtype=np.int32),  # Area 3 Stop Zone
    np.array([(0, 520), (120, 428), (273, 547), (42, 547)], dtype=np.int32)    # Area 4 Stop Zone
]

# Object classification based on size and aspect ratio
def classify_object(contour):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    if area > 5000 and aspect_ratio > 1.2:
        return "Arac"
    elif area > 500:
        return "Yaya"
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale and blurring
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Background subtraction
    fg_mask = background_subtractor.apply(blurred)
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # Noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw pedestrian areas and stop zones
    for area in pedestrian_areas:
        cv2.polylines(frame, [area], isClosed=True, color=(255, 255, 0), thickness=2)
    for stop_zone in stop_zones:
        cv2.polylines(frame, [stop_zone], isClosed=True, color=(0, 255, 255), thickness=2)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        classification = classify_object(contour)

        if classification == "Yaya":
            # Pedestrian center
            center_x, center_y = x + w // 2, y + h // 2
            inside_crosswalk = any(
                cv2.pointPolygonTest(area, (center_x, center_y), False) >= 0
                for area in pedestrian_areas
            )
            color = (0, 255, 0) if inside_crosswalk else (0, 0, 255)
            status = "Dogru Yolda" if inside_crosswalk else "Yanlis Yolda"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elif classification == "Arac":
            # Vehicle center
            center_x, center_y = x + w // 2, y + h // 2
            inside_stop_zone = any(
                cv2.pointPolygonTest(stop_zone, (center_x, center_y), False) >= 0
                for stop_zone in stop_zones
            )
            if inside_stop_zone:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Durmadi", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Arac", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Yaya ve Tasit Tespiti", frame)

    # Exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
