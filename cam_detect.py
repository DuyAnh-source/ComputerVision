import cv2
import math

W_real = 15.0  # cm
h_real = 20.0  # cm
focal_length = 600  # lấy từ bước hiệu chuẩn
real_diag = math.sqrt(W_real ** 2 + h_real ** 2)

# Load mô hình CNN
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Mở webcam (0 = camera mặc định)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Tiền xử lý ảnh đầu vào
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 1.0, (300, 300), 
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # Duyệt qua các detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            width = endX - startX
            height = endY - startY
            centreX = (startX + width // 2) // w
            centreY = (startY + height // 2) // h
            perceived_diag = math.sqrt(width ** 2 + height ** 2)

            distance = (real_diag * focal_length) / perceived_diag

            # Vẽ bounding box và hiển thị độ tin cậy
            text = f"{confidence:.2f} {distance:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Real-Time Face Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
