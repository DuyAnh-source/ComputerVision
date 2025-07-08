import cv2

# Load mô hình MobileNet SSD huấn luyện sẵn
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "mobilenet_iter_73000.caffemodel")

# Mở webcam hoặc video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 scalefactor=0.007843, size=(300, 300),
                                 mean=127.5)

    net.setInput(blob)
    detections = net.forward()

    person_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        
        # Class ID 15 là "person" trong mô hình này
        if class_id == 15 and confidence > 0.6:
            person_detected = True
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print("Có người" if person_detected else "Không có người")
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
