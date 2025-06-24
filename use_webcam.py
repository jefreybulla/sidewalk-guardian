import cv2

cap = cv2.VideoCapture(0)              # 0 = /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("C960 preview", frame)
    if cv2.waitKey(1) & 0xFF == 27:    # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
