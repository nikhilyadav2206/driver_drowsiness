# record_sample.py
# Run to record frames and save images for open-eye / closed-eye dataset.
import cv2
import os
import time

os.makedirs("dataset/open", exist_ok=True)
os.makedirs("dataset/closed", exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press 'o' to save open eye, 'c' to save closed eye, 'q' to quit")

counter_open = len(os.listdir("dataset/open"))
counter_closed = len(os.listdir("dataset/closed"))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("record", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('o'):
        fname = f"dataset/open/open_{counter_open:04d}.jpg"
        cv2.imwrite(fname, frame)
        counter_open += 1
        print("Saved", fname)
    elif k == ord('c'):
        fname = f"dataset/closed/closed_{counter_closed:04d}.jpg"
        cv2.imwrite(fname, frame)
        counter_closed += 1
        print("Saved", fname)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
