# realtime.py
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from utils import eye_aspect_ratio

# Settings (tune if needed)
EAR_THRESHOLD = 0.21      # below this -> eye considered closed (tweak per camera)
CONSEC_FRAMES = 20        # consecutive frames the eye must be below threshold to trigger
ALERT_COOLDOWN = 3.0      # seconds between voice alerts

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Text-to-speech engine for alerts
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# Landmarks indices for left and right eyes from Mediapipe (refine_landmarks True offers iris)
# We'll pick 6 points similar to EAR definition
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # approximate set
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # approximate set

def landmarks_to_array(landmarks, image_w, image_h):
    return np.array([(int(lm.x * image_w), int(lm.y * image_h)) for lm in landmarks])

def speak_alert():
    try:
        tts.say("Wake up! Driver drowsiness detected.")
        tts.runAndWait()
    except Exception as e:
        # fallback: just print
        print("ALERT: Wake up! (TTS failed)", e)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    closed_frames = 0
    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract eye coords
            left_pts = landmarks_to_array([face_landmarks[i] for i in LEFT_EYE_IDX], w, h)
            right_pts = landmarks_to_array([face_landmarks[i] for i in RIGHT_EYE_IDX], w, h)

            left_ear = eye_aspect_ratio(left_pts.astype(np.float32))
            right_ear = eye_aspect_ratio(right_pts.astype(np.float32))
            ear = (left_ear + right_ear) / 2.0

            # Draw eye polylines
            for p in left_pts:
                cv2.circle(frame, tuple(p), 1, (0,255,0), -1)
            for p in right_pts:
                cv2.circle(frame, tuple(p), 1, (0,255,0), -1)

            # Display EAR
            cv2.putText(frame, f"EAR: {ear:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Check drowsiness condition
            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            cv2.putText(frame, f"Closed_Frames: {closed_frames}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if closed_frames >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10,110),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 3)

                # audible alert with cooldown
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN:
                    # speak in non-blocking way: use small thread or pyttsx3 (blocking)
                    # here we call speak_alert (may block briefly)
                    speak_alert()
                    last_alert_time = now

                # (optional) draw a red rectangle around frame
                cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 6)

        else:
            # No face found: reset closed counters
            closed_frames = 0
            cv2.putText(frame, "Face not detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
