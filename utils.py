# utils.py
import numpy as np

# Utility to compute Eye Aspect Ratio (EAR)-like value using landmarks (x,y)
# Input: eye_points - list/array of 6 (x,y) points in image coordinates (ordered)
# We'll compute vertical distances / horizontal distance similar to original EAR.
def eye_aspect_ratio(eye_points):
    # eye_points: np.array shape (6,2)
    A = np.linalg.norm(eye_points[1] - eye_points[5])  # p2 - p6
    B = np.linalg.norm(eye_points[2] - eye_points[4])  # p3 - p5
    C = np.linalg.norm(eye_points[0] - eye_points[3])  # p1 - p4 (horiz)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear
