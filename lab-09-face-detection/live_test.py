import cv2 as cv
import dlib
from time import perf_counter

PREDICTOR_PATH = 'data/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

WINDOW = 'Live face detection  —  press Q to quit'
cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
cv.resizeWindow(WINDOW, 1280, 720)

prev_time = perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror for natural selfie view
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # Draw face bounding box
        cv.rectangle(
            frame,
            (face.left(), face.top()),
            (face.right(), face.bottom()),
            (0, 255, 0), 2,
        )

        # Draw 68 landmark points
        landmarks = predictor(gray, face)
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # FPS overlay
    now = perf_counter()
    fps = 1.0 / (now - prev_time)
    prev_time = now
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow('Live face detection  —  press Q to quit', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
