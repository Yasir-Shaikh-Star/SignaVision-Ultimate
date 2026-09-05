import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

HAND_MODEL = MODEL_DIR / "gesture_recognizer.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"


# =========================================================
# CHECK MODELS
# =========================================================

for model in [HAND_MODEL, FACE_MODEL, POSE_MODEL]:
    if not model.exists():
        print(f"ERROR: Model not found:")
        print(model)
        exit()


# =========================================================
# HAND LANDMARKER
# =========================================================

hand_base = python.BaseOptions(
    model_asset_path=str(HAND_MODEL)
)

hand_options = vision.GestureRecognizerOptions(
    base_options=hand_base,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

hand_recognizer = vision.GestureRecognizer.create_from_options(
    hand_options
)


# =========================================================
# FACE LANDMARKER
# =========================================================

face_base = python.BaseOptions(
    model_asset_path=str(FACE_MODEL)
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# POSE LANDMARKER
# =========================================================

pose_base = python.BaseOptions(
    model_asset_path=str(POSE_MODEL)
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Show your hands, face and upper body.")
print("Press Q to quit.")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================================
    # HAND DETECTION
    # =====================================================

    hand_result = hand_recognizer.recognize(mp_image)

    hand_count = len(hand_result.hand_landmarks)


    # Draw hand landmarks
    for hand_landmarks in hand_result.hand_landmarks:

        for landmark in hand_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        for start, end in connections:

            x1 = int(hand_landmarks[start].x * frame.shape[1])
            y1 = int(hand_landmarks[start].y * frame.shape[0])

            x2 = int(hand_landmarks[end].x * frame.shape[1])
            y2 = int(hand_landmarks[end].y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # =====================================================
    # FACE DETECTION
    # =====================================================

    face_result = face_landmarker.detect(mp_image)

    face_count = len(face_result.face_landmarks)


    # Draw face landmarks
    for face_landmarks in face_result.face_landmarks:

        for landmark in face_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                1,
                (255, 0, 255),
                -1
            )


    # =====================================================
    # POSE DETECTION
    # =====================================================

    pose_result = pose_landmarker.detect(mp_image)

    pose_count = len(pose_result.pose_landmarks)


    # Draw pose landmarks
    for pose_landmarks in pose_result.pose_landmarks:

        for landmark in pose_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )


    # =====================================================
    # INFORMATION
    # =====================================================

    cv2.putText(
        frame,
        f"Hands: {hand_count}/2",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Face: {face_count}/1",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pose: {pose_count}/1",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "SignaVision Ultimate - Full Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

hand_recognizer.close()
face_landmarker.close()
pose_landmarker.close()

print("Detection test finished.")