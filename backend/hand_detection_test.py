import cv2
import mediapipe as mp
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gesture_recognizer.task"


# =========================================================
# CHECK MODEL
# =========================================================

if not MODEL_PATH.exists():
    print("ERROR: gesture_recognizer.task not found.")
    print("Expected:")
    print(MODEL_PATH)
    exit()

print("Model found:")
print(MODEL_PATH)


# =========================================================
# MEDIAPIPE GESTURE RECOGNIZER
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

recognizer = vision.GestureRecognizer.create_from_options(options)

print("MediaPipe Gesture Recognizer started.")
print("Maximum hands: 2")
print("Press Q to quit.")


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Convert OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect gestures
    result = recognizer.recognize(mp_image)

    # -----------------------------------------------------
    # DRAW DETECTED HANDS
    # -----------------------------------------------------

    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

        # Get handedness
        if hand_index < len(result.handedness):
            handedness = result.handedness[hand_index][0]

            hand_name = handedness.display_name
            hand_score = handedness.score

        else:
            hand_name = "Unknown"
            hand_score = 0.0

        # Draw landmarks
        for landmark in hand_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Display hand information
        first_point = hand_landmarks[0]

        text_x = int(first_point.x * frame.shape[1])
        text_y = int(first_point.y * frame.shape[0]) - 20

        cv2.putText(
            frame,
            f"{hand_name} ({hand_score:.2f})",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # -------------------------------------------------
        # DISPLAY GESTURE
        # -------------------------------------------------

        if hand_index < len(result.gestures):

            gestures = result.gestures[hand_index]

            if gestures:

                gesture = gestures[0]

                gesture_name = gesture.category_name
                gesture_score = gesture.score

                cv2.putText(
                    frame,
                    f"Gesture: {gesture_name} ({gesture_score:.2f})",
                    (20, 40 + hand_index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


    # -----------------------------------------------------
    # HAND COUNT
    # -----------------------------------------------------

    hand_count = len(result.hand_landmarks)

    cv2.putText(
        frame,
        f"Hands detected: {hand_count}/2",
        (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # SHOW CAMERA
    # -----------------------------------------------------

    cv2.imshow(
        "SignaVision Ultimate - Two Hand Test",
        frame
    )


    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()
cv2.destroyAllWindows()
recognizer.close()

print("Camera closed.")