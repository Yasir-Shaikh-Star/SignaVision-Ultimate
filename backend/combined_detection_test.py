import os
import cv2
import mediapipe as mp

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
# API KEY
# =========================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

client = genai.Client(api_key=API_KEY)


# =========================================================
# CHECK MODELS
# =========================================================

for model in [HAND_MODEL, FACE_MODEL, POSE_MODEL]:

    if not model.exists():
        print("ERROR: Model not found:")
        print(model)
        exit()


# =========================================================
# HAND DETECTOR
# =========================================================

hand_options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

hand_recognizer = vision.GestureRecognizer.create_from_options(
    hand_options
)


# =========================================================
# FACE DETECTOR
# =========================================================

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(FACE_MODEL)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# POSE DETECTOR
# =========================================================

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# =========================================================
# GEMINI PROMPT
# =========================================================

PROMPT = """
You are assisting a Pakistani Sign Language (PSL) interpretation system.

Analyze the sign shown in the image.

Use all visible information:

1. Hand shape
2. Finger positions
3. Palm orientation
4. Hand position
5. One or two hands
6. Facial expression
7. Body posture
8. Visible signing context

Important:

Pakistani Sign Language is not necessarily identical to ASL or
international gestures.

Do not automatically label a gesture as PSL just because it is
common internationally.

If a static image is insufficient because the sign depends on movement,
direction, repetition, or context, clearly say that.

Return exactly:

Gesture:
Hands:
Face:
Pose:
Possible PSL Meaning:
English:
Urdu:
Confidence:
Explanation:

Use "Unknown / Ambiguous" when appropriate.

Do not claim that a result is an official PSL definition unless
there is reliable evidence supporting it.
"""


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")
    exit()


print()
print("==============================================")
print("       SIGNAVISION ULTIMATE")
print("       Combined AI Detection Test")
print("==============================================")
print()
print("MediaPipe:")
print("  Hands + Face + Pose")
print()
print("Gemini:")
print("  PSL interpretation")
print()
print("SPACE = Analyze current gesture")
print("Q     = Quit")
print()


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # =====================================================
    # CREATE MEDIAPIPE IMAGE
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================================
    # HAND DETECTION
    # =====================================================

    hand_result = hand_recognizer.recognize(mp_image)

    hand_count = len(
        hand_result.hand_landmarks
    )


    # Draw hands

    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]


    for hand_landmarks in hand_result.hand_landmarks:

        for landmark in hand_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        for start, end in hand_connections:

            x1 = int(
                hand_landmarks[start].x *
                frame.shape[1]
            )

            y1 = int(
                hand_landmarks[start].y *
                frame.shape[0]
            )

            x2 = int(
                hand_landmarks[end].x *
                frame.shape[1]
            )

            y2 = int(
                hand_landmarks[end].y *
                frame.shape[0]
            )

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

    face_result = face_landmarker.detect(
        mp_image
    )

    face_count = len(
        face_result.face_landmarks
    )


    for face_landmarks in face_result.face_landmarks:

        for landmark in face_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

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

    pose_result = pose_landmarker.detect(
        mp_image
    )

    pose_count = len(
        pose_result.pose_landmarks
    )


    for pose_landmarks in pose_result.pose_landmarks:

        for landmark in pose_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )


    # =====================================================
    # STATUS
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

    cv2.putText(
        frame,
        "SPACE = Analyze",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "SignaVision Ultimate - Combined",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # SEND CURRENT FRAME TO GEMINI
    # =====================================================

    if key == ord(" "):

        print()
        print("==============================================")
        print("Capturing gesture...")
        print("==============================================")

        image_path = BASE_DIR / "combined_gesture.jpg"

        cv2.imwrite(
            str(image_path),
            frame
        )

        print(f"Hands detected : {hand_count}/2")
        print(f"Face detected  : {face_count}/1")
        print(f"Pose detected  : {pose_count}/1")
        print()
        print("Sending image to Gemini...")


        try:

            image_part = types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type="image/jpeg"
            )


            # Add MediaPipe information to Gemini
            detection_info = f"""
MediaPipe detection information:

Hands detected: {hand_count} out of 2
Face detected: {face_count} out of 1
Pose detected: {pose_count} out of 1

Use this information together with the actual image.
"""


            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    image_part,
                    detection_info,
                    PROMPT
                ]
            )


            print()
            print("==============================================")
            print("             GEMINI PSL RESULT")
            print("==============================================")
            print()
            print(response.text)
            print()
            print("==============================================")
            print()
            print("Camera ready.")
            print("Make another gesture and press SPACE.")


        except Exception as e:

            print()
            print("ERROR communicating with Gemini:")
            print(e)


    # =====================================================
    # QUIT
    # =====================================================

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()
cv2.destroyAllWindows()

hand_recognizer.close()
face_landmarker.close()
pose_landmarker.close()

print()
print("SignaVision combined test finished.")