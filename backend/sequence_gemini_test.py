from pathlib import Path
from collections import deque
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

HAND_MODEL = MODEL_DIR / "gesture_recognizer.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_full.task"


# =========================================================
# GEMINI
# =========================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-3.5-flash-lite"


# =========================================================
# MEDIAPIPE
# =========================================================

# ---------- HANDS ----------
hand_base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL)
)

hand_options = vision.GestureRecognizerOptions(
    base_options=hand_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

hand_recognizer = vision.GestureRecognizer.create_from_options(
    hand_options
)


# ---------- FACE ----------
face_base_options = python.BaseOptions(
    model_asset_path=str(FACE_MODEL)
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)


# ---------- POSE ----------
pose_base_options = python.BaseOptions(
    model_asset_path=str(POSE_MODEL)
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_detector = vision.PoseLandmarker.create_from_options(
    pose_options
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================================
# FRAME BUFFER
# =========================================================

# Keep approximately the last 1.5 seconds
frame_buffer = deque(maxlen=45)

timestamp_ms = 0

last_result = "Press SPACE to analyze movement"


# =========================================================
# GEMINI SEQUENCE ANALYSIS
# =========================================================

def analyze_sequence(frames, detection_summary):

    if len(frames) < 3:
        return "Not enough frames captured."

    print("\nSending sequence to Gemini...")
    print(f"Frames: {len(frames)}")

    prompt = f"""
You are an AI assistant helping interpret Pakistani Sign Language (PSL).

I am providing a sequence of camera frames in chronological order.

Analyze the ENTIRE sequence, not just one frame.

Pay special attention to:

1. Hand shape
2. One hand or two hands
3. Which hand moves
4. Direction of movement
5. Movement between the hands
6. Repeated movement
7. Hand position relative to face/body
8. Facial expression
9. Body/pose context
10. Changes between consecutive frames

The goal is to identify a POSSIBLE Pakistani Sign Language meaning.

Do NOT automatically assume that a common international gesture is PSL.

If the sequence is ambiguous, clearly say that it is ambiguous.

MediaPipe detection summary:
{detection_summary}

Return exactly this format:

Gesture:
Hands:
Movement:
Face:
Pose:
Possible PSL Meaning:
English:
Urdu:
Confidence:
Explanation:

Important:
- Confidence must be Low, Medium, or High.
- If you cannot confidently identify the PSL sign, say "Ambiguous / Not enough evidence".
- Do not claim that an interpretation is officially confirmed PSL.
- Explain what visual evidence led to your interpretation.
"""

    contents = [prompt]

    # Add frames in chronological order
    for frame in frames:

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        image_part = types.Part.from_bytes(
            data=encoded.tobytes(),
            mime_type="image/jpeg"
        )

        contents.append(image_part)

    # Retry if Gemini temporarily returns 503
    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            return response.text

        except Exception as e:

            print(f"Gemini error (attempt {attempt + 1}/3):")
            print(e)

            if attempt < 2:
             wait_time = 10 * (attempt + 1)
             print(f"Waiting {wait_time} seconds before retry...")
             time.sleep(wait_time)
            else:
                return "Gemini analysis failed."


# =========================================================
# MAIN LOOP
# =========================================================

print("=" * 60)
print("SIGNAVISION ULTIMATE - SEQUENCE GEMINI TEST")
print("=" * 60)
print()
print("Camera started.")
print("SPACE = Analyze the recent movement")
print("Q     = Quit")
print()


try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            continue

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # -------------------------------------------------
        # RAW FRAME
        # -------------------------------------------------

        raw_frame = frame.copy()

        # Add raw frame to buffer
        frame_buffer.append(raw_frame.copy())

        # -------------------------------------------------
        # MEDIAPIPE IMAGE
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            raw_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms += 33

        # -------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------

        hand_result = hand_recognizer.recognize_for_video(
            mp_image,
            timestamp_ms
        )

        hand_count = 0

        if hand_result.hand_landmarks:
            hand_count = len(hand_result.hand_landmarks)

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        face_result = face_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_count = 0

        if face_result.face_landmarks:
            face_count = len(face_result.face_landmarks)

        # -------------------------------------------------
        # POSE DETECTION
        # -------------------------------------------------

        pose_result = pose_detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_count = 0

        if pose_result.pose_landmarks:
            pose_count = len(pose_result.pose_landmarks)

        # -------------------------------------------------
        # DRAWING
        # -------------------------------------------------

        display_frame = raw_frame.copy()

        # ---------- HANDS ----------

        if hand_result.hand_landmarks:

            for hand_landmarks in hand_result.hand_landmarks:

                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * display_frame.shape[1]
                    )

                    y = int(
                        landmark.y * display_frame.shape[0]
                    )

                    cv2.circle(
                        display_frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

        # ---------- FACE ----------

        if face_result.face_landmarks:

            for face_landmarks in face_result.face_landmarks:

                # Draw selected face landmarks
                for landmark in face_landmarks[::20]:

                    x = int(
                        landmark.x * display_frame.shape[1]
                    )

                    y = int(
                        landmark.y * display_frame.shape[0]
                    )

                    cv2.circle(
                        display_frame,
                        (x, y),
                        2,
                        (255, 0, 255),
                        -1
                    )

        # ---------- POSE ----------

        if pose_result.pose_landmarks:

            for pose_landmarks in pose_result.pose_landmarks:

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * display_frame.shape[1]
                    )

                    y = int(
                        landmark.y * display_frame.shape[0]
                    )

                    cv2.circle(
                        display_frame,
                        (x, y),
                        3,
                        (255, 0, 0),
                        -1
                    )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        cv2.putText(
            display_frame,
            f"Hands: {hand_count}/2",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Face: {face_count}/1",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Pose: {pose_count}/1",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Buffered frames: {len(frame_buffer)}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "SPACE = Analyze | Q = Quit",
            (20, display_frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # -------------------------------------------------
        # SHOW CAMERA
        # -------------------------------------------------

        cv2.imshow(
            "SignaVision Ultimate - Sequence Recognition",
            display_frame
        )

        # -------------------------------------------------
        # KEYBOARD
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:

            print("\n" + "=" * 60)
            print("CAPTURING RECENT MOVEMENT")
            print("=" * 60)

            all_frames = list(frame_buffer)

            # Select 8 evenly spaced frames from the recent movement
            if len(all_frames) >= 8:
                indices = [
                    int(i * (len(all_frames) - 1) / 7)                        for i in range(8)
                ]
                sequence = [all_frames[i] for i in indices]
            else:
                sequence = all_frames
                        # Save frames for debugging
            for i, saved_frame in enumerate(sequence):

                filename = (
                    BASE_DIR /
                    f"sequence_{i + 1:02d}.jpg"
                )

                cv2.imwrite(
                    str(filename),
                    saved_frame
                )

            detection_summary = (
                f"Current frame: "
                f"Hands={hand_count}/2, "
                f"Face={face_count}/1, "
                f"Pose={pose_count}/1"
            )

            result = analyze_sequence(
                sequence,
                detection_summary
            )

            print("\n" + "=" * 60)
            print("GEMINI RESULT")
            print("=" * 60)
            print(result)
            print("=" * 60)
            print()

            last_result = "Analysis complete - check terminal"

        # Q
        elif key == ord("q"):
            break


finally:

    camera.release()
    cv2.destroyAllWindows()

    hand_recognizer.close()
    face_detector.close()
    pose_detector.close()

    print("\nCamera closed.")
    print("Sequence test finished.")