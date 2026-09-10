from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import os
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# FLASK
# =========================================================

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env")


# =========================================================
# GEMINI
# =========================================================

GEMINI_MODEL = "gemini-1.5-flash"

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# =========================================================
# MODEL PATHS
# =========================================================

GESTURE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "gesture_recognizer.task"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)

POSE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_full.task"
)


# =========================================================
# CHECK MODEL FILES
# =========================================================

for model_path in [
    GESTURE_MODEL,
    FACE_MODEL,
    POSE_MODEL
]:

    if not os.path.exists(model_path):

        print(
            "WARNING: Model file not found:",
            model_path
        )


# =========================================================
# MEDIAPIPE - HANDS
# =========================================================

gesture_base_options = python.BaseOptions(
    model_asset_path=GESTURE_MODEL
)

gesture_options = vision.GestureRecognizerOptions(
    base_options=gesture_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

gesture_recognizer = (
    vision.GestureRecognizer.create_from_options(
        gesture_options
    )
)


# =========================================================
# MEDIAPIPE - FACE
# =========================================================

face_base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_landmarker = (
    vision.FaceLandmarker.create_from_options(
        face_options
    )
)


# =========================================================
# MEDIAPIPE - POSE
# =========================================================

pose_base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = (
    vision.PoseLandmarker.create_from_options(
        pose_options
    )
)


# =========================================================
# THREADING
# =========================================================

lock = threading.Lock()


# =========================================================
# DETECTION STATUS
# =========================================================

latest_detection = {
    "hands": 0,
    "face": 0,
    "pose": 0
}


# =========================================================
# LIVE FRAME BUFFER
# =========================================================

frame_buffer = deque(
    maxlen=45
)


# =========================================================
# LAST AI RESULT
# =========================================================

latest_ai_result = {
    "gesture": "—",
    "hands": "—",
    "movement": "—",
    "face": "—",
    "pose": "—",
    "possible_psl_meaning": "—",
    "english": "—",
    "urdu": "—",
    "confidence": "—",
    "explanation":
        "The AI explanation will appear here after gesture analysis."
}


# =========================================================
# MEDIA PIPE TIMESTAMP
# =========================================================

timestamp_ms = 0


def get_timestamp():

    global timestamp_ms

    current_time = int(
        time.monotonic() * 1000
    )

    with lock:

        timestamp_ms = max(
            timestamp_ms + 1,
            current_time
        )

        return timestamp_ms


# =========================================================
# DRAW HANDS
# =========================================================

def draw_hands(frame, result):

    if not result.hand_landmarks:
        return

    for landmarks in result.hand_landmarks:

        for landmark in landmarks:

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
                (80, 220, 180),
                -1
            )

        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (0, 17)
        ]

        for start, end in connections:

            x1 = int(
                landmarks[start].x *
                frame.shape[1]
            )

            y1 = int(
                landmarks[start].y *
                frame.shape[0]
            )

            x2 = int(
                landmarks[end].x *
                frame.shape[1]
            )

            y2 = int(
                landmarks[end].y *
                frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (80, 220, 180),
                2
            )


# =========================================================
# DRAW FACE
# =========================================================

def draw_face(frame, result):

    if not result.face_landmarks:
        return

    for face in result.face_landmarks:

        for i, landmark in enumerate(face):

            if i % 5 != 0:
                continue

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (210, 120, 220),
                -1
            )


# =========================================================
# DRAW POSE
# =========================================================

def draw_pose(frame, result):

    if not result.pose_landmarks:
        return

    for pose in result.pose_landmarks:

        for landmark in pose:

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
                (100, 160, 230),
                -1
            )


# =========================================================
# PROCESS BROWSER FRAME
# =========================================================

def process_frame(frame):

    raw_frame = frame.copy()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    current_timestamp = get_timestamp()

    hand_result = (
        gesture_recognizer.recognize_for_video(
            mp_image,
            current_timestamp
        )
    )

    face_result = (
        face_landmarker.detect_for_video(
            mp_image,
            current_timestamp
        )
    )

    pose_result = (
        pose_landmarker.detect_for_video(
            mp_image,
            current_timestamp
        )
    )

    hands = len(
        hand_result.hand_landmarks
    )

    face = (
        1
        if face_result.face_landmarks
        else 0
    )

    pose = (
        1
        if pose_result.pose_landmarks
        else 0
    )

    with lock:

        latest_detection["hands"] = hands
        latest_detection["face"] = face
        latest_detection["pose"] = pose

        frame_buffer.append(
            raw_frame
        )

    draw_hands(
        frame,
        hand_result
    )

    draw_face(
        frame,
        face_result
    )

    draw_pose(
        frame,
        pose_result
    )

    cv2.putText(
        frame,
        f"HANDS   {hands}/2",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 230, 220),
        2
    )

    cv2.putText(
        frame,
        f"FACE    {face}/1",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (210, 170, 230),
        2
    )

    cv2.putText(
        frame,
        f"POSE    {pose}/1",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 190, 230),
        2
    )

    return frame


# =========================================================
# RECEIVE CAMERA FRAME FROM BROWSER
# =========================================================

@app.route(
    "/api/frame",
    methods=["POST"]
)
def receive_frame():

    if "frame" not in request.files:

        return jsonify({
            "success": False,
            "error": "No camera frame received."
        }), 400

    try:

        file = request.files["frame"]

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Empty camera frame."
            }), 400

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Could not decode camera frame."
            }), 400

        process_frame(frame)

        with lock:

            detection = {
                "hands": latest_detection["hands"],
                "face": latest_detection["face"],
                "pose": latest_detection["pose"]
            }

        return jsonify({
            "success": True,
            "detection": detection
        })

    except Exception as e:

        print(
            "Frame processing error:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# GEMINI PROMPT
# =========================================================

GEMINI_PROMPT = """
You are an AI assistant helping interpret Pakistani Sign Language (PSL).

You are receiving a short sequence of camera frames showing a person
possibly performing a sign.

Analyze the ENTIRE sequence chronologically.

Pay attention to:

- One or both hands
- Hand shape
- Finger positions
- Palm direction
- Hand location
- Movement
- Relative movement of both hands
- Facial expression
- Head movement
- Body/pose
- Context visible in the sequence

IMPORTANT:

Do NOT assume that every gesture has a confirmed PSL meaning.

If the sign is ambiguous, say so.

If the exact PSL sign cannot be determined, provide the most likely
possible interpretation and explain the uncertainty.

Do not invent an official PSL meaning.

Return ONLY this format:

Gesture: ...
Hands: ...
Movement: ...
Face: ...
Pose: ...
Possible PSL Meaning: ...
English: ...
Urdu: ...
Confidence: ...
Explanation: ...

Confidence should be:
High
Medium-High
Medium
Low

For Possible PSL Meaning, use wording such as:
"Possible..."
"Likely..."
"Could mean..."
when the sign is uncertain.

The goal is to help a user understand a possible PSL interpretation,
not to claim that the AI is an official PSL authority.
"""


# =========================================================
# PARSE GEMINI RESPONSE
# =========================================================

def parse_ai_response(text):

    result = {
        "gesture": "—",
        "hands": "—",
        "movement": "—",
        "face": "—",
        "pose": "—",
        "possible_psl_meaning": "—",
        "english": "—",
        "urdu": "—",
        "confidence": "—",
        "explanation": "—"
    }

    field_map = {

        "Gesture": "gesture",
        "Hands": "hands",
        "Movement": "movement",
        "Face": "face",
        "Pose": "pose",
        "Possible PSL Meaning": "possible_psl_meaning",
        "English": "english",
        "Urdu": "urdu",
        "Confidence": "confidence",
        "Explanation": "explanation"
    }

    current_field = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        matched = False

        for label, key in field_map.items():

            prefix = label + ":"

            if line.lower().startswith(
                prefix.lower()
            ):

                value = line[
                    len(prefix):
                ].strip()

                result[key] = value

                current_field = key

                matched = True

                break

        if not matched and current_field:

            result[current_field] += (
                " " + line
            )

    return result


# =========================================================
# ANALYZE GESTURE WITH GEMINI
# =========================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze_gesture():

    global latest_ai_result

    if not gemini_client:

        return jsonify({
            "success": False,
            "error": "Gemini API key is not configured."
        }), 500

    with lock:

        all_frames = list(
            frame_buffer
        )

    if len(all_frames) < 3:

        return jsonify({
            "success": False,
            "error":
                "Not enough camera frames yet. "
                "Please wait a moment and try again."
        }), 400

    if len(all_frames) >= 8:

        indices = [
            int(
                i *
                (len(all_frames) - 1) /
                7
            )
            for i in range(8)
        ]

        sequence = [
            all_frames[i]
            for i in indices
        ]

    else:

        sequence = all_frames

    image_parts = []

    for frame in sequence:

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        if not success:
            continue

        image_parts.append(
            types.Part.from_bytes(
                data=encoded.tobytes(),
                mime_type="image/jpeg"
            )
        )

    if not image_parts:

        return jsonify({
            "success": False,
            "error": "Could not prepare camera frames."
        }), 500

    with lock:

        detection_info = (
            f"Hands detected: {latest_detection['hands']}/2\n"
            f"Face detected: {latest_detection['face']}/1\n"
            f"Pose detected: {latest_detection['pose']}/1"
        )

    prompt = (
        GEMINI_PROMPT
        + "\n\n"
        + "MediaPipe detection information:\n"
        + detection_info
    )

    contents = (
        image_parts
        + [
            types.Part.from_text(
                text=prompt
            )
        ]
    )

    # Exponential backoff retry loop for 503 errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            text = response.text.strip()
            parsed = parse_ai_response(text)
            latest_ai_result = parsed

            return jsonify({
                "success": True,
                "result": parsed
            })

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            print("Gemini analysis error:", e)
            return jsonify({
                "success": False,
                "error": "Gemini API is temporarily overloaded. Please try again in a few seconds."
            }), 503


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# DETECTION API
# =========================================================

@app.route("/api/detection")
def detection():

    with lock:

        return jsonify(
            latest_detection
        )


# =========================================================
# AI RESULT API
# =========================================================

@app.route("/api/result")
def ai_result():

    with lock:

        return jsonify(
            latest_ai_result
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "system": "SignaVision Ultimate"
    })


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("             SIGNAVISION ULTIMATE")
    print("             AI SIGN LANGUAGE SYSTEM")
    print("=" * 60)

    print("Camera: Browser camera mode")
    print("MediaPipe: Hands + Face + Pose")
    print("Gemini:", GEMINI_MODEL)
    print("Server: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )