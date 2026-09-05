import os
import cv2
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# PATHS + API
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

client = genai.Client(api_key=API_KEY)


# =========================================================
# PSL PROMPT
# =========================================================

PROMPT = """
You are an AI assistant helping interpret Pakistani Sign Language (PSL).

Analyze the sign shown in this camera image.

Consider:
- Both hands if visible
- Hand shape
- Finger positions
- Palm orientation
- Hand position
- Facial expression
- Body posture
- Possible movement/context visible in the image

Do NOT automatically assume that an ASL or internationally common gesture
has the same meaning in Pakistani Sign Language.

If the gesture is ambiguous, clearly say that it is ambiguous.

Return exactly:

Gesture:
Hands:
Possible PSL Meaning:
English:
Urdu:
Confidence:
Explanation:

Do not claim something is an official PSL definition unless there is reliable
evidence supporting that claim.
"""


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("==========================================")
print(" SignaVision Ultimate - Live Gemini Test")
print("==========================================")
print("SPACE = Analyze current gesture")
print("Q     = Quit")
print()


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Display instructions
    cv2.putText(
        frame,
        "SPACE = Analyze Gesture",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "SignaVision - Live Gemini",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # =====================================================
    # CAPTURE + SEND TO GEMINI
    # =====================================================

    if key == ord(" "):

        print("\nCapturing gesture...")

        # Save current frame
        image_path = BASE_DIR / "live_gesture.jpg"

        cv2.imwrite(
            str(image_path),
            frame
        )

        print("Image captured.")
        print("Sending to Gemini...")

        try:

            image_part = types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type="image/jpeg"
            )

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    image_part,
                    PROMPT
                ]
            )

            print("\n==========================================")
            print("        GEMINI PSL RESULT")
            print("==========================================\n")

            print(response.text)

            print("\n==========================================")

        except Exception as e:

            print("\nERROR communicating with Gemini:")
            print(e)

        print("\nCamera is ready again.")
        print("Make another gesture and press SPACE.")


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

print("\nLive Gemini test finished.")