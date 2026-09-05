import os
from pathlib import Path
import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =========================================================
# LOAD API KEY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    exit()

# =========================================================
# GEMINI
# =========================================================

client = genai.Client(api_key=api_key)

# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Make a sign in front of the camera.")
print("Press SPACE to capture the gesture.")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("SignaVision - Gemini Test", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------------------------------
    # CAPTURE IMAGE
    # -----------------------------------------------------

    if key == 32:  # SPACE

        print("\nCapturing gesture...")

        # Save temporary image
        image_path = BASE_DIR / "gesture_test.jpg"

        cv2.imwrite(str(image_path), frame)

        print("Image captured.")
        print("Sending image to Gemini...")

        try:

            # Read image
            with open(image_path, "rb") as f:
                image_data = f.read()

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type="image/jpeg"
                    ),
                    """
                    Analyze this image for a sign language gesture.

                    Try to identify:
                    1. What hand gesture is being performed.
                    2. Whether one or two hands are involved.
                    3. The likely meaning.
                    4. Whether the gesture could be Pakistani Sign Language (PSL).

                    Be careful not to claim certainty if the image is
                    ambiguous.

                    Return a short answer with:
                    Gesture:
                    Meaning:
                    Language:
                    Confidence:
                    Explanation:
                    """
                ]
            )

            print("\n==============================")
            print("GEMINI RESULT")
            print("==============================")
            print(response.text)
            print("==============================\n")

        except Exception as e:
            print("\nGEMINI ERROR:")
            print(e)

    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()