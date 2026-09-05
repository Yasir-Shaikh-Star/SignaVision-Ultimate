import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# LOAD API KEY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    exit()

client = genai.Client(api_key=API_KEY)


# =========================================================
# IMAGE
# =========================================================

image_path = BASE_DIR / "gesture_test.jpg"

if not image_path.exists():
    print("ERROR: gesture_test.jpg not found.")
    exit()


# =========================================================
# PSL PROMPT
# =========================================================

prompt = """
You are an AI assistant helping with Pakistani Sign Language (PSL).

Analyze the person's visible sign-language gesture in this image.

IMPORTANT:
- Consider BOTH hands if both are visible.
- Analyze hand shape, finger positions, palm orientation and hand position.
- Consider facial expression and body posture when relevant.
- Do not assume that a common international or ASL gesture is automatically PSL.
- If the sign is ambiguous, say so instead of inventing an answer.

Return the result in exactly this format:

Gesture:
Hands:
Possible PSL Meaning:
English:
Urdu:
Confidence:
Explanation:

If you cannot confidently identify a PSL sign, write:

Possible PSL Meaning: Unknown / Ambiguous

Do not claim that a result is an official PSL definition unless you have reliable evidence.
"""


# =========================================================
# PREPARE IMAGE
# =========================================================

image_part = types.Part.from_bytes(
    data=image_path.read_bytes(),
    mime_type="image/jpeg"
)


# =========================================================
# SEND IMAGE TO GEMINI
# =========================================================

print("Sending image to Gemini...")
print("Please wait...\n")


for attempt in range(1, 4):

    try:

        print(f"Attempt {attempt}/3...")

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                image_part,
                prompt
            ]
        )

        print("\n========== GEMINI PSL RESULT ==========\n")
        print(response.text)
        print("\n=======================================")

        break

    except Exception as e:

        print(f"\nAttempt {attempt} failed:")
        print(e)

        if attempt < 3:
            print("\nRetrying...\n")
        else:
            print("\nGemini is currently unavailable after 3 attempts.")