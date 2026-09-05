import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Get the SignaVision-Ultimate project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root
env_file = BASE_DIR / ".env"
load_dotenv(env_file)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found.")
    print("Expected .env location:")
    print(env_file)
    exit()

print("API key found successfully.")
print("Connecting to Gemini...")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly: SignaVision API works!"
    )

    print("\nGemini response:")
    print(response.text)

except Exception as e:
    print("\nERROR:")
    print(e)