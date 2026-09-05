import mediapipe as mp

print("MediaPipe version:", mp.__version__)

print("\nChecking MediaPipe Tasks API...")

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    print("✓ mediapipe.tasks.python is available")
    print("✓ mediapipe.tasks.python.vision is available")

except Exception as e:
    print("✗ Tasks API error:")
    print(e)

print("\nChecking old Solutions API...")

if hasattr(mp, "solutions"):
    print("✓ mp.solutions is available")
else:
    print("✗ mp.solutions is NOT available")