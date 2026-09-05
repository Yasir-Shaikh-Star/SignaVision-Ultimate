import cv2

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started successfully.")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Show camera feed
    cv2.imshow("SignaVision Ultimate - Camera Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()