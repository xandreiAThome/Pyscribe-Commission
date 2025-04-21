import cv2
from google.genai import types
from google import genai
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
import io

# Load the environment variables from .env file
load_dotenv()

# constants
GEMINI_API_KEY = os.environ.get("GEMINI_KEY")
# MODEL_PATH = 'Pyscribe-Commission\\app\\yolov5mu.pt'
KEY_TRANSCRIBE = ord('t')  # 't' to trigger transcription
KEY_RETURN = ord('r')      # 'r' to return to live feed

# setup
# MODEL_PATH = hf_hub_download(local_dir=".",
#                              repo_id="armvectores/yolov8n_handwritten_text_detection",
#                              filename="best.pt")
# model = YOLO(MODEL_PATH)
gemini_vision_model = genai.Client(api_key=GEMINI_API_KEY)


cap = cv2.VideoCapture(0)
print("Press 't' to detect and transcribe text. Press 'q' to quit.")

annotated_frame = None
showing_annotated = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    display_frame = annotated_frame if showing_annotated and annotated_frame is not None else frame.copy()
    cv2.imshow("Live Feed", display_frame)

    key = cv2.waitKey(1)

    # Exit if window is closed
    if cv2.getWindowProperty("Live Feed", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed. Exiting...")
        break

    # Trigger transcription on button press
    if key == KEY_TRANSCRIBE and not showing_annotated:

        print("\n📸 Sending full frame to Gemini...")

        # Convert to PIL for Gemini
        full_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_bytes = io.BytesIO()
        full_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        try:
            response = gemini_vision_model.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
            types.Part.from_bytes(
                data=img_bytes.read(),
                mime_type='image/png',
            ),
            'Transcribe the text in this image, do not reply with unnecessary words, only the transcription'
            ]
        )
            text = response.text.strip()
            print(response.text.strip())
        except Exception as e:
            text = "[Error]"
            print(f"❌ Gemini API error: {e}")

        # Draw text overlay on frame
        annotated_frame = frame.copy()
        y0 = 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * 25
            cv2.putText(annotated_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        showing_annotated = True
        print("✅ Transcription complete. Press 'r' to return to live feed.")
    
    # return to live feed
    elif key == KEY_RETURN and showing_annotated:
        showing_annotated = False
        annotated_frame = None
        print("🔄 Returned to live feed.")

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()