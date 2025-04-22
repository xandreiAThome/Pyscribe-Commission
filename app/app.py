import cv2
from google.genai import types
from google import genai
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image, ImageTk
import numpy as np
import os
from dotenv import load_dotenv
import io
import tkinter as tk
from tkinter import ttk
import threading
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from tqdm import tqdm
import torch
import glob as glob
import matplotlib.pyplot as plt
import warnings
from thefuzz import fuzz

# Load the environment variables from .env file
load_dotenv()

# constants
GEMINI_API_KEY = os.environ.get("GEMINI_KEY")
# MODEL_PATH = 'Pyscribe-Commission\\app\\yolov5mu.pt'
KEY_TRANSCRIBE = ord('t')  # 't' to trigger transcription
KEY_RETURN = ord('r')      # 'r' to return to live feed
GROUND_WORDS = ["Amoxicillin", "Amoxicillin 500mg", "Amoxicillin 500m cap", "Cefalexin",
                "Cefalexin 500mg", "Cefalexin 500mg cap", "Cephalexin", "Cephalexin 500mg"]
# CHANGE TO LOCAL PATH
TROCR_PATH = "C:\\Users\\ellex\\OneDrive\\Documents\\Code Commisions\\Pyscribe-Commission\\app\\trocr_handwritten\\checkpoint-600"

# setup
# MODEL_PATH = hf_hub_download(local_dir=".",
#                              repo_id="armvectores/yolov8n_handwritten_text_detection",
#                              filename="best.pt")
# model = YOLO(MODEL_PATH)
gemini_vision_model = genai.Client(api_key=GEMINI_API_KEY)
warnings.filterwarnings('ignore')
device = torch.device('cpu')

trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH).to(device)


class TextDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pyscribe")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # video capture
        self.cap = cv2.VideoCapture(0)
        self.annotated_frame = None
        self.captured_frame = None
        self.showing_annotated = False

        # UI
        self.is_processing = False
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        button_frame = ttk.Frame(root)
        button_frame.pack(fill=tk.X, pady=10)

        self.transcribe_btn = ttk.Button(button_frame, text="Transcribe", command=self.transcribe_text)
        self.transcribe_btn.pack(side=tk.LEFT, padx=10)

        self.back_btn = ttk.Button(button_frame, text="Back to Live Feed", command=self.back_to_live_feed)
        self.back_btn.pack(side=tk.LEFT, padx=10)

        # Method Selector
        self.method_var = tk.StringVar(value="gemini")  # Default is Gemini
        method_frame = ttk.Frame(root)
        method_frame.pack(pady=5)

        ttk.Label(method_frame, text="Transcription Method:").pack(side=tk.LEFT)
        method_dropdown = ttk.OptionMenu(method_frame, self.method_var, "gemini", "gemini", "trocr")
        method_dropdown.pack(side=tk.LEFT, padx=5)
        # hide initially
        self.back_btn.pack_forget()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # show annotated frame
        if self.showing_annotated and self.annotated_frame is not None:
            frame_to_display = self.annotated_frame
        elif self.is_processing and self.captured_frame is not None:
            frame_to_display = self.captured_frame
            cv2.putText(frame_to_display, "Processing...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            frame_to_display = frame

        # convert to rgb and display to tkinter
        cv_img = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0,0, anchor=tk.NW, image=imgtk)

        self.root.after(10, self.update_frame)

    def transcribe_text(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.transcribe_btn.config(state=tk.DISABLED)     # Disable transcribe
        self.back_btn.pack(side=tk.LEFT, padx=10)         # Show back to live
        
        print("Transcribing...")
        self.is_processing = True
        self.captured_frame = frame.copy()

        # Convert to PIL for Gemini
        full_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_bytes = io.BytesIO()
        full_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        def run_transcription():
            selected_method = self.method_var.get()
            text = ""

            if selected_method == "gemini":
                try:
                    response = gemini_vision_model.models.generate_content(
                    model='gemini-2.5-flash-preview-04-17',
                    contents=[
                    types.Part.from_bytes(
                        data=img_bytes.read(),
                        mime_type='image/png',
                    ),
                    'Transcribe the text in this image, do not reply with unnecessary words, only the transcription'
                        ]
                    )
                    if response.text is None:
                        text = "No text detected"
                    else:
                        text = response.text.strip()
                    print("🧠 Transcribed Text:\n", text)
                except Exception as e:
                    text = "[Error]"
                    print(f"❌ Gemini API error: {e}")
            elif selected_method == "trocr":
                try:
                    # Convert frame to RGB and resize
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = pil_image.convert("RGB")
                    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
                    generated_ids = trocr_model.generate(pixel_values)
                    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    match = self.fuzzy_match(text)
                    if match:
                        print(f"Did you mean {match}")
                    else:
                        print("No similar word")

                    print("🧠 TrOCR Transcribed Text:\n", text)
                except Exception as e:
                    text = "[TrOCR Error]"
                    print("❌ TrOCR Error:", e)

            annotated = frame.copy()
            y0 = 30
            y = 0

            for i, line in enumerate(text.split('\n')):
                y = y0 + i  * 25
                cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2, cv2.LINE_AA)
                
            if selected_method == "trocr" and match:
                cv2.putText(annotated, f"Similar to {match}", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2, cv2.LINE_AA)
            elif selected_method == "trocr" and match is None: 
                 cv2.putText(annotated, "No Similar Word", (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2, cv2.LINE_AA)
                
            self.annotated_frame = annotated
            self.showing_annotated = True
            self.is_processing = False

        threading.Thread(target=run_transcription, daemon=True).start()

    def back_to_live_feed(self):
        self.showing_annotated = False
        self.annotated_frame = None
        self.captured_frame = None
        self.transcribe_btn.config(state=tk.NORMAL)
        self.back_btn.pack_forget()

    def on_close(self):
        print("Closing app")
        self.cap.release()
        self.root.destroy()

    def fuzzy_match(self, str):
        match = ""
        score = 0
        for word in GROUND_WORDS:
            score_temp = fuzz.ratio(str, word)
            if score_temp > score:
                match = word
                score = score_temp

        if score > 50:
            return match
        return None
                
if __name__ == "__main__":
    root = tk.Tk()
    app = TextDetectionApp(root)
    root.mainloop()