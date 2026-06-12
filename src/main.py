import os
import pickle
import subprocess
import sys
import time
import warnings

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import simpledialog

from src.firebase_service import FirebaseService

warnings.filterwarnings("ignore", message=".*Given image is not CTkImage.*")


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


CASCADE_PATH = os.path.join(PROJECT_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(PROJECT_DIR, "lbph_classifier.yml")
PEOPLE_PATH = os.path.join(PROJECT_DIR, "face_names.pickle")
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 85.0
ACCESS_LOG_COOLDOWN_SECONDS = 5.0


def load_people_mapping(path):
    if not os.path.exists(path):
        return {}

    with open(path, "rb") as file:
        return pickle.load(file)


class GymAuthenticatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AuraGym Access Control")
        self.geometry("1125x700")
        self.minsize(980, 640)

        self.camera = None
        self.running = False
        self.frame_photo = None
        self._current_image = None
        self.last_authenticated_member = None
        self.last_access_log_at = 0.0
        self.firebase_service = FirebaseService(PROJECT_DIR)

        self.face_detector = cv2.CascadeClassifier(CASCADE_PATH)
        self.recognizer = self._load_recognizer()
        self.people_mapping = load_people_mapping(PEOPLE_PATH)
        self.id_to_name = {person_id: name for name, person_id in self.people_mapping.items()}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_recognizer(self):
        if not os.path.exists(MODEL_PATH):
            return None

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        return recognizer

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self, corner_radius=18)
        self.video_frame.grid(row=0, column=0, padx=(24, 12), pady=24, sticky="nsew")
        self.video_frame.grid_rowconfigure(1, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.video_frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(18, 10), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="AuraGym",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Access control powered by face recognition",
            text_color="#AAB3C5",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.preview_label = ctk.CTkLabel(
            self.video_frame,
            text="Camera not started",
            corner_radius=14,
            fg_color="#10131A",
        )
        self.preview_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        self.controls_frame = ctk.CTkFrame(self, corner_radius=18)
        self.controls_frame.grid(row=0, column=1, padx=(12, 24), pady=24, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.controls_frame,
            text="Entry Status",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(22, 8), sticky="w")

        self.status_badge = ctk.CTkLabel(
            self.controls_frame,
            text="Waiting for face",
            fg_color="#203040",
            corner_radius=12,
            height=40,
        )
        self.status_badge.grid(row=1, column=0, padx=20, pady=(8, 16), sticky="ew")

        self.name_value = ctk.CTkLabel(
            self.controls_frame,
            text="Member: -",
            anchor="w",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.name_value.grid(row=2, column=0, padx=20, pady=(8, 4), sticky="ew")

        self.confidence_value = ctk.CTkLabel(
            self.controls_frame,
            text="Confidence: -",
            anchor="w",
        )
        self.confidence_value.grid(row=3, column=0, padx=20, pady=4, sticky="ew")

        self.camera_value = ctk.CTkLabel(
            self.controls_frame,
            text="Camera: stopped",
            anchor="w",
            text_color="#AAB3C5",
        )
        self.camera_value.grid(row=4, column=0, padx=20, pady=(4, 16), sticky="ew")

        self.log_box = ctk.CTkTextbox(self.controls_frame, height=220, corner_radius=14)
        self.log_box.grid(row=5, column=0, padx=20, pady=(8, 16), sticky="nsew")
        self.log_box.insert("end", "System ready. Start the camera to authenticate members.\n")
        self.log_box.configure(state="disabled")

        button_row = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        button_row.grid(row=6, column=0, padx=20, pady=(4, 20), sticky="ew")
        button_row.grid_columnconfigure((0, 1), weight=1)

        self.start_button = ctk.CTkButton(button_row, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        self.stop_button = ctk.CTkButton(button_row, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        self.refresh_button = ctk.CTkButton(
            self.controls_frame,
            text="Refresh Models",
            command=self.reload_models,
            fg_color="#374151",
            hover_color="#4B5563",
        )
        self.refresh_button.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.add_member_button = ctk.CTkButton(
            self.controls_frame,
            text="Add Member",
            command=self.add_member,
            fg_color="#0F766E",
            hover_color="#115E59",
        )
        self.add_member_button.grid(row=8, column=0, padx=20, pady=(0, 20), sticky="ew")

        self._update_model_state_message()

    def _update_model_state_message(self):
        if self.recognizer is None:
            self._append_log("LBPH model not found. The interface will show the camera only.")
        if not os.path.exists(CASCADE_PATH):
            self._append_log("Face cascade not found. Face detection will be unavailable.")
        if not self.people_mapping:
            self._append_log("face_names.pickle not found. Member names will show as IDs.")

    def _append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _record_access_event(self, member_name, confidence, granted, member_id=None):
        if not self.firebase_service.is_ready:
            return

        success = self.firebase_service.record_access_event(member_name, confidence, granted, member_id=member_id)
        if success:
            self._append_log(f"Firebase updated for {member_name}.")
        else:
            self._append_log("Firebase database update skipped or failed.")

    def reload_models(self):
        self.recognizer = self._load_recognizer()
        self.people_mapping = load_people_mapping(PEOPLE_PATH)
        self.id_to_name = {person_id: name for name, person_id in self.people_mapping.items()}
        self._append_log("Models reloaded.")
        self._update_model_state_message()

    def add_member(self):
        if self.running:
            self.stop_camera()

        member_name = simpledialog.askstring("Add Member", "Enter the member RA/name:", parent=self)
        if not member_name:
            self._append_log("Add member cancelled.")
            return

        capture_script = os.path.join(PROJECT_DIR, "src", "captura_foto.py")
        train_script = os.path.join(PROJECT_DIR, "src", "train_recognizers.py")
        python_executable = sys.executable

        try:
            self._append_log(f"Capturing photos for {member_name}...")
            capture_result = subprocess.run(
                [python_executable, capture_script, "--name", member_name],
                cwd=PROJECT_DIR,
                check=False,
            )
            if capture_result.returncode != 0:
                self._append_log("Photo capture did not complete successfully.")
                return

            self._append_log("Training recognizers...")
            train_result = subprocess.run(
                [python_executable, train_script],
                cwd=PROJECT_DIR,
                check=False,
            )
            if train_result.returncode != 0:
                self._append_log("Training failed.")
                return

            self.reload_models()
            self._append_log(f"Member {member_name} added successfully.")
        except Exception as error:
            self._append_log(f"Failed to add member: {error}")

    def start_camera(self):
        if self.running:
            return

        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            self.camera = None
            self._set_status("Camera not available", "#991B1B")
            self._append_log("Unable to open the camera.")
            return

        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.camera_value.configure(text="Camera: running")
        self._set_status("Scanning faces", "#14532D")
        self._append_log("Camera started.")
        self._process_frame()

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self._current_image = None
        self.frame_photo = None
        self.preview_label.image = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.camera_value.configure(text="Camera: stopped")
        self.preview_label.configure(image=None, text="Camera stopped")
        self._set_status("Camera stopped", "#203040")
        self._append_log("Camera stopped.")

    def _set_status(self, text, color):
        self.status_badge.configure(text=text, fg_color=color)

    def _process_frame(self):
        if not self.running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self._append_log("Failed to read a frame from the camera.")
            self.after(25, self._process_frame)
            return

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        recognized_name = None
        confidence_value = None

        if not self.face_detector.empty():
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        else:
            faces = ()

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (80, 220, 120), 2)

            face_gray = gray[y:y + h, x:x + w]
            if face_gray.size == 0:
                continue

            face_gray = cv2.resize(face_gray, (90, 120))

            if self.recognizer is not None:
                label_id, confidence = self.recognizer.predict(face_gray)
                confidence_value = confidence
                recognized_name = self.id_to_name.get(label_id, f"ID {label_id}")
                color = (40, 180, 99) if CONFIDENCE_THRESHOLD <= confidence <= 105 else (220, 80, 80)
                status_text = "Access granted" if CONFIDENCE_THRESHOLD <= confidence <= 105 else "Access denied"
                self._set_status(status_text, "#14532D" if CONFIDENCE_THRESHOLD <= confidence <= 105 else "#991B1B")
                cv2.putText(
                    display_frame,
                    f"{recognized_name} | {confidence:.1f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            else:
                recognized_name = "Recognition model missing"
                self._set_status("Camera only", "#203040")

            break

        if recognized_name is None:
            self.name_value.configure(text="Member: -")
            self.confidence_value.configure(text="Confidence: -")
            self._set_status("Waiting for face", "#203040")
            self.last_authenticated_member = None
        else:
            self.name_value.configure(text=f"Member: {recognized_name}")
            if confidence_value is not None:
                self.confidence_value.configure(text=f"Confidence: {confidence_value:.1f}")
                access_granted = CONFIDENCE_THRESHOLD <= confidence_value <= 105
                now = time.monotonic()
                cooldown_active = now - self.last_access_log_at < ACCESS_LOG_COOLDOWN_SECONDS
                if access_granted and not cooldown_active:
                    self._append_log(f"{recognized_name} authenticated with confidence {confidence_value:.1f}.")
                    self._record_access_event(recognized_name, confidence_value, True, member_id=label_id)
                    self.last_authenticated_member = recognized_name
                    self.last_access_log_at = now
            else:
                self.confidence_value.configure(text="Confidence: -")

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self._current_image = Image.fromarray(rgb_frame)
        self._current_image.thumbnail((720, 540))
        self.frame_photo = ImageTk.PhotoImage(self._current_image)
        self.preview_label.configure(image=self.frame_photo, text="")
        self.preview_label.image = self.frame_photo

        self.after(15, self._process_frame)

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = GymAuthenticatorApp()
    app.mainloop()