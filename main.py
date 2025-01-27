import cv2
import mediapipe as mp
import tkinter as tk
import random
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        # Создаем временное окно загрузки
        self.loading_window = tk.Toplevel(root)
        self.loading_window.title("Загрузка...")
        self.loading_label = ttk.Label(self.loading_window, text="Инициализация приложения...")
        self.loading_label.pack(padx=20, pady=20)
        self.loading_window.geometry("300x100")
        self.loading_window.grab_set()  # Блокируем взаимодействие с основным окном

        # Инициализация основных компонентов
        self.initialize_components()

        # Закрываем окно загрузки после завершения инициализации
        self.loading_window.destroy()

    def initialize_components(self):
        self.cap = cv2.VideoCapture(0)  # Webcam capture
        self.running = False
        self.last_frame = None
        self.last_detections = None
        self.captured_faces = []  # Store captured faces as images

        # Create UI elements
        # Video stream area
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Captured face area with scrollable view
        self.scroll_frame = ttk.Frame(self.root, width=100)  # Set width to 100 pixels
        self.scroll_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.canvas = tk.Canvas(self.scroll_frame, width=100)  # Set width to 100 pixels
        self.scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Create a frame for buttons to manage padding and layout
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=5)  # Add horizontal padding (padx=20)

        # Buttons arranged horizontally inside the button_frame
        self.start_button = ttk.Button(self.button_frame, text="Start Detection (S)", command=self.start_detection)
        self.start_button.pack(side="left", padx=10, pady=5)  # Add padding between buttons

        self.stop_button = ttk.Button(self.button_frame, text="Stop Detection (E)", command=self.stop_detection)
        self.stop_button.pack(side="left", padx=10, pady=5)  # Add padding between buttons

        self.capture_button = ttk.Button(self.button_frame, text="Capture Face (C)", command=self.capture_face, state=tk.DISABLED)
        self.capture_button.pack(side="left", padx=10, pady=5)  # Add padding between buttons

        self.clear_button = ttk.Button(self.button_frame, text="Clear Faces (W)", command=self.clear_captured_faces, state=tk.DISABLED)
        self.clear_button.pack(side="left", padx=10, pady=5)  # Add padding between buttons

        # Configure grid for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=4)  # Video area takes 4 parts
        self.root.grid_columnconfigure(1, weight=1)  # Scroll area takes 1 part

        # Bind keyboard shortcuts
        self.root.bind('<s>', lambda event: self.start_detection())      # Press 's' to start detection
        self.root.bind('<e>', lambda event: self.stop_detection())       # Press 'e' to stop detection
        self.root.bind('<c>', lambda event: self.capture_face())         # Press 'c' to capture face
        self.root.bind('<w>', lambda event: self.clear_captured_faces()) # Press 'w' to clear captured faces
        self.root.bind('<Escape>', lambda event: self.on_closing())      # Press 'Escape' to close the window

        # Start the video stream
        self.update_frame()
        self.root.after(100, lambda: messagebox.showinfo( "Welcome", "Добро пожаловать в программу Face Detection!", icon=messagebox.INFO, detail="Рекомендуемое расстояние использования: 5 метров." ))

    def start_detection(self):
        self.running = True
        self.capture_button.config(state=tk.NORMAL)

    def stop_detection(self):
        self.running = False
        self.capture_button.config(state=tk.DISABLED)
    
    def clear_captured_faces(self):
        # Clear the list of captured faces
        self.captured_faces = []

        # Remove all images from the interface
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()  # Remove each widget from scrollable_frame
        self.canvas.config(scrollregion=(0, 0, 0, 0))  # Reset the scroll region
        self.scrollable_frame.update_idletasks()  # Update the size after removing images

        # Deactivate the clear button
        self.clear_button.config(state=tk.DISABLED)

    def capture_face(self):
        """Capture and display one random face in the scrollable area."""
        if self.last_frame is not None and self.last_detections:
            faces = []
            ih, iw, _ = self.last_frame.shape
            for detection in self.last_detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin, ymin, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_crop = self.last_frame[ymin:ymin + h, xmin:xmin + w]
                faces.append(face_crop)

            # Choose a random face from the list
            if faces:
                random_face = random.choice(faces)

                if random_face.shape[0] > 0 and random_face.shape[1] > 0:
                    resized_face = cv2.resize(random_face, (100, 100))
                    face_image = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
                    self.captured_faces.append(face_image)

                    # Display in the scrollable area
                    face_imgtk = ImageTk.PhotoImage(image=face_image)
                    face_label = ttk.Label(self.scrollable_frame, image=face_imgtk)
                    face_label.image = face_imgtk
                    face_label.pack(pady=5)

            # Scroll to the bottom to show the newly added face
            self.scrollable_frame.update_idletasks()
            self.canvas.yview_moveto(1)  # 1 means scroll to the bottom
            self.clear_button.config(state=tk.NORMAL)

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame  # Save the last frame
            self.last_detections = None

            if self.running:
                result = face_detection.process(frame_rgb)
                if result.detections:
                    self.last_detections = result.detections
                    for detection in result.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        xmin, ymin, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 2)

            # Resize frame to fit video_label size
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            if label_width > 0 and label_height > 0:
                frame = cv2.resize(frame, (label_width, label_height))

            # Convert frame for Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# Run the application
root = tk.Tk()

# Set the minimum size of the window and allow resizing
root.minsize(800, 600)
root.geometry("1000x600")  # Default size
app = FaceApp(root)

# Handle window close event
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()