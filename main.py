import cv2
import tkinter as tk
from mediapipe.python.solutions.face_detection import FaceDetection
from random import choice as random_choice
from threading import Thread
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from os import path as os_path, system as os_system


class FaceApp:
    # Initialize the applications
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.root.geometry("1000x600")
        self.running = False
        self.last_frame = None
        self.last_detections = None
        self.captured_faces = []  # Store captured faces as images
        self.show_loading_screen()  # Show the loading screen while initializing the application

    def show_loading_screen(self):
        # Show the loading screen
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.geometry("400x200")
        self.loading_window.title("Загрузка...")
        label = ttk.Label(self.loading_window, text="Инициализация... Пожалуйста подождите.", font=("Arial", 14))
        label.pack(pady=20)

        # Add progress bar
        self.progressbar = ttk.Progressbar(self.loading_window, mode='indeterminate')
        self.progressbar.pack(expand=True, fill='x', padx=20, pady=20)
        self.progressbar.start()

        self.root.withdraw()  # Hide the main window
        Thread(target=self.initialize_app, daemon=True).start()  # Start the initialization in a separate thread

    def initialize_app(self):
        # Initialize the application
        self.cap = cv2.VideoCapture(0)
        self.face_detection = FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.create_ui()
        self.update_frame()

        # Hide the loading screen and show the main window
        self.loading_window.after(0, self.progressbar.stop)
        self.loading_window.after(0, self.loading_window.destroy)
        self.root.after(0, self.root.deiconify)
        messagebox.showinfo("Welcome", "Добро пожаловать в программу Face Detection!", icon=messagebox.INFO, detail="Рекомендуемое расстояние использования: 5 метров.")

    def create_ui(self):
        # Add menu
        self.menu_bar = tk.Menu(self.root)
        self.menu_bar.add_cascade(label="Помощь", command=self.show_help)
        self.menu_bar.add_cascade(label="Экспорт", command=self.download_images)
        self.root.config(menu=self.menu_bar)

        # Add video label
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Add scroll frame
        self.scroll_frame = ttk.Frame(self.root, width=100)
        self.scroll_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Add scrollable frame
        self.canvas = tk.Canvas(self.scroll_frame, width=100)
        self.scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Add canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the scrollable frame
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Add buttons
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=5)

        self.start_button = ttk.Button(self.button_frame, text="Start Detection (S)", command=self.start_detection)
        self.start_button.pack(side="left", padx=10, pady=5)

        self.stop_button = ttk.Button(self.button_frame, text="Stop Detection (E)", command=self.stop_detection)
        self.stop_button.pack(side="left", padx=10, pady=5)

        self.capture_button = ttk.Button(self.button_frame, text="Capture Face (C)", command=self.capture_face, state=tk.DISABLED)
        self.capture_button.pack(side="left", padx=10, pady=5)

        self.clear_button = ttk.Button(self.button_frame, text="Clear Faces (W)", command=self.clear_captured_faces, state=tk.DISABLED)
        self.clear_button.pack(side="left", padx=10, pady=5)

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=4)
        self.root.grid_columnconfigure(1, weight=1)

        # Bind keyboard shortcuts
        self.root.bind('<s>', lambda event: self.start_detection())
        self.root.bind('<e>', lambda event: self.stop_detection())
        self.root.bind('<c>', lambda event: self.capture_face())
        self.root.bind('<w>', lambda event: self.clear_captured_faces())
        self.root.bind('<Escape>', lambda event: self.on_closing())

    # Button functions
    def start_detection(self):
        self.running = True
        self.capture_button.config(state=tk.NORMAL)

    def stop_detection(self):
        self.running = False
        self.capture_button.config(state=tk.DISABLED)

    def clear_captured_faces(self):
        self.captured_faces = []

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.canvas.config(scrollregion=(0, 0, 0, 0))
        self.scrollable_frame.update_idletasks()

        self.clear_button.config(state=tk.DISABLED)

    def capture_face(self):
        if self.last_frame is not None and self.last_detections:
            faces = []
            ih, iw, _ = self.last_frame.shape
            for detection in self.last_detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin, ymin, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_crop = self.last_frame[ymin:ymin + h, xmin:xmin + w]
                faces.append(face_crop)

            if faces:
                random_face = random_choice(faces)

                if random_face.shape[0] > 0 and random_face.shape[1] > 0:
                    resized_face = cv2.resize(random_face, (100, 100))
                    face_image = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
                    self.captured_faces.append(face_image)

                    face_imgtk = ImageTk.PhotoImage(image=face_image)
                    face_label = ttk.Label(self.scrollable_frame, image=face_imgtk)
                    face_label.image = face_imgtk
                    face_label.pack(pady=5)

            self.scrollable_frame.update_idletasks()
            self.canvas.yview_moveto(1)
            self.clear_button.config(state=tk.NORMAL)

    # Video functions
    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame
            self.last_detections = None

            if self.running:
                result = self.face_detection.process(frame_rgb)
                if result.detections:
                    self.last_detections = result.detections
                    for detection in result.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        xmin, ymin, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 2)

            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            if label_width > 0 and label_height > 0:
                frame = cv2.resize(frame, (label_width, label_height))

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # Menu functions
    def download_images(self):
        if not self.captured_faces:
            messagebox.showinfo("Нет изображений", "Список изображений пуст.")
            return

        folder = filedialog.askdirectory(title="Выберите папку для сохранения изображений")
        if not folder:
            return

        for idx, face_image in enumerate(self.captured_faces):
            file_path = f"{folder}/face_{idx + 1}.png"
            face_image.save(file_path, "PNG")

        messagebox.showinfo("Сохранено", f"Изображения успешно сохранены в папке: {folder}")

    def show_help(self):
        help_file = "help.html"
        if os_path.exists(help_file):
            try:
                os_system(f"hh.exe {help_file}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл справки: {e}")
        else:
            messagebox.showerror("Ошибка", f"Файл справки {help_file} не найден.")

    # Window closing
    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# Run the application
root = tk.Tk()
root.minsize(800, 600)
app = FaceApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
