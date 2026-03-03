"""GUI for preference reward model for Obstacle Car Racing environment (with multiple safety justifications).

Shows paired-clip videos and asks:
1) Safety justification's name (e.g. Default/Grass/Chuckhole/Car)
2) Preference (Left/Equally/Right)

Notes:
    Controlled by the obstacle mode at the top.
"""

import tkinter as tk
from tkinter import messagebox
import cv2
import imageio
import threading
import time
import os
import re
import pickle
from PIL import Image, ImageTk
from datetime import datetime


# choose chuckholes (chuckc); or chuckholes + cars (chuckccar)
OBST_MODE = 'chuckccar' # options: 'chuckc', 'chuckccar'
if OBST_MODE == 'chuckc':
    obst = '_chuckcobst'
elif OBST_MODE == 'chuckccar':
    obst = '_chuckccarobst'
else:
    obst = ''

user = '45'  # Set user ID here

models_dir = os.path.join(os.getcwd(), 'models', 'obstcarracing')
data_dir = os.path.join(os.getcwd(), 'data', 'obstcarracing')
video_dir = os.path.join(data_dir, f'queries_s20{obst}')
responses_dir = os.path.join(data_dir, f'responses_s20{obst}')
os.makedirs(responses_dir, exist_ok=True)

class CarRacingGUI:
    """Tk GUI for obstacle-mode queries with multiple justifications.

    Loads .mp4 query clips, plays them in a loop, and records per-clip
    justification's name plus the preference, along with repeats/time.
    """

    def __init__(self, root):
        """Build the UI and initialise state.

        Args:
            root: Tk root window
        """
        self.root = root
        self.root.title("Obstacle Car Racing GUI")
        self.video_dir = video_dir
        self.user = user

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_mode = "mjust"
        self.responses_file = os.path.join(responses_dir,
                                           f"responses{obst}_user{self.user}_{file_mode}_time{self.timestamp}.pkl")

        self.video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')],
                                  key=self.numerical_sort)

        self.current_video_index = 0
        self.responses = []
        self.stop_event = threading.Event()
        self.repeat_count = 0
        self.query_start_time = None

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.video_message_label = tk.Label(root, text="", font=('Helvetica', 12), fg='blue')
        self.video_message_label.pack()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack()

        self.start_button = tk.Button(self.control_frame, text="Start",
                                      command=self.load_new_query, font=('Helvetica', 12))
        self.start_button.pack(side=tk.LEFT)

        self.repeat_button = tk.Button(self.control_frame, text="Repeat Video",
                                       command=self.repeat_video, state=tk.DISABLED, font=('Helvetica', 12))
        self.repeat_button.pack(side=tk.RIGHT)

        self.setup_question_section()

        self.skip_button = tk.Button(root, text="Skip query",
                                     command=self.skip_query, state=tk.DISABLED, font=('Helvetica', 12))
        self.skip_button.pack()

        self.questions = ["Please give Justification", "Please give Preference"]
        self.current_responses = []

    def numerical_sort(self, filename):
        """Sort filenames by the first number they contain.

        Args:
            filename: Filename string

        Returns:
            Integer key for sorting
        """
        return int(re.findall(r'\d+', filename)[0])

    def setup_question_section(self):
        """Create justification and preference row.

        Notes:
            Justification buttons are enabled/disabled per-query according to the obstacle mode.
        """
        self.justification_frame = tk.Frame(self.root)
        self.justification_label = tk.Label(self.justification_frame,
                                            text="Please give Justification:", font=('Helvetica', 12))
        self.justification_label.pack(side=tk.LEFT)

        self.just_buttons = {}
        if OBST_MODE == 'chuckcar' or OBST_MODE == 'chuckccar':
            for label in ['Default', 'Grass', 'Chuckhole', 'Car']:
                b = tk.Button(self.justification_frame, text=label,
                              command=lambda l=label: self.answer_justification(l), font=('Helvetica', 12))
                b.pack(side=tk.LEFT)
                self.just_buttons[label] = b
        elif OBST_MODE == 'chuck' or OBST_MODE == 'chuckc':
            for label in ['Default', 'Grass', 'Chuckhole']:
                b = tk.Button(self.justification_frame, text=label,
                              command=lambda l=label: self.answer_justification(l), font=('Helvetica', 12))
                b.pack(side=tk.LEFT)
                self.just_buttons[label] = b

        self.justification_frame.pack()

        # PREFERENCE ROW
        self.preference_frame = tk.Frame(self.root)
        self.preference_label = tk.Label(self.preference_frame,
                                         text="Please give Preference:", fg="black", font=('Helvetica', 12))
        self.preference_label.pack(side=tk.LEFT)
        self.left_button = tk.Button(self.preference_frame, text="Left",
                                     command=lambda: self.answer_preference('Left'),
                                     font=('Helvetica', 12), state=tk.DISABLED)
        self.equally_button = tk.Button(self.preference_frame, text="Equally",
                                        command=lambda: self.answer_preference('Equally'),
                                        font=('Helvetica', 12), state=tk.DISABLED)
        self.right_button = tk.Button(self.preference_frame, text="Right",
                                      command=lambda: self.answer_preference('Right'),
                                      font=('Helvetica', 12), state=tk.DISABLED)
        self.right_button.pack(side=tk.RIGHT)
        self.equally_button.pack(side=tk.RIGHT)
        self.left_button.pack(side=tk.RIGHT)
        self.preference_frame.pack()

        # Pre-select justification
        self.selected_justification = 'Default'
        self.highlight_justification('Default')

        # Buttons disabled until video starts
        for b in self.just_buttons.values():
            b.config(state=tk.DISABLED)
        self.left_button.config(state=tk.DISABLED)
        self.equally_button.config(state=tk.DISABLED)
        self.right_button.config(state=tk.DISABLED)

    def answer_justification(self, label):
        """Select a justification label.

        Args:
            label: Selected justification name (e.g. 'Default','Grass','Chuckhole','Car')
        """
        self.selected_justification = label
        self.highlight_justification(label)

    def highlight_justification(self, selected_label):
        """Visually mark the selected justification button."""
        for label, button in self.just_buttons.items():
            if label == selected_label:
                button.config(relief=tk.SUNKEN)
            else:
                button.config(relief=tk.RAISED)

    def load_new_query(self):
        """Load the next video, reset timers, and enable the right buttons.

        Notes:
            First collect per-clip justification; then enable preference.
            If skip is pressed, record and move to the next query.
        """
        if self.current_video_index >= len(self.video_files):
            messagebox.showinfo("Info", "No more videos to load.")
            return

        self.current_video = os.path.join(self.video_dir, self.video_files[self.current_video_index])

        self.video_message_label.config(text=f"{os.path.basename(self.current_video)} loaded")

        self.current_video_index += 1

        self.query_start_time = time.time()
        self.repeat_count = 0

        self.selected_justification = 'Default'
        self.highlight_justification('Default')

        for b in self.just_buttons.values():
            b.config(state=tk.NORMAL)
        self.left_button.config(state=tk.NORMAL)
        self.right_button.config(state=tk.NORMAL)
        self.equally_button.config(state=tk.NORMAL)

        self.skip_button.config(state=tk.NORMAL)
        self.repeat_button.config(state=tk.NORMAL)

        self.current_responses = []

        self.stop_event.clear()
        threading.Thread(target=self.play_video, daemon=True).start()

        self.start_button.config(state=tk.DISABLED)

    def play_video(self):
        """Play the current video asynchronously and update the image label."""
        self.stop_event.set()
        self.stop_event.clear()
        reader = imageio.get_reader(self.current_video)
        for frame in reader:
            if self.stop_event.is_set():
                break
            img = Image.fromarray(frame)  # PIL
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
            time.sleep(0.04)

    def repeat_video(self):
        """Increment repeat counter and replay the current video."""
        self.repeat_count += 1
        self.stop_event.set()
        threading.Thread(target=self.play_video, daemon=True).start()

    def answer_preference(self, preference):
        """Record the preference and save the full response set.

        Args:
            preference: 'Left', 'Right', or 'Equally'
        """
        total_time = time.time() - self.query_start_time

        self.current_responses = [
            (self.questions[0], self.selected_justification),
            (self.questions[1], preference)
        ]

        video_filename = os.path.basename(self.current_video)
        self.responses.append({
            'video': video_filename,
            'responses': self.current_responses,
            'repeat_count': self.repeat_count,
            'total_time': total_time
        })
        with open(self.responses_file, 'wb') as f:
            pickle.dump(self.responses, f)

        # Reset UI and move to next
        self.left_button.config(state=tk.DISABLED)
        self.equally_button.config(state=tk.DISABLED)
        self.right_button.config(state=tk.DISABLED)
        for b in self.just_buttons.values():
            b.config(state=tk.DISABLED)

        self.stop_event.set()  # Stop video
        self.load_new_query()

    def skip_query(self):
        """Mark the current query as skipped, save, and move on."""
        total_time = time.time() - self.query_start_time
        video_filename = os.path.basename(self.current_video)
        self.responses.append({
            'video': video_filename,
            'responses': 'Skipped',
            'repeat_count': self.repeat_count,
            'total_time': total_time
        })
        with open(self.responses_file, 'wb') as f:
            pickle.dump(self.responses, f)
        self.stop_event.set()
        self.load_new_query()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarRacingGUI(root)
    root.mainloop()
