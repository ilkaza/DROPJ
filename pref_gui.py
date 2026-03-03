"""GUI for preference reward model (DROP, DROPe, DROPJ) for Car Racing environment (with a single safety justification).

DROPJ shows paired-clip videos and asks:
1) Does the left car stay safe?
2) Does the right car stay safe?
3) Which clip do you prefer? (Left/Equally/Right)

DROP and DROPe asks only 3).

Flags 'use_justifications' and 'use_equal_in_plain_prefs' control the choice of the algorithm.
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


user = '31'
use_justifications = True  # True to use a safety justification (Yes/No questions) or False to use only preferences
use_equal_in_plain_prefs = False # True to use equalities in plain preferences as well

models_dir = os.path.join(os.getcwd(), 'models', 'carracing')
data_dir = os.path.join(os.getcwd(), 'data', 'carracing')
video_dir = os.path.join(data_dir, 'queries_s20')
responses_dir = os.path.join(data_dir, 'responses_s20')
os.makedirs(responses_dir, exist_ok=True)

class CarRacingGUI:
    """Tk GUI to label queries with safety answers and a preference.

    Loads .mp4 query clips, plays them in a loop, and records responses,
    repeat counts, and total feedback time to a pickle file.
    """

    def __init__(self, root):
        """Build the UI and initialise state.

        Args:
            root: Tk root window
        """
        self.root = root
        self.root.title("Car Racing GUI")
        self.video_dir = video_dir
        self.user = user
        self.use_justifications = use_justifications
        if self.use_justifications == False:
            self.use_equal_in_plain_prefs = use_equal_in_plain_prefs
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_mode = "just" if self.use_justifications else "nojust"
        if use_justifications==False:
            file_mode = file_mode + ("_noeq" if self.use_equal_in_plain_prefs==False else "_eq")
        self.responses_file = os.path.join(responses_dir,
                                           f"responses_user{self.user}_{file_mode}_time{self.timestamp}.pkl")

        self.video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')],
                                  key=self.numerical_sort)
        self.current_video_index = 0
        self.responses = []
        self.stop_event = threading.Event()
        self.repeat_count = 0
        self.query_start_time = None

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.video_message_label = tk.Label(root, text="", font=('Helvetica', 14), fg='blue')
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

        self.skip_button = tk.Button(root, text="Skip query", command=self.skip_query,
                                     state=tk.DISABLED, font=('Helvetica', 12))
        self.skip_button.pack()

        self.questions = ["Does the left car stay safe?", "Does the right car stay safe?", "Which clip do you prefer?"]
        self.current_responses = []
        self.start_time = None

    def numerical_sort(self, filename):
        """Sort filenames by the first number they contain.

        Args:
            filename: Filename string

        Returns:
            Integer key for sorting
        """
        return int(re.findall(r'\d+', filename)[0])

    def setup_question_section(self):
        """Create the three question rows and their buttons.

        Notes:
            Buttons are initially disabled and enabled per query based on flags.
        """
        self.question_frame_1 = tk.Frame(self.root)
        self.question_label_1 = tk.Label(self.question_frame_1,
                                         text="Does the left car stay safe?", fg="grey", font=('Helvetica', 12))
        self.question_label_1.pack(side=tk.LEFT)
        self.yes_button_1 = tk.Button(self.question_frame_1, text="Yes",
                                      command=lambda: self.answer_question(1, 'Yes'),
                                      font=('Helvetica', 12), state=tk.DISABLED)
        self.no_button_1 = tk.Button(self.question_frame_1, text="No",
                                     command=lambda: self.answer_question(1, 'No'),
                                     font=('Helvetica', 12), state=tk.DISABLED)
        self.no_button_1.pack(side=tk.RIGHT)
        self.yes_button_1.pack(side=tk.RIGHT)
        self.question_frame_1.pack()

        self.question_frame_2 = tk.Frame(self.root)
        self.question_label_2 = tk.Label(self.question_frame_2, text="Does the right car stay safe?",
                                         fg="grey", font=('Helvetica', 12))
        self.question_label_2.pack(side=tk.LEFT)
        self.yes_button_2 = tk.Button(self.question_frame_2, text="Yes",
                                      command=lambda: self.answer_question(2, 'Yes'),
                                      font=('Helvetica', 12), state=tk.DISABLED)
        self.no_button_2 = tk.Button(self.question_frame_2, text="No",
                                     command=lambda: self.answer_question(2, 'No'),
                                     font=('Helvetica', 12), state=tk.DISABLED)
        self.no_button_2.pack(side=tk.RIGHT)
        self.yes_button_2.pack(side=tk.RIGHT)
        self.question_frame_2.pack()

        self.question_frame_3 = tk.Frame(self.root)
        self.question_label_3 = tk.Label(self.question_frame_3, text="Which clip do you prefer?", fg="grey",
                                         font=('Helvetica', 12))
        self.question_label_3.pack(side=tk.LEFT)
        self.left_button = tk.Button(self.question_frame_3, text="Left",
                                     command=lambda: self.answer_preference('Left'),
                                     font=('Helvetica', 12), state=tk.DISABLED)
        self.equally_button = tk.Button(self.question_frame_3, text="Equally",
                                        command=lambda: self.answer_preference('Equally'),
                                        font=('Helvetica', 12), state=tk.DISABLED)
        self.right_button = tk.Button(self.question_frame_3, text="Right",
                                      command=lambda: self.answer_preference('Right'),
                                      font=('Helvetica', 12), state=tk.DISABLED)
        self.right_button.pack(side=tk.RIGHT)
        self.equally_button.pack(side=tk.RIGHT)
        self.left_button.pack(side=tk.RIGHT)
        self.question_frame_3.pack()

        self.yes_button_1.config(state=tk.DISABLED)
        self.no_button_1.config(state=tk.DISABLED)
        self.yes_button_2.config(state=tk.DISABLED)
        self.no_button_2.config(state=tk.DISABLED)
        self.left_button.config(state=tk.DISABLED)
        self.equally_button.config(state=tk.DISABLED)
        self.right_button.config(state=tk.DISABLED)

    def load_new_query(self):
        """Load the next video, reset timers, and enable the right buttons.

        Notes:
            If 'use_justifications' is True, ask left/right safety first and open preference
            only if both answers are 'Yes'; otherwise skip preference (inferred automatically).
            If 'use_justifications' is False, ask only preference (optionally enabling 'Equally').
        """
        if self.current_video_index >= len(self.video_files):
            messagebox.showinfo("Info", "No more videos to load.")
            return

        self.current_video = os.path.join(self.video_dir, self.video_files[self.current_video_index])
        self.video_message_label.config(text=f"{os.path.basename(self.current_video)} loaded")
        self.current_video_index += 1

        self.query_start_time = time.time()
        self.repeat_count = 0

        if self.use_justifications:
            self.question_label_1.config(fg="black")
            self.yes_button_1.config(state=tk.NORMAL)
            self.no_button_1.config(state=tk.NORMAL)

            self.question_label_2.config(fg="grey")
            self.yes_button_2.config(state=tk.DISABLED)
            self.no_button_2.config(state=tk.DISABLED)

            self.question_label_3.config(fg="black")
            self.left_button.config(state=tk.DISABLED)
            self.equally_button.config(state=tk.DISABLED)
            self.right_button.config(state=tk.DISABLED)
        else:
            self.question_label_1.config(fg="grey")
            self.yes_button_1.config(state=tk.DISABLED)
            self.no_button_1.config(state=tk.DISABLED)

            self.question_label_3.config(fg="grey")
            self.yes_button_2.config(state=tk.DISABLED)
            self.no_button_2.config(state=tk.DISABLED)

            self.question_label_3.config(fg="black")
            self.left_button.config(state=tk.NORMAL)
            self.right_button.config(state=tk.NORMAL)
            if self.use_equal_in_plain_prefs:
                self.equally_button.config(state=tk.NORMAL)

        self.skip_button.config(state=tk.NORMAL)
        self.repeat_button.config(state=tk.NORMAL)

        self.current_responses = []
        self.start_time = time.time()

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
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
            time.sleep(0.04)

    def repeat_video(self):
        """Increment repeat counter and replay the current video."""
        self.repeat_count += 1
        self.stop_event.set()
        threading.Thread(target=self.play_video, daemon=True).start()

    def answer_question(self, question_number, answer):
        """Record a safety answer and advance the flow.

        Args:
            question_number: 1 for left safety, 2 for right safety
            answer: 'Yes' or 'No'

        Notes:
            If answers differ or both are 'No', save and move to the next query.
            If both are 'Yes', enable the preference buttons.
        """
        end_time = time.time()
        self.current_responses.append((self.questions[question_number - 1], answer, end_time - self.start_time))
        self.start_time = time.time()

        if question_number == 1:
            self.question_label_1.config(fg="grey")
            self.yes_button_1.config(state=tk.DISABLED)
            self.no_button_1.config(state=tk.DISABLED)
            self.question_label_2.config(fg="black")
            self.yes_button_2.config(state=tk.NORMAL)
            self.no_button_2.config(state=tk.NORMAL)
        elif question_number == 2:
            self.question_label_2.config(fg="grey")
            self.yes_button_2.config(state=tk.DISABLED)
            self.no_button_2.config(state=tk.DISABLED)
            if self.current_responses[0][1] != self.current_responses[1][1]:
                self.save_responses()
                self.stop_event.set()
                self.load_new_query()
            else:
                if self.current_responses[0][1]=='Yes':
                    self.question_label_3.config(fg="black")
                    self.left_button.config(state=tk.NORMAL)
                    self.right_button.config(state=tk.NORMAL)
                    self.equally_button.config(state=tk.NORMAL)
                else:
                    self.save_responses()
                    self.stop_event.set()
                    self.load_new_query()

    def answer_preference(self, preference):
        """Record the preference and save the full response set.

        Args:
            preference: 'Left', 'Right', or 'Equally'
        """
        end_time = time.time()
        self.current_responses.append((self.questions[2], preference, end_time - self.start_time))
        self.save_responses()
        self.question_label_3.config(fg="grey")
        self.left_button.config(state=tk.DISABLED)
        self.equally_button.config(state=tk.DISABLED)
        self.right_button.config(state=tk.DISABLED)
        self.stop_event.set()
        self.load_new_query()

    def save_responses(self):
        """Append current responses and write the pickle file.

        Notes:
            Stores {'video', 'responses', 'repeat_count', 'total_time'}.
        """
        total_time = time.time() - self.query_start_time  # total time for one whole query
        video_filename = os.path.basename(self.current_video)
        self.responses.append({
            'video': video_filename,  # Store only the filename
            'responses': self.current_responses,
            'repeat_count': self.repeat_count,
            'total_time': total_time
        })
        with open(self.responses_file, 'wb') as f:
            pickle.dump(self.responses, f)

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
        self.stop_event.set()  # Stop the current video playback
        self.load_new_query()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarRacingGUI(root)
    root.mainloop()
