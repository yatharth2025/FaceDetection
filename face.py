import cv2
import face_recognition
import numpy as np
import json
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

# Database file to store face encodings
DB_FILE = "face_encodings.json"

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

    def load_encodings(self):
        """Load face encodings from JSON database"""
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                self.known_face_encodings = [np.array(enc) for enc in data["encodings"]]
                self.known_face_names = data["names"]

    def save_encodings(self):
        """Save face encodings to JSON database"""
        with open(DB_FILE, "w") as f:
            json.dump({"encodings": [enc.tolist() for enc in self.known_face_encodings], "names": self.known_face_names}, f)

    def register_face(self, frame):
        """Register a new face in real-time"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            name = self.prompt_for_name()
            if name:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                self.save_encodings()
                print(f"{name} registered successfully!")

    def delete_face(self):
        """Delete a registered face by name"""
        root = tk.Tk()
        root.withdraw()
        name = simpledialog.askstring("Delete Face", "Enter the name to delete:")
        root.destroy()

        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            self.save_encodings()
            messagebox.showinfo("Success", f"{name} has been deleted!")
        else:
            messagebox.showerror("Error", "Name not found!")

    def prompt_for_name(self):
        """Prompt the user for a name using a GUI pop-up"""
        root = tk.Tk()
        root.withdraw()
        name = simpledialog.askstring("Face Registration", "Enter your name:")
        root.destroy()
        return name

    def recognize_faces(self, frame):
        """Detect and recognize faces"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

            if best_match_index >= 0 and matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            else:
                if self.prompt_for_registration():
                    self.register_face(frame)
                    name = self.known_face_names[-1]  # Update with registered name

            face_names.append(name)

        return face_locations, face_names

    def prompt_for_registration(self):
        """Ask the user if they want to register their face"""
        root = tk.Tk()
        root.withdraw()
        response = messagebox.askyesno("Unknown Face", "Face not recognized. Do you want to register?")
        root.destroy()
        return response
        

# Initialize FaceRecognition
fr = FaceRecognition()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face_locations, face_names = fr.recognize_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)
    # if key == ord("r"):  # Press 'r' to manually register a new face
    #     fr.register_face(frame)
    if key == ord("d"):  # Press 'd' to delete a registered face
        fr.delete_face()
    elif key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()