import os
import json
import numpy as np
from pathlib import Path

root_dir = Path(__file__).parent
image_dir = os.path.join(root_dir, "images")
face_encodings_dir = os.path.join(root_dir, "face_encodings")

if not os.path.isdir(face_encodings_dir):
    os.mkdir(face_encodings_dir)

""" Extract Face encoding """

import os
import face_recognition

known_face_encodings = []
known_face_names = {}

def updateEncoding(dir_name=image_dir):
    """ 
        Input: 
                Dir_path for image folder
        output: 
                Known Face Names
                know Face Encodings
     """
    count = 0
    for file in os.listdir(dir_name):
        if file.endswith == ".jpeg" or ".jpg":
            input_face_name = file.split('.')[0]

            known_face_names[count] = input_face_name
            count += 1
            
            input_face = face_recognition.load_image_file(os.path.join(dir_name, file))
            input_face_encoding = face_recognition.face_encodings(input_face)[0]
            known_face_encodings.append(input_face_encoding)

    print("Updating Face Encoding to System")
    np.save(f"{face_encodings_dir}/known_face_encodings.npy", known_face_encodings)

    with open(f"{face_encodings_dir}/known_face_names.json", "w") as file:
        json.dump(known_face_names, file)
